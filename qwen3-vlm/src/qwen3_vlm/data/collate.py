from typing import List

from PIL import Image
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F


def expand_image_tokens(text, image_token, im_start, image_patch, im_end, num_image_tokens):
    parts = text.split(image_token)
    if len(parts) == 1:
        return text
    image_block = " ".join([im_start] + [image_patch] * num_image_tokens + [im_end])
    return image_block.join(parts)


def load_image(image):
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)}")


def _flatten_tokenizer_ids(enc):
    ids = enc["input_ids"]
    truncated = False
    if ids and isinstance(ids[0], list):
        truncated = len(ids) > 1
        ids = ids[0]
    truncated = truncated or bool(enc.get("overflowing_tokens")) or enc.get(
        "num_truncated_tokens", 0
    ) > 0
    return ids, truncated


class VLMDataCollator:
    def __init__(
        self,
        tokenizer,
        image_processor,
        image_token,
        image_patch_token,
        im_start_token,
        im_end_token,
        num_image_tokens,
        max_seq_len,
        pad_to_multiple_of=None,
    ):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_token = image_token
        self.image_patch_token = image_patch_token
        self.im_start_token = im_start_token
        self.im_end_token = im_end_token
        self.num_image_tokens = num_image_tokens
        self.max_seq_len = max_seq_len
        self.pad_to_multiple_of = pad_to_multiple_of

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _encode_text(self, text):
        expanded = expand_image_tokens(
            text,
            self.image_token,
            self.im_start_token,
            self.image_patch_token,
            self.im_end_token,
            self.num_image_tokens,
        )
        enc = self.tokenizer(
            expanded,
            add_special_tokens=False,
            padding=False,
            truncation=True,
            max_length=self.max_seq_len,
            return_overflowing_tokens=True,
        )
        ids, truncated = _flatten_tokenizer_ids(enc)
        return ids, truncated

    def _build_chat_tokens(self, messages):
        if not hasattr(self.tokenizer, "apply_chat_template"):
            raise ValueError("Tokenizer does not support chat templates")
        render = lambda msgs: self.tokenizer.apply_chat_template(  # noqa: E731
            msgs, tokenize=False, add_generation_prompt=False
        )
        full_text = render(messages)
        input_ids, was_truncated = self._encode_text(full_text)
        labels = [-100] * len(input_ids)
        for idx, message in enumerate(messages):
            if message.get("role") != "assistant":
                continue
            prefix_text = render(messages[:idx])
            full_text = render(messages[: idx + 1])
            prefix_ids, prefix_truncated = self._encode_text(prefix_text)
            full_ids, full_truncated = self._encode_text(full_text)
            was_truncated = was_truncated or prefix_truncated or full_truncated
            start = min(len(prefix_ids), len(input_ids))
            end = min(len(full_ids), len(input_ids))
            for pos in range(start, end):
                labels[pos] = input_ids[pos]
        return input_ids, labels, was_truncated

    def __call__(self, batch):
        input_ids_list = []
        labels_list = []
        attention_mask_list = []
        image_counts = []
        truncation_flags = []
        label_token_counts = []
        buckets = []
        label_lens = []
        sample_ids = []
        image_relpaths = []
        images_flat: List[Image.Image] = []

        for sample in batch:
            messages = sample.get("messages")
            prompt = sample.get("prompt")
            answer = sample.get("answer")
            images = sample.get("images", [])

            if messages is not None:
                num_placeholders = sum(
                    msg.get("content", "").count(self.image_token) for msg in messages
                )
                if num_placeholders != len(images):
                    raise ValueError(
                        f"Image placeholder mismatch: {num_placeholders} tokens vs {len(images)} images"
                    )
                input_ids, labels, was_truncated = self._build_chat_tokens(messages)
            else:
                if prompt is None or answer is None:
                    raise ValueError("Sample missing prompt/answer or messages")
                num_placeholders = prompt.count(self.image_token) + answer.count(self.image_token)
                if num_placeholders != len(images):
                    raise ValueError(
                        f"Image placeholder mismatch: {num_placeholders} tokens vs {len(images)} images"
                    )

                expanded_prompt = expand_image_tokens(
                    prompt,
                    self.image_token,
                    self.im_start_token,
                    self.image_patch_token,
                    self.im_end_token,
                    self.num_image_tokens,
                )
                expanded_answer = expand_image_tokens(
                    answer,
                    self.image_token,
                    self.im_start_token,
                    self.image_patch_token,
                    self.im_end_token,
                    self.num_image_tokens,
                )

                prompt_enc = self.tokenizer(
                    expanded_prompt,
                    add_special_tokens=False,
                    padding=False,
                    truncation=True,
                    max_length=self.max_seq_len,
                    return_overflowing_tokens=True,
                )
                prompt_ids, prompt_truncated = _flatten_tokenizer_ids(prompt_enc)
                answer_enc = self.tokenizer(
                    expanded_answer,
                    add_special_tokens=False,
                    padding=False,
                    truncation=True,
                    max_length=self.max_seq_len,
                    return_overflowing_tokens=True,
                )
                answer_ids, answer_truncated = _flatten_tokenizer_ids(answer_enc)
                if self.tokenizer.eos_token_id is not None:
                    answer_ids = answer_ids + [self.tokenizer.eos_token_id]

                allowed_answer_len = max(0, self.max_seq_len - len(prompt_ids))
                combined_truncated = len(answer_ids) > allowed_answer_len
                if combined_truncated:
                    answer_ids = answer_ids[:allowed_answer_len]

                was_truncated = prompt_truncated or answer_truncated or combined_truncated

                input_ids = prompt_ids + answer_ids
                labels = [-100] * len(prompt_ids) + answer_ids
            attention_mask = [1] * len(input_ids)

            input_ids_list.append(torch.tensor(input_ids, dtype=torch.long))
            labels_list.append(torch.tensor(labels, dtype=torch.long))
            attention_mask_list.append(torch.tensor(attention_mask, dtype=torch.long))

            image_counts.append(len(images))
            truncation_flags.append(was_truncated)
            label_token_counts.append(sum(1 for v in labels if v != -100))
            buckets.append(sample.get("bucket"))
            label_len = sample.get("label_len")
            if label_len is None:
                label_len = -1
            label_lens.append(int(label_len))
            sample_id = sample.get("id")
            if sample_id is None:
                sample_id = -1
            sample_ids.append(int(sample_id))
            image_relpaths.append(sample.get("image_relpath"))
            for img in images:
                images_flat.append(load_image(img))

        input_ids_padded = pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels_padded = pad_sequence(labels_list, batch_first=True, padding_value=-100)
        attention_mask_padded = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
        if self.pad_to_multiple_of:
            multiple = int(self.pad_to_multiple_of)
            if multiple > 0:
                max_len = input_ids_padded.size(1)
                pad_len = (-max_len) % multiple
                if pad_len:
                    input_ids_padded = F.pad(
                        input_ids_padded,
                        (0, pad_len),
                        value=self.tokenizer.pad_token_id,
                    )
                    labels_padded = F.pad(labels_padded, (0, pad_len), value=-100)
                    attention_mask_padded = F.pad(attention_mask_padded, (0, pad_len), value=0)

        if images_flat:
            pixel_values = self.image_processor(images=images_flat, return_tensors="pt")["pixel_values"]
        else:
            pixel_values = None

        batch_out = {
            "input_ids": input_ids_padded,
            "labels": labels_padded,
            "attention_mask": attention_mask_padded,
            "pixel_values": pixel_values,
            "image_counts": image_counts,
            "truncation_flags": torch.tensor(truncation_flags, dtype=torch.bool),
            "label_token_counts": torch.tensor(label_token_counts, dtype=torch.long),
        }
        if any(bucket is not None for bucket in buckets):
            batch_out["buckets"] = buckets
        if any(label_len >= 0 for label_len in label_lens):
            batch_out["label_lens"] = torch.tensor(label_lens, dtype=torch.long)
        if any(sample_id >= 0 for sample_id in sample_ids):
            batch_out["sample_ids"] = sample_ids
        if any(path is not None for path in image_relpaths):
            batch_out["image_relpaths"] = image_relpaths
        return batch_out
