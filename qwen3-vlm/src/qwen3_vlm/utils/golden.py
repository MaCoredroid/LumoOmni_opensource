import json
import hashlib
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

from qwen3_vlm.data.collate import expand_image_tokens, load_image
from qwen3_vlm.data.splits import collect_label_metadata, select_candidate_indices, stratified_sample


def ensure_golden_set(dataset, tokenizer, cfg):
    data_cfg = cfg["data"]
    golden_path = data_cfg.get("golden_set_path")
    if not golden_path:
        return None
    golden_path = Path(golden_path)
    if golden_path.exists():
        return golden_path

    golden_seed = int(data_cfg.get("golden_seed", cfg["train"]["seed"]))
    golden_size = int(data_cfg.get("golden_size", 64))
    candidate_indices = select_candidate_indices(
        len(dataset), int(cfg["train"]["seed"]), data_cfg.get("max_samples")
    )
    metadata, skipped_missing, bucket_names = collect_label_metadata(
        dataset, candidate_indices, tokenizer, cfg
    )
    if skipped_missing:
        print(f"[golden] skipped_missing={skipped_missing}")

    selected = stratified_sample(metadata, golden_size, golden_seed, bucket_names)
    golden_path.parent.mkdir(parents=True, exist_ok=True)
    with golden_path.open("w", encoding="utf-8") as f:
        for item in selected:
            record = {
                "id": item["id"],
                "image_relpath": item.get("image_relpath") or item.get("image_path"),
                "gt_text": item.get("answer"),
                "bucket": item.get("bucket"),
                "label_len": item.get("label_len"),
            }
            f.write(json.dumps(record) + "\n")

    return golden_path


def load_golden_set(golden_path):
    entries = []
    with Path(golden_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def _build_prompt_batch(samples, tokenizer, image_processor, tokens_cfg, num_image_tokens, max_seq_len):
    input_ids_list = []
    attention_mask_list = []
    image_counts = []
    images_flat = []
    prompt_lens = []

    for sample in samples:
        prompt = sample.get("prompt")
        messages = sample.get("messages")
        images = sample.get("images", [])

        if messages is not None:
            if not hasattr(tokenizer, "apply_chat_template"):
                raise ValueError("Tokenizer does not support chat templates")
            last_assistant = -1
            for idx, msg in enumerate(messages):
                if msg.get("role") == "assistant":
                    last_assistant = idx
            if last_assistant >= 0:
                prompt_messages = messages[:last_assistant]
            else:
                prompt_messages = messages
            prompt = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        if prompt is None:
            continue

        expanded_prompt = expand_image_tokens(
            prompt,
            tokens_cfg["image_token"],
            tokens_cfg["im_start_token"],
            tokens_cfg["image_patch_token"],
            tokens_cfg["im_end_token"],
            num_image_tokens,
        )
        enc = tokenizer(
            expanded_prompt,
            add_special_tokens=False,
            padding=False,
            truncation=True,
            max_length=max_seq_len,
        )
        input_ids = enc["input_ids"]
        prompt_lens.append(len(input_ids))
        input_ids_list.append(torch.tensor(input_ids, dtype=torch.long))
        attention_mask_list.append(torch.ones(len(input_ids), dtype=torch.long))

        image_counts.append(len(images))
        for img in images:
            images_flat.append(load_image(img))

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids_padded = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
    attention_mask_padded = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
    if images_flat:
        pixel_values = image_processor(images=images_flat, return_tensors="pt")["pixel_values"]
    else:
        pixel_values = None

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded,
        "pixel_values": pixel_values,
        "image_counts": image_counts,
        "prompt_lens": prompt_lens,
    }


def run_golden_generation(
    model,
    tokenizer,
    image_processor,
    dataset,
    golden_entries,
    cfg,
    device,
    global_step,
    output_dir,
):
    if not golden_entries:
        return

    tokens_cfg = cfg["tokens"]
    num_image_tokens = int(cfg["model"]["resampler"]["num_latents"])
    max_seq_len = int(cfg["train"]["max_seq_len"])

    batch_size = int(cfg["train"].get("golden_batch_size", 8))
    do_sample = bool(cfg["train"].get("golden_do_sample", False))
    temperature = float(cfg["train"].get("golden_temperature", 0.2))
    top_p = float(cfg["train"].get("golden_top_p", 0.9))
    max_new_tokens = int(cfg["train"].get("golden_max_new_tokens", 64))
    repetition_penalty = float(cfg["train"].get("golden_repetition_penalty", 1.0))
    golden_seed = int(cfg["data"].get("golden_seed", cfg["train"]["seed"]))
    precision = cfg["train"].get("precision", "fp32")
    use_autocast = device.type == "cuda" and precision in {"bf16", "fp16"}
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    model.eval()

    qual_dir = Path(output_dir) / "qual"
    qual_dir.mkdir(parents=True, exist_ok=True)
    out_path = qual_dir / f"step_{global_step}_golden64.jsonl"

    results = []
    if do_sample and temperature > 0:
        torch.manual_seed(golden_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(golden_seed)

    for start in range(0, len(golden_entries), batch_size):
        batch_entries = golden_entries[start : start + batch_size]
        samples = []
        for entry in batch_entries:
            meta = dataset.get_metadata(int(entry["id"])) if hasattr(dataset, "get_metadata") else None
            if meta is None:
                continue
            image_rel = entry.get("image_relpath")
            image_path = None
            if image_rel and hasattr(dataset, "image_root"):
                image_path = str(Path(dataset.image_root) / str(image_rel))
            elif meta.get("image_path"):
                image_path = meta["image_path"]
            images = [image_path] if image_path else []
            messages = meta.get("messages")
            if messages is not None:
                placeholders = sum(
                    msg.get("content", "").count(tokens_cfg["image_token"]) for msg in messages
                )
                if placeholders != len(images):
                    continue
                samples.append(
                    {
                        "messages": messages,
                        "images": images,
                        "id": entry.get("id"),
                        "bucket": entry.get("bucket"),
                        "label_len": entry.get("label_len"),
                        "gt_text": entry.get("gt_text"),
                        "image_relpath": entry.get("image_relpath"),
                    }
                )
            else:
                if meta.get("prompt", "").count(tokens_cfg["image_token"]) != len(images):
                    continue
                samples.append(
                    {
                        "prompt": meta.get("prompt"),
                        "images": images,
                        "id": entry.get("id"),
                        "bucket": entry.get("bucket"),
                        "label_len": entry.get("label_len"),
                        "gt_text": entry.get("gt_text"),
                        "image_relpath": entry.get("image_relpath"),
                    }
                )

        if not samples:
            continue

        batch = _build_prompt_batch(
            samples,
            tokenizer,
            image_processor,
            tokens_cfg,
            num_image_tokens,
            max_seq_len,
        )
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        pixel_values = batch["pixel_values"]
        if pixel_values is not None:
            pixel_values = pixel_values.to(device, non_blocking=True)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_autocast):
                gen_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_counts=batch["image_counts"],
                    **gen_kwargs,
                )

        for i, sample in enumerate(samples):
            prompt_len = batch["prompt_lens"][i]
            seq = gen_ids[i]
            if seq.numel() > prompt_len:
                gen_slice = seq[prompt_len:]
            else:
                gen_slice = seq
            pred_text = tokenizer.decode(gen_slice, skip_special_tokens=True)
            pred_hash = hashlib.sha1(pred_text.encode("utf-8")).hexdigest()[:8]
            results.append(
                {
                    "id": sample.get("id"),
                    "bucket": sample.get("bucket"),
                    "gt_text": sample.get("gt_text"),
                    "pred_text": pred_text,
                    "image_relpath": sample.get("image_relpath"),
                    "label_len": sample.get("label_len"),
                    "pred_tokens_len": int(gen_slice.numel()),
                    "pred_hash": pred_hash,
                }
            )

    with out_path.open("w", encoding="utf-8") as f:
        for record in results:
            f.write(json.dumps(record) + "\n")

    return out_path
