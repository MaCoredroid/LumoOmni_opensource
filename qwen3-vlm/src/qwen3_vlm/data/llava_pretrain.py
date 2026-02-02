import json
from pathlib import Path

from torch.utils.data import Dataset


class LlavaPretrainDataset(Dataset):
    def __init__(self, json_path, image_root, prompt, image_token):
        self.json_path = Path(json_path)
        self.image_root = Path(image_root)
        self.prompt = prompt
        self.image_token = image_token
        self.max_missing_retries = 50

        with self.json_path.open("r", encoding="utf-8") as f:
            raw_data = json.load(f)
        cleaned = []
        dropped = 0
        for item in raw_data:
            image_rel, prompt_text, answer_text = self._extract_fields(item)
            if image_rel is None or not prompt_text or not answer_text:
                dropped += 1
                continue
            cleaned.append(item)
        self.data = cleaned
        if dropped:
            print(f"[data] dropped {dropped} items with missing image/caption")

    def _extract_fields(self, item):
        image_rel = item.get("image") or item.get("image_path") or item.get("image_id")
        prompt_text = None
        answer_text = None
        conversations = item.get("conversations")
        if isinstance(conversations, list):
            for turn in conversations:
                role = turn.get("from") or turn.get("role")
                text = turn.get("value") or turn.get("text")
                if prompt_text is None and role in {"human", "user"}:
                    prompt_text = text
                elif answer_text is None and role in {"gpt", "assistant"}:
                    answer_text = text

        if prompt_text is None:
            prompt_text = item.get("prompt") or self.prompt
        if answer_text is None:
            answer_text = item.get("caption") or item.get("text") or item.get("description")

        return image_rel, prompt_text, answer_text

    def get_metadata(self, idx):
        item = self.data[idx]
        image_rel, prompt_text, answer_text = self._extract_fields(item)
        if image_rel is None or not prompt_text or not answer_text:
            return None
        if self.image_token not in prompt_text:
            prompt_text = f"{prompt_text} {self.image_token}"
        image_rel = str(image_rel)
        image_path = self.image_root / image_rel
        return {
            "id": idx,
            "image_relpath": image_rel,
            "image_path": str(image_path),
            "prompt": prompt_text,
            "answer": answer_text,
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = None
        image_rel = None
        prompt = None
        caption = None
        image_path = None
        for attempt in range(self.max_missing_retries):
            item = self.data[(idx + attempt) % len(self.data)]
            image_rel, prompt, caption = self._extract_fields(item)
            if image_rel is None or caption is None or caption == "":
                continue
            image_path = self.image_root / str(image_rel)
            if image_path.exists():
                break
        else:
            raise FileNotFoundError("No valid image found after retries")

        if image_path is None or prompt is None or caption is None:
            raise ValueError("Missing image or caption in LLaVA pretrain item")
        if self.image_token not in prompt:
            prompt = f"{prompt} {self.image_token}"

        return {
            "prompt": prompt,
            "answer": caption,
            "images": [str(image_path)],
        }
