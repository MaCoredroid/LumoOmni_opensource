import json
from pathlib import Path

from torch.utils.data import Dataset


class LlavaInstructDataset(Dataset):
    def __init__(self, json_path, image_root, image_token):
        self.json_path = Path(json_path)
        self.image_root = Path(image_root)
        self.image_token = image_token
        self.max_missing_retries = 50

        with self.json_path.open("r", encoding="utf-8") as f:
            raw_data = json.load(f)

        cleaned = []
        dropped = 0
        for item in raw_data:
            record = self._normalize_item(item)
            if record is None:
                dropped += 1
                continue
            cleaned.append(record)
        self.data = cleaned
        if dropped:
            print(f"[data] dropped {dropped} items with missing image/messages")

    def _normalize_role(self, role):
        if role in {"human", "user"}:
            return "user"
        if role in {"gpt", "assistant"}:
            return "assistant"
        if role == "system":
            return "system"
        return None

    def _ensure_single_image(self, messages):
        total = 0
        for msg in messages:
            count = msg["content"].count(self.image_token)
            if count and msg["role"] != "user":
                return None
            total += count

        if total == 0:
            for msg in messages:
                if msg["role"] == "user":
                    if msg["content"]:
                        msg["content"] = f"{self.image_token}\n{msg['content']}"
                    else:
                        msg["content"] = self.image_token
                    total = 1
                    break

        if total != 1:
            return None
        return messages

    def _normalize_item(self, item):
        image_rel = item.get("image") or item.get("image_path") or item.get("image_id")
        if image_rel is None:
            return None
        if isinstance(image_rel, list):
            if len(image_rel) != 1:
                return None
            image_rel = image_rel[0]
        image_rel = str(image_rel)

        conversations = item.get("conversations") or item.get("messages")
        if not isinstance(conversations, list):
            return None

        messages = []
        for turn in conversations:
            role = turn.get("from") or turn.get("role")
            content = turn.get("value") or turn.get("text") or turn.get("content")
            if not content:
                continue
            norm_role = self._normalize_role(role)
            if norm_role is None:
                continue
            messages.append({"role": norm_role, "content": content.strip()})

        if not messages:
            return None
        if not any(m["role"] == "assistant" and m["content"] for m in messages):
            return None

        messages = self._ensure_single_image(messages)
        if messages is None:
            return None

        return {"image_rel": image_rel, "messages": messages}

    def _build_answer_text(self, messages):
        parts = [m["content"].strip() for m in messages if m["role"] == "assistant"]
        return "\n".join([p for p in parts if p])

    def get_metadata(self, idx):
        record = self.data[idx]
        image_path = self.image_root / record["image_rel"]
        answer = self._build_answer_text(record["messages"])
        if not answer:
            return None
        return {
            "id": idx,
            "image_relpath": record["image_rel"],
            "image_path": str(image_path),
            "messages": record["messages"],
            "answer": answer,
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        for attempt in range(self.max_missing_retries):
            base_idx = (idx + attempt) % len(self.data)
            record = self.data[base_idx]
            image_path = self.image_root / record["image_rel"]
            if image_path.exists():
                return {
                    "messages": record["messages"],
                    "images": [str(image_path)],
                    "id": base_idx,
                    "image_relpath": record["image_rel"],
                }
        raise FileNotFoundError("No valid image found after retries")
