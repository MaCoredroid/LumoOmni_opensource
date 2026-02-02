import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict

import torch
from transformers import AutoModelForCausalLM

from stage3_uti.data.tokenized_jsonl import TokenizedJsonlDataset, collate_tokenized
from stage3_uti.tokenization.token_space import TokenSpace
from stage3_uti.utils.train_utils import _load_trainable_rows, _resize_and_init_embeddings, resolve_pad_id


def _load_model(model_path: Path, token_space: TokenSpace, device: torch.device, device_map: str | None):
    load_kwargs: Dict[str, object] = {
        "torch_dtype": torch.bfloat16 if device.type == "cuda" else torch.float32,
        "low_cpu_mem_usage": True,
    }
    if device_map:
        load_kwargs["device_map"] = device_map

    trainable_rows_path = model_path / "trainable_rows.pt"
    if trainable_rows_path.exists():
        base_llm = ""
        meta_path = model_path / "trainable_rows.json"
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            base_llm = str(meta.get("base_llm", ""))
        if not base_llm:
            payload = torch.load(trainable_rows_path, map_location="cpu")
            base_llm = str(payload.get("base_llm", ""))
        if not base_llm:
            raise ValueError("trainable_rows checkpoint missing base_llm metadata")
        model = AutoModelForCausalLM.from_pretrained(base_llm, **load_kwargs)
        _resize_and_init_embeddings(
            model,
            text_vocab_size=int(token_space.text_vocab_size),
            vocab_size_total=int(token_space.vocab_size_total),
            init_new_rows=True,
        )
        model.config.pad_token_id = resolve_pad_id(token_space)
        _load_trainable_rows(
            model,
            model_path,
            row_start=int(token_space.text_vocab_size),
            row_end=int(token_space.vocab_size_total),
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(str(model_path), **load_kwargs)
    if not getattr(model, "hf_device_map", None):
        model.to(device)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--token-space", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--truncation-side", default="left")
    parser.add_argument("--device-map", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    token_space = TokenSpace.load_json(args.token_space)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_map = args.device_map or None
    model = _load_model(Path(args.model), token_space, device, device_map)

    dataset = TokenizedJsonlDataset(args.jsonl)

    def _collate(batch):
        return collate_tokenized(
            batch,
            pad_id=resolve_pad_id(token_space),
            max_seq_len=args.max_seq_len,
            truncation_side=args.truncation_side,
        )

    batch_size = int(args.batch_size)
    totals = defaultdict(float)
    counts = defaultdict(int)
    with torch.no_grad():
        batch_items = []
        for item in dataset:
            batch_items.append(item)
            if len(batch_items) < batch_size:
                continue
            batch = _collate(batch_items)
            input_ids = torch.tensor(batch["input_ids"], dtype=torch.long, device=device)
            attention_mask = torch.tensor(batch["attention_mask"], dtype=torch.long, device=device)
            labels = torch.tensor(batch["labels"], dtype=torch.long, device=device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = float(outputs.loss.detach().cpu())
            totals["all"] += loss
            counts["all"] += 1
            if batch_size == 1:
                task = str(batch_items[0].get("task", "all"))
                totals[task] += loss
                counts[task] += 1
            batch_items = []
        if batch_items:
            batch = _collate(batch_items)
            input_ids = torch.tensor(batch["input_ids"], dtype=torch.long, device=device)
            attention_mask = torch.tensor(batch["attention_mask"], dtype=torch.long, device=device)
            labels = torch.tensor(batch["labels"], dtype=torch.long, device=device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = float(outputs.loss.detach().cpu())
            totals["all"] += loss
            counts["all"] += 1
            if batch_size == 1:
                task = str(batch_items[0].get("task", "all"))
                totals[task] += loss
                counts[task] += 1

    report = {
        "loss_mean": {k: totals[k] / counts[k] for k in counts},
        "counts": dict(counts),
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
