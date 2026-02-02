import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM

from stage3_uti.tokenization.token_space import TokenSpace
from stage3_uti.utils.train_utils import _load_trainable_rows, _resize_and_init_embeddings, resolve_pad_id


def _load_model(
    checkpoint: Path,
    token_space: TokenSpace,
    device: torch.device,
    device_map: Optional[str],
    lora_path: Optional[Path],
):
    trainable_rows_path = checkpoint / "trainable_rows.pt"
    if not trainable_rows_path.exists():
        raise FileNotFoundError(trainable_rows_path)

    base_llm = ""
    meta_path = checkpoint / "trainable_rows.json"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        base_llm = str(meta.get("base_llm", ""))
    if not base_llm:
        payload = torch.load(trainable_rows_path, map_location="cpu")
        base_llm = str(payload.get("base_llm", ""))
    if not base_llm:
        raise ValueError("trainable_rows checkpoint missing base_llm metadata")

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    load_kwargs: Dict[str, object] = {"torch_dtype": dtype, "low_cpu_mem_usage": True}
    if device_map:
        load_kwargs["device_map"] = device_map
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
        checkpoint,
        row_start=int(token_space.text_vocab_size),
        row_end=int(token_space.vocab_size_total),
    )
    if lora_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(lora_path))
    if not getattr(model, "hf_device_map", None):
        model.to(device)
    model.eval()
    return model


def _find_span(input_ids: List[int], start_tok: int, end_tok: int) -> Optional[Tuple[int, int]]:
    try:
        start_idx = input_ids.index(start_tok)
        end_idx = input_ids.index(end_tok, start_idx + 1)
    except ValueError:
        return None
    return start_idx, end_idx


def _rebuild_with_text(input_ids: List[int], text_start: int, text_end: int, new_text: List[int]) -> List[int]:
    span = _find_span(input_ids, text_start, text_end)
    if span is None:
        return input_ids
    start_idx, end_idx = span
    return input_ids[: start_idx + 1] + new_text + input_ids[end_idx:]


def _build_labels(input_ids: List[int], target_start: int, target_end: int) -> List[int]:
    labels = [-100] * len(input_ids)
    for i in range(target_start, target_end + 1):
        if 0 <= i < len(labels):
            labels[i] = input_ids[i]
    return labels


def _loss_for_example(model, input_ids: List[int], labels: List[int], device: torch.device) -> float:
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    label_tensor = torch.tensor([labels], dtype=torch.long, device=device)
    with torch.no_grad():
        outputs = model(input_ids=input_tensor, labels=label_tensor)
        logits = outputs.logits
    if logits.size(1) < 2:
        return float("nan")
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = label_tensor[:, 1:].contiguous()
    loss_mask = shift_labels != -100
    if not bool(loss_mask.any().item()):
        return float("nan")
    vocab = shift_logits.size(-1)
    loss_per_token = F.cross_entropy(
        shift_logits.view(-1, vocab),
        shift_labels.view(-1),
        reduction="none",
    ).view(shift_labels.size())
    token_sums = (loss_per_token * loss_mask).sum(dim=1)
    token_counts = loss_mask.sum(dim=1)
    return float((token_sums / token_counts).item())


def _run_ablation(
    items: List[Dict[str, object]],
    *,
    task: str,
    token_space: TokenSpace,
    model,
    rng: random.Random,
    device: torch.device,
) -> Dict[str, float]:
    special = token_space.special_tokens
    text_start = int(special["<|text_start|>"])
    text_end = int(special["<|text_end|>"])
    if task == "t2a":
        tgt_start = int(special["<|aud_start|>"])
        tgt_end = int(special["<|aud_end|>"])
    else:
        tgt_start = int(special["<|img_start|>"])
        tgt_end = int(special["<|img_end|>"])

    losses = {"correct": [], "shuffle": [], "empty": []}

    for item in items:
        input_ids = list(item["input_ids"])
        text_span = _find_span(input_ids, text_start, text_end)
        target_span = _find_span(input_ids, tgt_start, tgt_end)
        if text_span is None or target_span is None:
            continue
        t_start, t_end = text_span
        prompt_tokens = input_ids[t_start + 1 : t_end]

        # correct
        correct_ids = input_ids
        correct_labels = _build_labels(correct_ids, *target_span)
        losses["correct"].append(_loss_for_example(model, correct_ids, correct_labels, device))

        # shuffled
        shuffled = list(prompt_tokens)
        rng.shuffle(shuffled)
        shuffled_ids = _rebuild_with_text(input_ids, text_start, text_end, shuffled)
        target_span_shuf = _find_span(shuffled_ids, tgt_start, tgt_end)
        if target_span_shuf is not None:
            shuffled_labels = _build_labels(shuffled_ids, *target_span_shuf)
            losses["shuffle"].append(_loss_for_example(model, shuffled_ids, shuffled_labels, device))

        # empty prompt
        empty_ids = _rebuild_with_text(input_ids, text_start, text_end, [])
        target_span_empty = _find_span(empty_ids, tgt_start, tgt_end)
        if target_span_empty is not None:
            empty_labels = _build_labels(empty_ids, *target_span_empty)
            losses["empty"].append(_loss_for_example(model, empty_ids, empty_labels, device))

    def _mean(vals: List[float]) -> float:
        return sum(vals) / len(vals) if vals else float("nan")

    summary: Dict[str, float] = {
        "loss_correct_mean": _mean(losses["correct"]),
        "loss_shuffle_mean": _mean(losses["shuffle"]),
        "loss_empty_mean": _mean(losses["empty"]),
        "n_examples": float(len(losses["correct"])),
    }
    if losses["shuffle"]:
        summary["win_rate_shuffle"] = sum(
            1 for lc, ls in zip(losses["correct"], losses["shuffle"]) if lc < ls
        ) / len(losses["correct"])
    if losses["empty"]:
        summary["win_rate_empty"] = sum(
            1 for lc, le in zip(losses["correct"], losses["empty"]) if lc < le
        ) / len(losses["correct"])
    return summary


def _load_jsonl(path: Path) -> List[Dict[str, object]]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            items.append(json.loads(line))
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt ablation check for T2A/T2I.")
    parser.add_argument("--token-space-json", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--lora", default="")
    parser.add_argument("--golden-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device-map", default="")
    args = parser.parse_args()

    token_space = TokenSpace.load_json(args.token_space_json)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_map = args.device_map or None
    lora_path = Path(args.lora) if args.lora else None
    model = _load_model(Path(args.checkpoint), token_space, device, device_map, lora_path)

    rng = random.Random(int(args.seed))
    golden_dir = Path(args.golden_dir)
    t2a_path = golden_dir / "golden_t2a_64.jsonl"
    t2i_path = golden_dir / "golden_t2i_64.jsonl"
    t2a_items = _load_jsonl(t2a_path)
    t2i_items = _load_jsonl(t2i_path)

    report = {
        "t2a": _run_ablation(
            t2a_items,
            task="t2a",
            token_space=token_space,
            model=model,
            rng=rng,
            device=device,
        ),
        "t2i": _run_ablation(
            t2i_items,
            task="t2i",
            token_space=token_space,
            model=model,
            rng=rng,
            device=device,
        ),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
