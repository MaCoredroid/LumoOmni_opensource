import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM

from stage3_uti.tokenization.token_space import TokenSpace
from stage3_uti.utils.train_utils import _load_trainable_rows, _resize_and_init_embeddings


def _load_examples(
    path: Path, *, task: str, count: int, seed: int
) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    seen = 0
    reservoir: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("task") != task:
                continue
            seen += 1
            if len(reservoir) < count:
                reservoir.append(obj)
            else:
                j = rng.randrange(seen)
                if j < count:
                    reservoir[j] = obj
    if len(reservoir) < count:
        raise RuntimeError(f"Only found {len(reservoir)} rows for task={task}, need {count}")
    return reservoir


def _span_indices(input_ids: List[int], start_tok: int, end_tok: int) -> Optional[Tuple[int, int]]:
    try:
        start_idx = input_ids.index(start_tok)
        end_idx = input_ids.index(end_tok, start_idx + 1)
    except ValueError:
        return None
    return start_idx, end_idx


def _build_ranges(token_space: TokenSpace, modality: str) -> List[Tuple[int, int]]:
    if modality == "image":
        spec = token_space.ranges["IMAGE"]
        return [(int(spec["start"]), int(spec["end"]))]
    audio_ranges = []
    for name, spec in token_space.ranges.items():
        if name.startswith("AUDIO_CB"):
            audio_ranges.append((int(spec["start"]), int(spec["end"])))
    if not audio_ranges:
        raise ValueError("No audio ranges found in token space")
    return audio_ranges


def _sample_from_ranges(rng: random.Random, ranges: List[Tuple[int, int]]) -> int:
    total = sum(end - start + 1 for start, end in ranges)
    r = rng.randrange(total)
    for start, end in ranges:
        size = end - start + 1
        if r < size:
            return start + r
        r -= size
    return ranges[-1][1]


def _apply_ablation(
    input_ids: List[int],
    *,
    start_idx: int,
    end_idx: int,
    mode: str,
    rng: random.Random,
    ranges: List[Tuple[int, int]],
) -> List[int]:
    if mode == "correct":
        return list(input_ids)
    out = list(input_ids)
    span = out[start_idx + 1 : end_idx]
    if not span:
        return out
    if mode == "shuffle":
        rng.shuffle(span)
        out[start_idx + 1 : end_idx] = span
        return out
    if mode == "noise":
        out[start_idx + 1 : end_idx] = [_sample_from_ranges(rng, ranges) for _ in span]
        return out
    if mode == "const":
        const_tok = ranges[0][0]
        out[start_idx + 1 : end_idx] = [const_tok for _ in span]
        return out
    raise ValueError(f"Unknown mode: {mode}")


def _eval_examples(
    model: AutoModelForCausalLM,
    examples: Iterable[Dict[str, object]],
    *,
    token_space: TokenSpace,
    modality: str,
    seed: int,
    device: torch.device,
) -> Dict[str, object]:
    rng = random.Random(seed)
    modes = ["correct", "shuffle", "noise"]
    losses: Dict[str, List[float]] = {m: [] for m in modes}

    start_tok = int(token_space.special_tokens["<|img_start|>"] if modality == "image" else token_space.special_tokens["<|aud_start|>"])
    end_tok = int(token_space.special_tokens["<|img_end|>"] if modality == "image" else token_space.special_tokens["<|aud_end|>"])
    ranges = _build_ranges(token_space, modality)

    for ex in examples:
        input_ids = ex["input_ids"]
        labels = ex["labels"]
        span = _span_indices(input_ids, start_tok, end_tok)
        if span is None:
            continue
        start_idx, end_idx = span
        for mode in modes:
            mod_ids = _apply_ablation(
                input_ids,
                start_idx=start_idx,
                end_idx=end_idx,
                mode=mode,
                rng=rng,
                ranges=ranges,
            )
            input_tensor = torch.tensor([mod_ids], dtype=torch.long, device=device)
            label_tensor = torch.tensor([labels], dtype=torch.long, device=device)
            with torch.no_grad():
                outputs = model(input_ids=input_tensor, labels=label_tensor)
            losses[mode].append(float(outputs.loss.detach().cpu()))

    if not losses["correct"]:
        raise RuntimeError("No valid examples found for evaluation")

    summary: Dict[str, object] = {}
    for mode in modes:
        summary[f"loss_{mode}_mean"] = sum(losses[mode]) / len(losses[mode])

    win_shuffle = sum(
        1 for lc, ls in zip(losses["correct"], losses["shuffle"]) if lc < ls
    ) / len(losses["correct"])
    win_noise = sum(
        1 for lc, ln in zip(losses["correct"], losses["noise"]) if lc < ln
    ) / len(losses["correct"])
    summary["win_rate_shuffle"] = win_shuffle
    summary["win_rate_noise"] = win_noise
    summary["n_examples"] = len(losses["correct"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--token-space", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--audio-n", type=int, default=128)
    parser.add_argument("--image-n", type=int, default=128)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--attn-impl", default="sdpa")
    parser.add_argument("--device-map", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    token_space = TokenSpace.load_json(args.token_space)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_kwargs: Dict[str, object] = {
        "torch_dtype": torch.bfloat16 if device.type == "cuda" else torch.float32,
        "low_cpu_mem_usage": True,
    }
    if args.attn_impl:
        load_kwargs["attn_implementation"] = args.attn_impl
    if args.device_map:
        load_kwargs["device_map"] = args.device_map
        load_kwargs["low_cpu_mem_usage"] = True

    model_path = Path(args.model)
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
        model.config.pad_token_id = int(token_space.special_tokens.get("<|pad_mm|>", 0))
        _load_trainable_rows(
            model,
            model_path,
            row_start=int(token_space.text_vocab_size),
            row_end=int(token_space.vocab_size_total),
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    if not getattr(model, "hf_device_map", None):
        model.to(device)
    model.eval()

    path = Path(args.jsonl)
    audio_examples = _load_examples(path, task="a2t", count=args.audio_n, seed=args.seed)
    image_examples = _load_examples(path, task="i2t", count=args.image_n, seed=args.seed)

    audio_metrics = _eval_examples(
        model, audio_examples, token_space=token_space, modality="audio", seed=args.seed, device=device
    )
    image_metrics = _eval_examples(
        model, image_examples, token_space=token_space, modality="image", seed=args.seed, device=device
    )

    report = {
        "audio": audio_metrics,
        "image": image_metrics,
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
