import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from stage3_uti.pipeline.sequence_builder import SequenceBuilder, build_supervised_labels, find_span
from stage3_uti.stage2.wds_io import iter_tar_samples
from stage3_uti.tokenization.token_space import TokenSpace


def _iter_shards(root: Path, split: str) -> Iterable[Path]:
    split_dir = root / split
    if not split_dir.exists():
        return []
    return sorted(split_dir.glob("shard-*.tar"))


def _center_crop_start(length: int, max_len: int, align: int = 1) -> int:
    if length <= max_len:
        return 0
    start = (length - max_len) // 2
    if align > 1:
        start = start - (start % align)
    return max(0, min(start, length - max_len))


def _random_crop_start(
    length: int, max_len: int, rng: np.random.Generator, align: int = 1
) -> int:
    if length <= max_len:
        return 0
    max_start = length - max_len
    if align > 1:
        max_start = max_start - (max_start % align)
    if max_start <= 0:
        return 0
    if align > 1:
        steps = max_start // align
        return int(rng.integers(0, steps + 1)) * align
    return int(rng.integers(0, max_start + 1))


def _crop_tokens(
    tokens: np.ndarray,
    max_len: Optional[int],
    *,
    rng: np.random.Generator,
    mode: str,
    align: int = 1,
) -> Tuple[np.ndarray, Dict[str, int]]:
    if max_len is None or max_len <= 0 or tokens.size <= max_len:
        return tokens, {"crop_start": 0, "crop_len": int(tokens.size)}
    if mode == "center":
        start = _center_crop_start(int(tokens.size), int(max_len), align=align)
    else:
        start = _random_crop_start(int(tokens.size), int(max_len), rng, align=align)
    end = start + int(max_len)
    return tokens[start:end], {"crop_start": int(start), "crop_len": int(end - start)}


def _as_list(arr: Optional[np.ndarray]) -> List[int]:
    if arr is None:
        return []
    if isinstance(arr, np.ndarray):
        return [int(x) for x in arr.tolist()]
    return [int(x) for x in arr]


def _label_span_for_task(builder: SequenceBuilder, task: str, input_ids: List[int]) -> Optional[slice]:
    special = builder.special
    if task in {"a2t", "i2t"}:
        return find_span(input_ids, int(special["<|text_start|>"]), int(special["<|text_end|>"]))
    if task == "t2a":
        return find_span(input_ids, int(special["<|aud_start|>"]), int(special["<|aud_end|>"]))
    if task == "t2i":
        return find_span(input_ids, int(special["<|img_start|>"]), int(special["<|img_end|>"]))
    return None


def _build_record(
    *,
    builder: SequenceBuilder,
    task: str,
    text_tokens: List[int],
    audio_tokens: Optional[np.ndarray],
    image_tokens: Optional[np.ndarray],
    audio_max: Optional[int],
    image_max: Optional[int],
    rng: np.random.Generator,
    mode: str,
    audio_align: int,
) -> Optional[Dict[str, object]]:
    crop_meta: Dict[str, int] = {}
    if audio_tokens is not None:
        audio_tokens, meta = _crop_tokens(
            audio_tokens,
            audio_max,
            rng=rng,
            mode=mode,
            align=audio_align,
        )
        crop_meta.update({f"audio_{k}": v for k, v in meta.items()})
    if image_tokens is not None:
        image_tokens, meta = _crop_tokens(
            image_tokens,
            image_max,
            rng=rng,
            mode=mode,
            align=1,
        )
        crop_meta.update({f"image_{k}": v for k, v in meta.items()})

    if task == "a2t":
        seq = builder.build_a2t(_as_list(audio_tokens), text_tokens)
    elif task == "i2t":
        seq = builder.build_i2t(_as_list(image_tokens), text_tokens)
    elif task == "t2a":
        seq = builder.build_t2a(text_tokens, _as_list(audio_tokens))
    elif task == "t2i":
        seq = builder.build_t2i(text_tokens, _as_list(image_tokens))
    else:
        return None

    span = _label_span_for_task(builder, task, seq)
    if span is None:
        return None
    labels = build_supervised_labels(seq, span)
    return {
        "input_ids": seq,
        "labels": labels,
        "length": len(seq),
        "task": task,
        "crop": crop_meta,
    }


def _iter_dataset_samples(
    root: Path, split: str, audio_max: Optional[int], image_max: Optional[int]
) -> Iterable[Dict[str, object]]:
    if not root.exists():
        return []
    for shard in _iter_shards(root, split):
        for _, sample in iter_tar_samples(shard):
            meta = sample.get("json", {})
            task = meta.get("task")
            if task not in {"a2t", "i2t", "t2a", "t2i"}:
                continue
            yield {
                "id": meta.get("id"),
                "source": meta.get("source"),
                "task": task,
                "audio": sample.get("audio"),
                "image": sample.get("image"),
                "text_out": sample.get("text_out"),
            }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage 3 JSONL from Stage 2 tokenized shards.")
    parser.add_argument("--token-space-json", required=True)
    parser.add_argument("--data-root", default="stage3_uti/data/tokenized")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--output-train", required=True)
    parser.add_argument("--output-eval", required=True)
    parser.add_argument("--audio-max", type=int, default=1024)
    parser.add_argument("--image-max", type=int, default=576)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    token_space = TokenSpace.load_json(args.token_space_json)
    token_space.validate()
    builder = SequenceBuilder(token_space)

    n_codebooks = int(token_space.audio_codec.get("n_codebooks", 1))
    audio_align = max(1, n_codebooks)
    if args.audio_max % audio_align != 0:
        args.audio_max = args.audio_max - (args.audio_max % audio_align)

    rng = np.random.default_rng(args.seed)

    output_train = Path(args.output_train)
    output_eval = Path(args.output_eval)
    output_train.parent.mkdir(parents=True, exist_ok=True)
    output_eval.parent.mkdir(parents=True, exist_ok=True)

    counts: Dict[str, int] = {"train": 0, "eval": 0, "skipped": 0}

    for split, output_path in (("train", output_train), ("eval", output_eval)):
        mode = "random" if split == "train" else "center"
        with output_path.open("w", encoding="utf-8") as dst:
            for ds in args.datasets:
                ds_root = Path(args.data_root) / ds
                for item in _iter_dataset_samples(ds_root, split, args.audio_max, args.image_max):
                    if args.max_samples and counts[split] >= args.max_samples:
                        break
                    text_tokens = _as_list(item.get("text_out"))
                    record = _build_record(
                        builder=builder,
                        task=item["task"],
                        text_tokens=text_tokens,
                        audio_tokens=item.get("audio"),
                        image_tokens=item.get("image"),
                        audio_max=args.audio_max if item["task"] in {"a2t", "t2a"} else None,
                        image_max=args.image_max if item["task"] in {"i2t", "t2i"} else None,
                        rng=rng,
                        mode=mode,
                        audio_align=audio_align,
                    )
                    if record is None:
                        counts["skipped"] += 1
                        continue
                    record.update({"id": item.get("id"), "source": item.get("source"), "dataset": ds})
                    dst.write(json.dumps(record) + "\n")
                    counts[split] += 1
                if args.max_samples and counts[split] >= args.max_samples:
                    break

    print(
        "[stage3_jsonl] train={train} eval={eval} skipped={skipped}".format(**counts)
    )


if __name__ == "__main__":
    main()
