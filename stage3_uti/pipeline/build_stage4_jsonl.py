import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from stage3_uti.pipeline.sequence_builder import SequenceBuilder, build_supervised_labels, find_span
from stage3_uti.tokenization.token_space import TokenSpace


def _extract_between(input_ids: List[int], start_tok: int, end_tok: int) -> Optional[List[int]]:
    try:
        start_idx = input_ids.index(start_tok)
        end_idx = input_ids.index(end_tok, start_idx + 1)
    except ValueError:
        return None
    if end_idx <= start_idx + 1:
        return []
    return input_ids[start_idx + 1 : end_idx]


def _label_span_for_task(builder: SequenceBuilder, task: str, input_ids: List[int]) -> Optional[slice]:
    special = builder.special
    if task in {"a2t", "i2t"}:
        return find_span(input_ids, int(special["<|text_start|>"]), int(special["<|text_end|>"]))
    if task == "t2a":
        return find_span(input_ids, int(special["<|aud_start|>"]), int(special["<|aud_end|>"]))
    if task == "t2i":
        return find_span(input_ids, int(special["<|img_start|>"]), int(special["<|img_end|>"]))
    return None


def _sample(items: List[Dict[str, object]], count: int, rng: random.Random) -> List[Dict[str, object]]:
    if count <= 0 or count >= len(items):
        return list(items)
    indices = rng.sample(range(len(items)), count)
    return [items[i] for i in indices]


def _build_record(
    builder: SequenceBuilder,
    token_space_sha: str,
    task: str,
    text_tokens: List[int],
    audio_tokens: Optional[List[int]],
    image_tokens: Optional[List[int]],
    base: Dict[str, object],
) -> Optional[Dict[str, object]]:
    if task == "a2t":
        seq = builder.build_a2t(audio_tokens or [], text_tokens)
    elif task == "i2t":
        seq = builder.build_i2t(image_tokens or [], text_tokens)
    elif task == "t2a":
        seq = builder.build_t2a(text_tokens, audio_tokens or [])
    elif task == "t2i":
        seq = builder.build_t2i(text_tokens, image_tokens or [])
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
        "source": base.get("source"),
        "id": base.get("id"),
        "base_task": base.get("task"),
        "token_space_sha256": token_space_sha,
    }


def _load_stage3_records(path: Path, token_space: TokenSpace) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    audio_items: List[Dict[str, object]] = []
    image_items: List[Dict[str, object]] = []
    special = token_space.special_tokens
    aud_start = int(special["<|aud_start|>"])
    aud_end = int(special["<|aud_end|>"])
    img_start = int(special["<|img_start|>"])
    img_end = int(special["<|img_end|>"])
    text_start = int(special["<|text_start|>"])
    text_end = int(special["<|text_end|>"])

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            task = item.get("task")
            input_ids = item.get("input_ids")
            if not isinstance(input_ids, list):
                continue
            if task == "a2t":
                audio = _extract_between(input_ids, aud_start, aud_end)
                text = _extract_between(input_ids, text_start, text_end)
                if audio is None or text is None:
                    continue
                audio_items.append(
                    {
                        "id": item.get("id"),
                        "source": item.get("source"),
                        "task": task,
                        "audio_tokens": audio,
                        "text_tokens": text,
                    }
                )
            elif task == "i2t":
                image = _extract_between(input_ids, img_start, img_end)
                text = _extract_between(input_ids, text_start, text_end)
                if image is None or text is None:
                    continue
                image_items.append(
                    {
                        "id": item.get("id"),
                        "source": item.get("source"),
                        "task": task,
                        "image_tokens": image,
                        "text_tokens": text,
                    }
                )
    return audio_items, image_items


def _write_jsonl(path: Path, records: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def build_split(
    *,
    input_path: Path,
    output_path: Path,
    token_space: TokenSpace,
    seed: int,
    per_task: int,
    shuffle: bool,
) -> Dict[str, int]:
    rng = random.Random(seed)
    builder = SequenceBuilder(token_space)
    token_space_sha = token_space.sha256()

    audio_items, image_items = _load_stage3_records(input_path, token_space)

    audio_sel = _sample(audio_items, per_task, rng)
    image_sel = _sample(image_items, per_task, rng)

    records: List[Dict[str, object]] = []
    counts: Dict[str, int] = {"a2t": 0, "t2a": 0, "i2t": 0, "t2i": 0, "skipped": 0}

    for item in audio_sel:
        text_tokens = list(item["text_tokens"])
        audio_tokens = list(item["audio_tokens"])
        rec_a2t = _build_record(
            builder,
            token_space_sha,
            "a2t",
            text_tokens,
            audio_tokens,
            None,
            item,
        )
        rec_t2a = _build_record(
            builder,
            token_space_sha,
            "t2a",
            text_tokens,
            audio_tokens,
            None,
            item,
        )
        if rec_a2t is not None:
            records.append(rec_a2t)
            counts["a2t"] += 1
        else:
            counts["skipped"] += 1
        if rec_t2a is not None:
            records.append(rec_t2a)
            counts["t2a"] += 1
        else:
            counts["skipped"] += 1

    for item in image_sel:
        text_tokens = list(item["text_tokens"])
        image_tokens = list(item["image_tokens"])
        rec_i2t = _build_record(
            builder,
            token_space_sha,
            "i2t",
            text_tokens,
            None,
            image_tokens,
            item,
        )
        rec_t2i = _build_record(
            builder,
            token_space_sha,
            "t2i",
            text_tokens,
            None,
            image_tokens,
            item,
        )
        if rec_i2t is not None:
            records.append(rec_i2t)
            counts["i2t"] += 1
        else:
            counts["skipped"] += 1
        if rec_t2i is not None:
            records.append(rec_t2i)
            counts["t2i"] += 1
        else:
            counts["skipped"] += 1

    if shuffle:
        rng.shuffle(records)

    _write_jsonl(output_path, records)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage 4 JSONL from Stage 3 JSONL.")
    parser.add_argument("--token-space-json", required=True)
    parser.add_argument("--input-train", required=True)
    parser.add_argument("--input-eval", required=True)
    parser.add_argument("--output-train", required=True)
    parser.add_argument("--output-eval", required=True)
    parser.add_argument("--train-per-task", type=int, default=0)
    parser.add_argument("--eval-per-task", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-shuffle", action="store_true")
    args = parser.parse_args()

    token_space = TokenSpace.load_json(args.token_space_json)
    token_space.validate()

    train_counts = build_split(
        input_path=Path(args.input_train),
        output_path=Path(args.output_train),
        token_space=token_space,
        seed=int(args.seed),
        per_task=int(args.train_per_task),
        shuffle=not args.no_shuffle,
    )
    eval_counts = build_split(
        input_path=Path(args.input_eval),
        output_path=Path(args.output_eval),
        token_space=token_space,
        seed=int(args.seed),
        per_task=int(args.eval_per_task),
        shuffle=not args.no_shuffle,
    )
    print(
        "[stage4_jsonl] train={train} eval={eval}".format(
            train=train_counts,
            eval=eval_counts,
        )
    )


if __name__ == "__main__":
    main()
