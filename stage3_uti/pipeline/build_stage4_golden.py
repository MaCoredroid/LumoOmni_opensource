import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def _load_by_task(path: Path) -> Dict[str, List[Dict[str, object]]]:
    buckets: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            task = obj.get("task")
            if not task:
                continue
            buckets[str(task)].append(obj)
    return buckets


def _sample(items: List[Dict[str, object]], count: int, rng: random.Random) -> List[Dict[str, object]]:
    if count <= 0 or count >= len(items):
        return list(items)
    return rng.sample(items, count)


def _write_jsonl(path: Path, records: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage 4 golden sets from Stage 4 JSONL.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    rng = random.Random(int(args.seed))
    buckets = _load_by_task(Path(args.input_jsonl))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = ["a2t", "i2t", "t2a", "t2i"]
    for task in tasks:
        items = buckets.get(task, [])
        if len(items) < args.count:
            raise RuntimeError(f"Only found {len(items)} items for task={task}, need {args.count}")
        sample = _sample(items, int(args.count), rng)
        out_path = output_dir / f"golden_{task}_{args.count}.jsonl"
        _write_jsonl(out_path, sample)

    print(
        "[stage4_golden] wrote="
        + ", ".join(str(output_dir / f"golden_{task}_{args.count}.jsonl") for task in tasks)
    )


if __name__ == "__main__":
    main()
