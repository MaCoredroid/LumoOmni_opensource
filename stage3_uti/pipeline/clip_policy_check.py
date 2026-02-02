import argparse
import json
from pathlib import Path
from typing import Dict, List


def _percentiles(values: List[int], ps: List[float]) -> Dict[str, float]:
    if not values:
        return {f"p{int(p)}": 0.0 for p in ps}
    vals = sorted(values)
    n = len(vals)
    out = {}
    for p in ps:
        k = int(round((p / 100.0) * (n - 1)))
        out[f"p{int(p)}"] = float(vals[k])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--max-seq-len", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    path = Path(args.jsonl)
    lengths: List[int] = []
    worst: List[Dict[str, object]] = []
    max_len = 0
    over_limit = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            length = int(obj.get("length", len(obj.get("input_ids", []))))
            lengths.append(length)
            if length > max_len:
                max_len = length
            if length > args.max_seq_len:
                over_limit += 1
            if args.topk > 0:
                worst.append(
                    {
                        "id": obj.get("id"),
                        "task": obj.get("task"),
                        "source": obj.get("source"),
                        "length": length,
                    }
                )

    if args.topk > 0:
        worst = sorted(worst, key=lambda x: int(x["length"]), reverse=True)[: args.topk]

    stats = {
        "jsonl": str(path),
        "count": len(lengths),
        "max_seq_len": int(args.max_seq_len),
        "max_length": int(max_len),
        "over_limit": int(over_limit),
        "percentiles": _percentiles(lengths, [50, 90, 99]),
        "worst": worst,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
