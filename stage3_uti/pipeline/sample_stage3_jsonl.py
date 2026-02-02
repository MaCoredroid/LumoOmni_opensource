import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List


PRESETS: Dict[str, Dict[str, int]] = {
    "smoke_2048": {
        "wavcaps_audioset": 791,
        "audiocaps": 311,
        "clotho": 156,
        "llava_pretrain": 790,
    },
    "iter_8192": {
        "wavcaps_audioset": 3162,
        "audiocaps": 1245,
        "clotho": 623,
        "llava_pretrain": 3162,
    },
    "prefull_51200": {
        "wavcaps_audioset": 19766,
        "audiocaps": 7782,
        "clotho": 3892,
        "llava_pretrain": 19760,
    },
}


def _load_targets(args: argparse.Namespace) -> Dict[str, int]:
    if args.preset:
        if args.preset not in PRESETS:
            raise ValueError(f"Unknown preset: {args.preset}")
        return dict(PRESETS[args.preset])
    if args.targets_json:
        return json.loads(args.targets_json)
    if args.targets_file:
        with open(args.targets_file, "r", encoding="utf-8") as f:
            return json.load(f)
    raise ValueError("Provide --preset, --targets-json, or --targets-file")


def sample_stage3_jsonl(in_path: Path, out_path: Path, targets: Dict[str, int], seed: int) -> None:
    rng = random.Random(seed)
    seen = defaultdict(int)
    reservoir: Dict[str, List[str]] = {k: [] for k in targets}
    totals = Counter()

    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            src = obj.get("source")
            if src not in targets:
                continue
            totals[src] += 1
            seen[src] += 1
            buf = reservoir[src]
            k = int(targets[src])

            if len(buf) < k:
                buf.append(line)
            else:
                j = rng.randrange(seen[src])
                if j < k:
                    buf[j] = line

    all_lines: List[str] = []
    for src, buf in reservoir.items():
        if len(buf) != int(targets[src]):
            raise RuntimeError(f"Source {src}: got {len(buf)} / {targets[src]}")
        all_lines.extend(buf)

    rng.shuffle(all_lines)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for line in all_lines:
            out.write(line)

    print(f"Wrote {len(all_lines)} lines to {out_path}")
    print("Source counts:")
    for src, count in sorted(targets.items()):
        print(f"  {src}: {count} (seen {totals[src]})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out", dest="out_path", required=True)
    parser.add_argument("--preset", default="")
    parser.add_argument("--targets-json", default="")
    parser.add_argument("--targets-file", default="")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    targets = _load_targets(args)
    sample_stage3_jsonl(Path(args.in_path), Path(args.out_path), targets, int(args.seed))


if __name__ == "__main__":
    main()
