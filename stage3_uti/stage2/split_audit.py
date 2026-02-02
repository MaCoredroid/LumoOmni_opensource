import argparse
import json
from pathlib import Path
from typing import Dict, List

from stage3_uti.stage2.utils import assign_split


def _load_ids(path: Path) -> List[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _count_dict_inc(counter: Dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit train/eval split integrity.")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--data-dir", default="stage3_uti/data")
    parser.add_argument("--eval-mod", type=int, default=1000)
    parser.add_argument("--eval-cutoff", type=int, default=10)
    parser.add_argument("--output-report")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    dataset = args.dataset_name
    splits_dir = data_dir / "splits" / dataset
    report_path = (
        Path(args.output_report)
        if args.output_report
        else data_dir / "reports" / f"{dataset}_split_audit.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)

    train_ids = _load_ids(splits_dir / "train_ids.txt")
    eval_ids = _load_ids(splits_dir / "eval_ids.txt")

    train_set = set(train_ids)
    eval_set = set(eval_ids)

    overlap = train_set.intersection(eval_set)
    dup_train = len(train_ids) - len(train_set)
    dup_eval = len(eval_ids) - len(eval_set)

    mismatch_counts = {"train": 0, "eval": 0}
    mismatch_examples: Dict[str, List[str]] = {"train": [], "eval": []}
    for split_name, ids in (("train", train_ids), ("eval", eval_ids)):
        for sample_id in ids:
            expected = assign_split(sample_id, eval_mod=args.eval_mod, eval_cutoff=args.eval_cutoff)
            if expected != split_name:
                mismatch_counts[split_name] += 1
                if len(mismatch_examples[split_name]) < 20:
                    mismatch_examples[split_name].append(sample_id)

    report = {
        "dataset": dataset,
        "train_count": len(train_ids),
        "eval_count": len(eval_ids),
        "duplicate_train": dup_train,
        "duplicate_eval": dup_eval,
        "overlap_count": len(overlap),
        "overlap_sample": sorted(list(overlap))[:20],
        "eval_mod": args.eval_mod,
        "eval_cutoff": args.eval_cutoff,
        "split_mismatch_counts": mismatch_counts,
        "split_mismatch_examples": mismatch_examples,
    }

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
