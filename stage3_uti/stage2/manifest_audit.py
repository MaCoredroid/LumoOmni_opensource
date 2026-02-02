import argparse
import json
from pathlib import Path
from typing import Dict

from stage3_uti.stage2.utils import min_max, percentile_stats


def _count_dict_inc(counter: Dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit raw manifest JSONL for Stage 2.")
    parser.add_argument("--manifest-jsonl", required=True)
    parser.add_argument("--dataset-name")
    parser.add_argument("--data-dir", default="stage3_uti/data")
    parser.add_argument("--output-report")
    args = parser.parse_args()

    manifest_path = Path(args.manifest_jsonl)
    dataset = args.dataset_name or manifest_path.stem
    data_dir = Path(args.data_dir)
    report_path = (
        Path(args.output_report)
        if args.output_report
        else data_dir / "reports" / f"{dataset}_manifest_stats.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    missing_id = 0
    missing_task = 0
    missing_modalities = 0
    missing_text = 0
    task_counts: Dict[str, int] = {}
    source_counts: Dict[str, int] = {}

    audio_paths = 0
    audio_exists = 0
    image_paths = 0
    image_exists = 0

    durations = []

    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            item = json.loads(line)
            if not item.get("id"):
                missing_id += 1
            task = item.get("task")
            if not task:
                missing_task += 1
            else:
                _count_dict_inc(task_counts, str(task))
            source = item.get("source")
            if source:
                _count_dict_inc(source_counts, str(source))

            modalities = item.get("modalities") or {}
            if not modalities:
                missing_modalities += 1
            audio = modalities.get("audio") or {}
            image = modalities.get("image") or {}
            audio_path = audio.get("path") if isinstance(audio, dict) else None
            image_path = image.get("path") if isinstance(image, dict) else None

            if audio_path:
                audio_paths += 1
                if Path(audio_path).exists():
                    audio_exists += 1
            if image_path:
                image_paths += 1
                if Path(image_path).exists():
                    image_exists += 1

            text = item.get("text")
            if not text:
                missing_text += 1

            meta = item.get("meta") or {}
            duration = meta.get("duration_s") if isinstance(meta, dict) else None
            if duration is not None:
                try:
                    durations.append(float(duration))
                except Exception:
                    pass

    report = {
        "dataset": dataset,
        "records_total": total,
        "missing_id": missing_id,
        "missing_task": missing_task,
        "missing_modalities": missing_modalities,
        "missing_text": missing_text,
        "task_counts": task_counts,
        "source_counts": source_counts,
        "audio_paths": audio_paths,
        "audio_path_exists": audio_exists,
        "audio_path_exists_rate": (audio_exists / audio_paths) if audio_paths else 0.0,
        "image_paths": image_paths,
        "image_path_exists": image_exists,
        "image_path_exists_rate": (image_exists / image_paths) if image_paths else 0.0,
        "duration_s": {**percentile_stats(durations), **min_max(durations)},
    }

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
