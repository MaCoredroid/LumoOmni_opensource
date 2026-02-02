import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


def _scan_audio_files(root: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if not name.lower().endswith((".wav", ".flac", ".mp3")):
                continue
            path = str(Path(dirpath) / name)
            if name not in mapping:
                mapping[name] = path
            stem = Path(name).stem
            if stem and stem not in mapping:
                mapping[stem] = path
    return mapping


def _write_jsonl(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
            count += 1
    return count


def _iter_clotho_rows(csv_path: Path) -> Iterator[Tuple[str, List[str]]]:
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_name = row.get("file_name")
            if not file_name:
                continue
            captions = []
            for i in range(1, 6):
                key = f"caption_{i}"
                if row.get(key):
                    captions.append(row[key])
            if captions:
                yield file_name, captions


def build_clotho(manifest_path: Path, audio_root: Path, data_root: Path) -> int:
    audio_map = _scan_audio_files(audio_root)
    records = []
    for split in ("development", "evaluation"):
        csv_path = data_root / f"clotho_captions_{split}.csv"
        if not csv_path.exists():
            continue
        for file_name, captions in _iter_clotho_rows(csv_path):
            audio_path = audio_map.get(file_name)
            if not audio_path:
                continue
            for idx, caption in enumerate(captions):
                records.append(
                    {
                        "id": f"clotho:{file_name}:{idx}",
                        "source": "clotho",
                        "modalities": {"audio": {"path": audio_path}},
                        "text": {"caption": caption},
                        "task": "a2t",
                        "meta": {"license_hint": "Clotho"},
                    }
                )
    return _write_jsonl(manifest_path, records)


def build_audiocaps(manifest_path: Path, audio_root: Path, data_root: Path) -> int:
    audio_map = _scan_audio_files(audio_root)
    records: List[dict] = []
    for split in ("train", "val", "test"):
        csv_path = data_root / "metadata" / f"{split}.csv"
        if not csv_path.exists():
            continue
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                audiocap_id = row.get("audiocap_id")
                caption = row.get("caption")
                if not audiocap_id or not caption:
                    continue

                # AudioCaps filenames typically include audiocap_id or youtube_id; try direct match first.
                audio_path = audio_map.get(f"{audiocap_id}.wav")
                if not audio_path:
                    # fallback: scan for files starting with audiocap_id
                    for name, path in audio_map.items():
                        if name.startswith(str(audiocap_id)):
                            audio_path = path
                            break
                if not audio_path:
                    continue

                records.append(
                    {
                        "id": f"audiocaps:{audiocap_id}",
                        "source": "audiocaps",
                        "modalities": {"audio": {"path": audio_path}},
                        "text": {"caption": caption},
                        "task": "a2t",
                        "meta": {"license_hint": "AudioCaps", "duration_s": 10.0},
                    }
                )
    return _write_jsonl(manifest_path, records)


def _iter_wavcaps_entries(json_path: Path) -> Iterator[dict]:
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    data = payload.get("data", [])
    for item in data:
        yield item


def build_wavcaps_audioset(
    manifest_path: Path, audio_root: Path, json_path: Path, max_samples: int
) -> int:
    audio_map = _scan_audio_files(audio_root)
    records: List[dict] = []
    entries = list(_iter_wavcaps_entries(json_path))
    entries.sort(key=lambda x: str(x.get("id", "")))
    if max_samples > 0:
        entries = entries[: max_samples]

    for item in entries:
        audio_id = str(item.get("id"))
        caption = item.get("caption")
        if not audio_id or not caption:
            continue
        audio_path = audio_map.get(audio_id)
        if not audio_path:
            audio_path = audio_map.get(Path(audio_id).stem)
        if not audio_path:
            continue
        records.append(
            {
                "id": f"wavcaps_as:{audio_id}",
                "source": "wavcaps_audioset",
                "modalities": {"audio": {"path": audio_path}},
                "text": {"caption": caption},
                "task": "a2t",
                "meta": {"license_hint": "WavCaps", "duration_s": float(item.get("duration", 0.0))},
            }
        )
    return _write_jsonl(manifest_path, records)


def build_llava_pretrain(
    manifest_path: Path, json_path: Path, images_root: Path, max_samples: int
) -> int:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data.sort(key=lambda x: str(x.get("image", "")))
    if max_samples > 0:
        data = data[: max_samples]

    records = []
    for idx, item in enumerate(data):
        image_name = item.get("image")
        caption = item.get("caption")
        if not caption:
            conversations = item.get("conversations", [])
            for convo in conversations:
                if convo.get("from") == "gpt" and convo.get("value"):
                    caption = convo["value"]
                    break
        if not image_name or not caption:
            continue
        image_path = images_root / image_name
        if not image_path.exists():
            continue
        records.append(
            {
                "id": f"llava_pretrain:{idx}",
                "source": "llava_pretrain",
                "modalities": {"image": {"path": str(image_path)}},
                "text": {"caption": caption},
                "task": "i2t",
                "meta": {"license_hint": "LLaVA-Pretrain"},
            }
        )
    return _write_jsonl(manifest_path, records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage 2 manifests.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    clotho = sub.add_parser("clotho")
    clotho.add_argument("--data-root", required=True)
    clotho.add_argument("--audio-root", required=True)
    clotho.add_argument("--out", required=True)

    audiocaps = sub.add_parser("audiocaps")
    audiocaps.add_argument("--data-root", required=True)
    audiocaps.add_argument("--audio-root", required=True)
    audiocaps.add_argument("--out", required=True)

    wavcaps = sub.add_parser("wavcaps_as")
    wavcaps.add_argument("--audio-root", required=True)
    wavcaps.add_argument("--json-path", required=True)
    wavcaps.add_argument("--out", required=True)
    wavcaps.add_argument("--max-samples", type=int, default=100000)

    llava = sub.add_parser("llava_pretrain")
    llava.add_argument("--json-path", required=True)
    llava.add_argument("--images-root", required=True)
    llava.add_argument("--out", required=True)
    llava.add_argument("--max-samples", type=int, default=100000)

    args = parser.parse_args()

    if args.cmd == "clotho":
        count = build_clotho(Path(args.out), Path(args.audio_root), Path(args.data_root))
    elif args.cmd == "audiocaps":
        count = build_audiocaps(Path(args.out), Path(args.audio_root), Path(args.data_root))
    elif args.cmd == "wavcaps_as":
        count = build_wavcaps_audioset(
            Path(args.out), Path(args.audio_root), Path(args.json_path), int(args.max_samples)
        )
    elif args.cmd == "llava_pretrain":
        count = build_llava_pretrain(
            Path(args.out), Path(args.json_path), Path(args.images_root), int(args.max_samples)
        )
    else:
        raise ValueError(f"Unknown command {args.cmd}")

    print(f"Wrote {count} records to {args.out}")


if __name__ == "__main__":
    main()
