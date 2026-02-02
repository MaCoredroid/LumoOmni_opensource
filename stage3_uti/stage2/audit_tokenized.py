import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from stage3_uti.stage2.utils import percentile_stats, sha256_int32
from stage3_uti.stage2.wds_io import iter_tar_samples
from stage3_uti.tokenization.unified_tokenizer import UnifiedTokenizer


class Reservoir:
    def __init__(self, size: int) -> None:
        self.size = int(size)
        self.items: List[Any] = []
        self.seen = 0

    def add(self, item: Any) -> None:
        if self.size <= 0:
            return
        self.seen += 1
        if len(self.items) < self.size:
            self.items.append(item)
            return
        idx = random.randint(0, self.seen - 1)
        if idx < self.size:
            self.items[idx] = item


def _audio_ranges(token_space) -> Dict[int, Tuple[int, int]]:
    ranges = {}
    for name, spec in token_space.ranges.items():
        if not name.startswith("AUDIO_CB"):
            continue
        cb = int(name.replace("AUDIO_CB", ""))
        ranges[cb] = (int(spec["start"]), int(spec["end"]))
    return ranges


def _validate_text(tokens: np.ndarray, token_space) -> Optional[str]:
    if tokens.size == 0:
        return None
    rng = token_space.ranges["TEXT"]
    start, end = int(rng["start"]), int(rng["end"])
    if tokens.min() < start or tokens.max() > end:
        return "text_range"
    return None


def _validate_image(tokens: np.ndarray, meta: Dict[str, Any], token_space) -> Optional[str]:
    rng = token_space.ranges["IMAGE"]
    start, end = int(rng["start"]), int(rng["end"])
    if tokens.min() < start or tokens.max() > end:
        return "image_range"
    expected = int(meta.get("n_tokens", tokens.size))
    if tokens.size != expected:
        return "image_length"
    return None


def _validate_audio(tokens: np.ndarray, meta: Dict[str, Any], token_space) -> Optional[str]:
    n_codebooks = int(meta.get("n_codebooks"))
    n_frames = int(meta.get("n_frames"))
    expected = n_codebooks * n_frames
    if tokens.size != expected:
        return "audio_length"
    ranges = _audio_ranges(token_space)
    for idx, tok in enumerate(tokens.tolist()):
        cb = idx % n_codebooks
        start, end = ranges[cb]
        if tok < start or tok > end:
            return "audio_range"
    return None


def _decode_check_image(uti: UnifiedTokenizer, tokens: np.ndarray, meta: Dict[str, Any]) -> Optional[str]:
    meta = dict(meta) if meta else {}
    meta.setdefault("decode_mode", "deterministic")
    img = uti.decode_image(tokens.tolist(), meta)
    proc_size = meta.get("proc_size")
    if proc_size and hasattr(img, "size"):
        if tuple(img.size) != (int(proc_size[1]), int(proc_size[0])):
            return "image_decode_size"
    return None


def _decode_check_audio(uti: UnifiedTokenizer, tokens: np.ndarray, meta: Dict[str, Any]) -> Optional[str]:
    wav, sr = uti.decode_audio(tokens.tolist(), meta)
    expected_sr = int(meta.get("sample_rate"))
    if sr != expected_sr:
        return "audio_decode_sr"
    clip_seconds = float(meta.get("clip_seconds", 0.0))
    expected_len = int(round(clip_seconds * expected_sr)) if clip_seconds else None
    if expected_len is not None:
        length = wav.shape[-1]
        if abs(length - expected_len) > 1:
            return "audio_decode_len"
    expected_channels = int(meta.get("channels", wav.shape[0]))
    if wav.ndim == 2 and wav.shape[0] != expected_channels:
        return "audio_decode_channels"
    return None


def _load_hashes(path: Optional[Path]) -> Dict[str, Dict[str, str]]:
    if not path or not path.exists():
        return {}
    data: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            sample_id = str(item.get("id"))
            data[sample_id] = item
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit tokenized WebDataset shards.")
    parser.add_argument("--uti-config", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--data-dir", default="stage3_uti/data")
    parser.add_argument("--range-samples", type=int, default=1000)
    parser.add_argument("--decode-samples", type=int, default=32)
    parser.add_argument("--compare-hashes")
    parser.add_argument("--hashes-jsonl")
    parser.add_argument("--output-report")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    dataset = args.dataset_name
    tokenized_root = data_dir / "tokenized" / dataset
    reports_dir = data_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    uti = UnifiedTokenizer.from_config(args.uti_config)
    token_space = uti.token_space
    token_space_sha = token_space.sha256()

    compare_hashes = _load_hashes(Path(args.compare_hashes)) if args.compare_hashes else {}
    hashes_out = Path(args.hashes_jsonl) if args.hashes_jsonl else None
    hashes_fp = hashes_out.open("w", encoding="utf-8") if hashes_out else None

    total = 0
    token_space_mismatch = 0
    per_task: Dict[str, int] = {}
    length_audio: List[int] = []
    length_image: List[int] = []
    length_text: List[int] = []

    failures: Dict[str, int] = {}
    hash_mismatches: Dict[str, int] = {}

    range_reservoir = Reservoir(args.range_samples)
    audio_decode_reservoir = Reservoir(args.decode_samples)
    image_decode_reservoir = Reservoir(args.decode_samples)

    audio_hash = hashlib.sha256()
    image_hash = hashlib.sha256()
    text_hash = hashlib.sha256()

    shards = sorted((tokenized_root / "train").glob("shard-*.tar")) + sorted(
        (tokenized_root / "eval").glob("shard-*.tar")
    )
    for shard in shards:
        for _, sample in iter_tar_samples(shard):
            meta = sample.get("json") or {}
            sample_id = str(meta.get("id"))
            total += 1

            per_task[str(meta.get("task"))] = per_task.get(str(meta.get("task")), 0) + 1
            if meta.get("token_space_sha256") != token_space_sha:
                token_space_mismatch += 1

            text_out = sample.get("text_out")
            audio = sample.get("audio")
            image = sample.get("image")

            if isinstance(text_out, np.ndarray) and text_out.size > 0:
                length_text.append(int(text_out.size))
                text_hash.update(sample_id.encode("utf-8"))
                text_hash.update(text_out.astype(np.int32).tobytes())
            if isinstance(audio, np.ndarray) and audio.size > 0:
                length_audio.append(int(audio.size))
                audio_hash.update(sample_id.encode("utf-8"))
                audio_hash.update(audio.astype(np.int32).tobytes())
            if isinstance(image, np.ndarray) and image.size > 0:
                length_image.append(int(image.size))
                image_hash.update(sample_id.encode("utf-8"))
                image_hash.update(image.astype(np.int32).tobytes())

            range_reservoir.add(sample)
            if isinstance(audio, np.ndarray) and audio.size > 0:
                audio_decode_reservoir.add(sample)
            if isinstance(image, np.ndarray) and image.size > 0:
                image_decode_reservoir.add(sample)

            if hashes_fp:
                hashes = {
                    "id": sample_id,
                    "audio_sha256": sha256_int32(audio) if isinstance(audio, np.ndarray) and audio.size > 0 else None,
                    "image_sha256": sha256_int32(image) if isinstance(image, np.ndarray) and image.size > 0 else None,
                    "text_out_sha256": sha256_int32(text_out)
                    if isinstance(text_out, np.ndarray) and text_out.size > 0
                    else None,
                }
                hashes_fp.write(json.dumps(hashes) + "\n")

            if compare_hashes:
                prev = compare_hashes.get(sample_id)
                if prev:
                    if "audio_sha256" in prev and prev["audio_sha256"]:
                        curr = sha256_int32(audio) if isinstance(audio, np.ndarray) else None
                        if curr != prev["audio_sha256"]:
                            hash_mismatches["audio"] = hash_mismatches.get("audio", 0) + 1
                    if "image_sha256" in prev and prev["image_sha256"]:
                        curr = sha256_int32(image) if isinstance(image, np.ndarray) else None
                        if curr != prev["image_sha256"]:
                            hash_mismatches["image"] = hash_mismatches.get("image", 0) + 1
                    if "text_out_sha256" in prev and prev["text_out_sha256"]:
                        curr = sha256_int32(text_out) if isinstance(text_out, np.ndarray) else None
                        if curr != prev["text_out_sha256"]:
                            hash_mismatches["text_out"] = hash_mismatches.get("text_out", 0) + 1

    if hashes_fp:
        hashes_fp.close()

    # Range + shape sanity
    for sample in range_reservoir.items:
        meta = sample.get("json") or {}
        text_out = sample.get("text_out")
        audio = sample.get("audio")
        image = sample.get("image")

        if isinstance(text_out, np.ndarray) and text_out.size > 0:
            err = _validate_text(text_out, token_space)
            if err:
                failures[err] = failures.get(err, 0) + 1
        if isinstance(image, np.ndarray) and image.size > 0:
            err = _validate_image(image, (meta.get("meta") or {}).get("image", {}), token_space)
            if err:
                failures[err] = failures.get(err, 0) + 1
        if isinstance(audio, np.ndarray) and audio.size > 0:
            err = _validate_audio(audio, (meta.get("meta") or {}).get("audio", {}), token_space)
            if err:
                failures[err] = failures.get(err, 0) + 1

    # Decode spot-check
    for sample in audio_decode_reservoir.items:
        meta = sample.get("json") or {}
        audio = sample.get("audio")
        if not isinstance(audio, np.ndarray) or audio.size == 0:
            continue
        err = _decode_check_audio(uti, audio, (meta.get("meta") or {}).get("audio", {}))
        if err:
            failures[err] = failures.get(err, 0) + 1

    for sample in image_decode_reservoir.items:
        meta = sample.get("json") or {}
        image = sample.get("image")
        if not isinstance(image, np.ndarray) or image.size == 0:
            continue
        err = _decode_check_image(uti, image, (meta.get("meta") or {}).get("image", {}))
        if err:
            failures[err] = failures.get(err, 0) + 1

    report = {
        "dataset": dataset,
        "records_total": total,
        "token_space_sha256": token_space_sha,
        "token_space_mismatch": token_space_mismatch,
        "task_counts": per_task,
        "range_samples_checked": len(range_reservoir.items),
        "decode_audio_checked": len(audio_decode_reservoir.items),
        "decode_image_checked": len(image_decode_reservoir.items),
        "failures": failures,
        "hash_mismatches": hash_mismatches,
        "audio_token_lengths": percentile_stats(length_audio),
        "image_token_lengths": percentile_stats(length_image),
        "text_out_token_lengths": percentile_stats(length_text),
        "audio_tokens_sha256": audio_hash.hexdigest(),
        "image_tokens_sha256": image_hash.hexdigest(),
        "text_out_tokens_sha256": text_hash.hexdigest(),
    }

    report_path = Path(args.output_report) if args.output_report else reports_dir / f"{dataset}_audit.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
