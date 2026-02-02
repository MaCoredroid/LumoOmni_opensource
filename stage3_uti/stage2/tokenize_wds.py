import argparse
import json
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from stage3_uti.stage2.utils import assign_split, percentile_stats, sha256_int32
from stage3_uti.stage2.wds_io import TarShardWriter
from stage3_uti.tokenization.unified_tokenizer import UnifiedTokenizer


def _load_audio(path: Path) -> Tuple[np.ndarray, int]:
    try:
        import torchaudio

        wav, sr = torchaudio.load(str(path))
        wav = wav.detach().cpu().numpy()
        return wav.astype(np.float32), int(sr)
    except Exception:
        pass

    try:
        import soundfile as sf

        data, sr = sf.read(str(path), always_2d=True)
        wav = data.T.astype(np.float32)
        return wav, int(sr)
    except Exception:
        pass

    try:
        from scipy.io import wavfile

        sr, data = wavfile.read(path)
        if data.dtype == np.int16:
            wav = data.astype(np.float32) / 32767.0
        else:
            wav = data.astype(np.float32)
        if wav.ndim == 1:
            wav = wav[None, :]
        else:
            wav = wav.T
        return wav, int(sr)
    except Exception:
        import wave

        with wave.open(str(path), "rb") as wf:
            sr = wf.getframerate()
            n_channels = wf.getnchannels()
            frames = wf.readframes(wf.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
            if n_channels > 1:
                audio = audio.reshape(-1, n_channels).T
            else:
                audio = audio[None, :]
            return audio, int(sr)


def _encode_text(uti: UnifiedTokenizer, text: Optional[str]) -> List[int]:
    if not text:
        return []
    return uti.encode_text(text)


def _encode_image(uti: UnifiedTokenizer, path: Optional[str]) -> Tuple[Optional[List[int]], Optional[Dict[str, Any]]]:
    if not path:
        return None, None
    img = Image.open(path).convert("RGB")
    tokens, meta = uti.encode_image(img)
    return tokens, meta


def _encode_audio(
    uti: UnifiedTokenizer, path: Optional[str], expected_sr: Optional[int]
) -> Tuple[Optional[List[int]], Optional[Dict[str, Any]]]:
    if not path:
        return None, None
    wav, sr = _load_audio(Path(path))
    if expected_sr and int(expected_sr) != int(sr):
        # Keep actual sr for encoding; report mismatch in meta.
        pass
    tokens, meta = uti.encode_audio(wav, sr)
    if expected_sr and int(expected_sr) != int(sr):
        meta = dict(meta)
        meta["manifest_sample_rate"] = int(expected_sr)
    return tokens, meta


def _text_range(token_space) -> Tuple[int, int]:
    rng = token_space.ranges["TEXT"]
    return int(rng["start"]), int(rng["end"])


def _image_range(token_space) -> Tuple[int, int]:
    rng = token_space.ranges["IMAGE"]
    return int(rng["start"]), int(rng["end"])


def _audio_ranges(token_space) -> Dict[int, Tuple[int, int]]:
    ranges = {}
    for name, spec in token_space.ranges.items():
        if not name.startswith("AUDIO_CB"):
            continue
        cb = int(name.replace("AUDIO_CB", ""))
        ranges[cb] = (int(spec["start"]), int(spec["end"]))
    return ranges


def _validate_text(tokens: List[int], token_space) -> None:
    if not tokens:
        return
    start, end = _text_range(token_space)
    arr = np.asarray(tokens, dtype=np.int64)
    if arr.min() < start or arr.max() > end:
        raise ValueError("text tokens out of TEXT range")


def _validate_image(tokens: List[int], meta: Dict[str, Any], token_space) -> None:
    if tokens is None:
        return
    start, end = _image_range(token_space)
    arr = np.asarray(tokens, dtype=np.int64)
    if arr.min() < start or arr.max() > end:
        raise ValueError("image tokens out of IMAGE range")
    expected = int(meta.get("n_tokens", len(tokens)))
    if len(tokens) != expected:
        raise ValueError("image token length mismatch")


def _validate_audio(tokens: List[int], meta: Dict[str, Any], token_space) -> None:
    if tokens is None:
        return
    n_codebooks = int(meta.get("n_codebooks"))
    n_frames = int(meta.get("n_frames"))
    expected = n_codebooks * n_frames
    if len(tokens) != expected:
        raise ValueError("audio token length mismatch")
    ranges = _audio_ranges(token_space)
    arr = np.asarray(tokens, dtype=np.int64)
    for idx, tok in enumerate(arr.tolist()):
        cb = idx % n_codebooks
        start, end = ranges[cb]
        if tok < start or tok > end:
            raise ValueError("audio tokens out of AUDIO range")


def _task_requirements(task: str) -> Dict[str, bool]:
    task = str(task)
    if task == "a2t":
        return {"audio": True, "image": False, "text_out": True}
    if task == "i2t":
        return {"audio": False, "image": True, "text_out": True}
    if task == "t2a":
        return {"audio": True, "image": False, "text_in": True}
    if task == "t2i":
        return {"audio": False, "image": True, "text_in": True}
    return {"audio": False, "image": False, "text_out": False}


def _first_text_value(text: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for key in keys:
        val = text.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize manifest JSONL into WebDataset shards.")
    parser.add_argument("--uti-config", required=True)
    parser.add_argument("--manifest-jsonl", required=True)
    parser.add_argument("--dataset-name")
    parser.add_argument("--data-dir", default="stage3_uti/data")
    parser.add_argument("--max-samples-per-shard", type=int, default=1000)
    parser.add_argument("--eval-mod", type=int, default=1000)
    parser.add_argument("--eval-cutoff", type=int, default=10)
    parser.add_argument("--errors-jsonl")
    parser.add_argument("--hashes-jsonl")
    args = parser.parse_args()

    uti = UnifiedTokenizer.from_config(args.uti_config)
    token_space = uti.token_space
    token_space_sha = token_space.sha256()

    manifest_path = Path(args.manifest_jsonl)
    dataset = args.dataset_name or manifest_path.stem
    data_dir = Path(args.data_dir)

    tokenized_root = data_dir / "tokenized" / dataset
    splits_dir = data_dir / "splits" / dataset
    reports_dir = data_dir / "reports"
    errors_dir = data_dir / "errors"
    outputs_dir = data_dir / "outputs"
    tokenized_root.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    errors_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    (outputs_dir / "token_space.json").write_text(
        json.dumps(token_space.to_json(), indent=2), encoding="utf-8"
    )
    (outputs_dir / "token_space.sha256").write_text(token_space_sha + "\n", encoding="utf-8")

    errors_path = Path(args.errors_jsonl) if args.errors_jsonl else errors_dir / "tokenize_errors.jsonl"
    hashes_path = Path(args.hashes_jsonl) if args.hashes_jsonl else reports_dir / f"{dataset}_token_hashes.jsonl"

    train_writer = TarShardWriter(tokenized_root / "train", "train", args.max_samples_per_shard)
    eval_writer = TarShardWriter(tokenized_root / "eval", "eval", args.max_samples_per_shard)

    train_ids = (splits_dir / "train_ids.txt").open("w", encoding="utf-8")
    eval_ids = (splits_dir / "eval_ids.txt").open("w", encoding="utf-8")

    kept = 0
    skipped = 0
    failure_reasons: Dict[str, int] = {}
    task_counts: Dict[str, int] = {}

    audio_lengths: List[int] = []
    image_lengths: List[int] = []
    text_lengths: List[int] = []

    def _fail(reason: str) -> None:
        nonlocal skipped
        skipped += 1
        failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

    with manifest_path.open("r", encoding="utf-8") as src, \
            errors_path.open("w", encoding="utf-8") as err_out, \
            hashes_path.open("w", encoding="utf-8") as hash_out:
        for line in src:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                _fail("json_decode")
                err_out.write(json.dumps({"error": "json_decode", "line": line[:200]}) + "\n")
                continue

            sample_id = item.get("id")
            if not sample_id:
                _fail("missing_id")
                err_out.write(json.dumps({"error": "missing_id", "item": item}) + "\n")
                continue
            sample_id_str = str(sample_id)

            task = item.get("task") or "unknown"
            task_counts[str(task)] = task_counts.get(str(task), 0) + 1

            text = item.get("text") or {}
            prompt = _first_text_value(text, ["prompt", "instruction"]) or ""
            caption = _first_text_value(text, ["caption", "response"]) or ""

            modalities = item.get("modalities") or {}
            audio = modalities.get("audio") or {}
            image = modalities.get("image") or {}
            audio_path = audio.get("path") if isinstance(audio, dict) else None
            image_path = image.get("path") if isinstance(image, dict) else None
            expected_sr = audio.get("sr") if isinstance(audio, dict) else None

            req = _task_requirements(str(task))
            if req.get("audio") and not audio_path:
                _fail("missing_audio")
                err_out.write(json.dumps({"id": sample_id, "error": "missing_audio"}) + "\n")
                continue
            if req.get("image") and not image_path:
                _fail("missing_image")
                err_out.write(json.dumps({"id": sample_id, "error": "missing_image"}) + "\n")
                continue
            if req.get("text_out") and not caption:
                _fail("missing_text_out")
                err_out.write(json.dumps({"id": sample_id, "error": "missing_text_out"}) + "\n")
                continue
            if req.get("text_in") and not prompt:
                _fail("missing_text_in")
                err_out.write(json.dumps({"id": sample_id, "error": "missing_text_in"}) + "\n")
                continue

            try:
                text_in_tokens = _encode_text(uti, prompt)
                text_out_tokens = _encode_text(uti, caption)
                image_tokens, image_meta = _encode_image(uti, image_path)
                audio_tokens, audio_meta = _encode_audio(uti, audio_path, expected_sr)

                if text_out_tokens:
                    _validate_text(text_out_tokens, token_space)
                if text_in_tokens:
                    _validate_text(text_in_tokens, token_space)
                if image_tokens is not None and image_meta is not None:
                    _validate_image(image_tokens, image_meta, token_space)
                if audio_tokens is not None and audio_meta is not None:
                    _validate_audio(audio_tokens, audio_meta, token_space)

                sample = {
                    "id": sample_id_str,
                    "source": item.get("source"),
                    "task": task,
                    "token_space_sha256": token_space_sha,
                    "tokens": {
                        "text_in": len(text_in_tokens),
                        "text_out": len(text_out_tokens),
                        "audio": len(audio_tokens) if audio_tokens is not None else 0,
                        "image": len(image_tokens) if image_tokens is not None else 0,
                    },
                    "special": token_space.special_tokens,
                    "meta": {
                        "audio": audio_meta,
                        "image": image_meta,
                        "text": {
                            "prompt_len": len(text_in_tokens),
                            "label_len": len(text_out_tokens),
                        },
                        "source_meta": item.get("meta"),
                    },
                }

                arrays = {}
                if text_in_tokens:
                    arrays["text_in"] = np.asarray(text_in_tokens, dtype=np.int32)
                if text_out_tokens:
                    arrays["text_out"] = np.asarray(text_out_tokens, dtype=np.int32)
                if image_tokens is not None:
                    arrays["image"] = np.asarray(image_tokens, dtype=np.int32)
                if audio_tokens is not None:
                    arrays["audio"] = np.asarray(audio_tokens, dtype=np.int32)

                split = assign_split(sample_id_str, eval_mod=args.eval_mod, eval_cutoff=args.eval_cutoff)
                writer = eval_writer if split == "eval" else train_writer
                writer.write_sample(sample, arrays)

                if split == "eval":
                    eval_ids.write(sample_id_str + "\n")
                else:
                    train_ids.write(sample_id_str + "\n")

                kept += 1
                if audio_tokens is not None:
                    audio_lengths.append(len(audio_tokens))
                if image_tokens is not None:
                    image_lengths.append(len(image_tokens))
                if text_out_tokens:
                    text_lengths.append(len(text_out_tokens))

                hashes = {
                    "id": sample_id_str,
                    "audio_sha256": sha256_int32(audio_tokens) if audio_tokens is not None else None,
                    "image_sha256": sha256_int32(image_tokens) if image_tokens is not None else None,
                    "text_out_sha256": sha256_int32(text_out_tokens) if text_out_tokens else None,
                    "text_in_sha256": sha256_int32(text_in_tokens) if text_in_tokens else None,
                }
                hash_out.write(json.dumps(hashes) + "\n")
            except Exception as exc:
                _fail("encode_or_validate")
                err_out.write(
                    json.dumps(
                        {
                            "id": sample_id,
                            "error": "encode_or_validate",
                            "exception": str(exc),
                            "traceback": traceback.format_exc(limit=2),
                        }
                    )
                    + "\n"
                )
                continue

    train_writer.close()
    eval_writer.close()
    train_ids.close()
    eval_ids.close()

    report = {
        "dataset": dataset,
        "records_kept": kept,
        "records_skipped": skipped,
        "failure_reasons": failure_reasons,
        "task_counts": task_counts,
        "audio_token_lengths": percentile_stats(audio_lengths),
        "image_token_lengths": percentile_stats(image_lengths),
        "text_out_token_lengths": percentile_stats(text_lengths),
    }
    report_path = reports_dir / f"{dataset}_token_stats.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
