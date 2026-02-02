import argparse
import json
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from stage3_uti.stage2.utils import sha256_int32
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
        idx = np.random.randint(0, self.seen)
        if idx < self.size:
            self.items[idx] = item


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


def _first_text_value(text: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for key in keys:
        val = text.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return None


def _tokens_equal(stored: Optional[np.ndarray], new_tokens: Optional[List[int]]) -> bool:
    stored_arr = stored if isinstance(stored, np.ndarray) else None
    new_list = new_tokens or []
    if stored_arr is None or stored_arr.size == 0:
        return len(new_list) == 0
    if len(new_list) != int(stored_arr.size):
        return False
    return np.array_equal(stored_arr.astype(np.int32), np.asarray(new_list, dtype=np.int32))


def main() -> None:
    parser = argparse.ArgumentParser(description="Retokenize consistency check against raw assets.")
    parser.add_argument("--uti-config", required=True)
    parser.add_argument("--manifest-jsonl", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--data-dir", default="stage3_uti/data")
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--output-report")
    parser.add_argument("--errors-jsonl")
    args = parser.parse_args()

    uti = UnifiedTokenizer.from_config(args.uti_config)

    manifest_path = Path(args.manifest_jsonl)
    dataset = args.dataset_name
    data_dir = Path(args.data_dir)
    tokenized_root = data_dir / "tokenized" / dataset
    reports_dir = data_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    report_path = (
        Path(args.output_report)
        if args.output_report
        else reports_dir / f"{dataset}_retokenize.json"
    )
    errors_path = Path(args.errors_jsonl) if args.errors_jsonl else reports_dir / f"{dataset}_retokenize_errors.jsonl"

    manifest_index: Dict[str, Dict[str, Any]] = {}
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("id"):
                manifest_index[str(item["id"])] = item

    reservoir = Reservoir(args.num_samples)
    shards = sorted((tokenized_root / "train").glob("shard-*.tar")) + sorted(
        (tokenized_root / "eval").glob("shard-*.tar")
    )
    for shard in shards:
        for _, sample in iter_tar_samples(shard):
            reservoir.add(sample)

    mismatches: Dict[str, int] = {}
    mismatch_examples: Dict[str, List[str]] = {}
    missing_manifest = 0
    checked = 0

    with errors_path.open("w", encoding="utf-8") as err_out:
        for sample in reservoir.items:
            meta = sample.get("json") or {}
            sample_id = str(meta.get("id"))
            checked += 1

            manifest = manifest_index.get(sample_id)
            if not manifest:
                missing_manifest += 1
                err_out.write(json.dumps({"id": sample_id, "error": "missing_manifest"}) + "\n")
                continue

            modalities = manifest.get("modalities") or {}
            audio = modalities.get("audio") or {}
            image = modalities.get("image") or {}
            audio_path = audio.get("path") if isinstance(audio, dict) else None
            image_path = image.get("path") if isinstance(image, dict) else None
            expected_sr = audio.get("sr") if isinstance(audio, dict) else None

            text = manifest.get("text") or {}
            prompt = _first_text_value(text, ["prompt", "instruction"]) or ""
            caption = _first_text_value(text, ["caption", "response"]) or ""

            try:
                text_in_tokens = _encode_text(uti, prompt)
                text_out_tokens = _encode_text(uti, caption)
                image_tokens, _ = _encode_image(uti, image_path)
                audio_tokens, _ = _encode_audio(uti, audio_path, expected_sr)

                stored_text_in = sample.get("text_in")
                stored_text_out = sample.get("text_out")
                stored_image = sample.get("image")
                stored_audio = sample.get("audio")

                if not _tokens_equal(stored_text_in, text_in_tokens):
                    mismatches["text_in"] = mismatches.get("text_in", 0) + 1
                    mismatch_examples.setdefault("text_in", []).append(sample_id)
                if not _tokens_equal(stored_text_out, text_out_tokens):
                    mismatches["text_out"] = mismatches.get("text_out", 0) + 1
                    mismatch_examples.setdefault("text_out", []).append(sample_id)
                if not _tokens_equal(stored_image, image_tokens):
                    mismatches["image"] = mismatches.get("image", 0) + 1
                    mismatch_examples.setdefault("image", []).append(sample_id)
                if not _tokens_equal(stored_audio, audio_tokens):
                    mismatches["audio"] = mismatches.get("audio", 0) + 1
                    mismatch_examples.setdefault("audio", []).append(sample_id)

            except Exception as exc:
                err_out.write(
                    json.dumps(
                        {
                            "id": sample_id,
                            "error": "retokenize_exception",
                            "exception": str(exc),
                            "traceback": traceback.format_exc(limit=2),
                        }
                    )
                    + "\n"
                )
                mismatches["exception"] = mismatches.get("exception", 0) + 1

    # trim examples
    for key, vals in list(mismatch_examples.items()):
        mismatch_examples[key] = vals[:20]

    report = {
        "dataset": dataset,
        "samples_checked": checked,
        "missing_manifest": missing_manifest,
        "mismatch_counts": mismatches,
        "mismatch_examples": mismatch_examples,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
