import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from stage3_uti.tokenization.token_space import TokenSpace
from stage3_uti.tokenization.unified_tokenizer import UnifiedTokenizer


def _hash_tokens(tokens) -> str:
    arr = np.asarray(tokens, dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _write_sha(path: Path, value: str) -> None:
    path.write_text(value + "\n", encoding="utf-8")


def _load_wav(path: Path) -> tuple[np.ndarray, int]:
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


def _write_wav(path: Path, wav: np.ndarray, sr: int) -> None:
    import wave

    if wav.ndim == 2:
        wav = wav[0]
    wav = np.clip(wav, -1.0, 1.0)
    pcm = (wav * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def _snr_db(ref: np.ndarray, est: np.ndarray) -> float:
    eps = 1e-8
    num = np.sum(ref ** 2)
    den = np.sum((ref - est) ** 2)
    return float(10.0 * np.log10((num + eps) / (den + eps)))


def _mae(ref: np.ndarray, est: np.ndarray) -> float:
    return float(np.mean(np.abs(ref - est)))


def _log_mel_distance(ref: np.ndarray, est: np.ndarray, sr: int) -> tuple[Optional[float], Optional[str]]:
    try:
        import torchaudio
        import torch

        n_fft = 1024
        hop = 256
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop, n_mels=80
        )
        ref_t = torch.from_numpy(ref)
        est_t = torch.from_numpy(est)
        ref_mel = mel(ref_t)
        est_mel = mel(est_t)
        ref_log = torch.log(ref_mel + 1e-6)
        est_log = torch.log(est_mel + 1e-6)
        dist = torch.mean(torch.abs(ref_log - est_log)).item()
        return float(dist), None
    except Exception as exc:
        return None, str(exc)


def _psnr(ref: np.ndarray, est: np.ndarray) -> float:
    mse = np.mean((ref - est) ** 2)
    if mse <= 1e-12:
        return float("inf")
    return float(20.0 * np.log10(1.0 / np.sqrt(mse)))


def _ssim_global(ref: np.ndarray, est: np.ndarray) -> float:
    c1 = (0.01 ** 2)
    c2 = (0.03 ** 2)
    mu_x = float(np.mean(ref))
    mu_y = float(np.mean(est))
    sigma_x = float(np.var(ref))
    sigma_y = float(np.var(est))
    sigma_xy = float(np.mean((ref - mu_x) * (est - mu_y)))
    num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    if den == 0:
        return 0.0
    return float(num / den)


def _tokens_in_range(tokens: List[int], start: int, end: int) -> bool:
    for tok in tokens:
        if tok < start or tok > end:
            return False
    return True


def _compare_metrics(metrics: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    base_metrics = baseline.get("metrics_v2", {})

    snr = metrics.get("audio_snr_db")
    base_snr = base_metrics.get("audio_snr_db")
    if snr is not None and base_snr is not None:
        checks["audio_snr_db"] = float(snr) >= float(base_snr) - 1.0

    mel = metrics.get("audio_log_mel_l1")
    base_mel = base_metrics.get("audio_log_mel_l1")
    if mel is not None and base_mel is not None:
        checks["audio_log_mel_l1"] = float(mel) <= float(base_mel) * 1.05

    psnr = metrics.get("image_psnr")
    base_psnr = base_metrics.get("image_psnr")
    if psnr is not None and base_psnr is not None:
        checks["image_psnr"] = float(psnr) >= float(base_psnr) - 0.5

    ssim = metrics.get("image_ssim")
    base_ssim = base_metrics.get("image_ssim")
    if ssim is not None and base_ssim is not None:
        checks["image_ssim"] = float(ssim) >= float(base_ssim) - 0.02

    return checks


def _runtime_size_check(uti: UnifiedTokenizer, report: Dict[str, Any]) -> None:
    img_size = int(getattr(uti.image_tokenizer, "codebook_size", 0))
    aud_size = int(getattr(uti.audio_codec, "codebook_size", 0))
    aud_cbs = int(getattr(uti.audio_codec, "n_codebooks", 0))

    report["runtime_sizes"] = {
        "image_codebook_size": img_size,
        "audio_codebook_size": aud_size,
        "audio_n_codebooks": aud_cbs,
    }

    ts = uti.token_space
    img_range = ts.ranges.get("IMAGE", {})
    report["runtime_size_match"] = {
        "image_codebook": int(img_range.get("size", -1)) == img_size,
    }
    audio_ok = True
    if aud_cbs:
        for i in range(aud_cbs):
            rng = ts.ranges.get(f"AUDIO_CB{i}", {})
            if int(rng.get("size", -1)) != aud_size:
                audio_ok = False
                break
    report["runtime_size_match"]["audio_codebook"] = audio_ok


def run_audit(
    *,
    config_path: str,
    outdir: str,
    smoke_assets: str,
    decode_mode: str,
    baseline_report: Optional[str] = None,
) -> Dict[str, Any]:
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    uti = UnifiedTokenizer.from_config(config_path)
    token_space = uti.token_space
    token_space.validate()
    token_space.save_json(str(out_path / "token_space.json"))
    _write_sha(out_path / "token_space.sha256", token_space.sha256())

    report: Dict[str, Any] = {
        "config": str(config_path),
        "decode_mode": decode_mode,
        "assets": str(smoke_assets),
    }
    if baseline_report:
        report["baseline_report"] = str(baseline_report)

    _runtime_size_check(uti, report)

    assets = Path(smoke_assets)
    image_path = assets / "image0.png"
    audio_path = assets / "audio0.wav"
    if not image_path.exists() or not audio_path.exists():
        raise FileNotFoundError("Missing image0.png or audio0.wav in smoke assets")

    img = Image.open(image_path).convert("RGB")
    wav, sr = _load_wav(audio_path)

    text_tokens_1 = uti.encode_text("hello")
    text_tokens_2 = uti.encode_text("hello")
    report["text_deterministic"] = text_tokens_1 == text_tokens_2

    img_tokens_1, img_meta_1 = uti.encode_image(img)
    img_tokens_2, img_meta_2 = uti.encode_image(img)
    report["image_deterministic"] = img_tokens_1 == img_tokens_2 and img_meta_1 == img_meta_2
    report["image_token_count"] = len(img_tokens_1)
    grid = img_meta_1.get("grid") or (img_meta_1.get("token_meta") or {}).get("grid")
    if grid:
        prod = 1
        for dim in grid:
            prod *= int(dim)
        report["image_shape_ok"] = prod == len(img_tokens_1)
    else:
        report["image_shape_ok"] = False

    aud_tokens_1, aud_meta_1 = uti.encode_audio(wav, sr)
    aud_tokens_2, aud_meta_2 = uti.encode_audio(wav, sr)
    report["audio_deterministic"] = aud_tokens_1 == aud_tokens_2 and aud_meta_1 == aud_meta_2
    report["audio_token_count"] = len(aud_tokens_1)

    # Token range checks
    text_range = token_space.ranges.get("TEXT", {})
    text_start = int(text_range.get("start", 0))
    text_end = int(text_range.get("end", -1))
    report["text_token_range_ok"] = _tokens_in_range(text_tokens_1, text_start, text_end)

    img_range = token_space.ranges.get("IMAGE", {})
    img_start = int(img_range.get("start", 0))
    img_end = int(img_range.get("end", -1))
    report["image_token_range_ok"] = _tokens_in_range(img_tokens_1, img_start, img_end)

    audio_range_ok = True
    audio_codebook_order_ok = True
    audio_code_idx_ok = True
    aud_size = int(getattr(uti.audio_codec, "codebook_size", 0))
    n_codebooks = int(aud_meta_1.get("n_codebooks", 0))
    for idx, tok in enumerate(aud_tokens_1):
        try:
            cb_idx, code = token_space.aud_from_global(int(tok))
        except Exception:
            audio_range_ok = False
            break
        expected_cb = idx % n_codebooks if n_codebooks else cb_idx
        if cb_idx != expected_cb:
            audio_codebook_order_ok = False
        if aud_size and (code < 0 or code >= aud_size):
            audio_code_idx_ok = False
    report["audio_token_range_ok"] = audio_range_ok
    report["audio_codebook_order_ok"] = audio_codebook_order_ok
    report["audio_code_idx_ok"] = audio_code_idx_ok

    _write_sha(out_path / "image_tokens.sha256", _hash_tokens(img_tokens_1))
    _write_sha(out_path / "audio_tokens.sha256", _hash_tokens(aud_tokens_1))

    # Decode smoke
    img_decode_ok = False
    if decode_mode in {"diffusion", "deterministic"}:
        try:
            img_meta_decode = dict(img_meta_1)
            img_meta_decode["decode_mode"] = decode_mode
            img_recon = uti.decode_image(img_tokens_1, img_meta_decode)
            img_recon.save(out_path / "recon.png")
            img_decode_ok = True
        except Exception as exc:
            report["image_decode_error"] = str(exc)

    wav_recon, sr_recon = uti.decode_audio(aud_tokens_1, aud_meta_1)
    _write_wav(out_path / "recon.wav", wav_recon, sr_recon)

    # Metrics (v2)
    metrics: Dict[str, Any] = {}
    try:
        ref = wav.astype(np.float32)
        est = wav_recon.astype(np.float32)
        if ref.ndim == 2:
            ref = np.mean(ref, axis=0)
        if est.ndim == 2:
            est = np.mean(est, axis=0)
        n = min(ref.shape[-1], est.shape[-1])
        ref = ref[:n]
        est = est[:n]
        metrics["audio_snr_db"] = _snr_db(ref, est)
        metrics["audio_mae"] = _mae(ref, est)
        log_mel, err = _log_mel_distance(ref, est, sr_recon)
        metrics["audio_log_mel_l1"] = log_mel
        if err:
            metrics["audio_log_mel_error"] = err
    except Exception as exc:
        metrics["audio_metric_error"] = str(exc)

    if img_decode_ok and decode_mode == "deterministic":
        try:
            ref_img = img.copy()
            if img_recon.size != ref_img.size:
                img_recon_resized = img_recon.resize(ref_img.size, Image.BICUBIC)
            else:
                img_recon_resized = img_recon
            ref_np = np.asarray(ref_img, dtype=np.float32) / 255.0
            est_np = np.asarray(img_recon_resized, dtype=np.float32) / 255.0
            metrics["image_psnr"] = _psnr(ref_np, est_np)
            metrics["image_ssim"] = _ssim_global(ref_np, est_np)
        except Exception as exc:
            metrics["image_metric_error"] = str(exc)

    if metrics:
        report["metrics_v2"] = metrics

    # Decode sanity checks
    sr_expected = int(aud_meta_1.get("sample_rate", 0))
    report["audio_sample_rate_ok"] = bool(sr_expected) and (sr_recon == sr_expected)
    channels_expected = int(aud_meta_1.get("channels", 0))
    if isinstance(wav_recon, np.ndarray) and wav_recon.ndim == 2:
        channels_actual = int(wav_recon.shape[0])
    else:
        channels_actual = 1
    report["audio_channels_ok"] = (channels_expected == 0) or (channels_actual == channels_expected)

    clip_seconds = float(aud_meta_1.get("clip_seconds", 0.0))
    if sr_expected and clip_seconds:
        expected_len = int(round(clip_seconds * sr_expected))
        actual_len = int(wav_recon.shape[-1]) if hasattr(wav_recon, "shape") else 0
        tol = max(1, int(0.01 * sr_expected))
        report["audio_length_expected"] = expected_len
        report["audio_length_actual"] = actual_len
        report["audio_length_tol"] = tol
        report["audio_length_ok"] = abs(actual_len - expected_len) <= tol
    else:
        report["audio_length_ok"] = False

    if img_decode_ok:
        proc_size = img_meta_1.get("proc_size") or img_meta_1.get("orig_size")
        if isinstance(proc_size, (list, tuple)) and len(proc_size) == 2:
            expected_size = (int(proc_size[1]), int(proc_size[0]))
            report["image_size_expected"] = list(expected_size)
            report["image_size_actual"] = list(img_recon.size)
            report["image_size_ok"] = img_recon.size == expected_size
        else:
            report["image_size_ok"] = False

    # Shape sanity
    report["audio_shape_ok"] = len(aud_tokens_1) == int(aud_meta_1.get("n_frames")) * int(
        aud_meta_1.get("n_codebooks")
    )

    # Idempotence checks
    report["audio_idempotent"] = False
    try:
        aud_tokens_re, _ = uti.encode_audio(wav_recon, sr_recon)
        report["audio_idempotent"] = aud_tokens_re == aud_tokens_1
    except Exception as exc:
        report["audio_idempotent_error"] = str(exc)

    if img_decode_ok and decode_mode == "deterministic":
        try:
            img_tokens_re, _ = uti.encode_image(img_recon)
            report["image_idempotent"] = img_tokens_re == img_tokens_1
        except Exception as exc:
            report["image_idempotent_error"] = str(exc)

    # Round-trip stability (signal domain)
    try:
        aud_tokens_re, aud_meta_re = uti.encode_audio(wav_recon, sr_recon)
        wav_round, sr_round = uti.decode_audio(aud_tokens_re, aud_meta_re)
        if sr_round == sr_recon:
            ref = wav_recon.astype(np.float32)
            est = wav_round.astype(np.float32)
            if ref.ndim == 2:
                ref = np.mean(ref, axis=0)
            if est.ndim == 2:
                est = np.mean(est, axis=0)
            n = min(ref.shape[-1], est.shape[-1])
            ref = ref[:n]
            est = est[:n]
            report["audio_roundtrip_snr_db"] = _snr_db(ref, est)
            report["audio_roundtrip_mae"] = _mae(ref, est)
    except Exception as exc:
        report["audio_roundtrip_error"] = str(exc)

    report["image_decode_ok"] = img_decode_ok
    report["audio_decode_ok"] = True

    # Optional baseline regression checks
    baseline_path = Path(report.get("baseline_report", "")) if report.get("baseline_report") else None
    if baseline_path and baseline_path.exists():
        try:
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            report["metrics_v2_baseline"] = baseline.get("metrics_v2", {})
            if "metrics_v2" in report:
                checks = _compare_metrics(report["metrics_v2"], baseline)
                if checks:
                    report["metrics_v2_checks"] = checks
                    report["metrics_v2_pass"] = all(checks.values())
        except Exception as exc:
            report["metrics_v2_baseline_error"] = str(exc)

    report_path = out_path / "report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--token-space-out", required=True)
    parser.add_argument("--smoke-assets", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--decode-mode", default="diffusion", choices=["diffusion", "deterministic", "none"])
    parser.add_argument("--baseline-report", default=None)
    args = parser.parse_args()

    outdir = args.outdir
    token_space_out = Path(args.token_space_out)
    if token_space_out.parent:
        token_space_out.parent.mkdir(parents=True, exist_ok=True)

    report = run_audit(
        config_path=args.config,
        outdir=outdir,
        smoke_assets=args.smoke_assets,
        decode_mode=args.decode_mode,
        baseline_report=args.baseline_report,
    )

    # Copy token_space.json to requested path for convenience
    token_space_src = Path(outdir) / "token_space.json"
    if token_space_src.exists():
        token_space_out.write_text(token_space_src.read_text(encoding="utf-8"), encoding="utf-8")
        sha_src = Path(outdir) / "token_space.sha256"
        if sha_src.exists():
            token_space_out.with_suffix(".sha256").write_text(
                sha_src.read_text(encoding="utf-8"), encoding="utf-8"
            )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
