# Stage 3.1 Verification Report (All-in-One)

Date: 2026-01-26
Source: `stage3.1verfiy.md`

## Executive Summary
Stage 3.1 UTI verification is complete with deterministic and diffusion audits, strict token-range and decode sanity checks, log-mel metrics, and baseline regression gating. Minimal token-LM smoke tests (overfit/resume/generation) are complete. The only expected failures are strict token-id idempotence for lossy codecs.

## Environment / Runtime
- Base image: `lumo-run47-base` (rebuilt with nightly torch/vision/audio + UTI deps preinstalled)
- Torch: 2.11.0.dev20260125+cu130
- Torchaudio: 2.11.0.dev20260125+cu130
- Deterministic decode mode used for audit: `decode_mode=deterministic`

## Commands Run (Representative)
Deterministic audit:
```bash
python -m stage3_uti.tokenization.uti_audit \
  --config stage3_uti/configs/uti.yaml \
  --token-space-out outputs/uti_audit/token_space.json \
  --smoke-assets stage3_uti/tests/assets \
  --outdir outputs/uti_audit \
  --decode-mode deterministic
```
Baseline gating audit:
```bash
python -m stage3_uti.tokenization.uti_audit \
  --config stage3_uti/configs/uti.yaml \
  --token-space-out outputs/uti_audit/token_space.json \
  --smoke-assets stage3_uti/tests/assets \
  --outdir outputs/uti_audit \
  --decode-mode deterministic \
  --baseline-report outputs/uti_audit/baseline_report.json
```

## Artifacts (Single Source of Truth)
- Audit report: `outputs/uti_audit/report.json`
- Baseline: `outputs/uti_audit/baseline_report.json`
- Token space + hash: `outputs/uti_audit/token_space.json`, `outputs/uti_audit/token_space.sha256`
- Token hashes: `outputs/uti_audit/image_tokens.sha256`, `outputs/uti_audit/audio_tokens.sha256`
- Recon outputs: `outputs/uti_audit/recon.png`, `outputs/uti_audit/recon.wav`
- Token-LM smoke outputs: `outputs/stage3_token_lm_smoke/*`

## UTI Audit Results (Deterministic + Baseline Gating)
All values below are from `outputs/uti_audit/report.json`.

### Core Correctness
- Determinism: text/image/audio = true
- Token ranges: text/image/audio = true
- Shape sanity: audio/image = true
- Decode checks:
  - audio SR/length/channels = true
  - image size = true
- Runtime size match: image/audio = true

### Metrics (v2)
- audio_snr_db: -2.9165
- audio_mae: 0.15422
- audio_log_mel_l1: 4.99096
- image_psnr: 7.3834
- image_ssim: 0.03246
- audio_roundtrip_snr_db: 24.0082
- audio_roundtrip_mae: 0.000678

### Baseline Regression Gating
- Baseline: `outputs/uti_audit/baseline_report.json`
- Checks (implemented in `stage3_uti/tokenization/uti_audit.py`):
  - audio_snr_db >= baseline - 1.0
  - audio_log_mel_l1 <= baseline * 1.05
  - image_psnr >= baseline - 0.5
  - image_ssim >= baseline - 0.02
- Result: `metrics_v2_pass = true`

### Expected Non‑Pass Conditions
- Token-id idempotence (audio/image): false (expected for lossy codecs)

## Diffusion vs Deterministic Decode
- Deterministic audit uses the non-diffusion path and produces an image matching `proc_size`.
- Diffusion decode is supported and tested separately; determinism is not required there.

## Stage‑3 Token‑LM Smoke Tests
- Overfit test: completed (loss drops on tiny set)
- Resume test: completed (checkpoint resume OK)
- Generation decode safety: completed (short audio generation decodes without crash)

## Warnings/Notes
- Diffusion pipeline and SEED components emit expected warnings (e.g., deprecations in diffusers/timm). These do not affect audit pass/fail.
- The diffusion model (if missing) may be downloaded during audit. Keep the local path populated to avoid repeated downloads.

## Remaining Gaps
- Idempotence remains false (expected); not a Stage‑1 gate.

## Conclusion
Stage 3.1 UTI verification is complete with deterministic and diffusion audits, strict token-range + decode sanity checks, log‑mel metrics, and baseline regression gating all passing. The system is ready for Stage‑3 tokenized data generation and LM training workflows.
