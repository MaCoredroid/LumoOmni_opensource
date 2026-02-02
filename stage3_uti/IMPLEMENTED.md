# Stage 3 UTI + Minimal Token LM — Implemented Summary

This is a concise record of **what is implemented** in `stage3_uti/` so far.
Stage 1/2 code paths are untouched.

---

## 1) Unified Token Interface (UTI)

### Core
- `stage3_uti/tokenization/token_space.py`
  - `TokenSpace` with validation, JSON I/O, and mapping helpers.
  - `build_token_space_from_config(...)` builds contiguous ID ranges.

- `stage3_uti/tokenization/unified_tokenizer.py`
  - `UnifiedTokenizer` with:
    - `encode_text`, `encode_image`, `encode_audio`
    - `decode_image`, `decode_audio`
  - Deterministic preprocessing for audio + image when adapters don’t handle it.
  - JSON‑serializable metadata for all modalities.
  - Container path mapping for `/media/mark/SHARED/lumoOmni` → `/workspace/lumoOmni`.

### Real adapters
- **SEED** (image tokenizer)
  - `SeedImageTokenizerAdapter` loads `seed_quantizer.pt` and uses the AnyGPT `seed2` stack.
  - Optional diffusion decode (requires `load_diffusion=true` + `diffusion_model_path`).

- **EnCodec** (audio codec)
  - `EncodecAudioCodecAdapter` uses HF `EncodecModel` to encode/decode.
  - Handles codebook count via target bandwidth or explicit `n_codebooks`.
  - Handles newer `EncodecModel.encode` return shapes (frame‑wise codes + pad length).
  - Upmixes mono → stereo when codec expects 2 channels.

### Dummy adapters (tests)
- `DummyImageTokenizer`, `DummyAudioCodec` for deterministic tests without external weights.

---

## 2) UTI Config + Token Space

- `stage3_uti/configs/uti.yaml`
  - SEED + EnCodec wired to local paths.
  - SEED diffusion decode enabled (UnCLIP local path set).
  - Explicit codebook sizes for predictable token‑space building.

- Generated outputs:
  - `outputs/stage3_token_lm/token_space.json`
  - `outputs/stage3_token_lm/token_space.sha256`

---

## 3) Tokenized Dataset Pipeline

- `stage3_uti/pipeline/sequence_builder.py`
  - Canonical sequence assembly using UTI special tokens.
  - Helpers for supervised labels and span detection.

- `stage3_uti/pipeline/tokenize_dataset.py`
  - JSONL → UTI tokens for `t2i`, `t2a`, `i2t`, `a2t` tasks.
  - Emits `input_ids` with global IDs and modality metadata.

- `stage3_uti/pipeline/build_token_space.py`
  - Writes `token_space.json` from config + tokenizer sizes.

---

## 3.2) Stage 2 Tokenized Dataset Pipeline (WebDataset)

- `stage3_uti/stage2/tokenize_wds.py`
  - JSONL manifest → WebDataset tar shards (train/eval splits).
  - Emits `token_space.json` + `token_space.sha256`.
  - Writes tokenized samples with modality `.npy` arrays + metadata JSON.
  - Deterministic split policy via hash of `id`.
  - Token-length stats + per-sample token hashes.

- `stage3_uti/stage2/manifest_audit.py`
  - Raw manifest validation + dataset stats (paths, durations, task counts).

- `stage3_uti/stage2/audit_tokenized.py`
  - Range/shape checks on sampled shards.
  - Decode spot-checks (audio + image).
  - Reproducibility hashes + mismatch comparison.

- `stage3_uti/stage2/wds_io.py`
  - Minimal WebDataset tar writer/reader (no external dependency).

---

## 4) Minimal Stage‑3 Token LM Training

- `stage3_uti/train/stage3_token_lm.py`
  - CLI entrypoint.

- `stage3_uti/utils/train_utils.py`
  - Minimal training loop over tokenized JSONL.
  - Pads with `<|pad_mm|>` from `token_space.json`.
  - Resizes LLM embeddings to `vocab_size_total`.
  - Saves checkpoints in `outputs/stage3_token_lm/checkpoint_*`.

- `stage3_uti/data/tokenized_jsonl.py`
  - Dataset loader + simple collator.

- `stage3_uti/configs/stage3_token_lm.yaml`
  - Training config stub (batch, lr, precision, etc.).

- `stage3_uti/scripts/train_stage3_token_lm.sh`
  - Simple run script (inside CUDA container).

---

## 5) Tests + Assets

- `stage3_uti/tests/test_token_space.py`
- `stage3_uti/tests/test_uti.py`
- `stage3_uti/tests/assets/image0.png`
- `stage3_uti/tests/assets/audio0.wav`

---

## 6) UTI Audit (Verification Script)

- `stage3_uti/tokenization/uti_audit.py`
  - Determinism checks for text/image/audio.
  - Runtime size introspection vs `token_space.json`.
  - Encode/decode smoke (writes recon outputs + report).

Latest run (January 25, 2026, decode_mode=diffusion) wrote:
- `outputs/uti_audit/token_space.json`
- `outputs/uti_audit/token_space.sha256`
- `outputs/uti_audit/image_tokens.sha256`
- `outputs/uti_audit/audio_tokens.sha256`
- `outputs/uti_audit/recon.png`
- `outputs/uti_audit/recon.wav`
- `outputs/uti_audit/report.json`

Notes from report:
- `runtime_size_match` = true (image/audio)
- `text_deterministic`, `image_deterministic`, `audio_deterministic` = true
- `image_decode_ok`, `audio_decode_ok` = true
- `audio_idempotent` = false (expected with EnCodec)

---

## 7) Container Reminder

Training must run in the CUDA container (see top‑level `readme.md`).
Inside container:

```bash
cd /workspace/lumoOmni
source /opt/lumo/venv/bin/activate
python3 -m stage3_uti.train.stage3_token_lm --config stage3_uti/configs/stage3_token_lm.yaml
```

---

## 8) Files Added / Updated

Key paths (non‑exhaustive):

- `stage3_uti/tokenization/token_space.py`
- `stage3_uti/tokenization/unified_tokenizer.py`
- `stage3_uti/tokenization/__init__.py`
- `stage3_uti/configs/uti.yaml`
- `stage3_uti/pipeline/sequence_builder.py`
- `stage3_uti/pipeline/tokenize_dataset.py`
- `stage3_uti/pipeline/build_token_space.py`
- `stage3_uti/data/tokenized_jsonl.py`
- `stage3_uti/utils/train_utils.py`
- `stage3_uti/train/stage3_token_lm.py`
- `stage3_uti/configs/stage3_token_lm.yaml`
- `stage3_uti/scripts/train_stage3_token_lm.sh`
- `stage3_uti/tests/*`

---

## 9) Not Implemented Yet (Explicit)

- Full Stage‑3 data mixing curriculum or evaluation suite.
- LoRA / adapter‑only training for Stage‑3.
- Advanced sampling or modality‑specific heads.
