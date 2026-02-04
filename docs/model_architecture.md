# Model architecture

This project currently uses two complementary multimodal architectures.

---

## Track A — Continuous‑embedding VLM

### Overview

```
image
  └─ SigLIP2 vision tower (frozen) → patch embeddings (B, N_patches, 1152)
        └─ Perceiver resampler (trainable) → latents (B, 64, 512)
              └─ MLP projector (trainable) → tokens in LLM space (B, 64, 4096)
                    └─ injected into Qwen3 input embedding sequence at <image_patch>
                          └─ Qwen3 LLM (frozen in Stage 1; LoRA in Stage 2)
```

### Concrete connector configuration

- Vision tower: **SigLIP2 SO400M patch14‑384**
  - hidden size: 1152
  - image size: 384
  - patch size: 14
  - `N_patches` = 27×27 = 729

- Resampler (Perceiver‑style):
  - `num_latents=64`
  - `depth=2`
  - `num_heads=8`
  - `head_dim=64`
  - latent dim = 512

- Projector (MLP):
  - `512 → 16384 → 4096` (LLM hidden size 4096)

### Parameter counts (connector only)

- Resampler: **6,927,872**
- Projector: **75,517,952**
- Total: **82,445,824** trainable parameters
- Optional `vision_ln`: 2,304 parameters (disabled in Stage 1)

### Code locations (as implemented)

- Resampler: `qwen3_vlm/models/resampler.py`
- Projector: `qwen3_vlm/models/projector.py`
- VLM wrapper / injection: `qwen3_vlm/models/vlm.py`
- Training entry: `qwen3_vlm/train/train_utils.py`

---

## Track B — Discrete tokens (UTI + token LM)

### Unified Token Interface (UTI)

UTI defines:
- how text is tokenized (lossless),
- how images are encoded to discrete token IDs + decode metadata,
- how audio is encoded to discrete token IDs + decode metadata,
- a single **global token space** (ranges per modality) stored in `token_space.json`.

The `token_space.json` + hash is treated as part of the model ABI.

### Token LM architecture

The token LM is conceptually:

- Base: a text‑only causal LLM (Qwen3)
- Modification: **resize vocab** to include image/audio token ID ranges
- Training (Stage 3): learn only the **new vocabulary rows**
  - input embeddings (added rows)
  - LM head (added rows)

Later stages can add:
- LoRA on attention/MLP blocks,
- full fine‑tuning,
- modality‑aware sampling/decoding.

### Trainable‑rows checkpointing

To keep distribution lightweight:
- Save only the trained rows (`trainable_rows.pt`) + metadata (`trainable_rows.json`)
- Reconstruct full weights by loading the base model and patching in the trained rows
- Always load the matching `token_space.json` (and ideally verify the hash)

This is the discrete‑token analogue of “connector‑only checkpoints” in the VLM track.
