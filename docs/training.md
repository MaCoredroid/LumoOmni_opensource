# Training

This page summarizes how training is staged and why. The emphasis is on **controlling degrees of freedom** (freeze most things early), plus **audits/regression gates**.

---

## Track A — Continuous‑embedding VLM

### Stage 1: Connector alignment (captioning‑style)

**Goal:** Make the frozen LLM attend to injected visual tokens.

**Trainable**
- resampler + projector (connector)
- optional vision LayerNorm (typically off initially)

**Frozen**
- Qwen3 LLM
- SigLIP2 vision tower

**Typical config highlights**
- `num_image_tokens=64`
- `max_seq_len=512` for early trials
- bf16 precision
- teacher‑forced next token prediction

**Key sanity gates**
- ablation loss ordering: correct < shuffled < zero/noise
- truncation rate and label coverage

### Stage 1.1: Stability sweep + qualitative harness

**Goal:** Select stable hyperparameters and harden evaluation.

Common sweep knobs:
- learning rate
- optional `vision_ln`

Added artifacts:
- stratified eval set by target length
- “golden set” qualitative caption dumps at checkpoints

### Stage 2: Visual instruction tuning (SFT)

**Goal:** Turn the aligned model into a chat VLM via instruction tuning.

Typical approach:
- Keep connector training
- Add **LoRA** on the LLM (selected attention/MLP modules)
- Train on **single‑image** instruction data (e.g., LLaVA‑Instruct‑150K)

---

## Track B — Discrete tokens (UTI + token LM)

### Stage 1: UTI implementation + verification

**Goal:** Define a stable ABI for tokenizing and decoding text/image/audio.

Core requirements:
- deterministic tokenization paths (for audit mode)
- strict token range checks
- decode sanity (sizes, sample rates, channel counts)
- regression gating on metrics

### Stage 2: Tokenized dataset generation

**Goal:** Produce tokenized shards (train/eval) that are stable and verifiable.

Key gates (dataset‑scale):
- token space hash matches
- split integrity (no overlap)
- decode spot checks
- retokenize consistency (when applicable)
- dataloader smoke tests

### Stage 3: Warm‑start multimodal vocabulary (“trainable rows”)

**Goal:** Teach the LLM to treat new modality token IDs as meaningful symbols while keeping text behavior stable.

Trainable:
- newly added rows in
  - input embedding
  - LM head

Frozen:
- all original weights (initially)

Deliverable:
- `trainable_rows.pt` + `trainable_rows.json` + `token_space.json`

### Stage 4: Multimodal token pretraining (MMPT)

**Goal:** Teach the model to **emit** image/audio tokens:
- T2I: text → image tokens
- T2A: text → audio tokens

while preserving:
- I2T / A2T

This is where LoRA/full‑tune and curriculum decisions start to matter.

---

## Design choices that keep iteration fast

- “One epoch ≈ 1 hour” sizing is enforced by measuring step time on real sequences and back‑computing epoch sample counts.
- Early stages focus on interface correctness and learning signal, not end‑quality.
- Checkpoints are designed to be light (connector‑only / trainable‑rows‑only) until later stages.
