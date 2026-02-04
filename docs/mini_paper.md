# Mini paper (living document)

> This is a “mini paper” intended for researchers who want a compact technical view of the project.
> It is a living document: numbers and stage status may evolve.

---

## Title

**LumoOmni: Staged construction of an omni‑modal model from a text‑only LLM backbone**

## Abstract

We describe a staged approach for building an omni‑modal model (vision + audio + text) starting from a high‑quality text‑only causal LLM. The approach has two parallel tracks. First, we build a continuous‑embedding VLM by freezing a vision tower and the LLM and training only a connector (Perceiver‑style resampler + projector) that injects a small number of visual tokens into the LLM sequence. Second, we establish a Unified Token Interface (UTI) that converts images and audio into discrete global token IDs with deterministic audits and regression gates; then we warm‑start the LLM’s vocabulary by training only the new modality embedding/head rows (“trainable‑rows‑only”). This decomposition keeps early iterations fast, isolates failure modes, and enables strict compatibility/versioning (token space hashes, retokenize checks, decode sanity).

## 1. Motivation

Omni‑modal models are hard to debug because failures can be caused by:
- data integrity (paths, serialization, truncation, split leakage),
- tokenization interface drift (codec settings, resizers, sample rates),
- placeholder or injection bugs (VLM token positions),
- training instability from changing too many parameters at once.

LumoOmni attempts to reduce these risks by:
1. **freezing most of the model** in early stages,
2. adding **audits** as first‑class gates,
3. packaging weights in **patchable, versioned units**.

## 2. Method overview

### 2.1 Track A: Continuous VLM connector alignment

We use a frozen vision encoder (SigLIP2) to produce patch embeddings and a trainable connector:

- **Resampler:** compresses `N_patches` embeddings into a fixed `N_latents` (e.g., 64) via cross‑attention.
- **Projector:** maps the resampler latents into the LLM hidden size.

The projected tokens are inserted into the LLM input embeddings at `<image_patch>` positions.

### 2.2 Track B: Discrete token omni (UTI + vocab warm‑start)

We define a global token space that includes:
- text token IDs (lossless),
- image token IDs (codec‑based, lossy),
- audio token IDs (codec‑based, lossy).

UTI is treated like an ABI: the token space JSON + hash becomes part of every checkpoint and dataset artifact. Before training, we run an audit that checks determinism, token ranges, decode success, and regression metrics. Then we train the LLM to handle new modality tokens by learning only the new vocabulary rows (input embeddings + LM head), before later stages that add LoRA or full fine‑tuning.

## 3. Experiments and results (selected)

### 3.1 VLM connector sanity via ablation

Teacher‑forced loss is evaluated on held‑out samples under four visual conditions:
- correct image,
- shuffled image,
- zeroed visual tokens,
- random/noise visual tokens.

A passing run shows `loss(correct) < loss(shuffled) < loss(zero/noise)`, providing evidence that visual embeddings are injected and attended to.

### 3.2 Stage‑1.1 sweep for stable hyperparameters

A small sweep over learning rate and optional vision LayerNorm is used to pick stable settings before scaling.

### 3.3 UTI audit and regression gating

UTI verification reports determinism, range checks, decode sanity, and metric‑based regression gates (image PSNR/SSIM, audio log‑mel + SNR). This is treated as a promotion gate before large‑scale tokenized dataset generation.

### 3.4 Trainable‑rows adapter deliverable

A token‑LM checkpoint can be delivered as trainable‑rows‑only weights plus token space metadata, enabling lightweight distribution without bundling base model weights.

## 4. Limitations

- Image/audio tokenizations are **lossy**; strict token‑id idempotence is not expected.
- Current evaluations emphasize **interface correctness** and training signal; generative quality requires later‑stage decoding + conditional generation verification.
- Dataset licenses and redistribution constraints must be respected; tokenized derivatives may still inherit restrictions.

## 5. Future work

- Scale tokenized data generation and run larger MMPT (Stage 4+).
- Expand evaluation: conditional generation tests, human preference tests, and modality‑specific benchmarks.
- Explore joint training where the continuous VLM track and discrete token track share a backbone.

---

## Suggested citation

See the **Citation** page for a BibTeX template.
