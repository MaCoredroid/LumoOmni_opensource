# LumoOmni

**LumoOmni** is a staged research build toward an **omni‑modal foundation model** (vision + audio + text), starting from a strong **text‑only LLM** backbone and adding multimodal capability in increments that are **auditable, reproducible, and regression‑gated**.

This documentation set is written like a small research website: it includes a mini paper, dataset notes, model/weight formats, training & eval results, and a forward roadmap.

---

## What this project is building

### Track A — Continuous‑embedding VLM (understanding / chat)

A classic “frozen towers + trainable connector” VLM stack:

- **Text backbone:** Qwen3‑8B‑Base (frozen in Stage 1)
- **Vision tower:** SigLIP2 SO400M patch14‑384 (frozen in Stage 1)
- **Connector:** Perceiver‑style resampler + MLP projector (trainable)

The connector produces a fixed number of **visual tokens** (e.g., 64) in the LLM hidden space and injects them at `<image_patch>` positions.

### Track B — Discrete‑token omni model (generation + understanding)

A “token‑first” path:

1. **Unified Token Interface (UTI)** defines a stable ABI for converting **text / image / audio** ↔ **global token IDs**.
2. A **token LM** expands the LLM vocabulary to include those modality token IDs.
3. Training begins by learning only the **new vocabulary rows** (“trainable‑rows‑only”) before moving to multimodal pretraining.

---

## Current milestones (high level)

### VLM connector sanity and scaling

- Stage‑1 ablation tests show the connector is being used (teacher‑forced loss: `correct < shuffled < zero/noise`) and truncation/label coverage are clean at `max_seq_len=512`.

### UTI verified with regression gates

- Stage‑3.1 UTI audit passes determinism, token‑range, decode sanity, and metric‑based regression gates (`metrics_v2_pass=true`).

### Weights packaging for token‑LM adapters

- Stage‑3 deliverable packages the adapter as **trainable‑rows‑only weights**, plus `token_space.json`, with a simple patch‑load procedure.

---

## Where to start

- **Mini paper:** see **Mini paper** in the navigation.
- **If you want to reproduce:** start with **Reproducibility** and **Checkpoints & weights**.
- **If you want to extend:** start with **Roadmap & future work**.

---

## Terminology cheatsheet

- **Connector (VLM):** resampler + projector that maps continuous vision embeddings into the LLM hidden space.
- **UTI:** unified token interface; stable token IDs and metadata for text/image/audio.
- **Trainable rows:** only the newly added embedding + LM‑head rows for non‑text tokens.
- **A2T/I2T/T2A/T2I:** audio/image → text and text → audio/image tasks used in token‑LM stages.
