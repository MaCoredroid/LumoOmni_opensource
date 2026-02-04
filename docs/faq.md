# FAQ

## Is this a single model or multiple models?

Today it is a staged system with two tracks:

- a continuous‑embedding VLM stack (vision tower + connector + LLM)
- a discrete token track (UTI + token‑LM vocabulary expansion)

Longer term, these can converge into one unified backbone.

## Why two tracks?

They solve different problems:

- Continuous embeddings are a strong default for *understanding* and chat VLMs.
- Discrete tokens are a practical path for *generation* (emit image/audio tokens) once tokenization is stable.

## Why so many audits and “gates”?

Because multimodal training failures are often silent. The goal is to catch:
- injection bugs,
- token‑space drift,
- split leakage,
- decode incompatibilities,
before expensive training runs.

## What does “trainable‑rows‑only” mean?

Only the newly added vocabulary rows (input embedding + LM head) are trained and saved.
You reconstruct the full model by loading the base LLM and patching those rows.

## Are PSNR/SSIM numbers low in UTI audit?

They can be, depending on codec settings and decode path. Stage‑1 UTI is about **interface correctness and stability**, not perceptual fidelity. Perceptual quality becomes a later‑stage concern.
