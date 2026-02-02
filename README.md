# LumoOmni Open Source

LumoOmni is a staged research program for building discrete‑token multimodal foundation models. The specs define a reproducible path from a VLM baseline (Qwen3‑8B + SigLIP + resampler/projector) to a unified token space (SEED + EnCodec), tokenized dataset pipelines, and generation‑quality gates. The repository is structured so researchers can reproduce the system and HR teams can assess scope, rigor, and outcomes.

## Program Scope (High‑Level)

- **Stage 1**: Alignment pretraining for a VLM baseline (connector‑only training, deterministic evaluation).
- **Stage 1.1**: Evaluation hardening with token‑weighted loss, bucketed metrics, and golden set tracking.
- **Stage 2**: Dataset normalization and deterministic tokenized shards (WebDataset + token‑space hashing).
- **Stage 3.1**: Unified Token Interface with SEED/EnCodec adapters and deterministic audits.
- **Stage 3.2**: Tokenized dataset verification (range/shape checks, retokenize consistency, decode spot checks).
- **Stage 3.3**: Warm‑start token LM with modality‑token integration and prompt ablations.
- **Stage 3.4**: Decode compliance and generation quality gates before Stage 4.

## What This Repo Contains

- Stage‑by‑stage specs and verification plans for multimodal tokenization.
- Implementation notes, audits, and run reports.
- A GitHub Pages site in `docs/` that presents the research program in a public‑facing format.

## Start Here

- Open the site locally at `docs/index.html`.
- Stage overview: `docs/stages/overview.html`.
- Runbook (operational workflow and container setup): `RUNBOOK.md`.

## Repo Map

- `docs/` GitHub Pages site and public‑facing research overview.
- `stage1.md`, `stage1.1spec.md`, `stage2.md`, `stage3.*.md` core specs.
- `reports/` runbooks and verification reports.
- `qwen3-vlm/` implementation code and training scripts.
- `stage3_uti/` unified token interface and dataset pipeline.

## Notes

This repository focuses on open specifications and reproducible methodology. If you are running training or audits, follow the container workflow and pinned versions in `RUNBOOK.md`.
