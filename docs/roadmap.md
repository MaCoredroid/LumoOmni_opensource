# Roadmap & future work

This roadmap is organized around **incremental capability** and **auditability**.

---

## Near term (next milestones)

### 1) Scale Stage 2 tokenized data generation

- Expand tokenized audio and image corpora beyond “iteration subsets”.
- Implement **dataset‑scale verify** (split integrity, decode spot checks, retokenize consistency, regression baselines).

### 2) Run a larger Stage 3 warm‑start

- Train trainable‑rows on larger tokenized shards (more diversity, longer clips).
- Track text‑only regression invariance when patching rows.

### 3) Stage 4 multimodal token pretraining (MMPT)

Goal: learn conditional generation (T2I/T2A) while preserving I2T/A2T.

Key verification requirements:
- decode artifacts are within expected bounds (no shape errors, stable sample rates)
- prompt ablations demonstrate conditional dependence (not unconditional memorization)
- mixing ratios prevent “read” collapse

---

## Mid term

### Expand evaluation

- Conditional generation evaluation (prompt fidelity, diversity, failure rate).
- Modality‑specific benchmarks:
  - image captioning/VQA (for Track A)
  - audio captioning and retrieval (for Track B)
- Human preference tests for edit/generate tasks if deploying a tool‑augmented system.

### Unify tracks

A longer‑term direction is to unify continuous VLM understanding and discrete token generation into a single backbone, with a curriculum that stabilizes both.

---

## Longer term

### Replace “frozen tools” with learned generators

If using a tool‑augmented image generation/editing system:
- keep tools frozen at first (fast iteration),
- later train internal generator/editor models,
- keep strict artifact logging for reproducibility (prompt, seed, mask, params, verifier scores).

### Speech output

The architecture and staging are designed to eventually support speech output (and potentially speech input) without rebuilding earlier pieces.

---

## Open questions

- Best curriculum for mixing A2T/I2T with T2A/T2I.
- How to allocate context length between text and long audio token sequences.
- Which evaluation metrics best predict perceptual quality for decoded tokens (especially music).
- How to prevent text regression when expanding vocab and adding multimodal objectives.
