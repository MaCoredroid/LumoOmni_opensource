# Datasets

This page documents the datasets and data products used across stages. It is split into:

- **Track A (VLM, continuous embeddings)**
- **Track B (UTI + token LM, discrete tokens)**

> Note: dataset licenses vary. Treat this as documentation of *usage* and *intended usage*, not permission to redistribute.

---

## Track A — VLM (continuous embeddings)

### Stage 1: Alignment / captioning‑style pretraining

**Primary dataset**
- **LLaVA‑Pretrain (558k)**: image + caption/conversation JSON.

**Typical sample fields**
- `image`: relative image path
- `conversations`: list of turns (human, gpt)

**Key notes**
- Prompt normalization ensures exactly one image placeholder; `<image>` is appended if missing.
- Early runs used `max_seq_len=512` with no truncation pressure in the trial config.

### Stage 2: Visual instruction tuning (single image)

**Primary dataset**
- **LLaVA‑Instruct‑150K** (single‑image filtered)

**Filtering rules (high level)**
- Exactly one resolvable image on disk.
- Exactly one `<image>` placeholder in user content (or normalize into that format).
- Skip missing/corrupt image files.
- Skip empty assistant responses.

---

## Track B — UTI + token LM (discrete tokens)

### Stage 1 (UTI): Token space + codec outputs

The goal is to define a stable internal token interface for:
- text tokens (lossless),
- image tokens + metadata required for decoding,
- audio tokens + metadata required for decoding.

A **token space JSON + hash** is treated as a first‑class artifact and must be consistent across:
- tokenized datasets,
- checkpoints,
- evaluation scripts.

### Audio datasets (recommended pipeline order)

A practical progression for tokenization and early training:

1. **Clotho** — small and clean: great for pipeline validation.
2. **AudioCaps** — mid‑sized general audio captions.
3. **WavCaps** — large; tokenize a subset first for iteration speed.
4. **Music caption data** — include any clean music‑caption sources you have and oversample during training (often small).

### Image dataset

For early tokenization iterations, using the **already‑wired** image‑text pipeline is recommended:

- Start with **LLaVA‑Pretrain** (tokenize a manageable subset, e.g., 100k–200k).
- Expand to larger/less‑curated sources after the pipeline is stable.

---

## Tokenized data products (Stage 2 in the token track)

### Output types

You typically want two separable outputs:

1. **Manifests**
   - canonical IDs
   - file paths
   - modality tags
   - split assignment (train/eval)
2. **Tokenized shards**
   - token IDs (global space)
   - per‑modality metadata needed for decode
   - optional cached feature stats

### Verification (Stage 2 Verify)

Before training on tokenized shards, run dataset‑scale verification analogous to the UTI audit:
- token space hash matches
- strict token range checks
- decode spot checks
- split integrity (no overlap)
- retokenize consistency (when applicable)
- dataloader smoke tests
- baseline regression limits

---

## Common pitfalls

- **Path drift:** manifests point to paths that moved after extraction (e.g., `.7z` archives not extracted).
- **Codec config drift:** sample rate / clip length changes silently and breaks retokenize assumptions.
- **Split leakage:** train/eval split generated from unstable IDs.
- **Silent truncation:** sequences exceed `max_seq_len` but labels are masked incorrectly.
