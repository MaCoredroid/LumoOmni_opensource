Yes — based on what you reported, **Stage 1 (UTI) is complete and you are cleared to start Stage 2 (tokenized data generation / normalized shards)**.

Your Stage 3.1 verification report covers the exact acceptance criteria we set for Stage 1, and it does so with the right kinds of gates (range checks, deterministic audits, decode sanity, regression baselines, and smoke LM tests). The only “non-pass” is **token-id idempotence**, which is **expected** for lossy codecs and should remain a **non-gate**.

### Why Stage 1 is “green”

* **Determinism**: text/image/audio = true
* **Token-space correctness**: no collisions; strict range validation passes
* **Shape sanity**: audio token count consistent with codec settings; image token count consistent with proc size/grid
* **Decode sanity**: audio SR/length/channels OK; image size OK
* **Regression gating**: `metrics_v2_pass = true` ensures you didn’t accidentally change preprocessing or codec/tokenizer versions
* **Operational smoke**: overfit + resume + generation decode safety complete (this is critical for the next stage)

The low absolute PSNR/SSIM is not a Stage‑1 blocker; Stage‑1 is about **interface correctness and stability**, not reconstruction fidelity. You already have baseline gating to ensure fidelity doesn’t regress.

---

# Stage 2 Specification — Tokenized Data Generation and Normalization

Stage 2’s purpose is to produce **stable, reproducible, model-ready tokenized datasets** so Stage 3 (warm-start multimodal vocab) and Stage 4 (joint token pretraining) can run without on-the-fly codec/tokenizer overhead and without “mystery drift.”

## Stage 2 Goals

1. Convert each dataset into a **single canonical sample schema** (raw manifest).
2. Produce **tokenized shards** containing:

   * global token IDs (already offset into unified token space)
   * meta needed for decoding
   * textual fields (prompt/caption) tokenized or raw
3. Produce **dataset reports** with token length distributions, error rates, and reproducibility hashes.
4. Ensure splits are deterministic and compatible across runs/checkpoints.

---

## 2.1 Canonical dataset schema

### 2.1.1 Raw manifest schema (pre-tokenization)

Store as JSONL (one record per “training example”):

```json
{
  "id": "dataset:clip_id:caption_idx",
  "source": "clotho|audiocaps|wavcaps|song_describer|llava_pretrain|...",
  "modalities": {
    "audio": {"path": "...", "sr": 44100},
    "image": {"path": "..."}
  },
  "text": {
    "prompt": "optional (for t2a/t2i later)",
    "caption": "target text for a2t/i2t",
    "instruction": "optional (later SFT)",
    "response": "optional (later SFT)"
  },
  "task": "a2t|i2t|t2a|t2i|a2t_music|...",
  "meta": {
    "license_hint": "...",
    "duration_s": 10.0
  }
}
```

**Stage 2 minimum requirement:** you must support at least these tasks:

* `a2t` (audio→text caption)
* `i2t` (image→text caption)

You *can* also prepare `t2a` and `t2i` manifests now, but they are not required until Stage 4.

---

## 2.1.2 Tokenized record schema (post-tokenization)

For each manifest record, produce a tokenized record (JSON + binary arrays). Logical schema:

```json
{
  "id": "...",
  "source": "...",
  "task": "a2t",
  "token_space_sha256": "...",

  "tokens": {
    "text_in":   [ ... ],   // optional (if you store prompt tokens)
    "text_out":  [ ... ],   // caption tokens (TEXT range only)
    "audio":     [ ... ],   // global IDs in AUDIO_CB* ranges
    "image":     [ ... ]    // global IDs in IMAGE range
  },

  "special": {
    "text_start": 151936,
    "text_end":   151937,
    "aud_start":  151940,
    "aud_end":    151941,
    "img_start":  151938,
    "img_end":    151939,
    "gen_text":   151942,
    "gen_aud":    151944,
    "gen_img":    151943
  },

  "meta": {
    "audio": {...},   // from UTI encode_audio meta
    "image": {...},   // from UTI encode_image meta
    "text":  {...}    // optional: label_len, etc.
  }
}
```

**Important:** include `token_space_sha256` in every record so any training job can hard-fail if the token space changed.

---

## 2.2 Storage format for tokenized datasets

You have two viable options. Pick one and commit to it.

### Option A (recommended): WebDataset tar shards

* `tokenized/<dataset>/<split>/shard-{000000..}.tar`
* Each sample inside tar:

  * `sample.json` (metadata)
  * `audio.npy` / `image.npy` / `text_out.npy` (int32 arrays)

**Why:** streaming-friendly, widely used at scale, easy to resume.

### Option B: Parquet

* `tokenized/<dataset>/<split>.parquet`
* Columns contain list<int32> arrays

**Why:** easy analytics and filtering; sometimes heavier I/O patterns.

Given you already mentioned WebDataset earlier, I’d standardize on **WebDataset** now.

---

## 2.3 Deterministic split policy (Stage 2)

Stage 2 must create splits that are stable across time.

### Split contract

* Use a deterministic hash of `id` (or `source + path`) to assign split:

  * e.g., `hash(id) % 1000 < 10` → eval (1%)
* Store the list of IDs for each split:

  * `splits/<dataset>/train_ids.txt`
  * `splits/<dataset>/eval_ids.txt`

Do not rely on “random shuffle then slice,” because it breaks reproducibility if any example is added/removed.

---

## 2.4 Tokenization pipeline behavior

### Stage 2 pipeline steps

1. Load raw manifest record.
2. Apply UTI:

   * `encode_audio(wav, sr)` if audio exists
   * `encode_image(img)` if image exists
   * `encode_text(caption)` for labels
3. Validate:

   * all audio tokens in AUDIO ranges
   * all image tokens in IMAGE range
   * all label tokens in TEXT range
   * lengths match meta (`n_frames * n_codebooks`, grid sizes, etc.)
4. Write tokenized sample to shard.
5. Update stats counters (token length percentiles, failures by reason).

### Error handling

* Missing/corrupt files:

  * log to `tokenize_errors.jsonl`
  * skip sample
* Decoder exceptions during *spot-check*:

  * mark sample as failed (do not keep silently)

---

## 2.5 Stage 2 acceptance tests (gates)

These are the gates you must pass before Stage 3 training starts.

### Gate A — Token space consistency

* Every tokenized sample includes `token_space_sha256`
* Training loader refuses to load mismatched token space

### Gate B — Range + shape sanity on tokenized shards

On a random sample of N=1,000 per dataset:

* tokens are in correct global ranges
* `len(audio_tokens) == n_codebooks * n_frames`
* `len(image_tokens) == meta.n_tokens`

### Gate C — Decode spot-check

Decode a small set (e.g., 32 audio + 32 images per dataset) and verify:

* audio SR/length/channels match meta
* image size matches proc_size

This is not “quality,” this is “no schema bugs.”

### Gate D — Distribution report

Write `reports/<dataset>_stats.json` including:

* counts: total, kept, skipped, failure reasons
* token length percentiles for each modality:

  * audio token lengths (p50/p90/p99)
  * image token lengths
  * label lengths
* optional: per-task counts

### Gate E — Reproducibility

Re-run tokenization on the same manifest subset and confirm:

* token hashes match (`audio_tokens.sha256`, `image_tokens.sha256`)
* shard content stable (or at least sample-level stable)

You already did this style of gating for UTI; Stage 2 simply extends it to *dataset-scale*.

---

## 2.6 Recommended directory layout

```text
stage3_uti/
  data/
    manifests/
      clotho.jsonl
      audiocaps.jsonl
      wavcaps.jsonl
      song_describer.jsonl
      image_text.jsonl
    tokenized/
      clotho/
        train/shard-000000.tar
        eval/shard-000000.tar
      song_describer/
        train/...
    splits/
      clotho/train_ids.txt
      clotho/eval_ids.txt
    reports/
      clotho_stats.json
      song_describer_stats.json
    errors/
      tokenize_errors.jsonl
  outputs/
    token_space.json
    token_space.sha256
```

---

# 2.7 Stage 2 stage breakdown (run it like a research project)

## Stage 2.0 — Build raw manifests

Deliverables:

* `manifests/<dataset>.jsonl`
* `reports/<dataset>_manifest_stats.json` (counts, durations, path existence rate)

## Stage 2.1 — Tokenize and write shards

Deliverables:

* `tokenized/<dataset>/<split>/shard-*.tar`
* `reports/<dataset>_token_stats.json`
* `errors/tokenize_errors.jsonl`

## Stage 2.2 — Dataset-level audits

Deliverables:

* `reports/<dataset>_audit.json` (range checks, decode spot-check, hash checks)
* Promote dataset to “ready” only if all gates pass.

---

# Decision point: what you should tokenize first

Given your “general audio + music from the start” target, the most productive sequencing is:

1. **Clotho** (small, clean, captioned general audio)
2. **Song Describer Dataset** (clean music captions with downloadable audio)
3. Then scale: **AudioCaps**, **WavCaps**
4. Then image-text corpora (if not already tokenized)

This keeps early failures cheap and makes the stats meaningful.

---

## Answer to your question

**Yes — Stage 1 is complete and you should start Stage 2 now.**

Optional: provide the current `uti.yaml` (or key fields: EnCodec model name, sample rate, clip length, channels, SEED resolution) to generate a Stage 2 “build manifests + tokenize shards + audit” runbook that matches the existing CLI style (`uti_audit`-like), including the reports/fields needed for Stage 3 training.
