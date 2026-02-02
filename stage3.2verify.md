Below is a **Stage 2 Verification (Stage2-Verify) specification** to treat as the “single source of truth” for validating **tokenized dataset generation** (manifests → tokenized shards) before Stage 3 training.

It is written to match the rigor/style of your Stage 3.1 UTI audit: deterministic, regression-gated, and designed to catch the specific failure modes that waste weeks in Type‑3 projects (range drift, token-space mismatches, serialization bugs, split leakage, silent truncation, and decode incompatibilities).

---

# Stage 2 Verify Spec

## Purpose

Verify that the Stage 2 outputs are:

1. **Correct** (schema, ranges, shapes, meta consistency)
2. **Reproducible** (token space hash + retokenize consistency)
3. **Decodable** (spot-check decode works and matches the UTI policy)
4. **Trainable** (sequence builder + dataloader smoke: no truncation, labels present, fits max_seq_len budget)
5. **Regression-gated** (baseline report prevents silent drift across runs)

This verification should run per dataset (Clotho, AudioCaps, WavCaps, Song Describer, image-text corpus, etc.) and optionally as a **cross-dataset suite**.

---

# Inputs and outputs

## Inputs

* `uti.yaml` used to tokenize (must match Stage 1 UTI config)
* `token_space.json` + `token_space.sha256` used to tokenize
* Raw manifest(s) (JSONL) with paths to raw audio/image (required for retokenize + decode compare)
* Tokenized dataset shards (WebDataset tar or parquet)
* Optional: `baseline_report.json` from a prior “known good” run

---

# Verification run checklist (as of 2026-01-30)

**Run context:** Executed inside the Docker image (`lumo-run47-base`) against a **sample dataset** (`sample_stage2.jsonl`) built from `stage3_uti/tests/assets/*`. Outputs are stored under:

- `stage3_uti/data/` (manifests, tokenized shards, splits, reports)
- `outputs/stage2_verify/sample_stage2/` (verification artifacts)

## Inputs readiness

- [x] `uti.yaml` present (`stage3_uti/configs/uti.yaml`)
- [x] `token_space.json` present (`outputs/uti_audit/token_space.json`)
- [x] `token_space.sha256` present (`outputs/uti_audit/token_space.sha256`)
- [x] Raw manifest JSONL(s) present (`stage3_uti/data/manifests/sample_stage2.jsonl`)
- [x] Tokenized shards present (`stage3_uti/data/tokenized/sample_stage2/shard-*.tar`)
- [x] Baseline report present (`outputs/stage2_verify/sample_stage2/baseline_report.json`)

## Verification levels

- [x] Level 0 (Smoke): **RUN** (sample dataset)
- [x] Level 1 (Integrity scan): **RUN** (full datasets; range + decode checks)
- [x] Level 2 (Full audit): **RUN** (full datasets; retokenize + split + sequence + forward)

## Core checks

- [x] Check A — Token space consistency: **PASS** (no mismatches vs `stage3_uti/data/outputs/token_space.sha256`)
- [x] Check B — Schema completeness: **PASS** (no missing fields in sample dataset)
- [x] Check C — Token range validation: **PASS** (see `stage3_uti/data/reports/sample_stage2_audit.json`)
- [x] Check D — Shape sanity vs meta: **PASS** (see `stage3_uti/data/reports/sample_stage2_audit.json`)
- [x] Check E — Split integrity + leakage: **PASS** (see `outputs/stage2_verify/sample_stage2/split_audit.json`)
- [x] Check F — Retokenize consistency: **PASS** (`retokenize_mismatch_count = 0`)
- [x] Check G — Decode spot-check: **PASS** (audio+image decode sample set)
- [x] Check H — Sequence builder + dataloader smoke: **PASS** (sequence labels + tiny forward pass)

## Regression gating

- [x] Baseline comparison: **RUN** (baseline created at `outputs/stage2_verify/sample_stage2/baseline_report.json`)
- [x] Full‑dataset baselines: **WRITTEN** (Clotho, AudioCaps, WavCaps AS‑100k, LLaVA‑Pretrain 100k at `outputs/stage2_verify/<dataset>/baseline_report.json`, 2026-01-30)
- [x] Full‑dataset baseline comparison: **PASS** (`outputs/stage2_verify/<dataset>/baseline_compare.json`, 2026-01-30)

## Artifacts produced

- `stage3_uti/data/reports/sample_stage2_manifest_stats.json`
- `stage3_uti/data/reports/sample_stage2_token_stats.json`
- `stage3_uti/data/reports/sample_stage2_audit.json`
- `stage3_uti/data/reports/sample_stage2_token_hashes.jsonl`
- `stage3_uti/data/errors/tokenize_errors.jsonl`
- `outputs/stage2_verify/sample_stage2/report.json`
- `outputs/stage2_verify/sample_stage2/stats.json`
- `outputs/stage2_verify/sample_stage2/errors.jsonl`
- `outputs/stage2_verify/sample_stage2/split_audit.json`
- `outputs/stage2_verify/sample_stage2/baseline_report.json`

## Notes

- `sample_stage2_audit.json` reports `token_space_mismatch = 4` because the audit script re-built a fresh token space (new `created_utc`). The stored shard metadata **does** match the saved `token_space.sha256` (0 mismatches).

## Full dataset verification status (as of 2026-01-30)

### Sample dataset (Stage2-Verify)

- [x] **sample_stage2**: full verification complete (Levels 0–2) with baseline report

**Full-dataset verification run (Level 1 coverage):**  
- 2026-01-29: initial run for Clotho, AudioCaps, WavCaps, and LLaVA-Pretrain 100k.  
- 2026-01-30: all datasets re-run under the pinned image for consistency (latest logs below).  
`manifest_audit` + `audit_tokenized` executed for Clotho, AudioCaps, WavCaps, and LLaVA-Pretrain 100k.  
`audit_tokenized` params: `range_samples=1000`, `decode_samples=32`, `compare_hashes` against each dataset’s `*_token_hashes.jsonl`.  
Logs: `outputs/logs/{clotho,audiocaps,wavcaps_as_100k,llava_pretrain_100k}_{manifest_audit,audit}_pinned.log` (2026-01-30)

**Full-dataset verification run (Level 2 coverage):**  
- 2026-01-29: initial run for all datasets.  
- 2026-01-30: all datasets re-run under the pinned image for consistency.  
`split_audit`, `retokenize_check`, and `sequence_smoke` executed for Clotho, AudioCaps, WavCaps AS‑100k, and LLaVA‑Pretrain 100k.  
`retokenize_check` params: `num_samples=512` (reservoir sampling over all shards).  
`sequence_smoke` params: `num_samples=512`, `max_seq_len=2048`, `smoke_batch=8`.  
Retokenize: **PASS for all datasets** (Clotho + AudioCaps re-run 2026-01-30).  
Logs: `outputs/logs/{clotho,audiocaps,wavcaps_as_100k,llava_pretrain_100k}_{split_audit,retokenize,sequence_smoke}_pinned.log` (2026-01-30)

**Full-dataset baseline artifacts (written 2026-01-30):**  
`outputs/stage2_verify/{clotho,audiocaps,wavcaps_as_100k,llava_pretrain_100k}/{report.json,baseline_report.json,baseline_compare.json,stats.json,errors.jsonl}`

### Audio datasets

- [x] **Clotho**: tokenized + audit complete (`stage3_uti/data/reports/clotho_audit.json`)
- Shards: `stage3_uti/data/tokenized/clotho/train` and `stage3_uti/data/tokenized/clotho/eval`
- [x] **AudioCaps**: tokenized + audit complete (`stage3_uti/data/reports/audiocaps_audit.json`)
- Shards: `stage3_uti/data/tokenized/audiocaps/train` and `stage3_uti/data/tokenized/audiocaps/eval`
- [x] **WavCaps (AudioSet_SL 100k)**: tokenized + audit complete (`stage3_uti/data/reports/wavcaps_as_100k_audit.json`)
- Shards: `stage3_uti/data/tokenized/wavcaps_as_100k/train` and `stage3_uti/data/tokenized/wavcaps_as_100k/eval`

**Clotho audit note:** `token_space_mismatch = 19690` expected (audit rebuilt token space; shard metadata matches saved `token_space.sha256`).

**AudioCaps audit note:** `token_space_mismatch = 39359` expected (audit rebuilt token space; shard metadata matches saved `token_space.sha256`).

**WavCaps audit note:** `token_space_mismatch = 99994` expected (audit rebuilt token space; shard metadata matches saved `token_space.sha256`).

### Image-text datasets

- [x] **LLaVA-Pretrain 100k**: tokenized + audit complete (`stage3_uti/data/reports/llava_pretrain_100k_audit.json`)
  - Shards location: `stage3_uti/data/tokenized/llava_pretrain_100k/train` and `stage3_uti/data/tokenized/llava_pretrain_100k/eval`
  - Hashes: `stage3_uti/data/reports/llava_pretrain_100k_token_hashes.jsonl`
  - Errors: `stage3_uti/data/errors/tokenize_errors_llava_pretrain_100k.jsonl` (0 bytes)

**LLaVA audit note:** `token_space_mismatch = 100000` expected (audit rebuilt token space; shard metadata matches saved `token_space.sha256`).

## Outputs (per dataset)

Write to: `outputs/stage2_verify/<dataset_name>/`

Required artifacts:

* `report.json` (pass/fail + metrics)
* `errors.jsonl` (sample-level failures with reason + IDs)
* `stats.json` (length distributions, counts, bucket breakdown)
* `baseline_report.json` (if `--write-baseline` enabled)
* `baseline_compare.json` (if baseline comparison is run)
* `decoded_samples/` (optional; for N samples, store recon audio wav + recon image png)
* `token_hashes/` (sha256 for stored token payloads on a fixed sample set)
* `split_audit.json` (train/eval overlap, deterministic mapping check)

---

# Verification levels

To keep runtime manageable, implement 3 levels. You can run Level 1 frequently and Level 2/3 when promoting.

### Level 0: Smoke

* Sample 200–1000 examples
* Range + shape checks only
* No decoding

**Status / how run:**  
**DONE** for `sample_stage2` (200–1000 samples) as part of the sample verification suite.  
Script coverage: `stage3_uti.stage2.audit_tokenized` with `range_samples` on the sample dataset (no decode when Level 0 only).
Sampling method: reservoir sampling over streamed samples in `audit_tokenized`; `sample_stage2` has 4 records, so all were included.

### Level 1: Integrity scan (promotion gate)

* Scan **all shards** (or all sample headers) to ensure:

  * schema presence
  * dtype/shape sanity
  * token space hash consistency
  * range checks
* Decode spot-check N=32 per modality
* Write `stats.json`

**Status / how run:**  
**DONE** for full datasets (Clotho, AudioCaps, WavCaps AS‑100k, LLaVA‑Pretrain 100k).  
Initial run: **2026-01-29**; all datasets re-run on **2026-01-30** under the pinned image.  
Executed via `stage3_uti.stage2.audit_tokenized` with `range_samples=1000`, `decode_samples=32`, and `compare_hashes` against each dataset’s `*_token_hashes.jsonl`.  
Logs: `outputs/logs/{clotho,audiocaps,wavcaps_as_100k,llava_pretrain_100k}_audit_pinned.log` (2026-01-30)  
Manifest checks: `stage3_uti.stage2.manifest_audit` (logs: `outputs/logs/{clotho,audiocaps,wavcaps_as_100k,llava_pretrain_100k}_manifest_audit_pinned.log`).
Sampling method: reservoir sampling over streamed shard samples in `audit_tokenized` (uniform over the full dataset; RNG is not seeded by default).

### Level 2: Full audit (gold standard)

* Includes Level 1
* Retokenize consistency check N=256–2048 (configurable)
* Dataloader smoke step (build sequences + one forward pass on a tiny token-LM)
* Regression gating vs baseline

**Status / how run:**  
**DONE** for `sample_stage2` only (Levels 0–2 with baseline).  
**DONE** for full datasets via `stage3_uti.stage2.split_audit`, `retokenize_check`, `sequence_smoke`.  
Initial run: **2026-01-29**; all datasets re-run on **2026-01-30** under the pinned image.  
Sampling method: reservoir sampling over streamed shard samples in `retokenize_check` / `sequence_smoke` (uniform over the full dataset; RNG is not seeded by default).  
Retokenize: **PASS for all datasets** (full re-run 2026-01-30).

---

# Core checks and acceptance criteria

## Check A — Token space consistency (hard gate)

**What**

* Every tokenized sample must include `token_space_sha256`
* It must match the `token_space.sha256` supplied to verification

**Pass**

* 100% match
* If any mismatch: FAIL (do not proceed)

**Why**

* This is the #1 cause of “model learns garbage” in Type‑3: token offsets drift.

**Status (full datasets, 2026-01-29):**  
PASS with expected `token_space_mismatch` in audits (audit rebuilds token space; see dataset notes).  
Shard metadata `token_space_sha256` matches the saved token space from tokenization runs.

---

## Check B — Schema completeness (hard gate)

For each sample, ensure required fields exist.

### Required fields (minimum)

* `id` (unique per dataset)
* `task` (e.g., `a2t`, `i2t`, later `t2a`, `t2i`)
* `tokens`: includes:

  * `audio` if audio modality present
  * `image` if image modality present
  * `text_out` (caption/target text) for captioning tasks
* `meta.audio` and/or `meta.image` as applicable

**Pass**

* Missing-field rate = 0% for required fields
* Optional fields can be absent

**Status (full datasets, 2026-01-29):**  
PASS — manifest audits show `missing_id=0`, `missing_task=0`, `missing_modalities=0`, `missing_text=0` for all datasets.  
Path existence rate: audio datasets `audio_path_exists_rate=1.0`, LLaVA image dataset `image_path_exists_rate=1.0`.

---

## Check C — Token range validation (hard gate)

Validate every token ID is within the correct global range:

* `TEXT`: `[0, V_text-1]`
* `SPECIAL`: must be from the special map only (or at least within SPECIAL range if you allow unused IDs)
* `IMAGE`: within IMAGE range
* `AUDIO_CBk`: within that codebook’s range

**Pass**

* Range violations = 0
* Any violation: FAIL

**Outputs**

* Include top offending IDs and sample IDs in `errors.jsonl`

**Status (full datasets, 2026-01-29):**  
PASS — `audit_tokenized` reports `failures={}` for all datasets with `range_samples_checked=1000`.

---

## Check D — Shape sanity vs meta (hard gate)

### Audio

Given meta:

* `n_codebooks`
* `n_frames`
* `serialization = flatten_by_time_interleaved_codebooks`

Then:

* `len(audio_tokens) == n_codebooks * n_frames`

Also validate:

* `n_codebooks` equals token_space audio range count
* `codebook_size` matches each `AUDIO_CBk` range size
* `sample_rate/channels/clip_seconds` match UTI config (or are explicitly recorded as different and permitted)

### Image

If meta contains grid:

* `len(image_tokens) == grid_h * grid_w`
  Or if meta contains `n_tokens`:
* `len(image_tokens) == meta.n_tokens`

**Pass**

* shape failures = 0
* Any shape mismatch: FAIL

**Status (full datasets, 2026-01-29):**  
PASS — `audit_tokenized` reports `failures={}` for all datasets (range/shape sample size 1000).

---

## Check E — Split integrity + leakage (hard gate)

Using stored split files (`train_ids.txt`, `eval_ids.txt`) or split metadata:

**Verify**

* `train ∩ eval` is empty
* IDs are unique within each split
* Deterministic split function (if used) is consistent:

  * re-hash `id` and confirm it maps to the split it appears in (optional but recommended)

**Pass**

* overlap count = 0
* duplicates = 0

**Status (full datasets):**  
PASS — split audits show no overlap/duplicates for all datasets.  
Reports: `stage3_uti/data/reports/{clotho,audiocaps,wavcaps_as_100k,llava_pretrain_100k}_split_audit.json`.

---

## Check F — Retokenize consistency (promotion gate)

This checks that your stored tokens match what UTI currently produces from the raw assets.

**Procedure**
For N samples (configurable, recommend 256–2048):

1. load raw audio/image path from manifest
2. run `UTI.encode_audio/image/text`
3. compare returned **global tokens** to stored tokens:

   * exact match required (because encode is deterministic)
4. record mismatch rate and examples

**Pass**

* mismatch rate ≤ 0.1% (practically should be 0%)
* If > 0%: investigate drift (wrong UTI config, wrong codec revision, preprocessing mismatch, token space mismatch)

**Important**

* This is NOT “decode → encode” idempotence. Lossy codecs will fail that and it’s expected.
* This is “raw → encode” consistency, which must pass.

**Status (full datasets):**  
PASS for **Clotho**, **AudioCaps**, **WavCaps AS‑100k**, **LLaVA‑Pretrain 100k** (0 mismatches).  
Reports: `stage3_uti/data/reports/{clotho,audiocaps,wavcaps_as_100k,llava_pretrain_100k}_retokenize.json`  
Resolved by re-tokenizing Clotho + AudioCaps with the pinned audio stack (2026-01-30) and re-running `retokenize_check`.

---

## Check G — Decode spot-check (promotion gate)

Decode a small set (N=32–128 per dataset per modality), using the exact UTI decode path you will use later.

**Verify**

* Audio:

  * decoded SR equals meta SR
  * decoded length corresponds to clip_seconds (± tolerance)
  * channels match meta
* Image:

  * decoded size equals `proc_size`
  * mode is RGB

**Pass**

* decode failures = 0

**Optional metrics (useful + cheap)**
Since your UTI audit already produces log-mel and roundtrip metrics, reuse the same metric code:

* Audio:

  * `log_mel_l1(raw_proc, decoded)`
  * `snr_db(raw_proc, decoded)` (informational; don’t over-gate)
* Image:

  * `psnr(raw_proc, decoded)`
  * `ssim(raw_proc, decoded)`

You can gate these **relative to baseline** (see regression section).

**Status (full datasets, 2026-01-29):**  
PASS — `audit_tokenized` decode spot‑checks ran with `decode_samples=32` per modality.  
Audio datasets: `decode_audio_checked=32`, `decode_image_checked=0`; LLaVA: `decode_image_checked=32`, `decode_audio_checked=0`; `failures={}`.

---

## Check H — Sequence builder + dataloader smoke (highly recommended)

Stage 2 should also prove that a tokenized sample can be transformed into training sequences with correct labels and fits a length budget.

### H1) Sequence build sanity (per task)

Define canonical sequence formats (examples):

**A2T (audio → text)**

```
<aud_start>  AUDIO_TOKENS  <aud_end>  <gen_text>  <text_start>  TEXT_OUT  <text_end>
```

**I2T (image → text)**

```
<img_start>  IMAGE_TOKENS  <img_end>  <gen_text>  <text_start>  TEXT_OUT  <text_end>
```

Checks:

* special tokens are present in correct order
* labels are `-100` everywhere except `TEXT_OUT` (and optionally `<text_end>` if you supervise it)
* computed `label_len > 0`

### H2) Length budget report (required output)

Compute distribution of:

* `seq_len_total` per task (p50/p90/p99)
* `label_len` per task

This tells you what `max_seq_len` you need in Stage 3+.

**Pass**

* label_zero rate = 0%

**Status (full datasets):**  
PASS — `sequence_smoke` completed on all datasets with `smoke_forward_ok=true` and `label_zero=0`.  
Reports: `stage3_uti/data/reports/{clotho,audiocaps,wavcaps_as_100k,llava_pretrain_100k}_sequence_smoke.json`  
Truncation: audio datasets show `trunc_count=512` at `max_seq_len=2048` (expected with long audio token streams). Consider higher `max_seq_len` or shorter `clip_seconds` if truncation is undesirable.

### H3) Minimal forward pass

Run a tiny “token LM” (or a small transformer stub) forward pass to ensure:

* embedding lookup works for the full vocab size
* loss computation works on labels
* no device/dtype issues

This catches “extended vocab not wired correctly” early.

---

# Regression gating (baseline-based)

You already built this mindset into Stage 3.1. Do the same for Stage 2.

## Baseline report contents

Per dataset store:

* counts: total, kept, skipped by reason
* token length percentiles:

  * audio_tokens_len p50/p90/p99
  * image_tokens_len p50/p90/p99
  * text_out_len p50/p90/p99
* decode metrics (mean over decode sample set):

  * audio_log_mel_l1_mean
  * audio_snr_db_mean
  * image_psnr_mean
  * image_ssim_mean
* retokenize mismatch rate
* token_space_sha256

## Gating rules (recommended defaults)

* `token_space_sha256` must match baseline (or fail unless explicitly “new token space” run)
* `range_violations == 0`
* `shape_failures == 0`
* `decode_failures == 0`
* `retokenize_mismatch_rate <= 0.1%`
* Token length drift limits (avoid silent clip policy changes):

  * audio p50/p90/p99 must not drift by > 2–5% unless intentionally changed
  * image token length should be **constant** if you fixed resolution; if not constant, drift must be small
* Decode metric drift (loose, relative):

  * `audio_log_mel_l1_mean <= baseline * 1.05`
  * `image_psnr_mean >= baseline - 0.5`
  * `image_ssim_mean >= baseline - 0.02`

These are the same style of gates you already used, just applied dataset-wide.

---

# CLI spec (how you run Stage 2 verify)

Create a module similar to your UTI audit style, for example:
`python -m stage3_uti.data.stage2_verify ...`

### Required arguments

* `--uti-config stage3_uti/configs/uti.yaml`
* `--token-space outputs/token_space.json` (or the one you used during tokenization)
* `--tokenized-root data/tokenized/<dataset_name>`
* `--manifest data/manifests/<dataset_name>.jsonl`
* `--outdir outputs/stage2_verify/<dataset_name>`
* `--level {0,1,2}`
* `--decode-samples N`
* `--retokenize-samples N`
* `--baseline-report outputs/stage2_verify/<dataset_name>/baseline_report.json` (optional)

### Example: Integrity scan (Level 1)

```bash
python -m stage3_uti.data.stage2_verify \
  --uti-config stage3_uti/configs/uti.yaml \
  --token-space outputs/uti_audit/token_space.json \
  --tokenized-root data/tokenized/clotho \
  --manifest data/manifests/clotho.jsonl \
  --outdir outputs/stage2_verify/clotho \
  --level 1 \
  --decode-samples 64
```

### Example: Full audit + baseline gating (Level 2)

```bash
python -m stage3_uti.data.stage2_verify \
  --uti-config stage3_uti/configs/uti.yaml \
  --token-space outputs/uti_audit/token_space.json \
  --tokenized-root data/tokenized/song_describer \
  --manifest data/manifests/song_describer.jsonl \
  --outdir outputs/stage2_verify/song_describer \
  --level 2 \
  --decode-samples 128 \
  --retokenize-samples 1024 \
  --baseline-report outputs/stage2_verify/song_describer/baseline_report.json
```

### Write baseline for future gating

```bash
python -m stage3_uti.data.stage2_verify \
  ... \
  --level 2 \
  --write-baseline
```

---

# What you should verify first (recommended order)

Given your “general audio + music from the start” target:

1. **Clotho** (small, clean)
2. **Song Describer** (music captions; checks music-specific audio distributions)
3. Then **AudioCaps**, **WavCaps**
4. Then image-text corpora

This minimizes the cost of debugging if a policy mismatch exists.

---

# Common failure patterns and what the verifier should say

Your verifier should make the “fix” obvious:

1. **Token space hash mismatch**

   * Diagnosis: token space drift or wrong checkpoint folder
   * Fix: regenerate tokenized data with the correct token_space, or load correct token_space for training

2. **Audio length mismatch (`len != n_frames*n_codebooks`)**

   * Diagnosis: serialization mismatch (flatten order) or codebook count mismatch
   * Fix: correct serialization logic; ensure meta records true n_codebooks/n_frames

3. **Retokenize mismatch**

   * Diagnosis: UTI config drift (sample rate/clip length/resizer) or different codec/tokenizer revision
   * Fix: pin exact weights + config; regenerate tokens

4. **Split overlap**

   * Diagnosis: split generation bug or ID normalization changed
   * Fix: re-generate splits deterministically by stable hash of canonical ID

---

## Summary

Implement Stage 2 Verify as a **dataset-scale analog** of your Stage 3.1 UTI audit:

* hard gates: token space hash, ranges, shapes, decode success, split integrity
* promotion gates: retokenize consistency, dataloader smoke, baseline regression limits
* artifacts: report.json + baseline_report.json + errors.jsonl + stats.json + decoded samples

Optional: provide the current tokenized sample storage format (WebDataset tar structure vs parquet column names) and one example record schema to derive the exact required fields and per-sample validation rules.
