Your Stage‑4 2k‑step trial looks **materially successful** from a training‑dynamics standpoint. You have clear learning signal on the two things Stage 4 must deliver:

1. **“Read” remains intact and improves** (A2T/I2T losses drop).
2. **“Write” is learning rapidly** (T2A/T2I losses drop sharply from near-uniform‑ish to much lower).

Remaining work: **decode artifacts + prompt ablation**. Until those are done, conditional generation is not confirmed (teacher-forced loss alone can be misleading).

Below are (A) my interpretation of your log, then (B) a Stage‑4 verification checklist with acceptance thresholds, and (C) concrete “if it fails, what to do” fixes.

---

## A) What your Stage‑4 trial log indicates

### 1) The core outcome is correct: the model is learning to emit modality tokens

From step 50 → 2000:

* `t2i_avg`: **12.817 → 4.417** (very large improvement)
* `t2a_avg`: **9.811 → 5.581** (large improvement)

This is exactly what you want in Stage 4: the LM head and/or LoRA are learning the discrete token distributions for the generative modalities.

### 2) “Read” tasks did not collapse; they improved

* `a2t_avg`: **2.099 → 1.621**
* `i2t_avg`: **1.501 → 0.962**

This means your multi-task mixture (A2T/I2T/T2A/T2I) is not catastrophically forgetting captioning/understanding while learning generation.

### 3) Norms are healthy and consistent with “new vocab learning”

* `head_norm_mean`: **0.457 → 1.164**
* `emb_norm_mean`: **2.094 → 2.172**

Those trends are consistent with “new rows + LoRA are being used,” not a dead run where nothing changes.

### 4) No red flags in the loss curve shape

The loss decreases smoothly and monotonically without spikes, suggesting:

* label masks are probably correct,
* there are no obvious NaNs/instabilities,
* and the optimizer is doing something sensible.

**Conclusion:** Stage‑4 *plumbing and optimization* look correct. You are at the point where verification should focus on **(i) decode correctness** and **(ii) conditionality**.

---

## B) Stage‑4 Verification List (P0 → Promote)

### Stage‑4 Verify P0 (must pass to call the trial “valid”)

#### P0.1 Token-space / checkpoint consistency (hard gate)

**Check**

* `token_space.sha256` in:

  * Stage 2 shards
  * Stage 3 checkpoint initialization
  * Stage 4 checkpoint_2000
    is identical.

**Pass**

* 100% identical; otherwise fail-fast.

---

#### P0.2 Generation token-type compliance (hard gate)

For T2A and T2I generations, compute:

* `out_of_range_pct`: percentage of generated tokens that are **not** in the correct modality range

  * T2A: not in AUDIO_CB* ranges (plus optional `<aud_end>`)
  * T2I: not in IMAGE range (plus optional `<img_end>`)

**Pass (trial)**

* `out_of_range_pct == 0%` on golden set.

If this fails, you need constrained decoding (see fixes below).

---

#### P0.3 Decode success on golden set (hard gate)

Confirm that generated tokens can be decoded:

* T2I: SEED decode succeeds for every sample and output has correct `proc_size`.
* T2A: EnCodec decode succeeds and output has correct SR/channels; length is consistent with the token length you decode.

**Pass**

* `decode_success_rate == 100%` on golden T2A and golden T2I.

**Important note about length**
Your `--max-new-audio 1200` is only safe if your golden meta expects ~1200 audio tokens (i.e., `n_frames * n_codebooks ≈ 1200`).
If many golden examples expect ~2000 tokens (common for 10s @ 4 codebooks), you will truncate generation and decode may fail or become inconsistent.

**Recommendation (strong)**
In golden generation, do **per-sample dynamic length**:

* `expected_audio_tokens = meta.n_frames * meta.n_codebooks`
* generate up to `expected_audio_tokens + margin`
* then **slice exactly** `expected_audio_tokens` tokens for decoding

Same for image:

* `expected_image_tokens = meta.n_tokens`
* generate `expected + margin`, slice to expected for decoding

This single change is the most common reason teams go from “decode flaky” to “decode 100%”.

---

#### P0.4 Teacher-forced prompt ablation for T2A and T2I (hard gate for conditionality)

Teacher-forced loss can improve even if the model is learning an **unconditional** distribution over tokens. The prompt ablation tells you whether it uses the prompt.

**Procedure (per task, N≥256 examples recommended; 64 is noisy):**
Compute teacher-forced loss under:

* Correct prompt
* Shuffled prompt (swap prompts across batch)
* Empty prompt (or a fixed neutral prompt)

**Pass thresholds**

* `loss(correct) < loss(shuffled)` and `loss(correct) < loss(empty)`
* `win_rate(correct < shuffled) ≥ 0.70`
* `win_rate(correct < empty) ≥ 0.80`
* Also log deltas:

  * `Δshuffle = loss(shuffled) - loss(correct)`
  * `Δempty  = loss(empty) - loss(correct)`
  * Target deltas for a 2k-step trial can be modest; I’d still expect:

    * `Δempty` clearly positive (≥0.10–0.20) for at least one of T2A/T2I

---

### Stage‑4 Verify P1 (recommended before scaling beyond “trial”)

#### P1.1 End-token behavior (not mandatory for decode, but important for product-like use)

Compute:

* `% sequences that emit <aud_end> before max_new_audio`
* `% sequences that emit <img_end> before max_new_image`

If you’re doing fixed-length slicing for decoding, missing end tokens is not fatal, but for interactive generation it matters.

**Target**

* trending upward over time; not necessarily high at 2k steps.

---

#### P1.2 “Read” ablations still pass (regression guard)

Run your existing modality destruction ablations on A2T/I2T:

* correct vs shuffled/zero/noise

**Pass**

* correct remains better than noise/zero with meaningful delta.
  This ensures Stage 4 didn’t accidentally teach the model to ignore modalities while learning generation.

---

#### P1.3 Qualitative sanity checks on golden outputs

You do not need CIDEr/SPICE yet. Just check for obvious failure modes:

* T2I images: identical outputs across prompts, or severe collapse.
* T2A audio: silence, extreme noise, identical outputs, or repetition artifacts.

You can log a quick fingerprint:

* image: perceptual hash / simple pixel variance stats
* audio: RMS, peak, zero-crossing rate, mel-spectrogram variance

These catch collapse quickly.

---

## C) “If it fails, what to do” (fast fixes)

### If decode success < 100% due to wrong token types

Implement **constrained decoding** (recommended for all Type‑3 systems with a shared vocab):

* During T2A generation:

  * allow only AUDIO_CB* ranges + `<aud_end>` (and maybe `<pad_mm>`)
* During T2I generation:

  * allow only IMAGE range + `<img_end>`

In Transformers, this is typically implemented via a `LogitsProcessor` or masking logits before sampling.

This one change usually makes:

* decode success go to ~100%,
* generation far more stable,
* and it reduces “random text tokens inside audio/image streams.”

### If decode fails due to length mismatch

Use the per-sample dynamic expected length approach:

* generate slightly longer than needed
* slice exact expected token count for decode
* record how often `<aud_end>/<img_end>` is emitted (separately)

### If prompt ablation is weak (model is unconditional)

Actions in order:

1. Ensure ablation is actually testing *prompt influence* (not accidentally still using the correct prompt due to caching/format bug).
2. Increase proportion of **T2A/T2I** in the mix (or ramp them later but give them enough weight).
3. Use stronger prompt text (less generic) for the golden set, especially for audio/music (instrument/genre/mood constraints).

---

## My overall assessment of your current trial

* **Training dynamics:** PASS
* **Multi-task compatibility:** PASS
* **Generation validity:** **PENDING** until you produce:

  1. T2A/T2I decode-success report (100% target), and
  2. T2A/T2I prompt ablation report (correct < shuffled/empty with strong win-rates)

Once both pass, proceed to a Stage‑4 “P1 scale run” (either more steps or a larger sample pool than 8k).

---

## One practical note based on your command

Your golden generation uses:

* `--max-new-audio 1200`
* `--max-new-image 256`

That is fine only if it matches your tokenizer meta. For many EnCodec configs, audio token count for a clip can be >1200. For **100% decode success**, make max length **data-driven per example** (from the golden meta), then slice exactly to expected length.

---

Provide the **bundle_inspect outputs** (even just the top-level JSON summary if it produces one) to determine:

* whether token-type compliance is already good,
* whether your max-new lengths are sufficient,
* and whether you need constrained decoding now or can postpone it.
