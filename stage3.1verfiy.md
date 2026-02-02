You implemented two distinct things under `stage3_uti/`:

1. **Stage 1 (UTI)**: a *stable ABI* for turning text / image / audio into **global token IDs** + metadata, and back again (for image/audio).
2. **A minimal “token LM” harness**: a smoke‑training loop proving the system can (a) **resize** the LLM vocab to include those tokens, (b) train on tokenized JSONL, and (c) checkpoint with `token_space.json`.

Your question is essentially: **what is the UTI supposed to “buy us,” what does “done” mean, and how do we verify it?** Below is the concrete answer.

---

## Verification Status (as of 2026-01-26)

* [x] UTI audit completed with diffusion decode (`outputs/uti_audit/*` present)
* [x] UTI audit completed with decode_mode=deterministic (image_idempotent=false)
* [x] Token space JSON + SHA written for Stage-3 (`outputs/stage3_token_lm/token_space.json`)
* [x] Minimal token LM overfit/resume/generation smoke tests (`outputs/stage3_token_lm_smoke/*`)
* [x] Audit rerun after adding strict token-range + decode SR/length + image size checks
* [x] Baseline regression thresholds run (`outputs/uti_audit/baseline_report.json`)
* [x] Log-mel metric computed (torchaudio working in base image)
* [x] Round-trip stability (signal-domain) logged in audit

## 1) What we are trying to achieve with UTI (in plain terms)

### UTI is the “binary interface” between modalities and the Transformer core

In a Type‑3 architecture, the core model never sees pixels or waveforms. It sees only **token IDs**.

So UTI must guarantee:

* **Every modality has a deterministic tokenization** into a **single shared ID space** (global IDs).
* Those tokens can be **decoded back** (at least for modalities you want to generate, i.e., image/audio).
* The mapping is **stable across runs and checkpoints** (so training/inference doesn’t silently drift).

If UTI is wrong, everything later becomes un-debuggable:

* training for days only to discover “audio tokens” were colliding with text tokens,
* or that decoding expects a different codebook count,
* or that “generation” produces tokens that can’t be decoded.

### Are we “loading pretrained encoders/decoders” here?

Yes—**but not in the transformer sense**.

* SEED tokenizer + detokenizer (and optional diffusion decoder) is your **image codec**.
* EnCodec is your **audio codec**.
* These are your **fixed modality bridges**. In Type‑3, they play the role that “projector/resampler” played in your VLM work, except now they are **discrete** and support *generation*.

### Will we “pair with a decoder” in future stages?

You already did. The decoders are:

* **EnCodec decoder** for audio tokens → waveform
* **SEED detokenizer / diffusion decode** for image tokens → image

Future stages train the *Transformer* to output tokens that those decoders can turn into signals. You generally **keep the codecs fixed** for a long time (unless you later decide to train a better tokenizer/codec).

---

## 2) Definition of Done for Stage 1 UTI (what “finished” means)

Stage 1 is “done” when UTI is correct, stable, and checkpoint-safe—**not** when generated audio sounds good.

### Stage 1 UTI exit criteria (hard)

I would declare Stage 1 complete when all items below are true:

#### A) TokenSpace correctness

1. [x] `TokenSpace.validate()` passes and checks:

   * contiguous ranges
   * no overlap
   * special tokens inside SPECIAL range
   * total vocab size consistent (`max_id+1`)
2. [x] `token_space.json` and `token_space.sha256` are written.
3. [x] `token_space.json` and `token_space.sha256` are **loaded back** with exact equality.
4. [x] Any attempt to load a checkpoint with a different token space **fails fast** (or requires an explicit override flag).

**Verification evidence:** [x] `outputs/uti_audit/token_space_audit.txt` printed with ranges and sizes; [x] mismatch resume raises error.

#### B) Deterministic encode for each modality

5. [x] `encode_text("...")` is deterministic.
6. [x] `encode_image(image0.png)` is deterministic (tokens + meta, modulo non-essential fields).
7. [x] `encode_audio(audio0.wav)` is deterministic (tokens + meta).

**Verification evidence:** [x] audit report + SHA256 files in `outputs/uti_audit/` (image_idempotent=false, audio_idempotent=false).

7a. [x] **Token range checks** enforced for text/image/audio.

#### C) Shape sanity is enforced (no “silent wrong shapes”)

8. [x] Audio meta must satisfy:

   * `len(tokens) == n_frames * n_codebooks` (given your chosen serialization)
   * tokens decode only if this holds (validate before decode)
9. [x] Image meta must satisfy:

   * token count matches grid shape / expected count for the tokenizer/resolution policy.

**Verification evidence:** [x] `audio_shape_ok` + `image_shape_ok` in `outputs/uti_audit/report.json`.
**Metrics (v2):** `metrics_v2` now includes `audio_snr_db`, `audio_mae`, optional `audio_log_mel_l1`, and `image_psnr`/`image_ssim` for deterministic decode.

#### D) Decode is callable and non-guessy

10. [x] `decode_audio(tokens, meta)` returns `(wav, sr)` without guessing codebook count or sample rate.
11. [x] `decode_image(tokens, meta)` returns an image without guessing grid/resolution.
12. [x] **Audio decode asserts** `sr == meta["sample_rate"]` and length within tolerance.
13. [x] **Image decode asserts** size matches `meta["proc_size"]`.

**Verification evidence:** a smoke script writes:

* `outputs/uti_smoke/recon.wav`
* `outputs/uti_smoke/recon.png`

#### E) “Idempotence” check (optional, but very valuable)

12. [ ] **Encode → decode → encode** is stable *for deterministic decoders*:

* audio: should generally pass (codec decode+encode should be near-stable depending on codec)
* image: if you use **diffusion decode**, this is not expected to be stable (diffusion is stochastic unless you lock seeds and even then it is not a true inverse).
  So, for UTI validation, you should test idempotence using the **deterministic SEED detokenizer path**, not diffusion.

**Verification evidence:** [ ] audio idempotence (currently false); [ ] image idempotence (currently false).

#### F) Reconstruction thresholds (regression gating)

14. [x] **Baseline regression checks** for audio SNR/log-mel and image PSNR/SSIM (run via `--baseline-report`).

#### G) Round-trip stability in signal domain (optional)

15. [x] **Audio round-trip stability** metrics (SNR/MAE) logged from decode→encode→decode.

---

## 3) Where your current implementation is already strong

From what you described, you already have:

* `TokenSpace` validation + JSON/sha I/O
* `UnifiedTokenizer` encode/decode methods with JSON meta
* Real adapters for SEED + EnCodec, plus dummy adapters for deterministic tests
* Test assets and tests (`test_uti.py`, `test_token_space.py`)
* Token space written to outputs

That is 80–90% of “Stage 1 done.” The remaining piece is usually not code—it is **verification discipline** and a couple of “fail-fast” guards.

---

## 4) The two most common verification gaps (and how to close them)

### Gap 1: Diffusion decode breaks determinism assumptions

You enabled diffusion decode for SEED (`load_diffusion=true`). That’s fine as an option, but:

* It is **not** a reliable primitive for Stage 1 “UTI correctness” validation.
* It can make “round-trip” style tests fail even if token mapping is correct.

**Recommendation**

* For Stage 1 completion, require that:

  * the deterministic SEED path works (quantizer detokenizer)
  * diffusion decode is optional and tested as a **separate** “quality decode” mode

**How to verify**
Run two decode smoke tests:

* [x] `decode_mode=deterministic` (ran; idempotence did not pass; uses token-grid renderer to bypass diffusion)
* [x] `decode_mode=diffusion` (must decode without crashing; determinism not required)

### Gap 2: “Config says codebook sizes” but runtime may differ

You stated: “Explicit codebook sizes for predictable token-space building.” That’s good for reproducibility, but the #1 failure mode is:

* config says `n_codebooks=4`, runtime codec runs with different bandwidth → `n_codebooks != 4`

**Recommendation**
Even if you allow explicit overrides, UTI should always do a runtime sanity check:

* introspect the actual codec/tokenizer sizes
* if mismatch, either:

  * fail hard (recommended), or
  * log a loud warning and write the *runtime* truth into meta and token_space.json (less recommended for research reproducibility)

**How to verify**
Add a `--verify_runtime_sizes` run that prints:

* [x] EnCodec: codebook_size, n_codebooks actually used
* [x] SEED: codebook_size and token grid for your chosen resolution

---

## 5) How to verify Stage 1 is complete (recommended “UTI Audit Runbook”)

Add one CLI script (or a mode in `unified_tokenizer.py`) that produces a small artifact bundle:

### Command

```bash
python3 -m stage3_uti.tokenization.unified_tokenizer \
  --config stage3_uti/configs/uti.yaml \
  --token-space-out outputs/uti_audit/token_space.json \
  --smoke-assets stage3_uti/tests/assets \
  --outdir outputs/uti_audit \
  --decode-mode deterministic
```

### Outputs (what you should expect)

* `outputs/uti_audit/token_space.json`
* `outputs/uti_audit/token_space.sha256`
* `outputs/uti_audit/image_tokens.sha256`
* `outputs/uti_audit/audio_tokens.sha256`
* `outputs/uti_audit/recon.png`
* `outputs/uti_audit/recon.wav`
* `outputs/uti_audit/report.json` with:

  * token counts
  * codec sizes
  * image grid
  * pass/fail flags

**Stage 1 is “finished” when this audit passes and is reproducible.**

**Audit status:** [x] diffusion audit run completed; [x] deterministic audit run completed (image_idempotent=false).

---

## 6) What the “Minimal Token LM” is for (and how to declare it done)

Your minimal Stage‑3 token LM harness is not about quality. It is about proving:

* embeddings can be resized to `vocab_size_total`
* global token IDs can be fed through the model
* loss/backprop/checkpoint/resume are functional

### Minimal token LM exit criteria

1. [x] **Overfit test on tiny tokenized set**:

   * tokenized JSONL with ~256 sequences
   * train for 200–500 steps
   * training loss drops sharply (even if eval isn’t meaningful)

2. [x] **Checkpoint resume test**:

   * train 100 steps, save
   * resume, train 100 more steps
   * loss continues smoothly (no mismatch loading embeddings/head sizes)

3. [x] **Decode safety test for generation**

   * sample a very short generation in “gen_audio” mode:

     * it won’t sound good, but it must produce tokens in valid ranges
   * decode does not crash

This validates end-to-end plumbing.

**Verification evidence:** [x] `outputs/stage3_token_lm_smoke/checkpoint_2`, `checkpoint_4`; [x] `outputs/stage3_token_lm_smoke/gen_audio.wav`.

---

## 7) What you should *not* use to declare Stage 1 done

Do **not** use perceptual quality as a completion criterion yet:

* recon audio quality (EnCodec fidelity) is mostly fixed by the codec
* image decode quality depends on the tokenizer and whether diffusion is used
* the core LM isn’t trained yet to generate meaningful tokens

Stage 1 is a **systems/ABI** milestone, not a model-quality milestone.

---

## 8) “Stage is finished” statement for the repo

Suggested statement:

> Stage 1 (UTI) is complete when TokenSpace is collision-free and checkpointed, and UTI deterministically encodes/decodes text, image, and audio with strict shape validation and reproducible audits.

And attach the audit artifacts.

---

## One specific next improvement before calling it complete

Given your implementation, the single highest-value addition is:

**A strict runtime introspection check** that the *actual* codec/tokenizer sizes match `uti.yaml` and therefore match `token_space.json`.

That is the most common real-world failure that only shows up later (and is painful to debug once you have tokenized shards).

Optional: provide the relevant `uti.yaml` excerpt for EnCodec (sample rate, bandwidth/codebooks, codebook size) and the runtime codec config (or adapter code) to define the exact assertions and locations needed to prevent “silent mismatch.”
