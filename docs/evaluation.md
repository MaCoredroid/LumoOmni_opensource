# Evaluation

LumoOmni prioritizes evaluation that answers one question early:

> **“Is the model actually using the modality signal we think it is using?”**

That leads to **ablation‑first** evaluation (loss ordering) and **audit‑first** verification (token interfaces).

---

## Track A — Continuous‑embedding VLM

### Ablation test (teacher‑forced loss)

Evaluate loss on held‑out samples under:
- correct image
- shuffled image
- zeroed visual tokens
- random/noise visual tokens

**Pass criterion:** `loss(correct) < loss(shuffled) < loss(zero/noise)`.

This catches:
- placeholder mismatch
- injection bugs (tokens not inserted)
- masking bugs (loss computed on wrong tokens)
- “vision ignored” collapse

### Truncation + label coverage

Track per epoch:
- truncation rate (% samples truncated by `max_seq_len`)
- % samples with `label_tokens == 0`
- avg label tokens

This catches:
- prompt wrapper too long
- formatting issues that mask labels

### Stage 1.1 sweep metrics

For sweep runs, compare:
- eval loss (stratified by label length buckets)
- ablation deltas (Δshuffle, Δzero) as a proxy for “vision sensitivity”
- qualitative “golden set” captions

---

## Track B — Discrete tokens (UTI + token LM)

### UTI audit (promotion gate)

UTI audit checks:
- determinism (text/image/audio)
- token range correctness
- shape sanity (token counts)
- decode sanity (audio SR/length/channels; image sizes)
- metric regression gates (PSNR/SSIM; log‑mel; SNR)

**Important:** token‑id idempotence is not expected for lossy codecs and is treated as a non‑gate.

### Token LM smoke tests

Minimal “does it train” tests:
- overfit tiny set (loss drops)
- resume from checkpoint
- generation decode safety (short sequences decode without crash)

### Token LM eval

Report loss by task:
- `loss(all)`
- `loss(a2t)` (audio → text)
- `loss(i2t)` (image → text)
- (later) `loss(t2a)`, `loss(t2i)` for generation stages

---

## What is *not* claimed yet

- Strong conditional generation quality requires post‑run decode checks + prompt ablation, not just teacher‑forced loss.
- Final benchmarking (e.g., VQA, captioning, audio caption benchmarks) is out of scope for early stage gates.
