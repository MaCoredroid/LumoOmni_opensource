## Stage 3 — Warm-start multimodal vocabulary on a text LLM backbone

### Objective

Warm-start a **text-only causal LLM** so that **discrete audio/image token IDs** (your VQ / codec IDs from Stages 0–2) behave like *meaningful symbols* in the model’s vocabulary—**without changing language behavior** on text-only prompts.

This is exactly the “connector alignment” analogue, except the “connector” is now the **new vocabulary rows** (input embeddings + output head) rather than a cross-modal module.

---

## 1) Hard requirements and assumptions

### Backbone

* Initialize from **Qwen3-8B-Base** (or smaller Qwen3 variant for faster iteration). The model card recommends using a recent Transformers version; older versions can error (e.g., `KeyError: 'qwen3'` for older releases). ([Hugging Face][1])
* Qwen3’s config defaults include:

  * `tie_word_embeddings = False` (so **input embeddings and LM head are NOT tied** by default) ([Hugging Face][2])
  * `max_position_embeddings = 32768` (so you must enforce a strict clip policy; see §6) ([Hugging Face][2])

### Tokenizer/vocab caveat (important)

There has been a reported mismatch for Qwen3 between `config.json`’s `vocab_size` and the *actual tokenizer size* (example: config `151936` vs tokenizer `151669`), which can cause shape mismatches when resizing embeddings. ([Hugging Face][3])
**Rule:** always treat `len(tokenizer)` as the source of truth and ensure the model’s embedding shapes match it *before* adding modality tokens.

### Distributed training caveat

If you use FSDP / meta-device init flows, resizing embeddings can break when parts of the model are on meta tensors. The safer pattern is: **resize embeddings before wrapping with FSDP / `accelerate.prepare()`**. ([GitHub][4])

---

## 2) Vocabulary design (token ID layout)

### Define contiguous ID ranges

Let:

* `V_text = len(tokenizer)`  (after you’ve loaded tokenizer + fixed any config mismatch)
* `N_audio = size_of_audio_codebook`  (e.g., 1024/2048/4096; whatever Stage 0 produced)
* `N_image = size_of_image_codebook`  (e.g., 8192/16384; whatever Stage 0 produced)

Then define **new token IDs**:

* Audio token IDs: `[V_text, V_text + N_audio - 1]`
* Image token IDs: `[V_text + N_audio, V_text + N_audio + N_image - 1]`

New total vocab:

* `V_total = V_text + N_audio + N_image`

**No collisions** are possible if you append after `V_text`.

### Optional (recommended) “delimiter” tokens

Keep delimiters as *existing text special tokens* if you already have them, e.g.

* `<|audio|>`, `<|image|>`, `<|end_audio|>`, `<|end_image|>`

If you must add delimiters as new tokens, they land in the appended region too—which complicates “only train modality rows” unless you explicitly include them in the trainable set. Prefer reusing existing text specials to keep Stage 3 clean.

---

## 3) Model surgery

### 3.1 Resize input embeddings + LM head

Because Qwen3 defaults to **untied embeddings** (`tie_word_embeddings=False`), you should ensure **both**:

* `model.get_input_embeddings().weight` grows to `[V_total, d_model]`
* `model.get_output_embeddings()` / `lm_head.weight` grows to `[V_total, d_model]` (or `[V_total, d_model]` depending on implementation)

The Transformers Qwen3 config explicitly documents `tie_word_embeddings` and the default behavior. ([Hugging Face][2])

### 3.2 Initialize new rows

Goal: new rows should not spike activations.

Recommended init (stable, simple):

* Let `E = input_embedding.weight[:V_text]`
* Compute `μ = mean(E, dim=0)` and `σ = std(E)` (scalar or per-dim)
* Initialize each new row as: `μ + ε`, with `ε ~ Normal(0, 0.02 * σ)` (small noise)

For LM head new rows:

* Initialize to **0** or same distribution as existing head rows.
* If you plan to do *any* modality-token prediction (copy/denoise), don’t leave head rows too tiny for too long; a small normal init is fine.

---

## 4) Trainable parameters (freeze plan)

### Train **only**:

1. **Input embeddings rows** for modality token IDs:

   * audio IDs `[V_text : V_text + N_audio)`
   * image IDs `[V_text + N_audio : V_total)`

2. **LM head rows** for modality token IDs (same ranges)

3. Optional: **small LoRA** on top layers (only if needed)

   * Only last `K` transformer blocks (e.g., 4–8)
   * Very small rank (e.g., r=4 or r=8), low alpha, dropout 0–0.05
   * If LoRA is enabled, include a *tiny* text-only “anchor” batch mix to prevent any drift.

### Everything else frozen

* All transformer blocks
* All layer norms
* All text token rows in embeddings/head
* Positional/RoPE params, etc.

### Practical implementation detail: “row-level training”

PyTorch doesn’t let you set `requires_grad` per-row. You must do one of:

**Option A (recommended): gradient masking hooks**

* Keep `embedding.weight.requires_grad = True`
* Register a hook on `.weight` that zeros gradients for rows `< V_text`
* Same for `lm_head.weight`

**Option B: split embedding tables**

* Replace embedding lookup with:

  * frozen base embedding table for text tokens
  * trainable modality embedding table for new IDs
* Merge at runtime
* More code, but clean semantics.

Stick to Option A unless you have a strong reason.

---

## 5) Data/tasks

Stage 3 assumes access to `(modality_tokens, caption_text)` pairs from the downloaded datasets.

### 5.0 Stage‑3 JSONL outputs (from Stage‑2 shards)

We generate the training/eval JSONL used by the Stage‑3 token‑LM from the Stage‑2
tokenized WebDataset shards, with clip policy + label masking applied:

* `outputs/stage3_token_lm/train.jsonl`
* `outputs/stage3_token_lm/eval.jsonl`

These are large artifacts and should remain in `.gitignore`.

### 5.1 Captioning tasks (core)

These are the *primary* signal that modality IDs affect text prediction.

#### Audio captioning

* **Input**: audio token IDs (your codec stream) + short instruction
* **Target**: caption text
* **Loss**: computed **only on caption text tokens**

Example format (chat-style, teacher forcing):

* User content:
  `"<|audio|> {AUDIO_TOKENS} <|end_audio|>\nDescribe the audio."`
* Assistant content (target):
  `"{caption}"`

#### Image captioning

Same structure:

* User: `"<|image|> {IMAGE_TOKENS} <|end_image|>\nDescribe the image."`
* Assistant: caption

**Critical label masking rule:** ensure all tokens before the assistant answer (including modality tokens + prompt) have label `-100`, so you only train against caption text.

---

### 5.2 Optional stabilization tasks (only if needed)

These exist mainly to:

* force the model not to ignore modality tokens
* give gradients to **LM head modality rows** (since captioning trains the *input embeddings*, not necessarily the head rows)

#### (A) Copy task (simplest)

* Input: modality tokens
* Output: the same modality tokens
* Loss: only on the output modality tokens

Example:

* User: `"<|audio|> {AUDIO_TOKENS} <|end_audio|>\nCopy the audio tokens."`
* Assistant: `"{AUDIO_TOKENS}"`

This directly trains LM head modality rows.

#### (B) Denoise / span corruption (better than pure copy)

* Corrupt small spans of modality tokens (replace with random modality IDs or a fixed “mask-like” sentinel)
* Ask model to reconstruct the clean sequence at output
* Loss: only on reconstructed modality tokens

Recommended corruption:

* 10–15% tokens affected
* contiguous spans length 3–12
* corruption types: replace-with-random (70%), delete (15%), keep (15%) to avoid distribution shift

**Keep denoise/copy low-weight** (e.g., 5–20% of batches). Captioning should dominate.

---

## 6) Strict clip policy (non-negotiable)

Your failure mode list is correct: modality tokens can explode context length.

### Define per-modality caps

Pick caps so that:

* You leave headroom for prompt + caption
* You stay far below `max_position_embeddings` (32,768 by default) ([Hugging Face][2])

Recommended starting caps (adjust to your tokenizers):

* `T_audio_max = 1024` (or 2048 if absolutely needed)
* `T_image_max = 256` to `576` (depends on image tokenization scheme)

### Trimming strategy

* **Audio**: sample a contiguous window (random during train, center during eval)
* **Image**: if it’s a grid sequence, keep a consistent crop scheme (center or random crop in token-space if it preserves locality)

### Reject policy

If a sample exceeds hard limits and can’t be safely cropped (e.g., missing alignment), drop it rather than letting it destabilize batches.

---

## 7) Training recipe (defaults that usually work)

### Optimizer

* AdamW
* Two param groups:

  1. modality input embedding rows
  2. modality lm_head rows
* Weight decay: 0.0 (these are “symbol tables”, decay often hurts)

### LR and schedule (starting point)

Because you’re training a tiny fraction of params, LR can be higher than full fine-tune, but don’t go crazy.

Good starting point:

* `lr_embed = 1e-3`
* `lr_head  = 1e-3`
* warmup: 2–5% of steps
* schedule: cosine decay → 1e-4

### Batch construction

* Mix audio/image captioning, e.g. 50/50 or proportional to data sizes
* Optional denoise/copy: 5–20% of batches

### Precision

* bf16 preferred
* gradient clip: 1.0

### If you use FSDP/ZeRO

* **Resize embeddings before wrapping**, to avoid meta-device resize issues. ([GitHub][4])

---

## 8) Acceptance tests (must-pass)

You asked for the “full spec and verification list”—this is the pass/fail gate.

### 8.1 Modality sensitivity ablation (per modality)

For a held-out dev set, compute teacher-forced loss on the **caption text tokens** under different modality inputs.

For each sample, define:

* `L_correct`: loss with correct modality tokens
* `L_shuffle`: loss with same tokens randomly permuted (within the modality span)
* `L_noise`: loss with random modality token IDs (uniform within that modality range)
* `L_zero`: loss with a constant token ID repeated (e.g., first audio token ID)

**Pass criteria (recommend):**

* Mean gap:

  * `mean(L_shuffle - L_correct) >= 0.10` nats/token **or** `>= 5%` relative
  * `mean(L_noise  - L_correct) >= 0.15` nats/token **or** `>= 8%` relative
* Per-sample win rate:

  * `P(L_correct < L_shuffle) >= 0.80`
  * `P(L_correct < L_noise)  >= 0.85`

Run this separately for **audio-caption dev** and **image-caption dev**.

Interpretation:

* If gaps are tiny → model is ignoring modality tokens.
* If gaps exist but captions still bad → embeddings learned “something” but you may need better data/format.

### 8.2 Text-only regression (hard stability gate)

Because you froze the backbone, text-only behavior should be **identical** unless you enabled LoRA or accidentally updated text rows.

Two levels:

**(A) Strong check (if no LoRA):**

* Pick a fixed set of text-only prompts.
* Compare logits from base vs Stage-3 model.
* Require: `max_abs_diff == 0` (or < 1e-6 if you have nondeterminism).

**(B) Statistical check:**

* Evaluate perplexity on a small held-out text corpus.
* Require: relative Δppl ≤ 1% (≤2% if LoRA enabled).

### 8.3 Parameter integrity checks (must-pass)

After training:

* For all frozen params: **exact match** (bytewise if possible).
* For embeddings and lm_head:

  * text rows `[0:V_text)` unchanged (≈0 diff)
  * modality rows changed (non-zero diff)

### 8.4 Length compliance checks

On the training manifest:

* `max(seq_len) <= max_position_embeddings` (32,768 by default) ([Hugging Face][2])
* report p50/p90/p99 lengths
* enforce caps from §6

---

## 9) Verification checklist (runbook)

### Before training

* [x] Load tokenizer; set `V_text = len(tokenizer)`
* [x] Verify model embedding shapes match `V_text` (watch for Qwen3 vocab mismatch reports). ([Hugging Face][3])
* [x] Define `N_audio`, `N_image`, offsets, and `V_total`
* [x] Resize **both** input embeddings and LM head to `V_total` (Qwen3 defaults to untied embeddings). ([Hugging Face][2])
* [x] Initialize new rows with low-variance init
* [x] Freeze everything except modality rows (and optional LoRA)
* [x] Install gradient masking hooks so only modality rows update
* [x] Confirm clip policy is enforced in dataloader (unit test with worst-case samples)  
  - `outputs/stage3_token_lm_iter5/clip_policy_check_8192.json`

### During training (monitoring)

* [x] Track captioning loss (audio, image separately) (per‑task logging enabled)
* [x] Track modality ablation gaps on a small fixed dev batch every N steps  
  - log: `outputs/stage3_token_lm_iter5/ablation_log.jsonl`
* [x] Track norm stats for modality embeddings (mean/std shouldn’t explode) (logged)
* [x] If LoRA enabled: track text-only perplexity periodically (LoRA disabled)

### After training

* [x] Run full ablation suite (§8.1) (Tier‑B 8,192 run; iter3 final ablation)
  - `outputs/stage3_token_lm_iter/ablation_eval.json`
  - `outputs/stage3_token_lm_iter3/ablation_eval_final.json`
* [x] Run text-only regression (§8.2) (step 7000)
  - `outputs/stage3_token_lm_iter5/text_regression_invariance.json`
* [x] Run parameter diff audit (§8.3) (step 7000)
  - `outputs/stage3_token_lm_iter5/text_regression_invariance.json`
* [ ] Save:

  * model weights
  * updated config (vocab size)
  * tokenizer metadata + your modality-range spec (offsets, codebook sizes)
  * an “inference adapter” snippet that inserts modality IDs correctly

---

## 10) Typical failure modes and fixes

### Failure: model ignores modality tokens

**Symptoms**

* `L_correct ≈ L_shuffle ≈ L_noise`
* captions are generic, dominated by prompt priors

**Fixes**

* Increase captioning weight (reduce copy/denoise)
* Remove extra text hints in prompts (make prompts minimal)
* Add “modality dropout”: randomly drop 10–30% of modality tokens in input so model learns distributed reliance
* Add denoise (span corruption) if you weren’t training head rows at all

### Failure: instability / weird generations when modality tokens present

**Symptoms**

* loss spikes early
* captions become repetitive

**Fixes**

* lower LR 2–5×
* reduce modality token caps
* reduce init variance (make new rows closer to embedding mean)
* optionally add tiny LoRA on last 4 layers (and add text-only anchor batches)

### Failure: sequence length blow-ups

**Fix**

* enforce hard caps at dataset creation time, not only at collate time
* reject/trim aggressively (audio windows, image token crops)

---

## Deliverable definition (what “Stage 3 complete” means)

Stage 3 is done when:

1. **Ablation sensitivity passes** for audio + image (§8.1), and
2. **Text-only regression is stable** (§8.2), and
3. **Only modality rows changed** (§8.3).

Optional: provide Qwen3 + Transformers details to generate a *reference implementation skeleton* (row-gradient masking hooks, collator label masking, ablation eval script structure). The spec above is the full Stage‑3 contract.

---

## Stage 3 Status Checklist (as of 2026-02-01)

- [x] **Token space + UTI config finalized** (`outputs/stage3_token_lm/token_space.json`)
- [x] **Stage‑3 JSONL built** (train/eval):  
  - `outputs/stage3_token_lm/train.jsonl`  
  - `outputs/stage3_token_lm/eval.jsonl`
- [x] **Fast iteration Tier‑A (2,048) completed**
- [x] **Fast iteration Tier‑B (8,192) completed**
- [x] **Fast iteration Tier‑B (8,192) resumed 3 epochs (iter3) completed**
  - `outputs/stage3_token_lm_iter3/checkpoint_24576`
- [x] **Ablation eval (Tier‑A)** passes (correct < shuffle/noise)
- [x] **Ablation eval (Tier‑B)** passes (correct < shuffle/noise)  
  - `outputs/stage3_token_lm_iter/ablation_eval.json`
- [x] **Ablation eval (iter3 final, 128/128 sample)** passes (correct < shuffle/noise)  
  - `outputs/stage3_token_lm_iter3/ablation_eval_final.json`
- [x] **Eval set (epoch‑0 on Tier‑B)**  
  - `outputs/stage3_token_lm_iter/eval_epoch0.json`  
  - loss(all)=2.8004, a2t=2.4211, i2t=3.3803
- [x] **Eval set (iter5 step 7000)**  
  - `outputs/stage3_token_lm_iter5/eval_step7000.json`  
  - loss(all)=2.7462, a2t=2.3606, i2t=3.3357
- [x] **Eval set (iter3 final step 24576)**  
  - `outputs/stage3_token_lm_iter3/eval_final.json`  
  - loss(all)=2.8839, a2t=2.3209, i2t=3.7445
- [x] **Trainable‑rows‑only checkpointing implemented**
- [x] **Trainable‑rows resume smoke test passed**

- [ ] **Tier‑C (51,200) run** (not started)
- [ ] **Full‑dataset run** (not started)
- [x] **Text‑only regression check** (step 7000)
  - `outputs/stage3_token_lm_iter5/text_regression_invariance.json`
- [x] **Frozen‑weights invariance audit** (base vs patched model)
  - `outputs/stage3_token_lm_iter5/text_regression_invariance.json`
- [x] **Deliverable packaging** (final adapter + README + load snippet)
  - `stage3.3deliverable.md`

[1]: https://huggingface.co/Qwen/Qwen3-8B-Base "Qwen/Qwen3-8B-Base · Hugging Face"
[2]: https://huggingface.co/docs/transformers/en/model_doc/qwen3 "Qwen3"
[3]: https://huggingface.co/Qwen/Qwen3-8B/discussions/25 "Qwen/Qwen3-8B · Fix incorrect vocab_size in Qwen3-8B config.json"
[4]: https://github.com/huggingface/transformers/issues/40382 "Cannot resize token embeddings in distributed training setups with FSDP · Issue #40382 · huggingface/transformers · GitHub"
