You can (and should) start **Stage 4** from the **Stage 3 checkpoint** you just produced.

Using the **same 8,000-row subset** for Stage 4 is appropriate **for a Stage‑4 “trial / plumbing + signal” run**. It is not “enough” to judge final generative quality, but it is enough to verify:

* the **T2A / T2I** sequence formats are correct,
* **labels are routed correctly** to audio/image token IDs,
* **generation decodes safely** (no shape crashes),
* and the model learns a **non-trivial conditional signal** (loss/ablations move in the right direction).

Below is an implementation-ready **Stage 4 spec** and a **verification checklist** (pre-run, in-run, post-run) consistent with how you’ve been operating (ablations, golden sets, strict hashing, regression gating).

---

# Stage 4 Spec — Multimodal Token Pretraining (MMPT)

## Stage 4 objective

Teach the core LM to **emit** discrete modality tokens:

* **T2A**: text → audio tokens (then decode with EnCodec)
* **T2I**: text → image tokens (then decode with SEED)

while preserving the Stage‑3 “read” capability:

* **A2T**: audio tokens → text
* **I2T**: image tokens → text

This is the first stage where audio/image tokens appear on the **label** side.

---

## 4.0 Inputs

### Required

* **Stage 3 checkpoint** (must include resized embeddings + resized LM head + token_space.json + sha256)
* Stage 2 tokenized shards (audio + image) with `token_space_sha256` matching the checkpoint
* `uti.yaml` (for decode in eval harness, not for training)

### Hard invariant

* `token_space.sha256` must match across:

  * Stage 2 tokenized data
  * Stage 3 checkpoint
  * Stage 4 run config

Fail fast otherwise.

---

## 4.1 Model architecture (what we train)

### Base

* Qwen3 decoder-only LM with expanded vocab = text + specials + image tokens + audio tokens (per codebook ranges)

### Tokenizers/codecs

* EnCodec and SEED are **frozen** (used offline for data, and only for decoding during eval)

### Trainable parameters (recommended baseline)

You want more capacity than Stage 3 because now the LM must learn long-range high-entropy token streams.

**Train:**

1. **Modality token embeddings**

   * All rows corresponding to IMAGE + AUDIO token ID ranges
2. **Special token embeddings** used by Stage 4 formats
3. **LM head rows** for IMAGE + AUDIO token ID ranges (if untied)
4. **LoRA on the transformer blocks**

   * Target modules: attention projection + MLP projection (your existing LoRA convention is fine)
* Keep LoRA modest; scale if underfitting

**Freeze:**

* All other base weights initially (except LoRA)
* (Optional) allow LayerNorms to train; it can help stability, but not required

> If you tie `lm_head` to embeddings, training embeddings automatically trains the output weights; in that case LM-head-only rows are not separately trainable.

---

## 4.2 Tasks and how to build Stage‑4 examples from your existing tokenized data

You can reuse the same 8k rows and produce **bidirectional** tasks.

### Audio datasets (Clotho / AudioCaps / WavCaps / music captions)

Each tokenized record has:

* `audio_tokens` (global IDs in AUDIO_CB* ranges)
* `caption_text_out` (text tokens in TEXT range)

Create two examples per record:

#### A2T (audio → text)

Input: audio tokens
Target: caption text

#### T2A (text → audio)

Input: caption text
Target: audio tokens

### Image dataset (LLaVA-pretrain subset)

Each record has:

* `image_tokens` (global IDs in IMAGE range)
* `caption/assistant_text_out` (text)

Create two examples per record:

#### I2T (image → text)

Input: image tokens
Target: caption text

#### T2I (text → image)

Input: caption text
Target: image tokens

---

## 4.3 Sequence formats and label masks (contract)

Use explicit boundary + “gen” tokens from your token_space specials. All sequences are standard next-token LM training; correctness depends on label masking.

### Common rules

* Labels are `-100` for non-target positions.
* Only the target segment (text_out OR audio_tokens OR image_tokens) is supervised.

### A2T format

```
<aud_start> AUD... <aud_end> <gen_text> <text_start> TEXT_OUT... <text_end>
```

**Labels:** only `TEXT_OUT` (optionally include `<text_end>`)

### I2T format

```
<img_start> IMG... <img_end> <gen_text> <text_start> TEXT_OUT... <text_end>
```

**Labels:** only `TEXT_OUT`

### T2A format

```
<text_start> PROMPT_TEXT... <text_end> <gen_aud> <aud_start> AUD... <aud_end>
```

**Labels:** only `AUD...` (optionally include `<aud_end>`)

### T2I format

```
<text_start> PROMPT_TEXT... <text_end> <gen_img> <img_start> IMG... <img_end>
```

**Labels:** only `IMG...` (optionally include `<img_end>`)

### Length constraints (hard policy)

For Stage 4 you must **never silently truncate the target segment**.

* If the full sequence would exceed `max_seq_len`, **drop the sample** (or shorten *input* text, not target tokens).
* Log drop counts per task so you see whether `max_seq_len` is feasible.

---

## 4.4 Training mixture (task weights)

For a first Stage‑4 trial on 8k rows:

* 25% A2T
* 25% I2T
* 25% T2A
* 25% T2I

If you see instability (common), use a ramp:

### Ramp schedule (recommended)

* Steps 0 → N_ramp: **only A2T/I2T** (stabilize)
* After N_ramp: mix in T2A/T2I gradually until you reach your target weights

You can implement this as a function of global_step.

---

## 4.5 Optimization (high-level spec)

Because you want “epoch-like” runs bounded in wall time, make Stage 4 step-based for the trial:

### Suggested run modes

* **Stage4-trial**: `train_steps=2000` (enough to see direction + decode safety)
* **Stage4-promote**: increase steps and/or increase sample pool size once verification passes

### Optimizer / precision / stability requirements

* bf16 precision (as you’ve been doing)
* gradient clipping (recommended; prevents rare spikes on long token targets)
* token-weighted loss logging (so long audio targets don’t skew metrics incorrectly)

### Must log per-task losses

You need separate loss curves:

* `loss_a2t`, `loss_i2t`, `loss_t2a`, `loss_t2i`
  plus overall token-weighted loss.

---

## 4.6 Evaluation harness (golden sets + metrics)

Stage 4 requires evaluation that checks **generation + decode**, not just loss.

### 4.6.1 Golden sets (fixed, persisted)

Create and save once:

* `golden_a2t_64.jsonl` (audio tokens + gt caption)
* `golden_t2a_64.jsonl` (prompt text + gt audio tokens)
* `golden_i2t_64.jsonl`
* `golden_t2i_64.jsonl`

Stratify by caption length (short/med/long). For audio, also stratify by dataset source (speech-like vs music-like if available).

### 4.6.2 Golden generation at checkpoints

At each `save_steps` (or at least at end):

* Generate:

  * A2T caption text (decode text)
  * T2A audio tokens (decode to wav using EnCodec)
  * I2T caption text
  * T2I image tokens (decode to image using SEED)
* Write artifacts:

  * `qual/step_<k>_golden_a2t.jsonl` etc.
  * `qual/step_<k>_t2a_audio/*.wav`
  * `qual/step_<k>_t2i_image/*.png`

### 4.6.3 Generation mode (important)

Start with **greedy decoding** (`do_sample=False`) for determinism and comparability.

* When using temperature/top_p, ensure `do_sample=True`—otherwise those flags are ignored.

### 4.6.4 Quantitative metrics (minimal set)

For the trial, keep metrics simple and robust:

**Decode success metrics (hard gate):**

* `t2a_decode_success_rate == 100%` on golden set
* `t2i_decode_success_rate == 100%`

**Conditionality metrics (high-value):**

* Teacher-forced **prompt ablation** for T2A/T2I:

  * compute loss on a fixed set with:

    * correct prompt
    * shuffled prompt
    * empty prompt (or `<unk>`)
  * Expect: `loss(correct) < loss(shuffled/empty)` by a measurable margin

**Optional (recommended) embedding metrics:**

* Audio: CLAP similarity (prompt text vs generated audio)
* Image: CLIP similarity (prompt text vs generated image)

Treat CLAP/CLIP as **trend metrics**, not hard pass/fail on the first trial.

---

# Stage 4 Verification List (Go/No-Go)

This is the checklist I would use to decide “Stage 4 plumbing is correct” and you’re safe to scale.

## A) Pre-run verification (must pass before training)

### A1. Token space strict match (hard gate)

* Stage 4 run loads:

  * `token_space.json`
  * `token_space.sha256`
* Verifier checks:

  * dataset shards’ `token_space_sha256 == run_token_space_sha256`
  * checkpoint token space hash matches

### A2. Trainable-parameter coverage (hard gate)

Log and assert:

* All IMAGE token ID rows are trainable (embeddings)
* All AUDIO token ID rows are trainable (embeddings)
* Specials used in formats are trainable
* LoRA modules attached and trainable
* Total trainable parameter count printed (for reproducibility)

### A3. Sequence builder unit tests (hard gate)

For each task (A2T/I2T/T2A/T2I) on a small batch:

* special tokens appear in correct order
* labels are `-100` outside the intended target segment
* `label_len > 0`
* No truncation of target segments
* Range checks:

  * audio labels only in AUDIO ranges
  * image labels only in IMAGE range
  * text labels only in TEXT range

### A4. Decode smoke test (hard gate)

Before training starts:

* Take 4 samples from each modality:

  * decode stored tokens → wav/png
  * ensure no exceptions and sizes/SR match meta

---

## B) In-run verification (must observe during the trial)

### B1. Loss behavior (expected)

Within the first portion of training:

* `loss_t2a` and/or `loss_t2i` should decrease measurably (even modestly)
* `loss_a2t` and `loss_i2t` should not explode (they can move slightly)

### B2. Gradient / norm stability

* No NaNs
* Gradient norm not exploding (use clip; log “clipped fraction” if you have it)
* Embedding norms for modality rows drift smoothly (no sudden jumps)

### B3. T2A/T2I “prompt ablation” turns positive

On a fixed ablation set (use ≥256 items if feasible; 64 is noisy):

* `loss(correct prompt) < loss(shuffled prompt)`
* `loss(correct prompt) < loss(empty prompt)`
  This is the most direct proof that generation is conditional.

### B4. Golden generation artifacts exist and decode

At each save:

* T2A produces token sequences that decode into playable wav files
* T2I produces valid PNGs with correct proc_size

---

## C) Post-run verification (promotion gate to scale Stage 4)

### C1. Decode success is perfect on golden set (hard gate)

* 100% decode success for T2A and T2I on golden sets

### C2. Conditionality improvement vs baseline (strongly recommended)

Compare step 0 (or first checkpoint) vs final checkpoint:

* prompt ablation deltas increase (shuffle/empty become worse relative to correct)
* CLAP/CLIP trend improves (if enabled)

### C3. No catastrophic language regression

Run a small fixed text-only prompt set; ensure outputs remain sane.
(With LoRA + modality rows only, this should hold, but still verify.)

### C4. Save/load + resume works (hard gate)

* Resume training from a checkpoint without:

  * token_space mismatch
  * embedding shape mismatch
  * LoRA adapter mismatch

---

# Answering your assumption: “use the same 8k rows for Stage 4”

You are directionally right, with a small correction:

* **Use the same 8k rows for Stage‑4 trial** to validate correctness and get the first conditional generation signal.
* Once verification passes, you will likely want to **expand the sample pool** (or increase steps) for meaningful audio/music/image quality—because T2A/T2I are simply harder objectives than A2T/I2T.

That said, you do **not** need to run the full 250k before you start Stage 4. You need a correct Stage 4 pipeline first.

---

## Default Stage‑4 trial configuration

To keep this precise and immediately actionable, tell me two things from your current setup:

1. Your Stage 4 `max_seq_len` target (e.g., 2048 / 4096)
2. Your audio token length per sample (from meta: `n_codebooks * n_frames`)

With those, I’ll give you a concrete Stage‑4 trial YAML (tasks + weights + ramp + eval schedule + golden generation config) that will fit your constraints without silently dropping most samples.

---

# Stage 4 Status Checklist (live)

**Snapshot:** 2026-02-01

## Data + setup

- [x] Stage‑4 JSONL built (train_8k + eval_512)
- [x] Golden sets created (golden_*_64.jsonl)
- [x] Token space hash checked at load

## Trial run (2k steps)

- [x] Trial training completed (`outputs/stage4_mmpt_trial/checkpoint_2000`)
- [x] Trainable‑rows + LoRA checkpoints saved
- [x] Loss logging per task (a2t/i2t/t2a/t2i)

## Completed artifacts (paths)

- [x] Stage‑4 train/eval JSONL: `outputs/stage4_mmpt/train_8k.jsonl`, `outputs/stage4_mmpt/eval_512.jsonl`
- [x] Golden sets: `outputs/stage4_mmpt/golden/golden_{a2t,i2t,t2a,t2i}_64.jsonl`
- [x] Trial checkpoints: `outputs/stage4_mmpt_trial/checkpoint_{500,1000,1500,2000}`
- [x] Trial LoRA: `outputs/stage4_mmpt_trial/lora_{500,1000,1500,2000}`

## Golden generation (checkpoint_2000)

- [x] A2T golden decode (64/64)
- [x] I2T golden decode (64/64)
- [ ] T2A golden decode (64/64 rows, **56/64 wavs written**; 8 decode errors remain)
- [ ] T2I golden decode (**62/64 generated.png**; 2 decode errors remain)
- [x] `golden_summary.json` written (currently contains only t2i summary from the last run)

### Latest results (as of 2026-02-01)

- A2T: 64/64 JSONL rows written
- I2T: 64/64 JSONL rows written
- T2A: 64/64 JSONL rows written, **56/64 wavs saved** (`outputs/stage4_mmpt_trial/qual/step_2000/step_2000_t2a_audio/`)
- T2I: 64/64 JSONL rows written, **62/64 generated.png** (2 decode failures)
- Golden summary: `outputs/stage4_mmpt_trial/qual/step_2000/golden_summary.json`

**T2A decode failures (8):**
- `wavcaps_as:Y8wF-eZOypiI.wav` (audio token codebook mismatch)
- `wavcaps_as:YG8UqX3V6pSs.wav` (global_id not in any AUDIO range)
- `wavcaps_as:YALxn5-2bVyI.wav` (global_id not in any AUDIO range)
- `audiocaps:91918` (global_id not in any AUDIO range)
- `wavcaps_as:Yn6YuWOt5TVU.wav` (audio token codebook mismatch)
- `wavcaps_as:Y3yJIXnimurU.wav` (audio token codebook mismatch)
- `clotho:hort.wav:1` (audio token codebook mismatch)
- `wavcaps_as:YHGCrtv_03FY.wav` (no audio tokens; decoded_ok false)

**T2I decode failures (2):**
- `llava_pretrain:57174` (global_id not in IMAGE range)
- `llava_pretrain:58572` (global_id not in IMAGE range)

### Training log summary (stage4_mmpt_trial.log)

- Steps: 50 → 2000 (2k-step trial)
- loss_mean: 9.6916 → 5.4324
- a2t_avg: 2.0987 → 1.6209
- i2t_avg: 1.5007 → 0.9615
- t2a_avg: 9.8105 → 5.5810
- t2i_avg: 12.8174 → 4.4173
- emb_norm_mean: 2.0938 → 2.1719
- head_norm_mean: 0.4570 → 1.1641

**Per‑step log fields (each logged step = one optimizer update after grad accumulation):**

- `loss_mean`: token‑weighted loss across all tasks
- `a2t_avg`, `i2t_avg`, `t2a_avg`, `t2i_avg`: per‑task average losses
- `emb_norm_mean/std`: modality embedding row norms (mean/std)
- `head_norm_mean/std`: LM head new‑row norms (mean/std)

### Commands used

**Training (stage4_mmpt_trial):**

```
docker run --rm --gpus all --ipc=host --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /home/mark/shared/lumoOmni:/workspace/lumoOmni \
  -v /home/mark/.cache/huggingface:/root/.cache/huggingface \
  -e HF_HOME=/root/.cache/huggingface \
  -w /workspace/lumoOmni lumo-run47-base bash -lc \
  'source /opt/lumo/venv/bin/activate && \
   python -m stage3_uti.train.stage4_mmpt --config stage3_uti/configs/stage4_mmpt_trial.yaml \
   2>&1 | tee outputs/logs/stage4_mmpt_trial.log'
```

**Golden set generation (checkpoint_2000):**

```
docker run --rm --gpus all --ipc=host --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /home/mark/shared/lumoOmni:/workspace/lumoOmni \
  -v /home/mark/.cache/huggingface:/root/.cache/huggingface \
  -e HF_HOME=/root/.cache/huggingface \
  -w /workspace/lumoOmni lumo-run47-base bash -lc \
  'source /opt/lumo/venv/bin/activate && \
   python -m stage3_uti.pipeline.stage4_golden_generate \
     --token-space-json outputs/stage3_token_lm/token_space.json \
     --checkpoint outputs/stage4_mmpt_trial/checkpoint_2000 \
     --lora outputs/stage4_mmpt_trial/lora_2000 \
     --golden-dir outputs/stage4_mmpt/golden \
     --out-dir outputs/stage4_mmpt_trial/qual/step_2000 \
     --device-map cuda \
     --max-new-text 256 \
     --max-new-audio 1200 \
     --max-new-image 256 \
     --tasks a2t,i2t,t2a,t2i'
```

**Bundle inspect folders (all 4 tasks):**

```
docker run --rm --gpus all --ipc=host --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /home/mark/shared/lumoOmni:/workspace/lumoOmni \
  -v /home/mark/.cache/huggingface:/root/.cache/huggingface \
  -e HF_HOME=/root/.cache/huggingface \
  -w /workspace/lumoOmni lumo-run47-base bash -lc \
  'source /opt/lumo/venv/bin/activate && \
   python -m stage3_uti.pipeline.stage4_bundle_inspect \
     --token-space-json outputs/stage3_token_lm/token_space.json \
     --checkpoint outputs/stage4_mmpt_trial/checkpoint_2000 \
     --golden-dir outputs/stage4_mmpt/golden \
     --qual-dir outputs/stage4_mmpt_trial/qual/step_2000 \
     --out-dir outputs/stage4_mmpt_trial/qual/step_2000/inspect \
     --tasks a2t,i2t,t2a,t2i'
```

## Prompt ablation (T2A/T2I)

- [ ] Teacher‑forced ablation run (correct vs shuffle vs empty)
- [ ] Report saved to `qual/step_2000/prompt_ablation.json`

## Gaps to close

- Resolve T2A/T2I decode failures (8 audio, 2 image) or add constrained decoding
- Regenerate `golden_summary.json` after final T2A/T2I decode pass
- Run prompt‑ablation check
- Confirm 100% T2A/T2I decode success on golden set
