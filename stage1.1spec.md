Stage 1 P0 results show strong ablation separation (correct < shuffled < zero < noise), which rules out common placeholder/masking/injection failures. Truncation and label coverage are clean at `max_seq_len=512` for the current prompt format.

Stage 1.1 focuses on hardening the evaluation signal, adding a minimal qualitative harness, and running a small sweep to pick stable hyperparameters before a full 558K run.

This document is an implementation-ready **Stage 1.1 spec**, including a scaling plan that keeps **one epoch within ~1 hour** on DGX Spark. It also includes implementation notes and results from the first completed Stage 1.1 run.

---

## Stage 1.1 — Alignment P1 Hardening Spec

### Status (implemented + run)

Implemented in repo and executed a 4-epoch scaled P1 run:

* Config used: `qwen3-vlm/configs/stage1_align_p11.yaml`
* Output dir: `qwen3-vlm/outputs/stage1_align_p11/`
* Stratified split + bucketed eval metrics implemented (`split_mode: stratified_len`)
* Token-weighted eval loss with per-bucket loss reporting
* Golden set creation + checkpoint-time caption dumps (`run_golden_every: "save"`)
* Progress output matches Stage 1 style (tqdm disabled, custom 10% progress logs)

### Stage 1.1 P1 run results (4 epochs)

Run summary:

* Data: `max_samples=18000` -> `subset_train=16000`, `subset_eval=2000`
* Eval buckets: `short=666`, `medium=666`, `long=668`
* Global steps: 16,000 (4 epochs x 4,000 steps)
* Avg step time: 0.798s
* Eval (token-weighted, full 2,000):
  * `eval_loss=2.7808`
  * `eval_loss_short=2.7306`, `eval_loss_medium=2.7561`, `eval_loss_long=2.8146`
  * `eval_truncated_pct=0.00%`

Artifacts:

* Split metadata: `qwen3-vlm/outputs/stage1_align_p11/splits/split_seed42_eval2000.json`
* Metrics: `qwen3-vlm/outputs/stage1_align_p11/metrics.json`
* Golden set: `qwen3-vlm/data/golden/stage1_llava_pretrain_golden64.jsonl`
* Golden dumps: `qwen3-vlm/outputs/stage1_align_p11/qual/step_1000_golden64.jsonl` through `step_16000_golden64.jsonl`

### Sweep summary (trial-size train, eval=2000)

Sweep settings:

* Train subset: `max_samples=5000` -> `subset_train=3000`, `subset_eval=2000`
* Steps: `train_steps=2000` (eval at step 1000)
* Ablations: `num_samples=256`

Summary table (eval loss from step 1000 full eval; ablations from `eval_ablation.py`):

```
LR       | vision_ln | eval_loss_end | Δshuffle | Δzero
5e-5     | False     | 3.6957        | 0.8309   | 1.8127
5e-5     | True      | 3.7513        | 0.6908   | 1.7659
1e-4     | False     | 3.7193        | 0.7759   | 1.7908
1e-4     | True      | 3.6753        | 0.7978   | 1.7479
2e-4     | False     | 3.9106        | 0.0084   | 1.4253
2e-4     | True      | 3.7383        | 0.6472   | 1.7506
```

Raw sweep artifact list:

* `qwen3-vlm/outputs/stage1_align_p11_sweep_trial_summary.csv`
* Per-run logs:
  * `qwen3-vlm/outputs/stage1_align_p11_sweep_trial_lr5e-5_ln0/train.log`
  * `qwen3-vlm/outputs/stage1_align_p11_sweep_trial_lr5e-5_ln1/train.log`
  * `qwen3-vlm/outputs/stage1_align_p11_sweep_trial_lr1e-4_ln0/train.log`
  * `qwen3-vlm/outputs/stage1_align_p11_sweep_trial_lr1e-4_ln1/train.log`
  * `qwen3-vlm/outputs/stage1_align_p11_sweep_trial_lr2e-4_ln0/train.log`
  * `qwen3-vlm/outputs/stage1_align_p11_sweep_trial_lr2e-4_ln1/train.log`

### Requested outputs (epoch 2 final checkpoint)

Eval loss (epoch 2, token-weighted full 2,000):

* `eval_loss=2.7972`
* `eval_loss_short=2.7595`, `eval_loss_medium=2.7771`, `eval_loss_long=2.8229`

Golden64 sample output (step 12000, 10 items):

```
id=156989 | gt=the kardah sisters | pred=the cast of the new season of the real housewives of new york
id=484063 | gt=a child hands that are covered in pink clay | pred=A girl is making a pink slime with her hands
id=319751 | gt=johnnie green label | pred=A bottle of green label scotch whisky
id=16800 | gt=a blue and white feather tree pillow | pred=A pair of blue and white feather pillows
id=51926 | gt=a tablet computer with connected social media circles over it | pred=A mobile phone with a network icon
id=540159 | gt=memphis nylon gloves, large, gray / white | pred=a pair of men's cut resistant gloves with a cut resistant cut
id=282548 | gt=an old fashioned coin machine in an office setting | pred=A vintage chicken game machine with a red and white background
id=437603 | gt=a set of three printable train bathroom art | pred=A set of three train themed nursery wall art prints
id=433498 | gt=the valley whisperer logo with red fish postcards | pred=A woman with a fishing rod and a fish in her mouth postcard
id=421403 | gt=the jordan backpack with its zipper closure | pred=a black nike backpack with a red and white logo
```

### Implementation map (what changed in repo)

Core code changes (Stage 1.1 features):

* `qwen3-vlm/src/qwen3_vlm/data/splits.py`:
  * stratified split builder
  * persisted split metadata
  * eval bucket counts
* `qwen3-vlm/src/qwen3_vlm/data/llava_pretrain.py`:
  * `get_metadata()` for split + golden selection
* `qwen3-vlm/src/qwen3_vlm/data/collate.py`:
  * pass through `bucket`, `label_len`, `id`, `image_relpath`
* `qwen3-vlm/src/qwen3_vlm/train/train_utils.py`:
  * token-weighted eval loss + per-bucket losses
  * quick eval support
  * golden generation hooks
  * `metrics.json` output
  * `train_steps` support for sweeps
* `qwen3-vlm/src/qwen3_vlm/utils/golden.py`:
  * golden set creation
  * generation routine
  * autocast for bf16
  * fixed `inputs_embeds` generation slicing (see 3.6)

Config + scripts:

* `qwen3-vlm/configs/stage1_align_p11.yaml` (scaled P1 run)
* `qwen3-vlm/configs/stage1_align_p11_sweep.yaml` (sweep template)
* `qwen3-vlm/scripts/train_stage1_p11.sh`
* `qwen3-vlm/scripts/train_stage1_p11_sweep.sh`

### Run configuration (Stage 1.1 P1)

Key settings used for the 4-epoch run:

* Data:
  * `max_samples=18000`, `eval_size=2000`, `split_mode=stratified_len`
  * `len_buckets=[0,10,20,9999]`
  * golden set: `golden_size=64`, `golden_seed=123`
* Train:
  * `num_epochs=4`, `batch_size=4`, `gradient_accum=4` (effective batch 16)
  * `lr=1e-4`, `weight_decay=0.0`
  * `precision=bf16`
  * `save_every=1000`, `full_eval_every="epoch"`
  * `quick_eval_size=200`, `quick_eval_steps=200`
  * `tqdm_disable=true`, `progress_percent=0.1`
* Model:
  * `resampler.num_latents=64`
  * `vision_ln=false`
  * frozen LLM + frozen vision

Estimated wall time:

* avg step `0.798s` x `4000` steps/epoch ≈ **53.2 min/epoch**

### Run log highlights (for traceability)

* epoch 2 eval (token-weighted, full 2,000):
  * `[eval] epoch=2 loss=2.7972 ... short=2.7595 medium=2.7771 long=2.8229`
* epoch 3 eval (token-weighted, full 2,000):
  * `[eval] epoch=3 loss=2.7808 ... short=2.7306 medium=2.7561 long=2.8146`
* train data sanity:
  * `[data] epoch=3 truncated=0.00% label_zero=0.00% avg_label_tokens=12.6`

### Observed issues + fixes (Stage 1.1 run)

* Golden generation dtype mismatch (bf16 vs fp32) during early runs:
  * fixed by running golden generation under autocast (bf16) in `qwen3-vlm/src/qwen3_vlm/utils/golden.py`
* `generator` kwarg rejected by `transformers.generate`:
  * removed generator kwarg; seeded via `torch.manual_seed` instead
* Empty `pred_text` in golden dumps:
  * root cause: `generate()` with `inputs_embeds` returns only new tokens
  * fix: avoid slicing by `prompt_len` when output length is shorter than prompt
  * regenerated golden dump for epoch 2 from `checkpoint_12000.pt`:
    `qwen3-vlm/outputs/stage1_align_p11/qual/step_12000_golden64.jsonl`
* Sweep trial output_dir collision:
  * per-run configs did not override `train.output_dir`, so checkpoints/metrics were overwritten in
    `qwen3-vlm/outputs/stage1_align_p11_sweep`
  * eval losses were recovered from per-run `train.log` files; ablation logs are per-run

### Objectives (P1)

A) Increase eval set size and stratify by caption length
B) Add qualitative “golden set” caption dump at checkpoints
C) Run a small hyperparameter sweep (LR × Vision LN) using stable eval + ablation scoring
D) Scale trial data so **epoch wall time ≤ 1 hour** (train + one full eval + golden dump)

### Non-goals (explicitly out of scope for Stage 1.1)

* WebDataset tar shards
* Sequence packing
* CIDEr/SPICE/CLIPScore (optional later; not required now)

---

## 1) Dataset split: stratified eval by caption length

### 1.1 Definition of “caption length” for LLaVA-Pretrain

Because LLaVA-Pretrain is stored as “conversation” samples, define caption length as:

* `label_len = token_count(assistant_text)`
  where `assistant_text` is the **model/assistant turn** used as the supervised target.

Do **not** measure the full prompt length; we only care about target token length distribution in eval.

### 1.2 Buckets (simple and robust)

Use three buckets based on `label_len` (tunable, but start here):

* **short**: `label_len <= 10`
* **medium**: `11 <= label_len <= 20`
* **long**: `label_len >= 21`

Rationale: your observed `avg_label_tokens ~13.1` indicates a heavy short/medium skew. These buckets will stabilize eval variance and help diagnose whether improvements come only from “easy short targets.”

### 1.3 Split policy

* Total eval size: **2,000 samples**
* Eval composition: **balanced across buckets** (default: 33/33/34; if a bucket is underrepresented, fill remaining slots from the next bucket).
* Train: everything else in the subset (or full dataset, later).

### 1.4 Determinism and reproducibility

Persist split indices to disk so runs are comparable across sweeps:

* `outputs/<run>/splits/split_seed{seed}_eval2000.json`
* Include `id`, `bucket`, `label_len`, `image_path` for auditing.

### 1.5 Required implementation changes

Add a split builder that runs once before training:

* Input: dataset metadata + tokenizer
* Output: `train_indices`, `eval_indices`, and per-bucket counts

Where to implement (suggested):

* `qwen3_vlm/data/splits.py` (new)
* Called from `train_utils.py` when `split_mode: stratified_len` is enabled

### 1.6 Eval reporting

In addition to overall eval loss:

* report **loss per bucket**: `eval_loss_short`, `eval_loss_medium`, `eval_loss_long`
* report bucket counts to confirm stable evaluation

---

## 2) Eval loss computation: ensure it is “full dataloader averaged”

This must be true (and should be enforced via tests/logging):

### 2.1 Correct averaging (token-weighted)

During eval:

* accumulate `sum_loss_tokens = Σ(loss_per_token * num_label_tokens)`
* accumulate `sum_label_tokens = Σ(num_label_tokens)`
* report `eval_loss = sum_loss_tokens / sum_label_tokens`

Do not average per-batch losses equally (caption lengths vary and will bias the metric).

### 2.2 Additional eval stats to log

* `eval_label_tokens_total`
* `eval_examples_total`
* `eval_truncated_pct` (should remain 0% in Stage 1.1, but log anyway)

---

## 3) Qualitative harness: “golden set” caption dump

### 3.1 Golden set selection

Create a fixed set of **64 samples** (saved once, reused forever):

* 24 short, 20 medium, 20 long (or evenly split 21/21/22)
* Deterministic sampling with a separate `golden_seed`
* Stored at: `data/golden/stage1_llava_pretrain_golden64.jsonl`

Each line contains:

* `id`
* `image_relpath`
* `gt_text` (assistant target text)
* `bucket`
* `label_len`

### 3.2 Generation prompt template

Because Stage 1 training is “captioning-like,” keep generation prompt minimal and consistent:

Option A (recommended, stable):

* Use the same conversation wrapper you train on, but at inference time stop before the assistant target.
* Generate the assistant response.

Option B (if you want explicit captioning):

```
<image>
Describe the image in one sentence.
```

Pick one and keep it constant across checkpoints.

### 3.3 Generation parameters (for comparability)

* `temperature = 0.2`
* `top_p = 0.9`
* `max_new_tokens = 64`
* `repetition_penalty = 1.0` (unless you see looping)

### 3.4 Artifact format

At each `save_steps` (or at least at each epoch end), write:

`outputs/<run>/qual/step_<global_step>_golden64.jsonl`

Each record:

* `id, bucket, gt_text, pred_text, image_relpath, label_len`
  Optionally include:
* `pred_tokens_len`
* `hash(pred_text)` for quick diffing

This is low-effort and catches failure modes that loss cannot (generic captions, template collapse, image ignoring).

### 3.5 Implementation hooks

Add a lightweight callback in `train_utils.py`:

* After saving connector checkpoint, run golden generation on 64 items with `torch.no_grad()`
* Keep it on GPU; batch it (e.g., batch=8) to finish quickly

### 3.6 Generation nuance (inputs_embeds) + fix

`model.generate()` with `inputs_embeds` returns **only newly generated tokens** (no prompt prefix).
Initially, we were slicing generated output by `prompt_len`, which produced empty `pred_text`.

Fix applied:

* if generated sequence length <= prompt length, treat the entire sequence as prediction.
* see: `qwen3-vlm/src/qwen3_vlm/utils/golden.py`

After the fix, golden dumps contain non-empty predictions (e.g., step 12000).

---

## 4) Hyperparameter sweep spec (cheap, decisive)

### 4.1 Sweep matrix

Hold fixed:

* `N_visual_tokens = 64`
* frozen LLM + frozen vision tower
* same stratified eval set (2000)
* same golden set (64)

Sweep:

* LR ∈ {`5e-5`, `1e-4`, `2e-4`}
* vision LN ∈ {`off`, `on`}

Total: **6 runs**

### 4.2 Run budget

Because you want this cheap and comparable:

* `train_steps = 2,000` (or `3,000` if extra time is acceptable)
* Evaluate:

  * quick eval (optional): 200 examples every 200 steps (for curve shape)
  * full stratified eval (2,000): at step 1,000 and at end
* Golden dump: end only (or end + mid if you want)

### 4.3 Scoring (choose winner objectively)

For each run, compute:

1. `full_eval_loss_end` (token-weighted)
2. Ablation deltas on **256 samples**:

   * `Δshuffle = loss(shuffled) - loss(correct)`
   * `Δzero = loss(zero) - loss(correct)`

Define a simple score:

* `score = full_eval_loss_end - 0.1*(Δshuffle + Δzero)`

Interpretation:

* lower loss is better
* larger deltas indicate the model is using vision more strongly (but do not overweight it)

Also sanity-check the golden generations for “generic caption collapse.”

### 4.4 Output artifacts

For each run:

* `metrics.json` (eval loss per bucket, global eval loss, ablation deltas, step time stats)
* `golden64.jsonl` at end
* connector checkpoint at end

---

## 5) Scaling plan: keep one epoch within ~1 hour

Observed: **4,900 train + 100 eval ≈ 15 minutes/epoch**. That implies training samples can scale roughly 4× and still remain within an hour, but Stage 1.1 adds a heavier eval (2,000) and a golden dump.

### 5.1 Target subset sizing for Stage 1.1 “1-hour epoch”

I recommend:

* `subset_eval = 2,000` (stratified)
* `subset_train ≈ 16,000`
* Total `max_samples = 18,000`
* `eval_ratio ≈ 0.1111` (only used if you still use ratio; stratified split overrides ratio for eval sizing)

This should land close to:

* Train time ~ 4× baseline ≈ 60 minutes *minus* some efficiency gains from better loader settings
* Plus full eval overhead (2,000 items) and golden dump; you should still be near the 1-hour target if you do **full eval once per epoch**, not every 200 steps.

### 5.2 Eval frequency policy (important)

To keep time bounded:

* **Full eval (2,000)**: end of epoch only (or every 2,000 steps if you run step-based)
* Optional **quick eval (200)**: every 200 steps for trend signal

This gives stable metrics without blowing up wall time.

---

## 6) Concrete config changes (what to add to YAML)

Create: `configs/stage1_align_p11.yaml` (or similar)

Key additions:

* `split_mode: stratified_len`
* `eval_size: 2000`
* `len_buckets: [0,10,20,9999]` (or explicit thresholds)
* `golden_set_path: data/golden/stage1_llava_pretrain_golden64.jsonl`
* `run_golden_every: "save"` (or `"epoch"`)
* `full_eval_every: "epoch"` and `quick_eval_size: 200` + `quick_eval_steps: 200` (optional)
* `max_samples: 18000` (for 1-hour epoch target)
* `train_steps`: only for sweep runs; leave unset for epoch-based runs
* `tqdm_disable: true` + `progress_percent: 0.1` to match Stage 1 logging style

---

## 7) Next steps checklist (in order)

### Step 1 — Implement stratified split + bucketed eval metrics

* Done: split builder added, indices persisted, bucket counts logged.
* Done: eval loss is token-weighted over the full eval loader.

### Step 2 — Add golden set generator + checkpoint-time caption dump

* Done: `data/golden/stage1_llava_pretrain_golden64.jsonl` created.
* Done: generation runs at each checkpoint save.

### Step 3 — Run the 6-run sweep (2,000 steps each)

* Pending: fixed subset + fixed stratified eval.
* Pending: pick winner by score + qualitative sanity.

### Step 4 — Run Stage 1.1 “scaled epoch” (16k train / 2k eval) with winner config

* Done: 4-epoch scaled run completed (see results above).
* Pending: ablation deltas for final checkpoint.

After this, you are ready to promote to **Stage 2** with high confidence.

---

## What I would want to see from you (no more than a small snippet)

After you implement Stage 1.1, share:

1. One run’s `eval_loss_{short,medium,long}` + overall `eval_loss`
2. Golden64 sample output for ~10 items (id + gt + pred)
3. Sweep summary table (LR, vision_ln, eval_loss_end, Δshuffle, Δzero)

Use these inputs to determine which sweep setting to promote and whether any data-format or prompt-template issue is still limiting learning signal before Stage 2.
