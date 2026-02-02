Understood. Based on your P0/P1 and the sweep table, you are in a good position to promote to Stage 2 now. The sweep also surfaced one important “do-not-use” region (LR=2e-4 with `vision_ln=False` effectively killed the shuffle sensitivity), which is exactly the kind of outcome you want the sweep to catch before you introduce LoRA.

Below is a **full Stage 2 specification** (single-image visual instruction tuning) tailored to your repo as implemented.

---

# Stage 2 Spec — Single-image Visual Instruction Tuning (SFT)

## 0) Stage 2 goals and success gates

### Primary goal

Transform the aligned connector + frozen vision tower into a **chat VLM** by training **LoRA on the Qwen3 LLM** (and continuing to train the connector) on **instruction-following multimodal conversations** (e.g., LLaVA-Instruct-150K).

### What changes vs Stage 1

* Stage 1 taught the LLM “how to interpret visual tokens.”
* Stage 2 teaches the system “how to behave as an assistant” (instruction following, conversation formatting, short reasoning chains) while staying grounded in the image.

### Stage 2 success gates (concrete)

You should require all of the below before moving to Stage 3 (multi-image SFT):

1. **Stable train + eval curves** (no divergence; eval loss improving then plateauing).
2. **Golden instruction set** outputs become:

   * grounded (mentions correct objects),
   * instruction-following (answers the asked question rather than generic captioning),
   * non-degenerate (no looping phrases).
3. **Image reliance remains**:

   * `loss(correct image) < loss(zero/noise)` on ablation,
   * and ideally `loss(correct) < loss(shuffled)` (shuffle still meaningful).
4. **No truncation failure**:

   * `label_zero == 0%`,
   * and truncation stays low; if truncation rises, you increase `max_seq_len` rather than silently training on partial labels.

---

# 1) Data: LLaVA-Instruct-150K (single-image)

## 1.1 Dataset requirements (filtering rules)

You want Stage 2 to be *strictly single-image*, because Stage 3 will be your multi-image specialization.

For each sample:

* Must resolve **exactly one image** on disk.
* The user content must include **exactly one `<image>` placeholder**, or you must normalize it into that format:

  * If missing `<image>`, insert it at the start of the first user turn (or prepend a one-token `<image>` line).
  * If more than one `<image>`, skip in Stage 2.

Other hygiene:

* Skip samples whose image file is missing/corrupt.
* Skip samples with empty assistant response.
* Optional: cap assistant length to reduce extreme outliers (e.g., drop >512 assistant tokens in Stage 2; you don’t need them yet).

## 1.2 Train/eval split and stratification

Use your existing stratified machinery, but update what you stratify on:

* Stage 2 stratify on **assistant target length** (label tokens) again (same concept as Stage 1, but lengths will be larger).

Recommended buckets for Stage 2 (more spread than Stage 1):

* short: `label_len <= 64`
* medium: `65–128`
* long: `>=129`

Eval sizing:

* `eval_size = 2000` is good and consistent with your Stage 1.1 practice.

Persist split metadata exactly like Stage 1.1 (you already do this).

---

# 2) Prompting and conversation formatting

Stage 2 lives or dies on consistent formatting. Do not invent a new schema; normalize the dataset into a consistent internal message list.

## 2.1 Internal canonical representation

Represent each sample as:

```json
{
  "id": "...",
  "image": "relative/path.jpg",
  "messages": [
    {"role": "user", "content": "<image>\n<user text>"},
    {"role": "assistant", "content": "<assistant answer>"},
    ...
  ]
}
```

Even if the dataset is 2-turn only, keep your pipeline multi-turn-capable now.

## 2.2 Qwen3 chat template

Use `tokenizer.apply_chat_template(...)` (your repo likely already has a wrapper). The core rule:

* **Labels only for assistant spans**, everything else `-100`:

  * system prompt tokens: `-100`
  * user tokens: `-100`
  * `<image>` token(s) and expanded visual token block: `-100`
  * assistant tokens: supervised

## 2.3 `<image>` expansion rule (keep identical to Stage 1)

Your stage1 connector learned a specific injection pattern. Do not change it in Stage 2.

* Text contains `<image>`
* Collator expands `<image>` → `<im_start> <im_patch>*N <im_end>`
* Embeddings for `<im_patch>` positions are overwritten by projected visual tokens

**N must match your connector checkpoint** (you have `num_latents=64`, so keep `N=64`).

---

# 3) Model: what is frozen vs trainable

## 3.1 Frozen

* **SigLIP vision tower** (still frozen)
* Qwen3 base weights **except LoRA adapters**

## 3.2 Trainable

* **Connector stack**: resampler + projector (+ vision LN if present in the checkpoint architecture)
* **LoRA adapters** on the Qwen3 LLM

### LoRA target modules (recommended)

Start with the standard set that almost always works on Qwen-family HF implementations:

* Attention: `q_proj`, `k_proj`, `v_proj`, `o_proj`
* MLP: `gate_proj`, `up_proj`, `down_proj`

If any names don’t match exactly in Qwen3’s HF module tree, adapt by inspecting `model.named_modules()` and mapping to the correct projection layers (keep the concept identical).

### LoRA hyperparameters (baseline)

* `r = 16`
* `alpha = 32`
* `dropout = 0.05`
* `bias = "none"`

This is conservative and stable for first SFT runs.

---

# 4) Training objective

Standard LM cross-entropy over assistant tokens:

* Input: prompt (system+user+image tokens) + assistant tokens
* Labels: `-100` for non-assistant tokens; assistant tokens are target ids
* Loss: token-weighted mean over label tokens (you already implement token-weighted eval—keep that)

---

# 5) Training configuration (detailed)

## 5.1 Sequence length

Stage 2 needs more context than Stage 1. Use:

* `max_seq_len = 2048` as the default.

If you later see truncation >0, increase to 3072 or 4096 (but do not silently accept truncation).

## 5.2 Precision and memory behaviors

* Use `bf16` (consistent with your Stage 1.1).
* Turn on `gradient_checkpointing = true` for the LLM.
* Set `use_cache = false` during training.

## 5.3 Optimizer groups (important)

Use **separate parameter groups** for:

1. connector params (resampler/projector[/LN])
2. LoRA params

Recommended starting LRs:

* Connector LR: `1e-4`
* LoRA LR: `5e-5` (or `1e-4` if learning is too slow)

Weight decay:

* Connector: `0.01` (or keep `0.0` if you prefer—both work; 0.01 can help generalization)
* LoRA: `0.0`

Gradient clipping:

* `1.0`

## 5.4 LR schedule

* Warmup ratio: `0.03`
* Scheduler: cosine decay

## 5.5 Batch sizing

Do not hardcode a “correct” batch size; Stage 2 is length-heavy. Instead define:

* `micro_batch_size` (per step)
* `gradient_accum`

Target a reasonably stable effective batch (e.g., 32–128 examples), but choose values that keep memory stable and avoid OOM.

You already have the throughput tooling; start small and scale.

## 5.6 Checkpoint frequency

* Save every `500–1000` steps early (until you trust stability), then relax.
* Save artifacts:

  * connector-only checkpoint (as you already do)
  * LoRA adapter weights (new)
  * run config + split metadata + metrics.json (you already have most of this)

---

# 6) Evaluation suite for Stage 2 (minimum viable, high signal)

You already have the scaffolding: stratified eval + golden dumps. Extend it with Stage 2 semantics.

## 6.1 Token-weighted eval loss (full eval=2000)

Keep what you built:

* `eval_loss` token-weighted
* `eval_loss_short/medium/long` (buckets defined for Stage 2)
* `eval_truncated_pct`
* `label_zero_pct`

## 6.2 Golden set for instruction-following (64 items)

Create `data/golden/stage2_llava_instruct_golden64.jsonl` with:

* balanced buckets by assistant length
* include a mix of:

  * “describe image”
  * “count”
  * “color”
  * “spatial”
  * “OCR-ish” (text in image)
  * “reasoning” (simple why/how)

Generation settings for stability (recommended for golden regression):

* greedy decoding (`temperature=0`)
* `max_new_tokens=128` (instruction answers can be longer than captions)

Store:

* `id, image, prompt_text, gt_text, pred_text, label_len, bucket`

## 6.3 Ablation (re-check image reliance under SFT)

Run teacher-forced ablation on a held-out set (256–512 samples), same as Stage 1:

* correct
* shuffled
* zero
* noise

This ensures LoRA didn’t teach the model to “answer from priors” and ignore vision.

---

# 7) Stage 2 implementation tasks in your repo

You likely need only a few concrete additions.

## 7.1 Add a LLaVA-Instruct dataset loader

New file (or extend existing):

* `qwen3_vlm/data/llava_instruct.py`

Responsibilities:

* load json
* resolve image filename → `image_root` path
* normalize messages (human/gpt → user/assistant)
* enforce exactly one `<image>` in the user prompt
* compute label_len metadata (assistant tokens length)

## 7.2 Add LoRA wiring and saving

Where:

* `qwen3_vlm/models/llm_qwen3.py` or `qwen3_vlm/models/vlm.py`

Add:

* `apply_lora(model, lora_config)` using PEFT (recommended) or your own adapters
* mark only LoRA params trainable in the LLM

Checkpointing:

* extend `utils/checkpointing.py` to save:

  * connector weights (existing)
  * LoRA adapter weights (new, ideally separate file like `lora.safetensors` or `lora.pt`)
  * metadata: base model name, lora config hash, connector config

## 7.3 Training loop updates

In `train_utils.py`:

* create optimizer param groups (connector vs LoRA)
* ensure grad scaler logic supports bf16 (it typically does; scaler is more for fp16)
* keep your token-weighted eval

---

# 8) Stage 2 YAML spec (drop-in style)

Concrete config spec to implement as `configs/stage2_sft_single.yaml`.

```yaml
run_name: stage2_sft_single
seed: 42
output_dir: outputs/stage2_sft_single

model:
  llm:
    name_or_path: Qwen/Qwen3-8B-Base
    attn_implementation: flash_attention_2   # or your preferred backend
    precision: bf16
    gradient_checkpointing: true
    use_cache: false

  vision:
    name_or_path: google/siglip-so400m-patch14-384
    freeze: true

  connector:
    load_checkpoint: outputs/stage1_align_p11/checkpoint_16000.pt
    trainable: true
    num_visual_tokens: 64
    vision_ln: false   # MUST match checkpoint architecture
    resampler:
      num_latents: 64

  lora:
    enable: true
    r: 16
    alpha: 32
    dropout: 0.05
    bias: none
    target_modules:
      - q_proj
      - k_proj
      - v_proj
      - o_proj
      - gate_proj
      - up_proj
      - down_proj

data:
  dataset: llava_instruct
  json_path: data/llava_instruct/llava_instruct_150k.json
  image_root: data/coco/train2017
  split_mode: stratified_len
  eval_size: 2000
  len_buckets: [0, 64, 128, 999999]
  max_samples: null
  num_workers: 8
  prefetch_factor: 2
  pin_memory: true
  persistent_workers: true

prompting:
  image_token: "<image>"
  max_images_per_sample: 1
  max_seq_len: 2048

train:
  num_epochs: 1
  train_steps: null
  micro_batch_size: 1
  gradient_accum: 16
  precision: bf16
  grad_clip: 1.0

  lr:
    connector: 1.0e-4
    lora: 5.0e-5

  weight_decay:
    connector: 0.01
    lora: 0.0

  warmup_ratio: 0.03
  lr_scheduler: cosine

  save_every: 1000
  log_every: 20
  full_eval_every: "epoch"
  quick_eval_size: 200
  quick_eval_steps: 200

eval:
  golden_set_path: data/golden/stage2_llava_instruct_golden64.jsonl
  run_golden_every: "save"
  golden_batch_size: 8
  golden_generate:
    temperature: 0.0
    max_new_tokens: 128

ablation:
  run_every: "epoch"
  num_samples: 256
```

Notes:

* **`vision_ln` must match the Stage 1 checkpoint** you load. Your best long run used `vision_ln=false`; stick to it for Stage 2 unless you retrain Stage 1 with LN enabled.
* If your sweep later convinces you to use `vision_ln=true`, do it by producing a new Stage 1 checkpoint with LN enabled and then promote that.

---

# 9) Stage 2 runbook (no ambiguity)

## Step 1: Smoke test (fast)

Run Stage 2 with:

* `max_samples=2000`
* `eval_size=200`
* `train_steps=200`
* `save_every=200`

Verify:

* loss decreases
* no NaNs
* golden dump produces non-empty assistant outputs
* ablation deltas are non-trivial

## Step 2: Trial run (decision-quality)

* `max_samples=20000`
* `eval_size=2000`
* `num_epochs=1` or `train_steps=2000–5000` (your preference)

Pick:

* best LR pair (connector vs LoRA) if needed
* ensure truncation is low; if truncation appears, increase `max_seq_len`

## Step 3: Full Stage 2 run

* Full dataset
* Save checkpoints periodically
* Promote the best checkpoint based on:

  * full eval loss trend
  * golden improvements
  * ablation deltas maintained

---

# 10) Immediate decisions (I’m choosing defaults for you)

Given your current results and the fact you already have a strong Stage 1.1 checkpoint:

* **Use** `outputs/stage1_align_p11/checkpoint_16000.pt` as the Stage 2 connector init.
* **Keep** `vision_ln=false` in Stage 2 to match that checkpoint.
* **Avoid** LR=2e-4 for connector-only regions (your sweep shows it can destroy shuffle sensitivity).
* Start Stage 2 with:

  * connector LR `1e-4`
  * LoRA LR `5e-5`
  * `max_seq_len=2048`
  * greedy golden generations

---

Optional: provide the current `configs/` structure used for Stage 2 (even just the keys) to map this spec to the exact field names your training script expects.
