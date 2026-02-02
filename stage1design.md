Below is a detailed, implementation-ready specification for **Stage 1: Alignment pretraining (captioning-style)** for your stack:

* **LLM**: `Qwen3-8B-Base` (frozen)
* **Vision**: `SigLIP` tower (frozen)
* **Connector**: `LayerNorm (optional) + Resampler + Projector MLP` (trainable)
* **Objective**: caption next-token prediction conditioned on injected visual tokens

I am going to be explicit about **data schema**, **prompt format**, **token interleaving**, **loss masking**, **batching**, **hyperparameters**, and **evaluation**, because Stage 1 succeeds or fails on these details.

---

## Status (repo as implemented)

### Done

- **Connector-only training** (resampler + projector, optional vision LN) with LLM + vision tower frozen. Implemented in `qwen3-vlm/src/qwen3_vlm/models/vlm.py` + `qwen3-vlm/src/qwen3_vlm/train/train_utils.py`.
- **Connector-only checkpoints** (no full LLM/vision weights) in `qwen3-vlm/src/qwen3_vlm/utils/checkpointing.py`.
- **Deterministic train/eval split** via `eval_ratio` + `seed` in configs; handled in `qwen3-vlm/src/qwen3_vlm/train/train_utils.py`.
- **Dynamic padding** and truncation at `max_seq_len` in collator; optional pad-to-multiple-of. Implemented in `qwen3-vlm/src/qwen3_vlm/data/collate.py`.
- **LLaVA-Pretrain support** (conversations format) with missing-caption filtering and missing-image skip; in `qwen3-vlm/src/qwen3_vlm/data/llava_pretrain.py`.
- **Input pipeline optimizations** (workers, prefetch, pin memory, persistent workers) and non-blocking transfers in `qwen3-vlm/src/qwen3_vlm/train/train_utils.py`.
- **Attention backend + TF32** controls via config (`attn_implementation`, TF32, matmul precision) in `qwen3-vlm/src/qwen3_vlm/train/train_utils.py` and `qwen3-vlm/src/qwen3_vlm/models/llm_qwen3.py`.
- **Stage 1 configs** updated for LLaVA image layout (`image_root: data/llava_pretrain`) and shorter context (`max_seq_len: 512`).
- **Trial run config** with subset (`max_samples`), eval split, logging, and progress throttling in `qwen3-vlm/configs/stage1_align_trial.yaml`.

### Not done yet

- **Multi-image mixing** in Stage 1 (small fraction of multi-image samples).
- **Sequence packing** of multiple samples into one context.
- **WebDataset tar shards** to reduce filesystem overhead (still using folder layout).
- **Formal evaluation suite** beyond simple eval loss (no metrics like CIDEr/SPICE or qualitative eval harness).
- **Full 558K run** with saved checkpoints and validation reports.
/usr/local/lib/python3.12/dist-packages/torch/cuda/__init__.py:63: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
/usr/local/lib/python3.12/dist-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /opt/pytorch/pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)
`torch_dtype` is deprecated! Use `dtype` instead!

Loading checkpoint shards:   0%|          | 0/5 [00:00<?, ?it/s]
Loading checkpoint shards: 100%|██████████| 5/5 [00:00<00:00, 307.41it/s]
`torch_dtype` is deprecated! Use `dtype` instead!
Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model. `use_fast=True` will be the default behavior in v4.52, even if the model was saved with a slow processor. This will result in minor differences in outputs. You'll still be able to use a slow processor with `use_fast=False`.
/workspace/lumoOmni/qwen3-vlm/src/qwen3_vlm/train/train_utils.py:277: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
  scaler = torch.cuda.amp.GradScaler(enabled=(precision == "fp16" and device.type == "cuda"))
[data] full_len=558128 subset_train=4900 subset_eval=100 full_train=546965
[estimate] avg_step=0.846s full_epoch≈32.12h full_total≈96.35h
[estimate] avg_step=0.822s full_epoch≈31.23h full_total≈93.69h
[progress] epoch=0 step=123/1225 (10%)
[estimate] avg_step=0.815s full_epoch≈30.94h full_total≈92.83h
[estimate] avg_step=0.810s full_epoch≈30.75h full_total≈92.26h
[eval] step=200 loss=3.9716
[progress] epoch=0 step=246/1225 (20%)
[estimate] avg_step=0.809s full_epoch≈30.74h full_total≈92.23h
[estimate] avg_step=0.810s full_epoch≈30.78h full_total≈92.34h
[estimate] avg_step=0.809s full_epoch≈30.71h full_total≈92.13h
[progress] epoch=0 step=369/1225 (30%)
[estimate] avg_step=0.805s full_epoch≈30.57h full_total≈91.70h
[eval] step=400 loss=3.8200
[estimate] avg_step=0.805s full_epoch≈30.59h full_total≈91.76h
[progress] epoch=0 step=492/1225 (40%)
[estimate] avg_step=0.806s full_epoch≈30.63h full_total≈91.88h
[estimate] avg_step=0.807s full_epoch≈30.67h full_total≈92.00h
[estimate] avg_step=0.809s full_epoch≈30.72h full_total≈92.15h
[eval] step=600 loss=3.7701
[progress] epoch=0 step=615/1225 (50%)
[estimate] avg_step=0.809s full_epoch≈30.72h full_total≈92.15h
[estimate] avg_step=0.808s full_epoch≈30.69h full_total≈92.08h
[progress] epoch=0 step=738/1225 (60%)
[estimate] avg_step=0.809s full_epoch≈30.71h full_total≈92.14h
[estimate] avg_step=0.808s full_epoch≈30.70h full_total≈92.09h
[eval] step=800 loss=3.6950
[estimate] avg_step=0.808s full_epoch≈30.67h full_total≈92.02h
[progress] epoch=0 step=861/1225 (70%)
[estimate] avg_step=0.807s full_epoch≈30.65h full_total≈91.95h
[estimate] avg_step=0.807s full_epoch≈30.65h full_total≈91.96h
[progress] epoch=0 step=984/1225 (80%)
[estimate] avg_step=0.807s full_epoch≈30.64h full_total≈91.91h
[eval] step=1000 loss=3.6672
[estimate] avg_step=0.807s full_epoch≈30.65h full_total≈91.94h
[estimate] avg_step=0.806s full_epoch≈30.63h full_total≈91.90h
[progress] epoch=0 step=1107/1225 (90%)
[estimate] avg_step=0.807s full_epoch≈30.64h full_total≈91.91h
[estimate] avg_step=0.807s full_epoch≈30.63h full_total≈91.90h
[eval] step=1200 loss=3.5889
[eval] epoch=0 loss=3.5800
[estimate] avg_step=0.806s full_epoch≈30.62h full_total≈91.85h
[estimate] avg_step=0.806s full_epoch≈30.62h full_total≈91.87h
[progress] epoch=1 step=123/1225 (10%)
[estimate] avg_step=0.806s full_epoch≈30.60h full_total≈91.81h
[estimate] avg_step=0.806s full_epoch≈30.60h full_total≈91.80h
[eval] step=1400 loss=3.5894
[estimate] avg_step=0.805s full_epoch≈30.59h full_total≈91.78h
[progress] epoch=1 step=246/1225 (20%)
[estimate] avg_step=0.805s full_epoch≈30.59h full_total≈91.77h
[estimate] avg_step=0.805s full_epoch≈30.59h full_total≈91.78h
[progress] epoch=1 step=369/1225 (30%)
[estimate] avg_step=0.806s full_epoch≈30.61h full_total≈91.82h
[eval] step=1600 loss=3.5128
[estimate] avg_step=0.806s full_epoch≈30.60h full_total≈91.81h
[estimate] avg_step=0.805s full_epoch≈30.60h full_total≈91.79h
[progress] epoch=1 step=492/1225 (40%)
[estimate] avg_step=0.805s full_epoch≈30.58h full_total≈91.75h
[estimate] avg_step=0.805s full_epoch≈30.57h full_total≈91.71h
[eval] step=1800 loss=3.6253
[progress] epoch=1 step=615/1225 (50%)
[estimate] avg_step=0.805s full_epoch≈30.56h full_total≈91.67h
[estimate] avg_step=0.804s full_epoch≈30.56h full_total≈91.67h
[estimate] avg_step=0.804s full_epoch≈30.54h full_total≈91.63h
[progress] epoch=1 step=738/1225 (60%)
[estimate] avg_step=0.804s full_epoch≈30.54h full_total≈91.62h
[eval] step=2000 loss=3.5452
[estimate] avg_step=0.804s full_epoch≈30.53h full_total≈91.59h
[progress] epoch=1 step=861/1225 (70%)
[estimate] avg_step=0.804s full_epoch≈30.53h full_total≈91.60h
[estimate] avg_step=0.804s full_epoch≈30.54h full_total≈91.61h
[estimate] avg_step=0.804s full_epoch≈30.54h full_total≈91.62h
[eval] step=2200 loss=3.4937
[progress] epoch=1 step=984/1225 (80%)
[estimate] avg_step=0.804s full_epoch≈30.54h full_total≈91.61h
[estimate] avg_step=0.804s full_epoch≈30.53h full_total≈91.59h
[progress] epoch=1 step=1107/1225 (90%)
[estimate] avg_step=0.804s full_epoch≈30.54h full_total≈91.61h
[estimate] avg_step=0.804s full_epoch≈30.54h full_total≈91.61h
[eval] step=2400 loss=3.4694
[eval] epoch=1 loss=3.5132
[estimate] avg_step=0.804s full_epoch≈30.52h full_total≈91.57h
[estimate] avg_step=0.804s full_epoch≈30.52h full_total≈91.57h
[estimate] avg_step=0.803s full_epoch≈30.51h full_total≈91.54h
[progress] epoch=2 step=123/1225 (10%)
[estimate] avg_step=0.803s full_epoch≈30.52h full_total≈91.55h
[eval] step=2600 loss=3.4970
[estimate] avg_step=0.803s full_epoch≈30.51h full_total≈91.53h
[progress] epoch=2 step=246/1225 (20%)
[estimate] avg_step=0.804s full_epoch≈30.52h full_total≈91.56h
[estimate] avg_step=0.803s full_epoch≈30.52h full_total≈91.56h
[estimate] avg_step=0.804s full_epoch≈30.52h full_total≈91.57h
[eval] step=2800 loss=3.4044
[progress] epoch=2 step=369/1225 (30%)
[estimate] avg_step=0.803s full_epoch≈30.52h full_total≈91.55h
[estimate] avg_step=0.803s full_epoch≈30.52h full_total≈91.55h
[progress] epoch=2 step=492/1225 (40%)
[estimate] avg_step=0.803s full_epoch≈30.52h full_total≈91.55h
[estimate] avg_step=0.803s full_epoch≈30.51h full_total≈91.54h
[eval] step=3000 loss=3.4338
[estimate] avg_step=0.803s full_epoch≈30.51h full_total≈91.53h
[progress] epoch=2 step=615/1225 (50%)
[estimate] avg_step=0.803s full_epoch≈30.50h full_total≈91.51h
[estimate] avg_step=0.803s full_epoch≈30.50h full_total≈91.50h
[progress] epoch=2 step=738/1225 (60%)
[estimate] avg_step=0.803s full_epoch≈30.51h full_total≈91.52h
[eval] step=3200 loss=3.4113
[estimate] avg_step=0.803s full_epoch≈30.51h full_total≈91.52h
[estimate] avg_step=0.803s full_epoch≈30.51h full_total≈91.53h
[progress] epoch=2 step=861/1225 (70%)
[estimate] avg_step=0.803s full_epoch≈30.51h full_total≈91.54h
[estimate] avg_step=0.803s full_epoch≈30.51h full_total≈91.54h
[eval] step=3400 loss=3.4288
[progress] epoch=2 step=984/1225 (80%)
[estimate] avg_step=0.803s full_epoch≈30.51h full_total≈91.54h
[estimate] avg_step=0.803s full_epoch≈30.51h full_total≈91.54h
[estimate] avg_step=0.804s full_epoch≈30.52h full_total≈91.56h
[progress] epoch=2 step=1107/1225 (90%)
[estimate] avg_step=0.803s full_epoch≈30.52h full_total≈91.56h
[eval] step=3600 loss=3.4043
[estimate] avg_step=0.803s full_epoch≈30.52h full_total≈91.56h
[eval] epoch=2 loss=3.5460

---

## 1) Stage 1 purpose and what “success” concretely means

### Purpose

Teach the frozen Qwen3 language model to interpret a block of “visual tokens” as meaningful context by learning **only** a small mapping from SigLIP embeddings into Qwen3’s hidden space.

### Success criteria (measurable)

On a held-out set of image-caption pairs:

1. **Caption quality**: captions are grounded and semantically correct (qualitative), and perplexity (or loss) consistently improves during training.
2. **Robustness**: the model answers “What is in the image?” with correct nouns/attributes for diverse samples.
3. **No collapse**: language fluency remains intact (easy sanity: run a small text-only prompt suite and ensure outputs remain coherent—since LLM is frozen, this should hold).

---

## 2) Model components and trainable parameters

### 2.1 Frozen

* **SigLIP vision encoder**
* **Qwen3-8B-Base** (all weights frozen, including embeddings and LN)

### 2.2 Trainable “projector” (recommended definition)

Although you wrote “projector only,” your architecture includes a resampler. For stability and multi-image scaling, treat **the entire connector stack** as “the projector module” for Stage 1:

1. **Vision feature LayerNorm (optional but recommended)**

   * `LN(vision_dim)` on SigLIP patch embeddings before resampling.
   * Helps when mixing images with varying distributions/augmentations.

2. **Resampler (trainable)**

   * Perceiver-style resampler with `N` learned latent queries (e.g., `N=64`) attending over SigLIP patch tokens.
   * Outputs a fixed number of tokens per image.

3. **Projector MLP (trainable)**

   * 2-layer MLP: `vision_token_dim -> hidden -> qwen_hidden`
   * Use GELU activation, and optionally dropout=0.0–0.1.

If you absolutely insist on “projector only” in the strictest sense, you must make the resampler **parameter-free** (e.g., mean-pool + linear). That works, but it is noticeably worse and wastes your multi-image headroom. For your stated end goal (Level 4), I strongly recommend **training the resampler + MLP** together in Stage 1.

### 2.3 Token budget decisions (lock now)

* **Visual tokens per image (`N`)**: start at **64**

  * Multi-image prompts remain feasible.
  * Leaves context headroom for future audio tokens.
* **Image resolution**: match the SigLIP checkpoint (commonly 384).

---

## 3) Input/output format and `<image>` interleaving

Stage 1 is easiest when the sequence structure is *simple and consistent*. Do **not** use full “chat SFT” formatting here—this stage is alignment, not instruction following.

### 3.1 Special tokens and placeholders

Define these special tokens in your tokenizer (or choose an existing convention and stick to it):

* `<image>`: placeholder in the raw text
* `<im_start>`, `<im_end>`: optional bracketing
* `<im_patch>`: “virtual” image token repeated `N` times

**Runtime expansion rule (in collator)**
Each `<image>` in the prompt expands to:

`<im_start> <im_patch> x N <im_end>`

But *you do not* want the LLM to see literal `<im_patch>` embeddings. Instead:

* Tokenize the text including the placeholder token IDs
* Replace the embeddings for the `<im_patch>` positions with the **projected visual token embeddings**

### 3.2 Multi-image mapping rule (even if Stage 1 is mostly single-image)

If your sample contains multiple images:

* The **k-th `<image>` occurrence** maps to `images[k]`.

Even in Stage 1, include a small multi-image fraction so the model learns the mapping and doesn’t develop “last image only” bias later.

---

## 4) Data spec (image-caption pairs)

### 4.1 Required dataset schema (JSONL recommended)

Each record:

```json
{
  "id": "unique_id",
  "images": ["relative/or/absolute/path.jpg"], 
  "caption": "A short description of the image.",
  "source": "dataset_name_optional"
}
```

For multi-image (optional in Stage 1):

```json
{
  "id": "unique_id",
  "images": ["img1.jpg", "img2.jpg"],
  "caption": "Two images: the first shows ..., the second shows ...",
  "source": "synthetic_multi"
}
```

### 4.2 Caption constraints (Stage 1 quality controls)

To keep alignment stable:

* Caption length: **5–60 tokens** preferred
* Avoid long story-like text
* Prefer literal, descriptive captions (“A dog jumping over a log”) over poetic text
* Deduplicate aggressively (near-duplicate captions cause shallow learning)

### 4.3 Train/val split

* `train`: 99% (or 98%)
* `val`: 1–2% with stratification by caption length (short/medium/long buckets)

### 4.4 “Clean and curated” guidance

If LLaVA-Pretrain / similar caption sets are already downloaded, start there. Regardless of source:

* Remove corrupted images (decode failures)
* Remove empty captions
* Remove “watermark / stock photo / download / click” spam lines
* Cap per-domain duplicates if applicable

---

## 5) Prompt template for Stage 1 (captioning-style)

Keep it extremely consistent. Recommended:

**Template A (single image)**

* Input text (user side):

```
<image>
Describe the image in one sentence.
```

* Target text (assistant side):
  `{caption}`

**Template B (multi-image, small fraction)**

```
<image>
<image>
Describe each image in order, one sentence per image.
```

Target:
`Image 1: ... Image 2: ...`

### Why this template

* Forces the model to condition on the image tokens
* Keeps language simple and close to caption distribution
* Does not require a chat-format model; works with base LLMs

### Loss masking rule (critical)

Compute loss **only on the target caption tokens** (assistant side), not on the prompt tokens.

---

## 6) Collation and packing (the part that usually breaks)

### 6.1 Batch collation steps

For each sample:

1. Load image(s)
2. Encode via SigLIP → patch embeddings (`[P, Dv]`)
3. Optional: LN on patch embeddings
4. Resampler → `[N, Dr]` (fixed N tokens per image)
5. Projector MLP → `[N, Dh]` where `Dh = Qwen3 hidden size`
6. Construct tokenized text sequence containing placeholders expanded to `<im_patch>` repeated N times per image
7. Build input embeddings:

   * text token embeddings from Qwen3 embedding table
   * overwrite the `<im_patch>` positions with projected visual embeddings
8. Attention mask is 1 for all positions (prompt + image tokens + caption tokens)
9. Labels:

   * `-100` for everything except caption tokens

### 6.2 Sequence packing

Packing multiple short samples into one long sequence improves throughput but complicates placeholder replacement.

For Stage 1, I recommend:

* Start **without packing** (one sample per sequence, padded)
* After it works and loss curves look healthy, add packing as an optimization

### 6.3 Length limits

* Max sequence length: **1024–2048** for Stage 1 is plenty
* Visual budget per image: `N=64`
* Multi-image in Stage 1: cap at 2 images per sample

---

## 7) Training hyperparameters (DGX Spark friendly)

Stage 1 trains very few parameters; you want stable, fast progress.

### 7.1 Optimizer

* AdamW
* Weight decay: 0.01 (projector) / 0.0 (LayerNorm parameters)

### 7.2 Learning rate

Start here and adjust based on loss curve smoothness:

* Projector MLP + resampler: **1e-4**
* If you include LN: same LR or slightly lower (5e-5)

### 7.3 Schedule

* Warmup: 2–5% of total steps
* Cosine decay to 10% of peak LR

### 7.4 Batch size

Target an effective batch that gives stable gradients:

* Micro-batch: as large as fits (depends on your implementation and resolution)
* Gradient accumulation to reach **effective batch = 256–1024 samples** if possible

### 7.5 Precision

* Use bf16 (preferred) or fp16
* Vision tower forward can be bf16; it’s frozen so numerical noise is less harmful

### 7.6 Regularization

* Dropout: 0.0–0.1 in projector MLP
* Gradient clip: 1.0

### 7.7 Duration

You should see meaningful progress quickly:

* First sanity curve: 10k–50k steps
* Better alignment: 100k–300k steps (depends on data size)

Stop when:

* Val loss plateaus
* Captions look grounded and consistent

---

## 8) Logging and artifacts (what you must record)

Record at least:

* Train/val loss
* Token-level perplexity on caption tokens (optional)
* Sample generations every N steps:

  * fixed 32-image eval set with prompts
  * log the generated captions
* Save checkpoints:

  * Keep only connector weights (resampler + projector + LN), not full LLM

This matters later because Stage 2/3 will reuse the connector checkpoint.

---

## 9) Evaluation protocol (simple, reliable)

### 9.1 Held-out captioning

On the val set:

* Generate captions with:

  * temperature 0.2–0.7
  * top_p 0.9
  * max_new_tokens 64
* Track:

  * qualitative correctness (manual spot checks)
  * caption length distribution
  * “hallucination rate” (obvious wrong objects)

### 9.2 “What is in the image?” QA sanity set

Create a small internal set (200–500 images) with short expected answers:

* “What is in the image?” → “a dog”, “a red car”, “a plate of food”
* “What color is the …?” etc.

Stage 1 is not a reasoning benchmark; it is grounding. Your metric is: does the model see the right objects/attributes.

---

## 10) Deliverables for Stage 1 (what you should produce)

1. `stage1_align.yaml` config file with:

   * model ids (qwen3, siglip)
   * `N` tokens per image
   * LR, schedule, batch, max_len
2. `train_stage1_align.py` script that:

   * loads dataset JSONL
   * runs training
   * saves connector-only checkpoints
3. `eval_stage1.py` script that:

   * generates captions on a fixed set
   * writes a single report artifact (JSONL + markdown summary)
4. A small “golden” image set committed to repo (or referenced) for reproducible eval output

---

## Recommended defaults (if you just want the “best first run”)

* `N=64` visual tokens per image
* Trainable: LN + resampler + MLP projector
* Max length: 1536
* LR: 1e-4
* Warmup: 2%
* Effective batch: 512
* Stage length: 150k–250k steps

---

Optional: provide the exact dataset you downloaded (e.g., LLaVA-Pretrain, your own JSONL, Mantis caption subset) to tailor the **prompt template**, **multi-image fraction**, and **bucketing strategy** to that dataset layout without rewriting the loader.
### 1) Stage 1 dataset: what we use, and whether you already downloaded it

**Stage 1 needs only image–caption pairs** (not instruction conversations). In the plan we laid out earlier, the default Stage 1 dataset is:

* **LLaVA-Pretrain (LCS-558K)** / `blip_laion_cc_sbu_558k`
  This is a turnkey “alignment” set explicitly used to train the *vision→LLM connector* before instruction tuning.

So the answer to “is it in the previous download session?” is:

* **If you downloaded LLaVA-Pretrain / LCS-558K, yes—that is your Stage 1 dataset.**
* If you only downloaded COCO + LLaVA-Instruct or Mantis, then **you have Stage 2/3 datasets**, but **not the Stage 1 caption-alignment set**.

**How to verify locally (quick sanity):**

* Look for **`blip_laion_cc_sbu_558k.json`** (or similarly named JSON) and an **`images.zip`** (or extracted image directory) under your data folder.

If LLaVA-Pretrain is not available, Stage 1 can still run using any clean caption dataset (e.g., COCO captions), but LLaVA-Pretrain is the most direct match to the canonical alignment stage.

---

### 2) How the projector actually “connects” vision into Qwen3

Qwen3 is a text-only causal LLM. The connection is done by feeding Qwen3 **`inputs_embeds`** where some token embeddings are replaced by projected vision embeddings.

#### The dataflow (single image)

1. **Image → SigLIP** (frozen)

   * Output: patch embeddings `V_patches` with shape `[P, Dv]`

2. **(Optional) LayerNorm on patch embeddings** (trainable or fixed; recommended)

   * Output: `[P, Dv]`

3. **Resampler** (trainable)

   * Converts variable-length patches into a fixed token budget
   * Output: `V_latents` with shape `[N, Dr]` (e.g., `N=64`)

4. **Projector MLP** (trainable)

   * Maps `Dr → Dh` where `Dh` is Qwen3 hidden size
   * Output: `V_tokens` with shape `[N, Dh]`

5. **Text prompt tokenization** produces token IDs for something like:

   ```
   <image>
   Describe the image in one sentence.
   ```

   In your **collator**, you expand `<image>` into **N placeholder positions** (e.g., `<im_patch>` repeated N times). Those placeholders exist only to reserve slots in the sequence.

6. **Build `inputs_embeds` for Qwen3**

   * Start with Qwen3’s normal token embeddings for the entire sequence: `E = embed(input_ids)` with shape `[T, Dh]`
   * Replace the embedding rows corresponding to the `<im_patch>` positions with `V_tokens`

7. **Forward pass into Qwen3 (frozen)**

   * `Qwen3(inputs_embeds=E, attention_mask=..., labels=...)`
   * Loss is computed only on caption tokens (labels for prompt + image slots are `-100`)

Because Qwen3 is frozen, **the only gradients flow back through the replaced embedding slots into the projector/resampler**. This is exactly what you want in Stage 1.

#### Multi-image is the same, just repeated

If the prompt contains `<image><image>`:

* You generate `V_tokens_1` and `V_tokens_2` (each `[N, Dh]`)
* You reserve **2×N slots** in the token sequence (in order)
* You replace the first block with `V_tokens_1`, second with `V_tokens_2`

---

### Critical implementation detail: avoid “new special token embeddings” in Stage 1

In Stage 1, Qwen3 is frozen. If you add brand-new tokenizer tokens like `<im_patch>` and expect Qwen3 to learn their embeddings, it cannot—because its embedding matrix is frozen.

The robust pattern is:

* Use placeholders purely as **reserved positions** in the collator (they can map to any existing token ID), and then **overwrite those positions’ embeddings** with vision embeddings in `inputs_embeds`.

In other words: the model does not need to “understand” `<im_patch>` as a token; it just needs consistent slots to receive the projected vision vectors.

---

### Inputs to lock the Stage 1 spec to the downloaded data

To confirm whether your Stage 1 dataset is already present without guessing, tell me which of these you downloaded:

* `liuhaotian/LLaVA-Pretrain` (LCS-558K)
* COCO images + LLaVA-Instruct-150K
* Mantis-Instruct

If you reply “I have LLaVA-Pretrain extracted at X,” I will provide the exact Stage 1 loader contract (paths, expected filenames, and a concrete batch collation algorithm with the placeholder index mapping and label masking).
