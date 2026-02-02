Below is a concrete Level-1 plan for **Qwen3-8B-Base + SigLIP + (Perceiver) resampler + projector + `<image>` interleaving**, with (1) a sample repo blueprint, (2) practical data acquisition paths, and (3) the key design choices (including why they matter for Level-4 later).

---

## 1) The “reference repos” you should copy patterns from (not necessarily fork)

These are the most useful public baselines for *how* people structure the data + training stages:

* **LLaVA**: canonical two-stage recipe (feature alignment → visual instruction tuning). ([GitHub][1])
* **LLaVA-Pretrain (LCS-558K)** dataset: packaged captions + an `images.zip` for reproducing the alignment stage. ([Hugging Face][2])
* **LLaVA-Instruct-150K** dataset: instruction tuning format with `<image>` embedded in the conversation. ([Hugging Face][3])
* **Mantis / Mantis-Instruct**: best-in-class *multi-image interleaved* instruction data (721K) and practical download scripts. ([Hugging Face][4])
* **TinyLLaVA**: “small-but-complete” engineering patterns for data prep, organization, prompt templates, and multi-image prompt conventions. ([Hugging Face][5])

You will implement your own model (Qwen3 backbone), but you should reuse these patterns for dataset plumbing and training staging.

---

## 2) Sample repo layout (what I would actually build)

You want a repo that (a) trains Level-1 cleanly and (b) doesn’t block Level-4 later. The most important long-term choice is: **make “modalities” a first-class abstraction now** (image today; audio later).

```text
qwen3-vlm/
  README.md
  pyproject.toml
  requirements.txt
  configs/
    stage0_sanity.yaml
    stage1_align.yaml
    stage2_sft_single.yaml
    stage3_sft_multi.yaml
  src/
    qwen3_vlm/
      models/
        llm_qwen3.py            # loads Qwen/Qwen3-8B-Base
        vision_siglip.py        # loads SigLIP/SigLIP2 vision tower
        resampler.py            # perceiver-style resampler: patches -> N tokens
        projector.py            # MLP: vision_tokens -> llm_hidden
        vlm.py                  # wraps everything; handles interleaving + loss mask
      data/
        formats.py              # "conversation" schema, image list schema
        llava_pretrain.py       # loader for LLaVA-Pretrain json + local images.zip
        llava_instruct.py       # loader for LLaVA-Instruct-150K + COCO images
        mantis_instruct.py      # loader for Mantis-Instruct (multi-image)
        collate.py              # expands <image> placeholders -> N image tokens
      train/
        stage0_sanity.py
        stage1_align.py
        stage2_sft.py
      eval/
        prompt_suite_single.jsonl
        prompt_suite_multi.jsonl
        run_eval.py
      utils/
        chat_template.py        # Qwen3 tokenizer.apply_chat_template wrapper
        logging.py
        checkpointing.py
  scripts/
    download_llava_pretrain.sh
    download_coco2017.sh
    download_mantis.sh
    train_stage1.sh
    train_stage2.sh
    train_stage3.sh
```

Key property: **the collator owns `<image>` expansion**, not the tokenizer. That makes multi-image, variable image counts, and future audio tokens manageable.

---

## 3) Model choices (and why they are “future-proof”)

### 3.1 LLM backbone: Qwen3-8B-Base

* **Apache-2.0** license and **32,768 context**. ([Hugging Face][6])
* Hugging Face explicitly warns you need **Transformers ≥ 4.51.0**, otherwise `KeyError: 'qwen3'`. ([Hugging Face][6])

Design implication: pin your environment early:

* `transformers>=4.51.0`
* `datasets` pinned per Mantis script needs (see below)

### 3.2 Vision tower: use SigLIP2 unless you have a reason not to

Default vision tower: **SigLIP2 so400m patch14 384**:

* SigLIP2 is explicitly released as a “better” encoder family; the HF blog notes SigLIP2 outperforms SigLIP1 across core capabilities and is intended as a VLM encoder. ([Hugging Face][7])
* Simplest baseline: SigLIP1 `siglip-so400m-patch14-384` (standard VLM vision tower). ([Hugging Face][8])

### 3.3 Resampler: Perceiver-style latent tokens (critical for multi-image + Level-4)

Without a resampler, SigLIP patch embeddings are hundreds of tokens per image; multi-image prompts explode context and compute.

**Decision:** convert “patch tokens” → fixed **N visual tokens per image**.

* Default N: **64** tokens/image (good starting point for multi-image and leaves headroom for audio later).
* You can make N a config knob: 64 (fast), 128 (balanced), 256 (quality).

### 3.4 Projector: 2-layer MLP with GELU (LLaVA-style)

A 2-layer MLP projector is a proven connector baseline; LLaVA-style recipes commonly use an `mlp2x_gelu` projector configuration. ([internvl.readthedocs.io][9])

---

## 4) `<image>` interleaving design (multi-image from day one)

### 4.1 Token strategy (what actually goes into the LLM)

You do **not** want `<image>` to be a single token that “magically holds an image.” You want it to expand to a block of reserved positions:

* In the raw text you keep a single sentinel token: `<image>`
* In collation you expand each `<image>` to:
  `<im_start> <image_patch> x N <im_end>`
  where `N` is your resampler token count (e.g., 64)

This makes the LLM see a consistent token budget per image, and it keeps alignment/training stable.

### 4.2 Mapping multiple images to multiple `<image>` slots

Each training example carries:

* `images: [path0, path1, ...]`
* `conversation: [{role, content-with-<image>}, ...]`

Collator rule:

* The k-th `<image>` occurrence maps to `images[k]`.

This is exactly the kind of multi-image prompt convention you see in practice; TinyLLaVA explicitly documents prompt templates using `<image>` as a placeholder token and notes multi-image support. ([Hugging Face][5])

---

## 5) How you get data (practical and reproducible)

You need three data buckets: (A) alignment, (B) single-image instruction tuning, (C) multi-image instruction tuning.

### A) Alignment data (Stage 1): LLaVA-Pretrain (LCS-558K)

This is the easiest “turnkey” alignment dataset to start with because it ships:

* `blip_laion_cc_sbu_558k.json` and meta
* **`images.zip`** for reproduction (with usage caveats) ([Hugging Face][2])

Important note: the dataset card explicitly says `images.zip` “should not be used for any other purpose” and must comply with LAION/CC/SBU licensing, and it may be taken down. ([Hugging Face][2])

Implementation: your `download_llava_pretrain.sh` does:

* `datasets.load_dataset("liuhaotian/LLaVA-Pretrain")` for JSON/meta
* download `images.zip`, extract to `data/llava_pretrain/images/`

### B) Single-image SFT (Stage 2): LLaVA-Instruct-150K + COCO images

LLaVA-Instruct-150K is a GPT-generated conversation dataset with `<image>` in the dialogue. ([Hugging Face][3])
The HF preview shows COCO-style filenames like `000000215677.jpg` in the `image` field, which implies you should point it at COCO images on disk. ([Hugging Face][3])

Practical approach:

* Download **COCO 2017 train** images and store them under `data/coco/train2017/`
* Dataset loader resolves `image` → `data/coco/train2017/{image}`

TinyLLaVA’s data prep section also treats COCO2017 as a standard dependency in VLM SFT setups. ([Hugging Face][5])

### C) Multi-image SFT (Stage 3): Mantis-Instruct (721K)

For multi-image, you want data that is *actually interleaved multi-image*, not just single-image examples with multiple images stuffed in.

Mantis-Instruct:

* is explicitly “fully text-image interleaved” and covers multi-image skills (coref, comparison, temporal, reasoning). ([Hugging Face][4])
* provides two loading modes:

  * load text only + download images manually
  * or use `revision="script"` to auto-download/process images (the dataset card mentions `datasets==2.18.0` for the script loader). ([Hugging Face][4])

This is the cleanest way to make “multi-image in one prompt” actually work.

---

## 6) Training stages (exactly what trains when)

This follows LLaVA’s logic (alignment → instruction tuning), but with your resampler in the loop. ([GitHub][1])

### Stage 0 (sanity): overfit 200 examples

* Freeze everything except projector
* Confirm:

  * `<image>` expansion shape correctness
  * loss masking (only assistant tokens contribute)
  * save/load/infer

### Stage 1 (alignment): LLaVA-Pretrain (558K)

* Freeze: Qwen3 + vision tower
* Train: resampler + projector
* Objective: next-token prediction on assistant tokens conditioned on image-token block

### Stage 2 (single-image SFT): LLaVA-Instruct-150K

* Freeze: vision tower
* Train: resampler + projector + **LoRA on Qwen3**
* Goal: chat behavior grounded in image

### Stage 3 (multi-image SFT): Mantis-Instruct

* Same as Stage 2, but with multi-image batches and careful bucketing by #images.

---

## 7) The design choices that matter most (and what can be safely deferred)

### Must decide now

1. **Resampler token budget (N tokens/image)**
   This controls multi-image feasibility and your future audio budget.

2. **Prompt/interleaving grammar**
   Lock the `<image>` mapping rule and stick to it across all datasets.

3. **Training masking rules**
   Always compute loss on assistant spans only; otherwise you teach the model to “predict the user.”

### Can defer

* Unfreezing the vision tower (can be done later for higher accuracy)
* Fancy fusion (cross-attention blocks) — you do not need it for Level-1
* Any audio integration decisions (Level-2+) as long as you keep the resampler token budget conservative

---

## 8) Minimal “get started” dependency pins (to avoid known foot-guns)

* `transformers>=4.51.0` for Qwen3 support. ([Hugging Face][6])
* If you use Mantis auto-download scripts, follow their dataset loader requirement (`datasets==2.18.0` noted in their card). ([Hugging Face][4])

---

Optional: convert this into a concrete “v0.1 implementation spec” with:

* the exact `<image>` expansion algorithm (including special token IDs),
* the resampler module definition (Perceiver-style),
* and stage-by-stage configs (batching, sequence lengths, LoRA target modules for Qwen3, and bucketing strategy for multi-image).

[1]: https://github.com/haotian-liu/LLaVA "GitHub - haotian-liu/LLaVA: [NeurIPS'23 Oral] Visual Instruction Tuning (LLaVA) built towards GPT-4V level capabilities and beyond."
[2]: https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain "liuhaotian/LLaVA-Pretrain · Datasets at Hugging Face"
[3]: https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K "liuhaotian/LLaVA-Instruct-150K · Datasets at Hugging Face"
[4]: https://huggingface.co/datasets/TIGER-Lab/Mantis-Instruct "TIGER-Lab/Mantis-Instruct · Datasets at Hugging Face"
[5]: https://huggingface.co/bczhou/tiny-llava-v1-hf "bczhou/tiny-llava-v1-hf · Hugging Face"
[6]: https://huggingface.co/Qwen/Qwen3-8B-Base "Qwen/Qwen3-8B-Base · Hugging Face"
[7]: https://huggingface.co/blog/siglip2 "SigLIP 2: A better multilingual vision language encoder"
[8]: https://huggingface.co/google/siglip-so400m-patch14-384 "google/siglip-so400m-patch14-384 · Hugging Face"
[9]: https://internvl.readthedocs.io/en/latest/internvl1.0/internvl_chat_llava.html?utm_source=chatgpt.com "InternVL for Multimodal Dialogue using LLaVA Codebase"
