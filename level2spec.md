Below is a concrete **Level 2 specification** (the “system first” nano-banana mini experience), followed by a **replacement program specification** (training your own image generator + image editor to replace the frozen tools), with **resource estimates** broken into: **data**, **compute**, and **engineering effort**.

I’m going to be explicit about one key reality:

* Building the **Level 2 system** is very feasible on DGX Spark (mostly inference + some light fine-tuning).
* Training a **frontier-capable** text-to-image generator from scratch is *not* feasible on a single workstation at the scale frontier models are trained; we can still define a **“from-scratch mini generator”** that is feasible, but it will be materially behind SDXL-class quality.
  The Databricks/Mosaic reproduction of Stable Diffusion 2 is a good public “scale anchor” for what “from scratch at competitive quality” looks like. ([Databricks][1])

---

# Level 2 Spec

## Objective

Deliver a **Nano-banana mini user experience** using your existing **Level 1 VLM (Qwen + vision encoder + projector)** as the “brain” and **image tools** as the “hands”:

1. **Text → Image** generation
2. **Image + Prompt → Image** editing (with strong detail preservation; localized edits)

## System boundaries

* **Qwen VLM does not generate pixels.**
* Qwen outputs structured tool calls; an image generator/editor produces pixels.
* We retain all artifacts (image, mask, seed, prompt, tool params) for reproducibility and iterative editing.

---

## Components (services/modules)

### 1) VLM Orchestrator (your Level 1 model)

**Inputs:** chat history + optional image(s) + user instruction
**Outputs:** either (a) a natural language response, or (b) a structured tool call plan.

**Responsibilities**

* Classify request: `generate` vs `edit` vs `ask clarification`
* Extract constraints: “preserve identity,” “keep background,” “only change bar color,” etc.
* Produce tool calls with explicit parameters (seed, steps, strength, mask policy)
* Post-edit critique: verify results and decide whether to refine automatically

### 2) Image Generator Tool

**API:** `image.generate(prompt, neg_prompt, width, height, seed, steps, cfg, style_refs?) -> image`

Recommended “frozen” baseline initially:

* SD-family latent diffusion is widely documented: Stable Diffusion uses a **frozen CLIP ViT-L/14** text encoder and an **~860M UNet** in latent space, trained on subsets of **LAION-5B**. ([Hugging Face][2])
  (You can choose SDXL/Flux/etc as the implementation; the system design is the same.)

### 3) Image Editor Tool (Inpainting / Edit)

**API:** `image.edit(image, instruction, mask=None, strength, seed, steps, cfg, preserve_outside_mask=true) -> image`

Key behavior requirement:

* If a mask is provided, the editor must focus changes inside the mask and minimize drift elsewhere. Stable Diffusion has explicit inpainting variants/pipelines. ([Hugging Face][2])

### 4) Masking / Localization Tool

**API:** `mask.propose(image, instruction) -> mask, confidence, rationale`

Three modes (choose at runtime):

* **VLM-proposed bbox/polygon** (fast; good when the user’s instruction is clear)
* **Segmentation-assisted mask** (best for objects)
* **User-confirmed mask** (for diagrams/UI where precision is critical)

### 5) Preservation Verifier

**API:** `verify(original, edited, mask) -> drift_score, text_drift_score, pass/fail`

Minimum checks:

* Compute pixel similarity **outside mask** (SSIM/LPIPS/PSNR—pick one or two)
* Optional OCR check for “diagram mode” to ensure labels did not change

### 6) Artifact Store / Versioning

Store every revision as:

* `image_id`, `parent_image_id`
* tool name + params
* prompts/instructions
* mask
* seed
* verifier scores

This enables iterative “nano-banana style” editing with rollback and reproducibility.

---

## Tool-call schema (recommended)

A minimal function-calling JSON schema (what Qwen emits):

```json
{
  "action": "image.edit",
  "args": {
    "image_id": "img_0123",
    "instruction": "Change the bar labeled 'Q2' from blue to orange. Keep everything else identical.",
    "mask_policy": "auto_then_verify",
    "strength": 0.55,
    "steps": 28,
    "cfg": 4.5,
    "seed": 123456,
    "diagram_mode": true
  }
}
```

The orchestrator then:

1. Calls `mask.propose(...)`
2. Calls `image.edit(...)`
3. Calls `verify(...)`
4. If drift fails, automatically re-run with tighter mask and/or lower strength.

---

## Level 2 dataset needs (system-first phase)

To ship the system, you do **not** need large training datasets. You need:

* A small tool-use tuning set (hundreds to low thousands) where the desired output is tool call JSON
* A prompt-suite for regression testing (hundreds)

Engineering value here is in:

* prompt compiler + mask logic + verifier + iteration loop

---

## Level 2 compute + storage (system-first)

* **Compute:** mostly inference; the only training is optional lightweight tool-call formatting alignment.
* **Storage:** modest (tens to hundreds of GB) unless you start archiving large datasets and many generated images.

**Engineering effort (order-of-magnitude):**

* Moderate: orchestration + tool APIs + verifier + UI/UX + logging
* This is “systems engineering + ML plumbing,” not large-scale training.

---

# Replacement Program Spec

Goal: replace the frozen **image generator tool** and **image editor tool** with **your own trained models**, while preserving the Level 2 system interface.

There are two subprojects:

1. **Train a text-to-image generator** (T → I)
2. **Train an edit/inpaint model** (I + prompt + mask → I)

You can either:

* train them as **separate models** (simpler), or
* train a **unified generator** that supports text-to-image + inpainting + img2img via conditioning (harder but elegant).

---

## Part A — Train your own Image Generator (T → I)

### Architecture (recommended baseline)

Use the classic “Stable Diffusion-style” latent diffusion recipe:

* **Image autoencoder (VAE)** to map pixels ↔ latent space
* **Text encoder** (often frozen) to condition
* **Denoiser** (UNet or DiT) trained in latent space
  Stable Diffusion v1 explicitly uses a downsampling-factor 8 autoencoder and CLIP ViT-L/14 for text conditioning. ([GitHub][3])

### Data requirements

At minimum you need **image–text pairs**.

For “competitive from-scratch quality,” the public reproduction reference is huge:

* Databricks/Mosaic trained on a subset of LAION-5B with **790M image-caption samples** (phase 1) and **300M** (phase 2), and note the dataset was **>100TB**. ([Databricks][1])

For a **mini from-scratch generator**, it is possible to aim much lower (tens of millions), but quality drops accordingly.

### Compute requirements (anchored to a published run)

Databricks/Mosaic report:

* Training Stable Diffusion 2 base from scratch used **20,051 A100 hours** for training plus **3,784 A100 hours** to precompute VAE+CLIP latents = **23,835 A100 hours total**. ([Databricks][1])

That is your “competitive from-scratch” anchor.

#### Practical scaling estimate

Compute for diffusion training scales roughly with:

* model size × resolution × number of training steps/samples seen

If you keep a similar architecture but reduce the number of samples seen, a **rough linear scale** gives:

* **100M image-text pairs** (vs ~1.09B exposures in the Mosaic run) → ~2,200 “A100-hours equivalent”
* **50M pairs** → ~1,100 A100-hours equivalent
* **10M pairs** → ~220 A100-hours equivalent

These are *order-of-magnitude* numbers derived by scaling the published 23,835 A100-hours by dataset size. ([Databricks][1])
They do **not** guarantee quality; they indicate feasibility bands.

### Engineering work (what you must build)

1. Data pipeline (sharding, decoding, resizing, caption normalization, filtering)
2. VAE training (if not reusing a known good one)
3. Latent precompute and caching (Mosaic explicitly did this to reduce training cost/time) ([Databricks][1])
4. Denoiser training loop (diffusion scheduler, EMA, checkpointing)
5. Eval harness (FID/CLIP alignment + human preference)
6. Safety filters (at minimum for public releases)

**Engineering effort (order-of-magnitude):**

* High. Training-from-scratch becomes an “ML infra + data engineering” project.

---

## Part B — Train your own Image Editor (I + prompt + mask → I)

There are two distinct “edit” paradigms; you likely want both.

### B1) Mask-guided inpainting (best for preservation)

**Training data:** (image, mask, mask-text description, target image)

A relevant public reference on scale and method:

* MTADiffusion introduces an **MTADataset** comprising **5 million images** and **25 million mask-text pairs**, built to improve inpainting semantic/structural stability. ([CVF Open Access][4])
  They also describe multi-task training (inpainting + edge prediction) and a style-consistency loss to preserve coherence. ([CVF Open Access][4])

You do not need to match their scale to learn the method, but this gives you a realistic sense of what “strong inpainting performance” data looks like.

### B2) Instruction-guided editing (no explicit mask)

**Training data:** (input image, instruction, edited image)

Public reference:

* InstructPix2Pix reports a generated dataset of **454,445 examples**, each with (input image, instruction, output image). ([Hugging Face][5])
* Newer datasets exist at larger scales, e.g., ImgEdit claims **1.2 million curated edit pairs**. ([arXiv][6])

### Compute requirements (editor)

You have two ways to train the editor:

#### Editor training path 1 (recommended): fine-tune from your generator

* Start from your base text-to-image diffusion model
* Add mask conditioning channels and train inpainting/editing heads

This is far cheaper than training an editor from scratch, and is how many practical stacks evolve.

Compute ballpark:

* **Tens to low hundreds of A100-hours equivalent** for initial inpainting capability (smallish dataset, moderate steps)
* **Hundreds to low thousands of A100-hours equivalent** for “good” editing behavior with broad coverage and low drift (especially if you include multi-task losses like MTADiffusion-style structure/style constraints) ([CVF Open Access][4])

#### Editor training path 2 (hard): train an editor model from scratch

Expect costs closer to the base generator (not recommended unless you have a strong reason).

### Engineering effort (editor)

1. Mask generation + augmentation (random masks + segmentation masks)
2. Data synthesis pipeline (if you generate edits using a teacher model)
3. Conditional architecture implementation (mask channels, masked-image channels)
4. Losses for structure and preservation (outside-mask preservation metrics; optionally multi-task structure losses)
5. Benchmarks for editing (EditBench exists as an evaluation dataset; Google describes it as 240 images for text-guided inpainting evaluation) ([Google Research][7])

---

# Resource Estimates Summary

## Phase 1: Level 2 system-first (frozen tools)

### Data

* 0–10k tool-call examples (handcrafted + self-generated)
* 200–1,000 evaluation prompts (internal)

### Compute

* Primarily inference on DGX Spark
* Optional light fine-tuning of Qwen for tool-call formatting (LoRA)

### Engineering effort

* Moderate:

  * tool schema + orchestrator + mask tool + verifier + artifact store
  * deterministic generation (seed management) and iterative editing loop

---

## Phase 2: Replace the generator (train your own T → I)

### Data

* **Mini generator:** 10M–100M image-text pairs
* **Competitive (SD2-like):** hundreds of millions to >1B exposures; Mosaic used 790M + 300M with >100TB data. ([Databricks][1])

### Compute

* **Competitive SD2-like baseline:** **23,835 A100-hours total** reported (training + latent precompute). ([Databricks][1])
* **Mini from-scratch (very rough):**

  * 10M pairs: ~220 A100-hours equivalent
  * 50M pairs: ~1,100 A100-hours equivalent
  * 100M pairs: ~2,200 A100-hours equivalent
    (Derived by scaling the published 23,835 A100-hours by sample count; actual depends on model size/resolution/steps.) ([Databricks][1])

### Engineering effort

* High:

  * data engineering dominates
  * training stability + eval + safety

---

## Phase 3: Replace the editor (train your own I+prompt(+mask) → I)

### Data

You’ll want two pools:

* Mask inpainting pool (ideally millions): MTADataset is an example at **5M images / 25M mask-text pairs**. ([CVF Open Access][4])
* Instruction editing pool:

  * InstructPix2Pix: **454,445** supervised edit triples ([Hugging Face][5])
  * ImgEdit: **1.2M** curated edit pairs ([arXiv][6])

### Compute

* If fine-tuning from your generator: **hundreds to low thousands of A100-hours equivalent**, depending on dataset scale and how aggressively you optimize drift/preservation.
* If from scratch: comparable to generator-scale (not recommended).

### Engineering effort

* High:

  * dataset synthesis/cleaning (especially mask-text alignment)
  * drift metrics and regression loops
  * multi-turn edit behavior

---

# Recommended “sane” execution strategy

To maximize learning and keep the project feasible:

1. **Ship Level 2 system with frozen tools** first (you already agreed).
2. **Train your own generator as a “mini SD”** on a manageable dataset (10–50M), accept that it won’t beat SDXL, but it will be “yours.”
3. **Fine-tune your editor from your generator**, using InstructPix2Pix + a mask dataset.
4. Only then consider scaling data/compute if you want to chase SD2/SDXL-quality.

---

# Two clarifying decisions that materially change the resource plan

These are non-blocking inputs (they can be answered later), but they determine the correct “from scratch” target:

1. Do you want the **replacement generator/editor** to be **general-purpose** (internet-style), or **domain-specialized** (e.g., diagrams/UI/product images)?
   Domain-specialized can cut data and compute by an order of magnitude.

2. What resolution is your “must support” target: **512-only** vs **768/1024**?
   Resolution is a major compute multiplier.

If you give me your answer for (1) and (2), I’ll turn the replacement program into a concrete milestone plan with:

* chosen architecture (UNet vs DiT),
* dataset mixture composition and filters,
* step counts per phase,
* and an “A100-hour equivalent” budget that matches your DGX Spark reality rather than paper-scale clusters.

[1]: https://www.databricks.com/blog/stable-diffusion-2 "Training Stable Diffusion from Scratch for $50k with MosaicML (Part 2) | Databricks Blog"
[2]: https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/overview "Stable Diffusion pipelines"
[3]: https://github.com/CompVis/stable-diffusion "GitHub - CompVis/stable-diffusion: A latent text-to-image diffusion model"
[4]: https://openaccess.thecvf.com/content/CVPR2025/papers/Huang_MTADiffusion_Mask_Text_Alignment_Diffusion_Model_for_Object_Inpainting_CVPR_2025_paper.pdf "MTADiffusion: Mask Text Alignment Diffusion Model for Object Inpainting"
[5]: https://huggingface.co/spaces/timbrooks/instruct-pix2pix/commit/2afcb7e45bd350765f21a58a0c135871e9dc5a78?utm_source=chatgpt.com "Add InstructPix2Pix · timbrooks/instruct-pix2pix at 2afcb7e"
[6]: https://arxiv.org/html/2505.20275v1?utm_source=chatgpt.com "ImgEdit: A Unified Image Editing Dataset and Benchmark"
[7]: https://research.google/blog/imagen-editor-and-editbench-advancing-and-evaluating-text-guided-image-inpainting/?utm_source=chatgpt.com "Imagen Editor and EditBench: Advancing and evaluating ..."
