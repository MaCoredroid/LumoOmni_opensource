Below is a concrete **Level 1 (Text + Multi-Image VLM)** plan using **Option 2** with a **Qwen3 text-only backbone**, designed to allow later extension to **Level 4 (speech output)** without repainting the system into a corner.

## 0) Model and architecture choices (chosen for Level 4 compatibility)

### Backbone: use a dense Qwen3 “Base” model (not MoE, not Qwen3-Next)

* **Recommended starting point:** **Qwen3-8B-Base** (dense, Apache-2.0, 32,768 context). ([Hugging Face][1])
* Rationale: Dense models are easier to fine-tune, easier to reason about, and less likely to require MoE-specific stabilization tricks (e.g., load balancing losses) that become annoying once you start injecting non-text tokens. Qwen3 explicitly spans dense + MoE; for your build-from-scratch multimodal stack, dense is the lowest-friction foundation. ([Qwen][2])
* Note: “Latest Qwen3” could also mean **Qwen3-Next** (newer hybrid MoE/hybrid attention, long-context efficiency). That line is real and newer, but it adds architectural complexity that is unnecessary for Level 1 and will slow your iteration. ([AlibabaCloud][3])

### Vision tower: SigLIP (frozen initially)

SigLIP is well-supported in Transformers and is a standard choice as a vision encoder for VLMs. ([Hugging Face][4])

### Connector: use a token-budgeted design (critical for multi-image now, audio later)

To keep multi-image scalable and leave room for future audio tokens, do **not** pass thousands of patch tokens per image into the LLM. Use:

1. Vision encoder → patch embeddings
2. **Resampler** (e.g., Perceiver-style) → fixed **N visual tokens per image** (typical N=64–256)
3. Small **projector MLP** → LLM hidden size

This single decision is what makes “multi-image in one prompt” practical today and makes Level 4 feasible later.

---

## 1) Multi-image prompt format (what you will actually support)

### Format

* Prompt contains multiple `<image>` placeholders, optionally interleaved with text.
* At inference you pass `images=[img1, img2, ...]` and the model fills embeddings in the order of `<image>` occurrences.

This is the same general mechanism used by LLaVA-style implementations; multi-image support requires correct placeholder usage. ([Hugging Face][5])

### Important constraint

If you only train on single-image data, many models will effectively ignore earlier images when multiple are provided. ([Hugging Face Forums][6])
So you must include **multi-image training data** (not just support multi-image at inference).

---

## 2) Training plan (Level 1)

You will do three stages. Each stage has a clear “what trains,” “what data,” and “what success looks like.”

### Stage A — Pipeline sanity (mandatory)

**Goal:** prove end-to-end correctness before burning compute.

* Train on ~100–1,000 samples until it overfits.
* Verify: checkpoint save/load, generation works, `<image>` token placement matches number of images, and your connector produces the right token count per image.

**Pass criterion:** near-100% accuracy on the tiny set, deterministic behavior.

---

### Stage B — Vision-language alignment (CPT-style, but narrow and stable)

**Goal:** teach Qwen3 to treat your injected visual tokens as meaningful conditioning.

**Trainable components**

* Train: resampler + projector (and optionally LoRA on the top few LLM layers later)
* Freeze: vision encoder + most/all of the LLM initially

**Data**

* Large image-caption / image-description pairs (clean is better than huge at this stage).
* Include a small but nontrivial fraction of *synthetic multi-image captioning* examples:

  * “Describe image 1 and image 2.” (two images)
  * “What are differences between image A and B?” (two images)

**Pass criterion**

* The model produces grounded captions and can answer basic questions about each image in a multi-image prompt.

---

### Stage C — Instruction tuning with true multi-image supervision

**Goal:** chat behavior + instruction following + multi-image reasoning.

**Core datasets**

* **LLaVA-Instruct-150K** for single-image chat behavior. ([Hugging Face][7])
* **Mantis-Instruct (721K)** for *interleaved multi-image instruction tuning* (co-reference, comparison, reasoning, temporal). This is purpose-built to teach multi-image skills without needing “hundreds of millions” of noisy interleaved pretraining examples. ([Hugging Face][8])

**Trainable components**

* Train: projector + (QLoRA/LoRA) on the LLM (at least on attention/MLP blocks)
* Keep: vision encoder frozen (it can be unfrozen later, but it is not necessary for a strong Level 1)

**Pass criterion**

* Multi-turn chat with multiple images works reliably.
* The model references the correct image when asked (not “latest-image bias”).

---

## 3) Evaluation harness (what you run every checkpoint)

### Multi-image evaluation

* **NLVR2** style paired-image reasoning is a good sanity task (it is explicitly grounded in pairs of photographs). ([LIL Lab][9])
* Add a multi-image benchmark like **MIRB** for relational multi-image skills if you want a standardized score. ([arXiv][10])

### Single-image regression

* Keep a fixed single-image prompt suite and (optionally) a benchmark like **MMMU** to ensure you didn’t regress general VLM competence while specializing on multi-image. ([MMMU Benchmark][11])

### Language-only regression

* Maintain a fixed text-only set to ensure Qwen3 did not lose its base language capability.

---

## 4) Practical tooling constraints you must pin early

* **Transformers versioning:** Qwen3 model cards explicitly warn that older Transformers versions will error (e.g., `KeyError: 'qwen3'` with older releases). Plan to pin a modern Transformers version in the repo and document it. ([Hugging Face][12])

---

## 5) What I recommend you implement first (to keep momentum)

1. Qwen3-8B-Base + SigLIP + resampler + projector + `<image>` interleaving
2. Stage A sanity overfit
3. Stage B alignment on captions (single + small multi-image augmentation)
4. Stage C instruction tuning with **LLaVA-Instruct-150K + Mantis-Instruct**

Once this is stable, Level 2/3/4 become “add modality adapters,” not “rewrite the stack.”

Optional: specify the **exact token budget** (visual tokens per image), the **interleaving grammar**, and the **training mixture ratios** for Stage B/C (optimized for multi-image and future audio headroom) using **Qwen3-8B-Base’s 32k context**. ([Hugging Face][1])

[1]: https://huggingface.co/Qwen/Qwen3-8B-Base "Qwen/Qwen3-8B-Base · Hugging Face"
[2]: https://qwenlm.github.io/blog/qwen3/ "Qwen3: Think Deeper, Act Faster | Qwen"
[3]: https://www.alibabacloud.com/blog/602536?utm_source=chatgpt.com "Qwen3-Next: A New Generation of Ultra-Efficient Model ..."
[4]: https://huggingface.co/docs/transformers/en/model_doc/siglip?utm_source=chatgpt.com "SigLIP"
[5]: https://huggingface.co/llava-hf/bakLlava-v1-hf/discussions/3?utm_source=chatgpt.com "llava-hf/bakLlava-v1-hf · How do i pass in multiple images ..."
[6]: https://discuss.huggingface.co/t/llava-multi-image-input-support-for-inference/68458?utm_source=chatgpt.com "LLaVA multi-image input support for inference - Models"
[7]: https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K?utm_source=chatgpt.com "liuhaotian/LLaVA-Instruct-150K · Datasets at Hugging Face"
[8]: https://huggingface.co/datasets/TIGER-Lab/Mantis-Instruct "TIGER-Lab/Mantis-Instruct · Datasets at Hugging Face"
[9]: https://lil.nlp.cornell.edu/nlvr/?utm_source=chatgpt.com "Natural Language for Visual Reasoning - LIL Lab"
[10]: https://arxiv.org/html/2406.12742v1?utm_source=chatgpt.com "Benchmarking Multi-Image Understanding in Vision and ..."
[11]: https://mmmu-benchmark.github.io/?utm_source=chatgpt.com "MMMU Benchmark"
[12]: https://huggingface.co/Qwen/Qwen3-1.7B "Qwen/Qwen3-1.7B · Hugging Face"
