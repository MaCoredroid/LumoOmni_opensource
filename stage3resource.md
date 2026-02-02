### 1) SigLIP: does it output text? What is the connector actually doing?

**SigLIP is a vision encoder, not a “vision-to-text” model.** It takes an image and outputs a sequence of **continuous feature vectors** (embeddings). By itself, it does not produce words or captions.

The **connector (resampler + projector, optionally a vision LayerNorm)** exists because a text LLM expects **token embeddings in its own hidden space**. So you need a learned mapping:

* **Resampler**: compresses many vision patch features into a fixed small set of “visual tokens” (e.g., 64 latents) so your context length stays manageable.
* **Projector**: maps vision features into the LLM’s embedding dimension / distribution so the frozen LLM can attend to them as if they were “special tokens.”
* **(Optional) Vision LN**: stabilizes feature scale / distribution before projection.

This “frozen vision + frozen LLM + train connector” pattern is extremely common for **text output conditioned on images** (VLM chat, captioning, VQA). It is *not* the same thing as “true omni” generation (speech/image output), which is where discrete tokenization becomes important.

---

### 2) Is the projector/resampler pattern “real-world multimodal”?

Yes—for **understanding** (image → text reasoning) it is a dominant, pragmatic pattern in open models because it’s easy to train, stable, and compute-efficient.

But if you want **generation in another modality** (audio output, image output) *inside the same core model*, you typically need one of:

* **Discrete tokenization** of that modality (codec/VQ tokens) + a transformer/flow model that predicts those tokens, or
* A **separate generator head** (e.g., diffusion for images, TTS module for speech) attached to the system.

Your current Qwen3-8B + SigLIP + resampler/projector is excellent for **Level 1/2** (text with image conditioning). It cannot natively “speak” or “draw” unless you attach a separate audio/image generator or switch to a tokenized multimodal framework.

---

### 3) Where do frontier systems “sit” (Gemini / GPT-class) relative to connector vs cross-attn vs discrete tokens?

Public details are limited for closed frontier systems, so nobody outside those labs can state their exact architecture with certainty.

What *is* clear from the open literature and open releases is the trend:

* For **perception / understanding**: systems often keep a **strong continuous encoder** (vision/audio) and fuse it into the core model (via projectors, resamplers, cross-attention, or more integrated fusion).
* For **high-quality generation** in a modality (speech, music, images): **discrete tokenization or latent generation** is very common, because it turns “generate waveform/pixels” into “generate tokens.”
  Open examples of this approach exist for audio and for omni models (see below). ([AudioCraft][1])

The “reads text in images better than OCR” effect is usually not magical OCR replacement; it’s typically a combination of:

* high-resolution visual processing,
* training on lots of text-rich imagery,
* instruction tuning on OCR-like tasks,
* and strong language/world knowledge priors.

---

## 4) Most recent public “Type-3” (discrete-token) resources for **image + audio** models

Below is a curated shortlist of **public weights + code + papers** that are actually relevant to *discrete tokenization* (or hybrid discrete+continuous) for image+audio “omni” systems.

### A) Full “omni / any-to-any” models with public weights

**HyperCLOVA X SEED Omni 8B (Jan 2026) — speech-to-speech, with discrete vision/audio codebook tokens**

* Paper: “HyperCLOVA X 8B Omni: A Full-Stack Speech-to-Speech Model with Multimodal Understanding and Generation” (Jan 2026). ([arXiv][2])
* Weights: `naver-hyperclovax/HyperCLOVAX-SEED-Omni-8B` (public model card). ([Hugging Face][3])
* Why it matters for you: it explicitly uses **text tokens + discrete vision/audio codebook entries** in the modeling stack, i.e., it’s in the “tokenized modalities” family (often hybridized with continuous embeddings/adapters). ([arXiv][2])

**MIO (EMNLP 2025) — “foundation model on multimodal tokens” (public weights + code)**

* Code: `meituan-longcat/MIO` GitHub repo (points to Hugging Face models). ([GitHub][4])
* Weights: `meituan-longcat/mio-base` and `meituan-longcat/mio-instruct` (Apache-2.0). ([Hugging Face][5])
* Why it matters: it is explicitly about training on **multimodal tokens** (the design space you asked for). ([ACL Anthology][6])

**AnyGPT (OpenMOSS) — unified discrete representations across modalities (speech/images/music) + public SFT data**

* Paper: “AnyGPT: Unified Multimodal LLM with Discrete Representations for Generative Tasks.” ([Hugging Face][7])
* Weights: `OpenMOSS/AnyGPT` (public model card). ([arXiv][8])
* Instruction data: `OpenMOSS/AnyInstruct` dataset (108k). ([Jun Zhan][9])
* Why it matters: it’s a clean public reference implementation of “tokenize modalities → train a single model to model/generate tokens.” ([Hugging Face][7])

**NExT-OMNI (Oct 2025) — discrete flow matching over multimodal tokens (paper-level blueprint)**

* Paper (arXiv): “NExT-OMNI: Towards Any-to-Any Omnimodal Foundation Models with Discrete Flow Matching.” ([arXiv][10])
* Why it matters: the paper describes a full recipe using **VQVAE-based modality encoders** to produce **discrete token sequences**, and trains via a discrete flow matching paradigm, with reconstruction/semantic alignment warmup for encoders. ([arXiv][11])
* Note: the sources cite the **training approach and the claim of open-sourcing**, but no official repo/weights link appears in the excerpts above—treat this as a **paper recipe to follow/replicate** and verify the authors’ release location separately. ([arXiv][11])

### B) Discrete-token “building blocks” for audio (tokenizers + token LMs)

For Type‑3 builds, the **audio tokenizer/codec** choice matters as much as the transformer.

**EnCodec (Meta) — open neural audio codec / tokenizer**

* Repo: `facebookresearch/encodec` (MIT). ([GitHub][12])
* Overview page: describes residual VQ bottleneck producing parallel streams of audio tokens. ([AudioCraft][13])

**AudioCraft + MusicGen — public reference for “EnCodec tokens → autoregressive transformer → decode”**

* AudioCraft repo: includes EnCodec tokenizer and MusicGen. ([GitHub][14])
* MusicGen doc: explicitly states it’s an **autoregressive Transformer trained over a 32kHz EnCodec tokenizer** (4 codebooks). ([GitHub][15])
* AudioCraft overview: explains the full pipeline (tokenize → LM over tokens → decode). ([AudioCraft][1])

**DAC (Descript Audio Codec) — alternative open tokenizer**

* Repo: training + inference scripts, weights link. ([GitHub][16])
* Transformers doc: describes DAC as compressing audio into tokens (useful for token-based modeling). ([Hugging Face][17])

---

## 5) Public datasets suitable for **audio-language** and “image+audio via video”

You asked for datasets “towards that goal.” For tokenized omni training, you typically need:

1. **Audio ↔ text** (captioning/retrieval)
2. **Image ↔ text** (captioning/instruction)
3. **Audio + image together** (usually via **video datasets**: audio track + sampled frames + optional text)

Here are **public, commonly used audio-text datasets** with solid primary references:

### Audio ↔ text datasets (good for audio token LM conditioning, captioning, retrieval)

**WavCaps**

* HF dataset card: ChatGPT-assisted weakly-labeled audio captioning dataset; includes sources like FreeSound/BBC/SoundBible and an AudioSet subset. ([Hugging Face][18])
* License note (important): WavCaps authors specify **research-only / academic-use constraints** for the dataset access process. ([GitHub][19])

**AudioCaps**

* Official site: audio captioning dataset; ~46K audio clips with human-written text pairs collected on AudioSet. ([AudioCaps][20])
* ACL Anthology record for the original paper. ([ACL Anthology][21])

**Clotho**

* Paper: “Clotho: An Audio Captioning Dataset.” ([arXiv][22])
* Zenodo distribution record. ([Zenodo][23])

**MusicCaps**

* HF dataset card: 5,521 music examples with musician-written captions + aspect lists. ([Hugging Face][24])
* Kaggle dataset page (commonly used access point). ([Kaggle][25])
* Practical note: MusicCaps is often distributed as metadata (e.g., IDs/timestamps) with scripts/notebooks used to fetch audio; an example “download underlying audio” gist exists. ([Gist][26])

### Image ↔ text datasets (for the image side)

For open large-scale image-text pairs and VLM instruction corpora, community “awesome” lists track the common choices (COYO, ShareGPT4V, etc.). ([GitHub][27])
(You already have LLaVA-Pretrain in your pipeline; that remains a practical Stage-1/Stage-2 resource.)

### How to get **paired image+audio** without a special dataset

The simplest public route is: **use “audio captioning on videos” sources** and extract frames:

* AudioCaps is built from AudioSet clips; that means the underlying unit is a video segment with an audio track. ([AudioCaps][20])

For “image+audio” training, options include:

* take the **audio** from the clip,
* sample **one or more frames** (or a short frame sequence),
* treat (frames, audio, caption) as a tri-modal sample.

That becomes a natural dataset for any-to-any tasks (audio→text, image→text, audio+image→text, etc.), and later (text→audio) if you tokenize the waveform.

---

## 6) What to fine-tune vs what to pretrain for a Type-3 (image+audio) stack

Given your current trajectory (Qwen3 VLM with a connector), the most feasible “Type-3” learning path is:

### Option A: Start from an existing token-omni model and fine-tune

If your goal is to **learn the training recipe** and ship a research-grade open demo:

* Start from **HyperCLOVAX-SEED-Omni-8B** (freshest public omni weight) ([Hugging Face][3])
  or **MIO** / **AnyGPT** if you want a more “tokens-first” codebase. ([Hugging Face][5])
* Fine-tune on:

  * Audio captioning + instruction data (WavCaps/AudioCaps/Clotho) ([Hugging Face][18])
  * Multi-turn multimodal instruction data (AnyInstruct) ([Jun Zhan][9])

This avoids months of destabilizing pretraining work.

### Option B: Build your own Type-3 “mini-omni” from a text LLM

This is more aligned with your “build it from the ground up” goal:

**Step 1 — Pick tokenizers**

* Audio: EnCodec (very standard + open) or DAC. ([GitHub][12])
* Image: either use a known image tokenizer (SEED-style tokenizers are a known public direction via SEED-LLaMA) ([Hugging Face][28])
  or train/borrow a VQ-VAE/VQGAN-like tokenizer (NExT-OMNI describes VQVAE-based encoders/codebooks for multimodal tokenization). ([arXiv][11])

**Step 2 — Define your unified token space**

* Treat image/audio codebook entries as:

  * either **new vocabulary IDs** in the LLM embedding table, or
  * separate heads that predict modality tokens (NExT-OMNI uses modality-specific heads rather than extending the LM head directly). ([arXiv][11])

**Step 3 — Training objective**

* Classic Type-3 is “next token prediction over a single interleaved sequence.”
* Newer variants use **discrete diffusion / flow matching** rather than pure AR (NExT-OMNI). ([arXiv][11])

**Step 4 — Data curriculum**

* Start with **audio↔text** and **image↔text** (captioning/instruction).
* Then add “paired audio+image” from video-derived samples (AudioCaps/AudioSet-style). ([AudioCaps][20])

**Step 5 — Evaluation (practical and hard to game)**

* Your existing **ablation loss tests** are exactly the right idea—extend them to audio tokens too.
* Add “golden sets” for:

  * audio captioning,
  * image captioning,
  * and later text→audio generation sanity (listen/ASR-based checks).

---

## 7) A direct recommendation for you (given your current repo + DGX Spark)

If your near-term objective is “learn the full omni recipe” without turning this into a multi-month infrastructure project:

1. Keep your current **Qwen3+SigLIP connector VLM** as your “image understanding” baseline (it’s already working well).
2. In parallel, pick **one** token-omni direction to study deeply, end-to-end:

   * **HyperCLOVAX-SEED-Omni-8B** (most recent public omni weight) ([Hugging Face][3])
   * or **AnyGPT** (clean discrete multi-modal design + AnyInstruct). ([Hugging Face][7])
3. Use **EnCodec + AudioCaps/WavCaps/Clotho** to build your audio-token pipeline and reproduce “text→audio tokens→decode.” ([AudioCraft][1])

That gets you to “image+audio” competence much faster than inventing every component at once.

---

Optional: provide the **Stage 2 YAML** used (or key flags: what’s unfrozen, LR, max_seq_len, prompt template) to align the **Stage 2 → Stage 3** plan with a “tokenized audio” roadmap so the current VLM work composes cleanly with a future Type-3 omni branch.

[1]: https://audiocraft.metademolab.com/ "https://audiocraft.metademolab.com/"
[2]: https://arxiv.org/abs/2601.01792 "https://arxiv.org/abs/2601.01792"
[3]: https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Omni-8B/tree/TAG-2025-12-31 "naver-hyperclovax/HyperCLOVAX-SEED-Omni-8B at TAG-2025-12-31"
[4]: https://github.com/mio-team/mio "GitHub - MIO-Team/MIO: MIO: A Foundation Model on Multimodal Tokens"
[5]: https://huggingface.co/m-a-p/MIO-7B-Base "m-a-p/MIO-7B-Base · Hugging Face"
[6]: https://aclanthology.org/2025.emnlp-main.255.pdf?utm_source=chatgpt.com "MIO: A Foundation Model on Multimodal Tokens"
[7]: https://huggingface.co/datasets/OpenMOSS-Team/AnyInstruct?utm_source=chatgpt.com "OpenMOSS-Team/AnyInstruct · Datasets at Hugging Face"
[8]: https://arxiv.org/html/2502.06490v2?utm_source=chatgpt.com "Recent Advances in Discrete Speech Tokens: A Review"
[9]: https://junzhan2000.github.io/AnyGPT.github.io/?utm_source=chatgpt.com "AnyGPT"
[10]: https://arxiv.org/abs/2510.13721 "https://arxiv.org/abs/2510.13721"
[11]: https://arxiv.org/html/2510.13721v1 "https://arxiv.org/html/2510.13721v1"
[12]: https://github.com/facebookresearch/encodec "https://github.com/facebookresearch/encodec"
[13]: https://audiocraft.metademolab.com/encodec.html "https://audiocraft.metademolab.com/encodec.html"
[14]: https://github.com/facebookresearch/audiocraft "https://github.com/facebookresearch/audiocraft"
[15]: https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md "https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md"
[16]: https://github.com/descriptinc/descript-audio-codec "https://github.com/descriptinc/descript-audio-codec"
[17]: https://huggingface.co/docs/transformers/en/model_doc/dac "https://huggingface.co/docs/transformers/en/model_doc/dac"
[18]: https://huggingface.co/datasets/cvssp/WavCaps "https://huggingface.co/datasets/cvssp/WavCaps"
[19]: https://github.com/XinhaoMei/WavCaps "https://github.com/XinhaoMei/WavCaps"
[20]: https://audiocaps.github.io/ "https://audiocaps.github.io/"
[21]: https://aclanthology.org/N19-1011/ "https://aclanthology.org/N19-1011/"
[22]: https://arxiv.org/abs/1910.09387 "https://arxiv.org/abs/1910.09387"
[23]: https://zenodo.org/records/3490684 "https://zenodo.org/records/3490684"
[24]: https://huggingface.co/datasets/google/MusicCaps "https://huggingface.co/datasets/google/MusicCaps"
[25]: https://www.kaggle.com/datasets/googleai/musiccaps "https://www.kaggle.com/datasets/googleai/musiccaps"
[26]: https://gist.github.com/nateraw/14818c6e4889fc13698e5ca1a2ace2ad "https://gist.github.com/nateraw/14818c6e4889fc13698e5ca1a2ace2ad"
[27]: https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models "https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models"
[28]: https://huggingface.co/docs/transformers/en/model_doc/chameleon?utm_source=chatgpt.com "Chameleon"
SigLIP (and similar vision towers) does **not** “output text.” It outputs **continuous visual embeddings** (feature vectors). The “connector” (projector / resampler / Q-Former / cross-attn bridge) exists to **translate those embeddings into a form the LLM can condition on** (shape, scale, token count, and distribution that the LLM can actually use). That “frozen vision + frozen LLM + train connector” pattern you implemented is a very common real‑world baseline for VLMs.

For **Type‑3 (discrete tokenization)** omni models (image+audio tokens + a transformer core), the “connector” concept mostly shifts to:

* **Tokenizers/decoders as the modality interface** (image/audio ↔ discrete tokens)
* The core transformer learns **next-token prediction over a unified token stream** (text tokens + image tokens + audio tokens), often with modality/type embeddings and special delimiters.

Assumption: **go with Option B** (build the Type‑3 pipeline), but **download Option A** as a reference. Below is a clean “resources to pull” manifest, prioritized for immediate start.

---

## A. Option A reference downloads (study + sanity-check baselines)

These are “known working” public Type‑3 / omni systems to inspect (model cards + code + tokenizers + dataset formats). No need to run them all; the goal is to have reference implementations and weight layouts.

### A1) HyperCLOVA X SEED Omni + OmniServe (production-style omni stack)

**Why pull:** shows an end-to-end “inputs: text/image/audio/video → outputs: text/image/audio” system, plus a server that decomposes unified weights into components. ([Hugging Face][1])

**Pull:**

```bash
# Inference system
git clone https://github.com/NAVER-Cloud-HyperCLOVA-X/OmniServe.git

# Model weights (requires accepting their custom license on HF)
huggingface-cli download naver-hyperclovax/HyperCLOVAX-SEED-Omni-8B \
  --local-dir ./models/HyperCLOVAX-SEED-Omni-8B
```

Notes:

* This model is under a custom license (“hyperclovax”), so review/accept before use. ([Hugging Face][1])
* OmniServe is Apache-2.0 (helpful as reference code). ([GitHub][2])

---

### A2) AnyGPT + AnyInstruct (classic “discrete tokens unified LM” reference)

**Why pull:** AnyGPT is one of the clearest public implementations of the “discrete sequence modeling” approach across **speech, images, music** (tokenizers + SoundStorm + seed-tokenizer). ([Hugging Face][3])

**Pull:**

```bash
# Core repo
git clone https://github.com/OpenMOSS/AnyGPT.git

# Model weights
huggingface-cli download fnlp/AnyGPT-base --local-dir ./models/AnyGPT-base
huggingface-cli download fnlp/AnyGPT-chat --local-dir ./models/AnyGPT-chat

# Multimodal modules (speech tokenizer + soundstorm)
huggingface-cli download fnlp/AnyGPT-speech-modules --local-dir ./models/AnyGPT-speech-modules

# Image tokenizer (SEED tokenizer v2)
huggingface-cli download AILab-CVC/seed-tokenizer-2 --local-dir ./models/seed-tokenizer-2

# Instruction dataset (108k multimodal interleaved)
huggingface-cli download OpenMOSS-Team/AnyInstruct --repo-type dataset --local-dir ./data/AnyInstruct
```

Key points:

* The AnyGPT model card explicitly lists the dependency chain: **SpeechTokenizer + SoundStorm + SEED tokenizer**. ([Hugging Face][4])
* AnyInstruct tells you what “multimodal interleaved instruction data” looks like in practice. ([Hugging Face][5])

---

### A3) MIO (foundation model on multimodal tokens)

**Why pull:** Another “multimodal tokens” reference system with open weights (Apache-2.0), plus a dedicated repo/paper. ([GitHub][6])

**Pull:**

```bash
# Code
git clone https://github.com/mio-team/mio.git

# Weights
huggingface-cli download m-a-p/MIO-7B-Base --local-dir ./models/MIO-7B-Base
huggingface-cli download m-a-p/MIO-7B-Instruct --local-dir ./models/MIO-7B-Instruct
```

Notes:

* The HF entries show Apache-2.0 licensing for these MIO weights. ([Hugging Face][7])

---

### A4) (Optional) LongCat Flash Omni (extreme-scale reference)

**Why optional:** It’s an “omni” open release, but it is far beyond what you will run/train locally; still useful as a *paper/report reference*. ([Hugging Face][8])

I would **not** recommend downloading these weights unless you specifically need them for reading config layouts.

---

## B. Option B build resources (what you need to implement Type‑3 yourself)

A practical Type‑3 system needs **(1) tokenizers**, **(2) datasets**, **(3) a unified-core training codebase**, and **(4) evaluation assets**.

### B1) Modality tokenizers (the single most important “pull” for Type‑3)

#### Image tokenizer options

**Recommended for fastest progress:** SEED tokenizer v2

* Designed exactly for “image ↔ discrete tokens” workflows used by SEED-LLaMA / AnyGPT-style systems. ([Hugging Face][4])

Pull:

```bash
huggingface-cli download AILab-CVC/seed-tokenizer-2 --local-dir ./tokenizers/image/seed-tokenizer-2
```

**Alternative (classic baseline): VQGAN (Taming Transformers)**

* Gives you a well-documented VQ tokenizer + pretrained checkpoints (ImageNet/OpenImages variants). ([GitHub][9])

Pull:

```bash
git clone https://github.com/CompVis/taming-transformers.git
```

Pretrained checkpoints are linked in the repo README (ImageNet/OpenImages VQGAN etc.). ([GitHub][9])

---

#### Audio tokenizer options

**Recommended:** EnCodec (AudioCraft)

* Very standard “audio ↔ discrete tokens” codec used in many autoregressive audio LMs. ([GitHub][10])

Pull:

```bash
git clone https://github.com/facebookresearch/audiocraft.git

# If you prefer pulling pretrained weights via HF:
huggingface-cli download facebook/encodec_48khz --local-dir ./tokenizers/audio/encodec_48khz
```

**Speech-specialized path:** SpeechTokenizer + SoundStorm (AnyGPT modules)

* AnyGPT already packages these modules as working references. ([Hugging Face][4])

Pull (already listed above in A2, but repeated for clarity):

```bash
huggingface-cli download fnlp/AnyGPT-speech-modules --local-dir ./tokenizers/speech/AnyGPT-speech-modules
```

---

### B2) Datasets to pull (image+audio, publicly accessible)

You have two broad strategies:

1. **Text-supervised per modality** (image↔text, audio↔text)
   Then learn cross-modal conversions via **instruction mixing** (like AnyInstruct), without needing a huge paired audio+image dataset.

2. **Paired audio+visual** (video datasets)
   Heavier operationally (download pipelines, licensing, YouTube IDs, etc.). You can add later.

For “start now,” do #1.

#### Audio-caption datasets (audio↔text)

* **WavCaps**: ~400k weakly labeled audio captions; metadata provides links; research-only constraints. ([Hugging Face][11])
* **AudioCaps**: ~46k captioned clips (AudioSet-derived). ([AudioCaps][12])
* **Clotho**: ~5k clips, 5 captions each; common AAC benchmark; available via DCASE/Zenodo. ([arXiv][13])
* **MusicCaps**: 5,521 music-text pairs; often requires downloading audio from YouTube IDs. ([Hugging Face][14])

Pull (metadata/cards):

```bash
# WavCaps (HF metadata)
huggingface-cli download cvssp/WavCaps --repo-type dataset --local-dir ./data/audio/WavCaps

# AudioCaps (HF mirror/metadata)
huggingface-cli download d0rj/audiocaps --repo-type dataset --local-dir ./data/audio/AudioCaps

# MusicCaps
huggingface-cli download google/MusicCaps --repo-type dataset --local-dir ./data/music/MusicCaps
```

Practical note:

* Some AudioCaps distributions are gated or metadata-only; you may need to pull audio separately (AudioSet/YouTube-based). ([Hugging Face][15])
* WavCaps explicitly restricts usage to research/academic and provides links rather than redistributing all audio directly. ([Hugging Face][11])

---

#### Image-caption / image-text datasets (image↔text)

With LLaVA-Pretrain downloaded, there is enough image-text signal to begin. Additional clean caption corpora (COCO captions, CC3M/CC12M, etc.) can be added later; they are omitted here to keep the list scoped by license simplicity vs scale.

---

#### Multimodal interleaved instruction data (teaches conversions + “chatty omni” behavior)

* **AnyInstruct (108k)** is a strong “format reference” for interleaving modalities and conversion tasks. ([Hugging Face][5])

Pull:

```bash
huggingface-cli download OpenMOSS-Team/AnyInstruct --repo-type dataset --local-dir ./data/instruct/AnyInstruct
```

---

### B3) Codebases worth cloning (even if you don’t adopt them)

These are “reference implementations” for Type‑3 mechanics: token interleaving, modality tags, decoding pathways, etc.

Pull:

```bash
# AnyGPT (very relevant)
git clone https://github.com/OpenMOSS/AnyGPT.git

# SEED (tokenizer + detokenizer + training notes)
# (HF repo also exists; pull it via git-lfs or use HF download)
# Use the SEED HF hub or project page for instructions.
```

SEED HF hub explicitly states it releases tokenizer/de-tokenizer + related checkpoints. ([Hugging Face][16])

Also:

```bash
# MIO (multimodal tokens foundation model)
git clone https://github.com/mio-team/mio.git
```

([GitHub][6])

---

## C. A single “pull script” to copy/paste

This gives you a coherent local layout and pulls the core reference stack + tokenizers + datasets.

```bash
set -e
mkdir -p ./refs ./models ./tokenizers ./data

# ---------- Option A references ----------
mkdir -p ./refs/OmniServe
git clone https://github.com/NAVER-Cloud-HyperCLOVA-X/OmniServe.git ./refs/OmniServe

mkdir -p ./refs/AnyGPT
git clone https://github.com/OpenMOSS/AnyGPT.git ./refs/AnyGPT

mkdir -p ./refs/MIO
git clone https://github.com/mio-team/mio.git ./refs/MIO

# HyperCLOVA X SEED Omni (license-gated)
mkdir -p ./models/HyperCLOVAX-SEED-Omni-8B
huggingface-cli download naver-hyperclovax/HyperCLOVAX-SEED-Omni-8B \
  --local-dir ./models/HyperCLOVAX-SEED-Omni-8B

# AnyGPT weights
huggingface-cli download fnlp/AnyGPT-base --local-dir ./models/AnyGPT-base
huggingface-cli download fnlp/AnyGPT-chat --local-dir ./models/AnyGPT-chat
huggingface-cli download fnlp/AnyGPT-speech-modules --local-dir ./models/AnyGPT-speech-modules

# MIO weights
huggingface-cli download m-a-p/MIO-7B-Base --local-dir ./models/MIO-7B-Base
huggingface-cli download m-a-p/MIO-7B-Instruct --local-dir ./models/MIO-7B-Instruct

# ---------- Option B tokenizers ----------
mkdir -p ./tokenizers/image ./tokenizers/audio
huggingface-cli download AILab-CVC/seed-tokenizer-2 --local-dir ./tokenizers/image/seed-tokenizer-2
huggingface-cli download facebook/encodec_48khz --local-dir ./tokenizers/audio/encodec_48khz

# ---------- Datasets ----------
mkdir -p ./data/audio ./data/music ./data/instruct
huggingface-cli download cvssp/WavCaps --repo-type dataset --local-dir ./data/audio/WavCaps
huggingface-cli download d0rj/audiocaps --repo-type dataset --local-dir ./data/audio/AudioCaps
huggingface-cli download google/MusicCaps --repo-type dataset --local-dir ./data/music/MusicCaps
huggingface-cli download OpenMOSS-Team/AnyInstruct --repo-type dataset --local-dir ./data/instruct/AnyInstruct
```

---

## D. What I’d pull first (if you want the shortest path to “Type‑3 working prototype”)

Minimum set that unlocks Type‑3 experimentation:

1. **AnyGPT repo + AnyInstruct + seed-tokenizer-2 + encodec**
2. Your chosen **text LLM backbone** (e.g., Qwen3‑8B‑Base)
3. **WavCaps + AudioCaps + MusicCaps** (metadata + captions) to start audio↔text learning
4. Optional: **taming-transformers** only if you want a VQGAN baseline tokenizer path

This gives you the complete toolbox: tokenize → interleave → train → detokenize.

---

## One small question (only because it changes the pull list a lot)

For your Type‑3 build, are you targeting:

* **(1) speech-only audio** first (TTS/ASR style), or
* **(2) general audio + music** from the start?

If you say “speech-only first,” we prioritize SpeechTokenizer/MLS/FLEURS; if you say “general+music,” we prioritize EnCodec + MusicCaps + WavCaps and will add a music/vocal decoder path early.

Either way, the resource list above remains valid; this just changes what you download *next*.

[1]: https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Omni-8B "naver-hyperclovax/HyperCLOVAX-SEED-Omni-8B · Hugging Face"
[2]: https://github.com/NAVER-Cloud-HyperCLOVA-X/OmniServe?utm_source=chatgpt.com "NAVER-Cloud-HyperCLOVA-X/OmniServe"
[3]: https://huggingface.co/OpenMOSS-Team/AnyGPT-base "OpenMOSS-Team/AnyGPT-base · Hugging Face"
[4]: https://huggingface.co/OpenMOSS-Team/AnyGPT-base?utm_source=chatgpt.com "OpenMOSS-Team/AnyGPT-base"
[5]: https://huggingface.co/datasets/OpenMOSS-Team/AnyInstruct?utm_source=chatgpt.com "OpenMOSS-Team/AnyInstruct · Datasets at Hugging Face"
[6]: https://github.com/mio-team/mio?utm_source=chatgpt.com "MIO-Team/MIO - A Foundation Model on Multimodal Tokens"
[7]: https://huggingface.co/m-a-p/MIO-7B-Base "m-a-p/MIO-7B-Base · Hugging Face"
[8]: https://huggingface.co/meituan-longcat/LongCat-Flash-Omni?utm_source=chatgpt.com "meituan-longcat/LongCat-Flash-Omni"
[9]: https://github.com/CompVis/taming-transformers "GitHub - CompVis/taming-transformers: Taming Transformers for High-Resolution Image Synthesis"
[10]: https://github.com/facebookresearch/audiocraft?utm_source=chatgpt.com "facebookresearch/audiocraft"
[11]: https://huggingface.co/datasets/cvssp/WavCaps?utm_source=chatgpt.com "cvssp/WavCaps · Datasets at Hugging Face"
[12]: https://audiocaps.github.io/?utm_source=chatgpt.com "AudioCaps: Generating Captions for Audios in the Wild ..."
[13]: https://arxiv.org/abs/1910.09387?utm_source=chatgpt.com "Clotho: An Audio Captioning Dataset"
[14]: https://huggingface.co/datasets/google/MusicCaps?utm_source=chatgpt.com "google/MusicCaps · Datasets at Hugging Face"
[15]: https://huggingface.co/datasets/jp1924/AudioCaps/tree/main?utm_source=chatgpt.com "jp1924/AudioCaps at main"
[16]: https://huggingface.co/AILab-CVC/SEED?utm_source=chatgpt.com "AILab-CVC/SEED"
Yes — if your end goal is a **Type‑3 (“discrete tokens for modalities + one transformer core”)** system that can eventually *generate* audio (including music), then it’s reasonable to include **general audio + music from day one**. The key is to **standardize the audio codec/tokenizer and clip policy up front**, otherwise you’ll end up with incompatible datasets (different sample rates, durations, stereo/mono assumptions, codebook layouts, etc.) and your training mix becomes messy.

Below I’ll do three things:

1. Answer your connector questions (SigLIP, “is this how real multimodal works?”).
2. Situate “frontier” systems in the public taxonomy (without guessing proprietary internals).
3. Give you a **concrete, prioritized “resources to pull” list** for **Option B (Type‑3 tokens)**, while also telling you what to download for **Option A references**.

---

## 1) SigLIP/SigLIP‑style towers: do they output text? What’s the connector for?

A SigLIP (or CLIP/SigLIP‑family) **vision encoder does not output text**. It outputs **continuous embeddings** (feature vectors) for the image (often per patch + pooled features). Those embeddings live in *the vision model’s representation space*, not in your LLM’s token embedding space.

The **connector** (projector + optionally a resampler/Q‑Former) is valuable because it:

* **Maps** vision features → **LLM “token-like” embeddings** (same hidden size as the LLM).
* **Controls token budget** (e.g., compress to 32/64/128 “visual tokens” per image).
* Lets you keep the vision tower and LLM **frozen** early (cheap/stable), then later unfreeze selectively.
* Gives a clean abstraction boundary so vision towers can be swapped without rewriting the LLM.

This “frozen tower + learned connector + LLM” pattern is absolutely a real-world workhorse for **VLM chat** models because it is compute-efficient and modular.

---

## 2) Is the connector pattern “how real multimodal systems work”?

It’s one of the dominant patterns for **image → text** (VLM chat). But once you want **generation in other modalities (audio/image)**, you typically move toward either:

* **Discrete-token modality modeling** (Type‑3): can *generate* image/audio tokens and decode them back.
* **Hybrid objectives**: text is autoregressive; images/audio may be diffusion or separate decoders (e.g., “Transfusion” / “Show‑o” style). ([arXiv][1])
* **Cross-attention** (Flamingo-style) or **Q‑Former** (BLIP‑2 style) remain common when your target is still **text**; they’re not inherently designed for **audio waveform** generation.

So: **connector-based VLM is “real”** and often the most practical first milestone, but **Type‑3 is the more “omni” direction** when you want unified *input+output* across modalities.

---

## 3) Where do “frontier” models sit (Gemini/GPT-class) in this taxonomy?

Publicly, we **do not have enough detail** to place proprietary systems precisely into one bucket (connector vs cross-attn vs tokenization). What we *can* say reliably:

* Capabilities like **strong OCR-like reading without explicit OCR** and **geo-location cues** can be achieved with **standard vision encoders + strong training mixtures** (high-res, doc/scene-text data, geo data, instruction tuning) — discrete tokenization is *not required* for that.
* “Omni” experiences (speech in/out, fast latency, duplex) often imply **separate audio encoders/decoders** plus a core model that coordinates them; some open systems explicitly use modular designs.

**Public, inspectable** systems in the “multimodal tokens” family include the open projects below.

---

# 4) Resources to pull (your request): **Option B (Type‑3) + also download Option A references**

Assumption: “go with Option B but download Option A as reference” and “general audio + music from the start.” The following list reflects that scope:

* **A. Reference implementations (Option A study set)**
* **B. Core building blocks for Option B (tokenizers/codecs)**
* **C. Datasets (audio + music, plus audio‑visual where useful)**
* **D. Evaluation tools/models**

**P0 / P1 / P2** priority markers are included to stage downloads.

---

## A) Option A references (download to study; don’t copy design blindly)

### A1) AnyGPT (OpenMOSS) — multimodal tokens (text + images + speech; music via EnCodec)

* Code + docs: AnyGPT repository and model card list the required weights:

  * `fnlp/AnyGPT-base`, `fnlp/AnyGPT-chat`
  * `fnlp/AnyGPT-speech-modules`
  * `AILab-CVC/seed-tokenizer-2` ([Hugging Face][2])
* Their docs also mention that **unCLIP SD‑UNet weights** (for image reconstruction) and **Encodec‑32k** (for music tokenization/reconstruction) are used/downloaded as part of setup. ([GitHub][3])

**Why pull it:** It is a very concrete “Type‑3-ish” reference that mixes **discrete image tokens** (SEED) and **audio tokens** (EnCodec), and it shows practical token detokenization paths.

**Priority:** **P0 reference**

---

### A2) HyperCLOVAX‑SEED‑Omni‑8B + OmniServe (another public “omni” reference)

The Hugging Face model card describes it as a multimodal LLM (image + speech + text) and points to an inference/serving stack (OmniServe). ([Hugging Face][4])

**Priority:** **P1 reference** (good to read, but less essential than AnyGPT for “token-based DIY”)

---

### A3) MIO (MIO‑7B) — “multimodal tokens” foundation model (public code + weights)

* Code: MIO-Team GitHub repo ([GitHub][5])
* Weights on HF: `m-a-p/MIO-7B-Base` and `m-a-p/MIO-7B-Instruct` ([Hugging Face][6])
* Paper describes “trained on a mixture of discrete tokens across modalities.” ([arXiv][7])

**Priority:** **P1 reference** (very relevant conceptually; good to compare training stage design)

---

## B) Option B core building blocks (what you actually build with)

For “general audio + music from the start,” the biggest decision is **audio tokenization**. I strongly recommend you pick **one** codec/tokenizer and enforce a consistent preprocessing pipeline.

### B1) Audio tokenizer / codec (pick one as your “language of sound”)

#### Choice 1 (recommended): **EnCodec** (Meta)

* EnCodec models exist on Hugging Face, including `facebook/encodec_32khz` and `facebook/encodec_48khz`. ([Hugging Face][8])
* MusicGen (and AudioCraft) are built around EnCodec tokenization; it’s a practical, well-tested path for music/audio generation. ([Hugging Face][9])

**Why this fits “audio + music”:**

* MusicGen is evidence that EnCodec tokens can drive high-quality music generation. ([Hugging Face][9])
* You can also handle general audio/sfx with the same codec.

**Priority:** **P0**

#### Choice 2: **DAC (Descript Audio Codec)**

* Descript’s DAC is another audio codec option with HF models. ([Hugging Face][10])

**Priority:** **P1** (good alternative; pick only if you have a reason)

---

### B2) Image tokenizer (for *image generation* in a Type‑3 core)

You have two realistic routes:

#### Route 1: **SEED tokenizer (discrete visual tokens designed for LLM compatibility)**

* SEED is explicitly a **discrete image tokenizer** intended to make LLMs “see and draw” via next-token prediction on visual tokens. ([AI Lab CVC][11])
* SEED-LLaMA work highlights the tokenizer/detokenizer concept. ([ICLR Proceedings][12])
* AnyGPT explicitly points you to `AILab-CVC/seed-tokenizer-2` for image tokenization. ([Hugging Face][13])

**Priority:** **P0** if you want “modern” discrete image tokens.

#### Route 2: **VQGAN (Taming Transformers)**

* Classic “discrete image codes + transformer” approach. ([arXiv][14])
* You can grab common checkpoints like `boris/vqgan_f16_16384` (widely used; easiest to make work end-to-end). ([GitHub][15])

**Priority:** **P1** (simpler engineering; good baseline; tokens are longer / less semantically aligned than SEED-style)

---

### B3) Audio/music reference generators (very useful for baselines and debugging)

* **AudioCraft** repo (Meta) is the main codebase that includes MusicGen/AudioGen tooling. ([GitHub][16])
* **MusicGen weights**: `facebook/musicgen-small`, `facebook/musicgen-medium`, `facebook/musicgen-large`, etc. ([Hugging Face][9])
* **AudioGen weights** (general audio): e.g., `facebook/audiogen-medium` and related models. ([Hugging Face][17])

**Priority:** **P0/P1** (not required to train your own model, but invaluable for sanity checks)

---

### B4) Audio-text embedding model for evaluation: **CLAP**

CLAP is to audio what CLIP is to images; it gives you:

* retrieval-style metrics (“does audio match the text prompt?”),
* zero-shot classification baselines.

Model cards:

* `laion/clap-htsat-fused` ([Hugging Face][18])
* `laion/larger_clap_general` and variants ([Hugging Face][19])
  CLAP is associated with the LAION-Audio-630K dataset in the original paper. ([arXiv][20])

**Priority:** **P0**

---

## C) Datasets to pull (audio + music, from the start)

I’ll split these into **captioned** vs **tagged/weak** vs **audio-visual**.

### C1) Captioned audio (general sounds) — best for “learn audio→text grounding”

**P0 set (start here):**

* **Clotho v2.x** (audio captioning dataset on Zenodo). ([Zenodo][21])
* **AudioCaps** (audio captioning dataset; commonly used; available via HF). ([Hugging Face][22])
* **WavCaps** (~403k audio-text pairs; large-scale caption set). ([Hugging Face][23])

**Why these first:** They give you clean “audio ↔ text” supervision for alignment and instruction tuning.

---

### C2) Large-scale audio-text (bigger, noisier) — good for scaling

* **LAION‑Audio‑630K** (633k audio-text pairs; paper + community mirrors). ([arXiv][20])

  * Note: LAION’s repo indicates that **Freesound** portion is released with audio+captions on HF, while other sources may require separate scraping/handling. ([GitHub][24])

**Priority:** **P1** (pull after the P0 caption sets work end-to-end)

---

### C3) Music captioning (this is the “music from the start” backbone)

You have three tiers:

#### Tier M1 (small but high-quality human captions)

* **MusicCaps** (5,521 examples with an aspect list + free text caption). ([Hugging Face][25])

  * Practical caveat: MusicCaps is tied to YouTube examples; many public scripts treat it as metadata and download audio separately (ensure you comply with source terms and licensing). ([Gist][26])

**Priority:** **P1** (excellent supervision, but operationally annoying if audio cannot be sourced legally/cleanly)

#### Tier M2 (fully downloadable audio + captions, permissive)

* **Song Describer Dataset (SDD)** — ~706 permissively licensed music recordings with ~1.1k captions; released via Zenodo and mirrored. ([arXiv][27])

**Priority:** **P0 for music** (because it’s clean and actually downloadable)

#### Tier M3 (scale via synthetic captions over open music corpora)

* **Jamendo‑FMA‑captions** — synthetic captions for MTG‑Jamendo + FMA (captioned with SALMONN then refined). ([Hugging Face][28])

**Priority:** **P1/P2** depending on your tolerance for synthetic text and license constraints inherited from sources.

---

### C4) Open music corpora (tags, not captions) — useful for *conditioning and diversity*

* **MTG‑Jamendo** (55k+ CC-licensed tracks with tags for genre/instrument/mood). ([MTG][29])

Tags can be converted into text prompts (either templated or model-generated) to make pseudo-caption training data.

**Priority:** **P1** (very good to have for music coverage)

---

### C5) “Audio-visual” datasets (optional, but useful if you truly want image+audio grounding)

* **VGGSound** (audio-visual clips from YouTube; usually distributed as metadata + scripts; you sample frames as images). ([GitHub][30])

**Priority:** **P2** (only after audio-only + image-only pipelines are stable)

---

### C6) Extra: labeled general audio (classification, tags)

* **FSD50K** (Freesound subset with AudioSet-style labels; CC-licensed content). ([Freesound Annotator][31])

**Priority:** **P1** (good for “what sound is this?” instruction data)

---

## D) “Pull list” to execute (models + repos)

Below is a practical list of what to download first. This is not the only way to do it, but it’s a clean P0/P1 staging.

### P0 (minimum viable Type‑3 kit: audio+music + evaluation + references)

**Models / weights**

* Qwen3 base (your text core): `Qwen/Qwen3-8B-Base` ([Hugging Face][32])
* Audio codec: `facebook/encodec_32khz` (or `facebook/encodec_48khz`) ([Hugging Face][8])
* Image tokenizer: `AILab-CVC/seed-tokenizer-2` ([Hugging Face][13])
* Eval embedder: `laion/clap-htsat-fused` (or `laion/larger_clap_general`) ([Hugging Face][18])

**Reference repo**

* AnyGPT (code + weight list) ([Hugging Face][2])

**Datasets (start small and clean)**

* Clotho (audio captioning) ([Zenodo][21])
* Song Describer Dataset (music captions with downloadable audio) ([Zenodo][33])

---

### P1 (scale up)

**Datasets**

* WavCaps ([Hugging Face][23])
* AudioCaps ([Hugging Face][22])
* LAION‑Audio‑630K (where available) ([arXiv][20])
* MTG‑Jamendo (tags) ([MTG][29])
* Jamendo‑FMA‑captions (synthetic captions) ([Hugging Face][28])

**More references**

* MIO code + weights ([GitHub][5])
* AudioCraft + MusicGen/AudioGen weights ([GitHub][16])

---

### P2 (audio-visual / YouTube-sourced)

* VGGSound ([GitHub][30])
* MusicCaps (if audio can be sourced legally/cleanly in the current environment) ([Hugging Face][25])

---

## 5) Direct answer to “(2) general audio + music from the start?”

If you do it, do it like this (so you don’t paint yourself into a corner):

* Treat “audio” as one modality, but **tag each sample** with a subtype: `speech | sfx | music`.
* Use **one codec** (EnCodec recommended) and **one clip policy** early:

  * e.g., fixed **10s** segments for caption datasets (matches MusicCaps’s typical 10s concept) ([Hugging Face][25])
  * fixed **mono/stereo** policy (don’t mix accidentally)
* Start with **captioned general audio (Clotho/AudioCaps/WavCaps)** plus **captioned music (Song Describer)**, then scale music coverage via **MTG‑Jamendo (+ synthetic captions)**.

This way, your core model learns:

* audio ↔ text grounding,
* music-specific descriptive language,
* image tokens can be added later without redesigning the audio stack.

---

## Optional: “download + verify” checklist tailored to the folder layout

Example outputs for a checklist:

* a single `resources_pull.sh` that downloads all **P0** weights and creates dataset folder stubs,
* a “dataset license ledger” template (so open-sourcing later is clean),
* a minimal smoke test script: encode/decode audio tokens + compute CLAP similarity for (audio,text).

But as a first step, the **P0 kit above** is the right “resource pull” set for **Type‑3 with audio+music from day one**.

[1]: https://arxiv.org/abs/2408.11039?utm_source=chatgpt.com "Transfusion: Predict the Next Token and Diffuse Images ..."
[2]: https://huggingface.co/OpenMOSS-Team/AnyGPT-base?utm_source=chatgpt.com "OpenMOSS-Team/AnyGPT-base"
[3]: https://github.com/OpenMOSS/AnyGPT?utm_source=chatgpt.com "Code for \"AnyGPT: Unified Multimodal LLM with Discrete ..."
[4]: https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Omni-8B?utm_source=chatgpt.com "naver-hyperclovax/HyperCLOVAX-SEED-Omni-8B"
[5]: https://github.com/mio-team/mio "GitHub - MIO-Team/MIO: MIO: A Foundation Model on Multimodal Tokens"
[6]: https://huggingface.co/m-a-p/MIO-7B-Base "m-a-p/MIO-7B-Base · Hugging Face"
[7]: https://arxiv.org/abs/2409.17692?utm_source=chatgpt.com "MIO: A Foundation Model on Multimodal Tokens"
[8]: https://huggingface.co/facebook/encodec_32khz/blame/2f57de751f17e7d69663025c4005d94657fd00ee/README.md?utm_source=chatgpt.com "README.md · facebook/encodec_32khz at ..."
[9]: https://huggingface.co/facebook/musicgen-large?utm_source=chatgpt.com "facebook/musicgen-large"
[10]: https://huggingface.co/descript/descript-audio-codec?utm_source=chatgpt.com "Descript Audio Codec"
[11]: https://ailab-cvc.github.io/seed/?utm_source=chatgpt.com "SEED Multimodal"
[12]: https://proceedings.iclr.cc/paper_files/paper/2024/file/97011c648eda678424f9292dadeae72e-Paper-Conference.pdf?utm_source=chatgpt.com "MAKING LLAMA SEE AND DRAW WITH SEED TOKENIZER"
[13]: https://huggingface.co/OpenMOSS-Team/AnyGPT-chat?utm_source=chatgpt.com "OpenMOSS-Team/AnyGPT-chat"
[14]: https://arxiv.org/abs/2012.09841?utm_source=chatgpt.com "Taming Transformers for High-Resolution Image Synthesis"
[15]: https://github.com/CompVis/taming-transformers?utm_source=chatgpt.com "Taming Transformers for High-Resolution Image Synthesis"
[16]: https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md?utm_source=chatgpt.com "audiocraft/docs/MUSICGEN.md at main"
[17]: https://huggingface.co/facebook/audiogen-medium?utm_source=chatgpt.com "facebook/audiogen-medium"
[18]: https://huggingface.co/laion/clap-htsat-fused?utm_source=chatgpt.com "laion/clap-htsat-fused"
[19]: https://huggingface.co/laion/larger_clap_general?utm_source=chatgpt.com "laion/larger_clap_general"
[20]: https://arxiv.org/abs/2211.06687?utm_source=chatgpt.com "Large-scale Contrastive Language-Audio Pretraining with ..."
[21]: https://zenodo.org/records/4783391?utm_source=chatgpt.com "Clotho dataset"
[22]: https://huggingface.co/api/resolve-cache/models/OpenMOSS-Team/AnyGPT-chat/c8999624b077b3837f32c3948f0f9e90bd67b860/README.md?download=true&etag=%2226630dc23c16d2e397570b4ce9c6ea11a99c6fd6%22&utm_source=chatgpt.com "https://huggingface.co/api/resolve-cache/models/Op..."
[23]: https://huggingface.co/datasets/cvssp/WavCaps?utm_source=chatgpt.com "cvssp/WavCaps · Datasets at Hugging Face"
[24]: https://github.com/LAION-AI/audio-dataset/blob/main/laion-audio-630k/README.md?utm_source=chatgpt.com "audio-dataset/laion-audio-630k/README.md at main"
[25]: https://huggingface.co/datasets/google/MusicCaps?utm_source=chatgpt.com "google/MusicCaps · Datasets at Hugging Face"
[26]: https://gist.github.com/nateraw/14818c6e4889fc13698e5ca1a2ace2ad?utm_source=chatgpt.com "Download the underlying audio files of the MusicCaps ..."
[27]: https://arxiv.org/abs/2311.10057?utm_source=chatgpt.com "The Song Describer Dataset: a Corpus of Audio Captions ..."
[28]: https://huggingface.co/datasets/disco-eth/jamendo-fma-captions?utm_source=chatgpt.com "disco-eth/jamendo-fma-captions · Datasets at ..."
[29]: https://mtg.github.io/mtg-jamendo-dataset/?utm_source=chatgpt.com "The MTG-Jamendo Dataset"
[30]: https://github.com/hche11/VGGSound?utm_source=chatgpt.com "VGGSound: A Large-scale Audio-Visual Dataset"
[31]: https://annotator.freesound.org/fsd/release/FSD50K/?utm_source=chatgpt.com "FSD50K"
[32]: https://huggingface.co/Qwen/Qwen3-8B-Base?utm_source=chatgpt.com "Qwen/Qwen3-8B-Base"
[33]: https://zenodo.org/records/10072001?utm_source=chatgpt.com "Song Describer Dataset"

## Checklist

- [x] Clone core reference repos into `refs/stage3` (OmniServe, AnyGPT, MIO, AudioCraft, Taming-Transformers, Descript-Audio-Codec).
- [x] Confirm which HF weights/datasets you want locally and whether auth tokens are available.
- [x] Pull image tokenizer weights (SEED tokenizer v2) into `tokenizers/image/`.
- [x] Pull audio tokenizer weights (EnCodec 48kHz) into `tokenizers/audio/`.
- [x] Mirror AnyInstruct into `data/` for preprocessing.
- [x] Mirror audio-caption datasets (WavCaps/AudioCaps/Clotho) into `data/`.
- [x] Pull model weights into `models/` (AnyGPT base/chat + speech modules, MIO-7B Base/Instruct, HyperCLOVAX-SEED-Omni-8B).
