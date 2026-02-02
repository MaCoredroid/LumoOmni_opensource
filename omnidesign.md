Below is a research‑project stage plan for **Option B: Type‑3 discrete tokenization** with **image + audio/music**, using what you already have:

* **Core compute**: DGX Spark
* **Tokenizers/decoders**: SEED image tokenizer (encode/decode) + EnCodec audio codec (encode/decode)
* **Datasets**: audio captioning (Clotho/AudioCaps/WavCaps), music captioning (Song Describer, etc.), multimodal instruction (AnyInstruct), plus your existing image-text resources
* **References**: AnyGPT / MIO / HyperCLOVA Omni (as “known working” token‑omni systems for sanity checks)

The plan is deliberately staged so every step has: **(a) deliverables, (b) acceptance tests, (c) failure modes**, and the stages are ordered so you avoid the common Type‑3 dead-ends (token length blowups, token ID collisions, decoders that don’t invert, and models that “ignore” non-text tokens).

---

## Project definition

### Target capability (Option B, Type‑3)

A single Transformer core (starting from a text LLM backbone) that can:

1. **Read**:

   * image tokens + text tokens → generate text
   * audio tokens + text tokens → generate text
   * image + audio + text → generate text

2. **Generate**:

   * text → image tokens → decode to image
   * text → audio tokens → decode to waveform (general audio + music)
   * (later) image+text → audio, audio+text → image (optional but natural once tri-modal training is stable)

### Non-goals (at least initially)

* “Frontier quality” image generation (photorealistic) and “Suno‑class” music coherence in one jump.
* Multi-minute audio. Keep clips short until the token economy and training stability are proven.
* Perfect OCR/geolocation. Those are data/scale problems more than architecture problems.

---

## Core design choices to lock early

Type‑3 projects fail most often because these choices are made late and inconsistently.

### 1) Canonical tokenizers and clip policies

* **Audio tokenizer**: EnCodec (you already pulled it).
* **Audio clip policy** (must be consistent across datasets):

  * fixed sample rate (choose one; do not mix)
  * fixed duration window (e.g., 5–10s to start; long music later)
  * fixed channel policy (mono or stereo; pick one)
* **Image tokenizer**: SEED tokenizer (you already pulled it).
* **Image policy**:

  * fixed resolution for tokenization (keep token length constant at first)
  * fixed color space (RGB)

### 2) Unified token space (ID layout)

You need a deterministic, collision-free mapping:

* Base text tokenizer IDs: `[0 … V_text-1]`
* Reserve a block for **special control tokens** (boundaries and task selectors):

  * `<|img_start|>`, `<|img_end|>`, `<|aud_start|>`, `<|aud_end|>`, `<|text_start|>`, `<|text_end|>`
  * `<|gen_img|>`, `<|gen_aud|>`, `<|gen_text|>`
  * optional: `<|aud_cb0|> ...` (if you want explicit codebook markers)
* Reserve contiguous blocks for:

  * **Image codebook tokens**: `IMG_0 … IMG_{V_img-1}`
  * **Audio tokens**: either

    * (recommended) **separate ranges per codebook stream**: `AUD0_*`, `AUD1_*`, `AUD2_*`, `AUD3_*`
      (avoids interference and makes reshaping trivial), or
    * one combined range with offsets encoded in the token id (works but is easier to mess up)

This layout is the backbone of everything: training, decoding, evaluation, and checkpoint portability.

### 3) How you serialize audio tokens

EnCodec yields **multiple codebooks per time step**. Pick one canonical serialization:

* **Flatten-by-time** (recommended):

  * for each frame `t`, append `[cb0[t], cb1[t], cb2[t], cb3[t]]`
  * decode by reshaping back to `[num_codebooks, num_frames]`

This is easy to implement and works with a single autoregressive Transformer.

### 4) Model head(s): unified LM head vs modality heads

Two viable designs:

* **Unified LM head** over the full combined vocabulary (simpler, first pass).
* **Modality-specific output heads** (more engineering, often cleaner optimization):

  * text head predicts only text vocab
  * image head predicts only image tokens
  * audio head predicts only audio tokens

For a research build, I recommend:

* start with **unified LM head** for velocity,
* keep the code structured so modality heads can be swapped in if interference becomes a limiting factor.

### 5) Training objective(s)

Start simple:

* **Autoregressive next-token prediction** over a single stream with boundaries and a task token that determines which modality is generated.

Later upgrades (optional):

* masked/denoising objectives for image/audio tokens (MaskGIT-style) once you have a working baseline.

---

# Staged research plan

## Stage 0 — Reproducibility + “reference runs”

**Goal**: Ensure your environment, decoders, and reference baselines are callable end-to-end.

### Deliverables

* `env/` with pinned versions (torch, transformers, tokenizers, audiocraft/encodec deps)
* Scripts that run inference on **Option A reference models** you downloaded (AnyGPT/MIO/Omni) and produce:

  * one generated audio clip + prompt
  * one generated image + prompt (if supported)
  * one audio caption output

### Acceptance tests

* You can encode and decode:

  * audio → tokens → audio (round-trip without errors)
  * image → tokens → image (round-trip without errors)
* You can run a reference model and produce a non-empty output artifact.

### Typical failure modes

* tokenizers installed but decoder mismatch (wrong checkpoint / wrong expected shape)
* audio codec expects a different sample rate than your preprocessing

---

## Stage 1 — Build the **Unified Token Interface** (UTI)

**Goal**: One internal representation and I/O contract that everything else uses.

### Deliverables

* `unified_tokenizer.py` with:

  * `encode_text(str) -> List[int]`
  * `encode_image(PIL/np) -> (List[int], meta)`  (meta includes grid shape if needed)
  * `encode_audio(wav) -> (List[int], meta)`    (meta includes codebooks/frames)
  * `decode_image(tokens, meta) -> image`
  * `decode_audio(tokens, meta) -> wav`
* A **token ID allocator** that:

  * assigns contiguous ranges,
  * validates no collisions,
  * writes `token_space.json` into checkpoints.

### Acceptance tests

* Deterministic encode/decode for fixed inputs
* “Shape sanity”:

  * audio token count corresponds to duration and codec settings
  * image token count corresponds to chosen resolution

### Typical failure modes

* off-by-one offsets, collisions, or silently mismatched vocab sizes across runs

---

## Stage 2 — Data normalization and “tokenized shards”

**Goal**: Convert all datasets into a single normalized format and optionally cache tokens.

### Deliverables

A unified dataset schema (JSONL or parquet), e.g.:

```json
{
  "id": "...",
  "modalities": {
    "image": {"path": "...", "meta": {...}},
    "audio": {"path": "...", "meta": {...}}
  },
  "text": {
    "caption": "...",
    "instruction": "...", 
    "response": "..."
  },
  "task": "caption_audio" | "caption_image" | "t2a" | "t2i" | "a2t" | "i2t" | ...
}
```

And **tokenized shards** (recommended) so training isn’t bottlenecked by codec inference:

* `*.tar` shards (WebDataset) or parquet with `audio_tokens`, `image_tokens`, `text_tokens`.

### Acceptance tests

* Dataset stats report:

  * distributions of token lengths per modality
  * fraction of missing/corrupt files
  * min/median/max durations and resolutions
* A “golden batch” can be loaded and passed through model input builder without any special casing.

### Typical failure modes

* mixing sample rates / durations
* dataset licensing constraints not tracked (solve via a license ledger now)

---

## Stage 3 — **Warm-start multimodal vocabulary** on a text LLM backbone

**Goal**: Teach the model that “audio/image token IDs” are meaningful, without destabilizing language.

This is the Type‑3 analogue of “connector alignment” you already mastered.

### Training approach

* Initialize from **Qwen3-8B-Base** (or a smaller Qwen3 variant for faster iteration).
* Extend embeddings and LM head for new token ranges.
* Train **only**:

  * embeddings rows for image/audio token IDs,
  * LM head rows for image/audio token IDs,
  * (optional) small LoRA on top layers if needed for stability.
* Keep the rest frozen.

### Data/tasks (use what you already downloaded)

* **Audio captioning**: (audio tokens as input) → (text as output)
* **Image captioning**: (image tokens as input) → (text as output)
* Add light “copy/denoise” tasks if you need stabilization:

  * mask small spans of modality tokens and reconstruct (optional)

### Acceptance tests

* **Ablation like you already do**, per modality:

  * audio correct vs shuffled/zero/noise affects teacher-forced loss on text
  * image correct vs shuffled/zero/noise affects teacher-forced loss on text
* Language regression check remains stable (text-only prompts unaffected).

### Typical failure modes

* model learns to ignore modality tokens because text priors dominate
* modality tokens explode context length due to too-long clips (fix via strict clip policy)

---

## Stage 4 — Joint multimodal pretraining (true Type‑3 mixed stream)

**Goal**: Train the core Transformer to generate **image tokens and audio tokens**, not only text.

### Key design: task-conditioned generation

Use explicit task tokens and boundaries so the model knows what to generate next.

Example formats:

**Text → Audio (T2A)**

```
<|text_start|> “a calm lo-fi beat with soft drums” <|text_end|>
<|gen_aud|> <|aud_start|> [AUDIO TOKENS...] <|aud_end|>
```

**Text → Image (T2I)**

```
<|text_start|> “a red car parked in snow” <|text_end|>
<|gen_img|> <|img_start|> [IMAGE TOKENS...] <|img_end|>
```

**Audio → Text (A2T)**

```
<|aud_start|> [AUDIO TOKENS...] <|aud_end|>
<|gen_text|> <|text_start|> “people cheering at a stadium” <|text_end|>
```

### What to train

* Unfreeze more capacity:

  * Connector-free Type‑3 means the Transformer must learn mixed token statistics.
  * Use LoRA first (for stability), then consider partial unfreeze later.

### Data mixture

Start with a conservative mixture:

* 40–50% audio↔text (Clotho/AudioCaps/WavCaps)
* 20–30% music captioning / music prompts (Song Describer + any music-text you have)
* 20–30% image↔text (whatever image-text you already have)

### Acceptance tests

* Audio generation sanity:

  * generated audio decodes without invalid shapes
  * CLAP similarity for (prompt, generated audio) beats a random baseline
* Image generation sanity:

  * generated image decodes
  * CLIP similarity beats random baseline (optional, if you have CLIP)
* “Mode separation”:

  * When prompted for audio, it produces audio tokens (not text tokens), and vice versa.

### Typical failure modes

* audio token degeneration (repetition, collapse) due to long sequences
* interference: adding audio generation hurts text quality (fix via schedules / LR / modality heads)

---

## Stage 5 — Instruction tuning (AnyInstruct + your own)

**Goal**: Convert the pretrained Type‑3 model into an “assistant” that can do multi-turn tasks.

### What changes

* Adopt a chat template (Qwen-style is fine), but now the content can include:

  * `<|aud_start|> ... <|aud_end|>`
  * `<|img_start|> ... <|img_end|>`

### Data

* AnyInstruct becomes your format anchor.
* Build a small “curated instruction set” for audio/music:

  * “describe this audio”
  * “classify the mood”
  * “generate a 5s drum loop in style X”
  * “make it faster / darker / add bass” (editing-like instructions; even if the model is weak, it learns the interface)

### Acceptance tests

* The model follows instructions reliably (format, modality requested).
* Ablation deltas remain non-trivial (it still uses modality tokens).

---

## Stage 6 — Quality and sampling research (where you get big jumps)

**Goal**: Improve perceptual quality without rewriting the whole system.

This is where Type‑3 systems usually need extra modeling tricks.

### Audio/music: practical upgrades

* **Codebook-aware sampling** (e.g., different temperatures per codebook stream)
* Length control and anti-loop heuristics
* Optional two-stage generation:

  * generate a coarse token stream first, refine later (if you find AR is too brittle)

### Image: practical upgrades

* If AR image tokens are too weak, consider masked iterative decoding for image tokens (still Type‑3, but not strictly AR).

### Acceptance tests

* A/B preference tests on a small curated prompt suite (human or your own judgments)
* CLAP/CLIP-based scores track improvements without becoming the only objective

---

# Cross-cutting evaluation plan (use the tools you already like)

You already do the right thing with **teacher-forced ablations**. Extend that to Type‑3:

## Required metrics (always)

1. **Token-weighted eval loss** per task type (A2T, I2T, T2A, T2I)
2. **Ablation deltas** per modality:

   * correct vs shuffled
   * correct vs zero
   * correct vs noise
3. **Truncation and label coverage** (critical once audio sequences enter)

## Generation metrics (lightweight but meaningful)

* **CLAP similarity** for audio/text prompts (prompt–generated audio)
* Optional: FAD later (once you have stable audio generation)
* Optional: CLIP similarity for image/text prompts

## Golden sets

Maintain 3 golden suites (64 each):

* `golden_a2t_64` (audio caption)
* `golden_t2a_64` (text→audio)
* `golden_t2i_64` (text→image)
  Each checkpoint logs:
* decoded audio/image artifacts
* text outputs
* scoring summaries

---

# Immediate next actions (what I recommend you do now)

1. **Lock your token policies**:

   * EnCodec variant + sample rate
   * fixed clip length for training
   * fixed image resolution for tokenization
   * unified token ID layout written to `token_space.json`

2. Implement **Stage 1 Unified Token Interface** + round-trip tests.

3. Start **Stage 2 tokenized shard building** for:

   * Clotho + Song Describer first (clean, manageable)
   * then expand to AudioCaps/WavCaps and your image-text corpus

4. Run **Stage 3 warm-start** with:

   * embeddings/head rows for modality token ranges trainable
   * the rest frozen
   * and your ablation suite as the main pass/fail gate

Once Stage 3 passes, you’ll have a credible platform to begin true Type‑3 mixed-stream pretraining for text→audio and text→image.

Optional: convert this into a concrete “experiment notebook” format (one page per stage with config templates, expected logs, and failure playbooks), aligned to the Stage 1/2 conventions (splits persisted, golden dumps, ablation logs).
