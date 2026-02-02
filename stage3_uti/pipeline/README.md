# Stage 3 Pipeline (UTI integration)

This folder is a **future Stage 3 training pipeline scaffold** that consumes the
Unified Token Interface (UTI). It does **not** touch Stage 1/2 code.

## What this provides

- `sequence_builder.py`: canonical functions to assemble **single-stream** token
  sequences using UTI special tokens and modality boundaries.
- A minimal pattern for **supervised labels** (mask non-target spans with `-100`).

## How to use (high-level)

1. Build a `UnifiedTokenizer` (with SEED + EnCodec adapters).
2. Tokenize per-sample text/image/audio into **global IDs**.
3. Use `SequenceBuilder` to assemble sequences like:

```
<|text_start|> ... <|text_end|> <|gen_aud|> <|aud_start|> ... <|aud_end|>
```

4. Use `build_supervised_labels(...)` to mask loss to the generated span.

## Tokenizing a JSONL dataset

`tokenize_dataset.py` expects a JSONL with fields:

- `id`
- `text`
- `image_path` (optional)
- `audio_path` (optional)
- `task` (`t2i`, `t2a`, `i2t`, `a2t`, or custom)

Example:

```bash
python3 -m stage3_uti.pipeline.tokenize_dataset \\
  --uti-config stage3_uti/configs/uti.yaml \\
  --input-jsonl data/stage3_samples.jsonl \\
  --output-jsonl outputs/stage3_token_lm/tokenized.jsonl
```

## Why separate from Stage 1/2

Stage 1/2 use **continuous vision embeddings** injected into Qwen3. Stage 3
switches to **discrete multimodal tokens** with a unified vocabulary. This
pipeline keeps the two systems cleanly separated.
