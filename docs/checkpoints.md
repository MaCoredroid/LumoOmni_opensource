# Checkpoints & weights

This page documents what artifacts exist, how they are saved, and how to load them.

---

## Track A — VLM (connector checkpoints)

### What is saved

In early stages, checkpoints typically include:
- connector weights (resampler + projector)
- training metadata (step, optimizer if desired)
- pointers to base model versions

The base LLM + vision tower are usually loaded from their original sources and kept frozen (Stage 1).

### Typical paths (examples)

- `outputs/stage1_align_trial/checkpoint_3675.pt`
- `outputs/stage1_align_p11/...` (scaled Stage‑1.1 runs)

> Adjust paths to match your repo layout. The important idea is “connector‑only” or “frozen‑tower” checkpoints until SFT.

---

## Track B — Token LM (“trainable rows” adapters)

### Stage‑3 deliverable format

A Stage‑3 adapter checkpoint is packaged as **trainable‑rows‑only** weights:

Inside `checkpoint_7000/`:
- `trainable_rows.pt` — modality rows for input embedding + LM head
- `trainable_rows.json` — row ranges + base model id
- `token_space.json` — global modality ranges + vocab sizes

Base model weights are **not** saved; you load them and patch the new rows.

### Loading example

```python
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM
from stage3_uti.tokenization.token_space import TokenSpace
from stage3_uti.utils.train_utils import (
    _load_trainable_rows,
    _resize_and_init_embeddings,
    resolve_pad_id,
)

base_llm = "Qwen/Qwen3-8B-Base"
ckpt_dir = "outputs/stage3_token_lm_iter5/checkpoint_7000"

# Load token space
space = TokenSpace.load_json("outputs/stage3_token_lm/token_space.json")
text_vocab = int(space.text_vocab_size)
vocab_total = int(space.vocab_size_total)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    base_llm,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)

# Resize to include modality ranges
_resize_and_init_embeddings(
    model,
    text_vocab_size=text_vocab,
    vocab_size_total=vocab_total,
    init_new_rows=True,
)
model.config.pad_token_id = resolve_pad_id(space)

# Patch in the trained rows
_load_trainable_rows(
    model,
    Path(ckpt_dir),
    row_start=text_vocab,
    row_end=vocab_total,
)

model.eval()
```

### Why this format exists

- It keeps deliverables small.
- It enforces ABI stability via `token_space.json`.
- It isolates what was actually trained in Stage 3.

---

## Compatibility rules (strongly recommended)

1. **Always ship `token_space.json` with any token‑LM checkpoint.**
2. Verify the token space hash matches the dataset you train/evaluate on.
3. Record codec/tokenizer versions used to generate tokens.
