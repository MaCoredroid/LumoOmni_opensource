# Stage 3 Deliverable (Checkpoint 7000)

This deliverable packages the Stage‑3 adapter as **trainable‑rows‑only weights**. It is meant to be applied on top of the base Qwen3 model.

## Delivery checkpoint

- Adapter checkpoint: `outputs/stage3_token_lm_iter5/checkpoint_7000`
- Token space: `outputs/stage3_token_lm/token_space.json`
- Eval (full eval set): `outputs/stage3_token_lm_iter5/eval_step7000.json`

## What is saved

Inside `checkpoint_7000/`:

- `trainable_rows.pt` — only the new modality rows for:
  - input embedding
  - lm_head
- `trainable_rows.json` — metadata (row ranges, base model id)
- `token_space.json` — modality ranges and vocab sizes

> Base model weights are **not** saved. They are loaded from the base LLM and patched with the trainable rows.

## How to load (Python)

```python
import torch
from transformers import AutoModelForCausalLM
from stage3_uti.tokenization.token_space import TokenSpace
from stage3_uti.utils.train_utils import _load_trainable_rows, _resize_and_init_embeddings, resolve_pad_id

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

## Eval result (epoch‑0, step 7000)

From `outputs/stage3_token_lm_iter5/eval_step7000.json`:

- loss(all)=2.7462
- loss(a2t)=2.3606
- loss(i2t)=3.3357

## Notes

- This is an **adapter‑style** checkpoint; optimizer state is not included.
- Use `outputs/stage3_token_lm/token_space.json` to keep modality ranges consistent.
- If you need a full model snapshot, load base + patch rows, then `save_pretrained` to a new directory.
