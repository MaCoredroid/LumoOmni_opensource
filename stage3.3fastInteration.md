Treating “1 epoch” as the unit of progress is the wrong mental model for Stage‑3. This stage is not learning general language or doing connector training; it is mostly fitting **new embedding rows (and maybe head rows)** so that the frozen Qwen backbone *reacts* to those IDs. That signal should show up **very early** if the setup is correct, and iteration should happen with **small, instrumented runs** instead of waiting 2.5 days.

Below is a concrete iteration plan that lets you validate the modeling approach fast, plus exactly how to carve a smaller dataset round from your existing JSONL.

---

## 1) What to do immediately at step ~650

Before changing anything, answer: “is the model learning to use modality tokens or ignoring them?”

### Run a quick modality ablation eval now (no need to wait to step 1000)

Pick a small fixed dev slice, e.g.:

* **audio:** 128 examples
* **image:** 128 examples

For each example, compute teacher-forced loss on the **text label region only** under:

* correct modality tokens
* shuffled modality tokens
* noise modality tokens
* (optional) constant token

If at step ~650 you already see **consistent** `L_correct < L_shuffle` and `L_correct < L_noise`, you’re on the right track and can decide whether you actually need the full dataset.

If all losses are basically identical, the model is likely ignoring modality IDs and you should iterate on the objective/format (not “just train longer”).

This single check is the fastest “go/no-go” indicator for Stage‑3.

---

## 2) You do *not* need a full epoch for Stage‑3

Given you’re only training a tiny parameter subset (new embedding/head rows), the dataset is massively redundant. A better workflow is:

* Train for a **fixed number of steps** (or a fraction of data)
* Run **ablation gap metrics**
* Stop as soon as the gap plateaus + text-only regression is clean

So instead of “1 epoch”, your unit is “**does the ablation gap exist and grow?**”

---

## 3) Fast iteration plan (3 tiers)

### Tier A — Smoke test (pipeline correctness)

Goal: prove gradients flow to the intended rows and the model *can* condition on modality IDs.

* Train set: **2,048 rows**
* Eval set: **256–512 rows**
* Steps: just enough to see loss move + ablation gap emerge (hundreds to low-thousands)

Target mixture (matches your full distribution):

* wavcaps_audioset: **791**
* audiocaps: **311**
* clotho: **156**
* llava_pretrain: **790**
  (total 2,048)

What you check:

* Only modality rows changed (embedding + head, if you train head)
* `L_correct` beats `L_noise` on a fixed dev slice

If this fails, a 2.5-day epoch won’t fix it.

---

### Tier B — “Modeling approach” iteration run (fast but meaningful)

Goal: validate that your exact formatting + masking + clip policy produces stable modality sensitivity.

* Train set: **8,192 rows**
* Eval set: **512–1,024 rows**
* Run until ablation gap stabilizes (a few thousand steps is usually enough to tell)

Target mixture (matches your full distribution):

* wavcaps_audioset: **3,162**
* audiocaps: **1,245**
* clotho: **623**
* llava_pretrain: **3,162**
  (total 8,192)

This is usually the sweet spot for rapid iteration.

---

### Tier C — Pre-full run (stability check)

Goal: make sure you’re not accidentally relying on dataset quirks and that training doesn’t drift.

* Train set: **51,200 rows**
* Eval set: **2,000 rows** (or keep full 2,635)
* Run until the acceptance metrics look “final enough”

Target mixture:

* wavcaps_audioset: **19,766**
* audiocaps: **7,782**
* clotho: **3,892**
* llava_pretrain: **19,760**
  (total 51,200)

Only after Tier C looks good do you bother with the full 256k.

---

## 4) Easiest way to shorten “epoch time” without changing hardware: increase effective batch

Your run showing `step = sample` (256,408 total steps) strongly suggests your effective batch is **1 sample/step**.

Even for long contexts, Stage‑3 often tolerates a higher effective batch because you’re not updating the whole transformer.

Levers that reduce steps-per-epoch / wall time *proportionally*:

* Increase **per-device batch size** if memory allows
* Or increase **gradient accumulation** (same memory, fewer optimizer steps)
* Ensure **dynamic padding** (avoid padding everything to max length)
* Filter or cap extreme-length samples more aggressively for iteration runs

If the effective batch increases from 1 → 4, optimizer steps drop by ~4×.

---

## 5) How to build smaller JSONLs from your current files

You already have:

* `outputs/stage3_token_lm/train.jsonl` (256,408 rows)
* `outputs/stage3_token_lm/eval.jsonl` (2,635 rows)

Since each row has `source` and `task`, do a **stratified sample by source** to preserve the exact audio/image mix.

Standalone script (reservoir sampling per source); paste into a file like `sample_stage3.py`:

```python
import json, random
from collections import defaultdict

IN_PATH  = "outputs/stage3_token_lm/train.jsonl"
OUT_PATH = "outputs/stage3_token_lm/train.small_8192.jsonl"
SEED = 123

# Choose one target preset:
targets = {
    # 8192 preset (recommended iteration run)
    "wavcaps_audioset": 3162,
    "audiocaps": 1245,
    "clotho": 623,
    "llava_pretrain": 3162,
}

rng = random.Random(SEED)
seen = defaultdict(int)
reservoir = {k: [] for k in targets}

with open(IN_PATH, "r", encoding="utf-8") as f:
    for line in f:
        ex = json.loads(line)
        src = ex.get("source")
        if src not in targets:
            continue
        seen[src] += 1
        buf = reservoir[src]
        k = targets[src]

        if len(buf) < k:
            buf.append(line)
        else:
            j = rng.randrange(seen[src])
            if j < k:
                buf[j] = line

# Merge + shuffle
all_lines = []
for src, buf in reservoir.items():
    if len(buf) != targets[src]:
        raise RuntimeError(f"Source {src}: got {len(buf)} / {targets[src]}")
    all_lines.extend(buf)

rng.shuffle(all_lines)

with open(OUT_PATH, "w", encoding="utf-8") as out:
    for line in all_lines:
        out.write(line)

print(f"Wrote {len(all_lines)} lines to {OUT_PATH}")
```

Do the same for eval (e.g., sample 512 out of 2,635) with a simpler uniform reservoir sampler, or just keep full eval since it’s small.

---

## 6) Make the *small run* maximally informative

If only one action is taken: add these two evaluations to enable early stopping with confidence.

### A) Ablation-gap metric (every N steps)

Every `eval_steps`:

* compute `Δ_shuffle = mean(L_shuffle - L_correct)`
* compute `Δ_noise   = mean(L_noise  - L_correct)`
* log win rates `P(L_correct < L_noise)`

This tells you if the modeling approach is working without waiting for “epoch completion”.

### B) Parameter-diff audit (once per run)

After a short run:

* verify text token rows `[0:V_text)` in embeddings/head are unchanged
* verify only modality ranges changed

If your masking/hook is wrong, you’ll catch it immediately.

---

## 7) Faster iteration option: use a smaller backbone (recommended)

For *modeling approach* changes (formatting, masking, copy task mix, clip policy), run Stage‑3 on:

* Qwen3 smaller variant (same tokenizer family + same surgery)

Once the approach passes the ablation/regression gates on the small backbone, scale to 8B for the “real” warm-start.

This is the highest-leverage way to iterate quickly because your bottleneck is transformer forward/backward, not dataset I/O.

---

## 8) Recommended decision rule

Use this so you don’t get trapped in “let it run longer”:

* If after a **Tier B** run (8,192 rows) your ablation gap is still near zero → **stop** and change modeling (prompting/masking/copy-mix/clip policy), not dataset size.
* If ablation gap is strong and text-only regression is clean → **don’t** wait for a full epoch; go directly to Tier C or full run depending on how stable it looks.

---

### Concrete recommendation

Do this next:

1. **Run ablation eval now** at your current checkpoint (step ~650).
2. Start a **Tier B** run (8,192 stratified rows) with frequent ablation logging.
3. Only launch the 256k full run once Tier B passes the gates.

Provide the trainer stack in use (HF Trainer / Accelerate / Deepspeed / custom loop) and whether `step` corresponds to a batch or a single sample to map this into exact knobs/flags (e.g., `max_steps`, `max_train_samples`, eval cadence, grad accumulation) without changing code structure.
