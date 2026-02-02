Here is how I would handle dataset selection and sizing for **Stage 2 (tokenized shards)** given your constraints:

* You want **general audio + music from the start**
* You want **iteration-speed** (keep “one epoch” ≈ **~1 hour** on DGX Spark)
* You already have UTI passing and the relevant tokenizers/weights downloaded

I’ll answer your two bullets directly (audio + image), then give you a concrete **“how big”** recommendation and an **hour-per-epoch sizing method** that will stay correct even as sequence lengths change.

---

## Audio datasets

### Yes: process **Clotho, AudioCaps, WavCaps**

These three give you a clean progression:

* **Clotho**: small, clean, excellent for pipeline validation and early grounding.
* **AudioCaps**: mid-size, captioned general audio; good step up in diversity.
* **WavCaps**: large; good for scale, but you should not tokenize all of it initially unless you have disk/time budget.

### About the `.7z` archives (Clotho)

For Stage 2 tokenization, you need the audio files accessible as real files on disk. If you only see `.7z` and no `.wav/.flac` files, then **you need to extract**.

**Fast check**

```bash
# should return >0 if extracted
find data/Clotho -type f \( -name "*.wav" -o -name "*.flac" \) | wc -l

# list archives
find data/Clotho -type f -name "*.7z" -maxdepth 2
```

**Extraction recommendation**

* Extract once into a stable folder (don’t tokenise directly from `.7z`).
* Keep the extracted structure unchanged, because manifests will point to paths.

Example:

```bash
mkdir -p data/Clotho/extracted
7z x data/Clotho/*.7z -odata/Clotho/extracted
```

If you extract and still have missing files at manifest-build time, that’s a manifest/path mapping issue—not UTI.

---

## Image dataset

### Best choice for you right now: use **the image-text dataset you already know is wired**

Given your earlier work, you already have:

* **LLaVA-Pretrain 558K** layout and loader support
* It’s already integrated into your repo style and you’ve used it successfully

So for Stage 2 tokenization I recommend:

1. **LLaVA-Pretrain (558K)** as your primary image→text source (for i2t tasks)
2. Optionally add a “clean small” dataset later (COCO Captions) if you want a high-quality eval slice, but it’s not required to start

### Why not CC12M/LAION full right now?

You *can* use them later, but for DGX Spark + “1 hour epoch” iteration:

* the incremental benefit over a good curated subset is smaller than the cost
* the pipeline complexity (manifests, filtering, duplicates, corrupt files) rises sharply

**Recommendation**: start with LLaVA-Pretrain and only introduce CC3M/LAION subsets once Stage 3/4 training is stable and you want more scale.

---

## How large should you process vs how large should you train on?

These are different decisions:

* **Stage 2 (processing/tokenizing)**: tokenize **more than is used per epoch**, because training can sample from tokenized shards.
* **Stage 3 training “one epoch ~ 1h”**: you choose a **subset size per epoch** via a sampler/max_samples.

### Practical recommendation: tokenize moderately, train small

This keeps iteration fast and avoids re-tokenizing later.

---

## Concrete dataset sizing plan

### Stage 2 tokenization targets (reasonable on your setup)

**Audio (tokenize)**

* Clotho: **100%** (small)
* AudioCaps: **100%** (assuming audio files exist locally; if it’s metadata-only, tokenize whatever subset has audio)
* WavCaps: start with **50k–100k** examples (not full)
* Music: include any clean music-caption dataset you have (Song Describer etc.) **100%**, and **oversample during training** (because it’s small)

**Images (tokenize)**

* LLaVA-Pretrain: tokenize **100k–200k** to start (not all 558k yet)

  * expand later without changing code paths

This gives you enough diversity while keeping tokenization time/disk reasonable.

---

## Keep one epoch ≈ 1 hour: the sizing method that won’t lie to you

For Type‑3, “epoch time” depends heavily on **sequence length** (audio tokens can be huge), so you should not guess. Use a short pilot to measure.

### Step 1: measure average step time for *your Stage 3 sequence builder*

Run a tiny training loop for ~200–500 steps on a representative mix (audio+i2t) and record:

* `avg_step_seconds`
* effective batch size `B_eff = micro_batch * grad_accum * num_gpus`

### Step 2: compute how many optimizer steps fit in ~1 hour

```text
steps_per_hour ≈ 3600 / avg_step_seconds
```

### Step 3: choose your “epoch sample count”

```text
epoch_samples ≈ steps_per_hour * B_eff
```

This is the only robust way to keep the “epoch ~ 1 hour” invariant across different max_seq_len, clip_seconds, and modality mix.

---

## Starting point values (what I would set before you measure)

A good initial “one-hour epoch” target for Stage 3 warm-start is usually:

* **Total train samples per epoch: 30k–50k**

  * e.g., **20k audio→text + 20k image→text = 40k total**
* **Eval: 2k total** (or 2k per modality if you want separated eval)

Why this is a reasonable default:

* It’s large enough to see stable curves.
* It’s small enough to iterate quickly.
* It aligns with your previous “subset_train=16000, subset_eval=2000” style, but gives more room for modality mix.

If your measured step time is higher than expected (common with audio tokens), drop to **20k total** per epoch.

---

## Recommended Stage 3 warm-start training mix (to match your “general audio + music from day one” goal)

Even though music datasets are smaller, enforce a *minimum share*:

* 45% **general audio captioning** (Clotho + AudioCaps + WavCaps subset)
* 10–20% **music captioning / music descriptions** (oversample)
* 35–45% **image captioning** (LLaVA-pretrain subset)

This prevents the model from becoming “speech/sfx only” early and gives music its own gradient budget.

---

## Implementation note that will matter for your epoch time

If your audio token length is blowing up sequence length:

* the dominant lever is **clip_seconds** (not dataset count)
* the second lever is **max_seq_len** (but truncation is dangerous)
* the third lever is **micro_batch_size**

Because UTI is already audited and stable, I would not change clip_seconds yet unless you prove you must. Instead, size the epoch by sampling fewer examples.

---

## Summary recommendation

### Process (Stage 2)

* Audio: **Clotho + AudioCaps fully**, **WavCaps 50k–100k**, include music dataset fully
* Image: **LLaVA-pretrain subset 100k–200k** (expand later)

### Train (Stage 3, keep epoch ~ 1h)

* Start at **40k train / 2k eval** per epoch
* Mix: **~50% audio, ~50% image**, ensure **10–20% music** via oversampling
* Then adjust `epoch_samples` using the pilot-derived formula above

---

If you tell me two numbers from your Stage 3 sequence builder pilot:

1. `avg_step_seconds`
2. your `micro_batch`, `grad_accum`, and number of GPUs used
   Use runtime measurements to select `max_samples` (train/eval) that land within ~60 minutes.
