# Stage 1 Connector Report (Qwen3 + SigLIP2)

## Summary
- Connector = Perceiver-style resampler + MLP projector.
- Frozen: Qwen3-8B-Base LLM and SigLIP2 vision tower.
- Trainable: resampler + projector (vision LN disabled for stage1).
- Trial config: `configs/stage1_align_trial.yaml`.

## Code Paths
- Connector modules: `qwen3-vlm/src/qwen3_vlm/models/resampler.py`, `qwen3-vlm/src/qwen3_vlm/models/projector.py`
- VLM wrapper / image token injection: `qwen3-vlm/src/qwen3_vlm/models/vlm.py`
- Training entry: `qwen3-vlm/src/qwen3_vlm/train/train_utils.py`

## Connector Architecture and Shapes

### Vision -> Resampler
- Vision encoder: SigLIP2 SO400M patch14-384
  - config: `hidden_size=1152`, `patch_size=14`, `image_size=384`
- Vision output: `(B_img, N_patches, 1152)`
  - `N_patches` depends on patching; for 384/14 it is 27x27 = 729.

### PerceiverResampler
Config from `configs/stage1_align_trial.yaml`:
- `num_latents=64`
- `depth=2`
- `num_heads=8`
- `head_dim=64`
- `latent_dim = num_heads * head_dim = 512`

Layers and shapes:
- `proj_in`: Linear **1152 -> 512**
- `latents`: learnable **(64, 512)**
- Each of 2 layers:
  - `ln1`: LayerNorm(512)
  - `attn`: MultiheadAttention(embed_dim=512, heads=8)
    - queries: latents `(B_img, 64, 512)`
    - keys/values: vision patches `(B_img, N_patches, 512)`
    - output: `(B_img, 64, 512)`
  - `ln2`: LayerNorm(512)
  - `ff`: MLP **512 -> 2048 -> 512**

Resampler output: `(B_img, 64, 512)`

### MLPProjector
Config from `configs/stage1_align_trial.yaml`:
- `mlp_ratio=4`
- LLM hidden size (Qwen3-8B): 4096

Layers and shapes:
- `fc1`: Linear **512 -> 16384**
- `fc2`: Linear **16384 -> 4096**

Projector output: `(B_img, 64, 4096)`

### Injection into LLM
- Each image expands to `num_image_tokens=64` patch tokens.
- Projected tokens are inserted into positions of `<image_patch>`.
- Code path: `qwen3-vlm/src/qwen3_vlm/models/vlm.py` (`build_inputs_embeds`).

## Parameter Counts (Connector)

### PerceiverResampler
- `proj_in`: 1152*512 + 512 = **590,336**
- `latents`: 64*512 = **32,768**
- Per layer (x2):
  - `ln1`: 512 + 512 = **1,024**
  - `attn`: 3*512*512 + 3*512 + 512*512 + 512 = **1,050,624**
  - `ln2`: 512 + 512 = **1,024**
  - `ff`: (512*2048 + 2048) + (2048*512 + 512) = **2,099,712**
  - Per-layer total = **3,152,384**
- Resampler total = 590,336 + 32,768 + (2 * 3,152,384) = **6,927,872**

### MLPProjector
- `fc1`: 512*16384 + 16384 = **8,404,992**
- `fc2`: 16384*4096 + 4096 = **67,112,960**
- Projector total = **75,517,952**

### Connector Total
- **82,445,824** params (resampler + projector)
- Optional `vision_ln` (disabled for stage1): 1152 + 1152 = **2,304** params

## Dataset Spec (Stage 1)
- Dataset JSON: `qwen3-vlm/data/llava_pretrain/blip_laion_cc_sbu_558k.json`
- Images root: `qwen3-vlm/data/llava_pretrain` (folders like `00000/`)
- Format: each item has `image` and `conversations` (human + gpt)
  - Prompt: first human turn
  - Answer: first gpt turn
  - `<image>` appended to prompt if missing
- Scale:
  - Full: 558,128 samples
  - Eval split: 2% (`eval_ratio=0.02`) => train ~546,965, eval ~11,163

## Trial Run Config
- Config: `qwen3-vlm/configs/stage1_align_trial.yaml`
- Subset: `max_samples=5000`
- Split: train 4,900 / eval 100
- Batch: 4, grad_accum: 4 (effective batch 16)
- Precision: bf16

## Training Speed (Trial)
- Log: `qwen3-vlm/outputs/stage1_align_trial/train.log`
- Estimate line observed:
  - `avg_step=0.797s` => ~1.25 steps/sec
  - Full epoch estimate (1 epoch): **~30.28 hours**
  - Approx throughput: ~5.0 samples/sec (batch_size=4)
