# qwen3-vlm

Qwen3-8B-Base + SigLIP + resampler + projector with multi-image `<image>` interleaving.

## Quick start (Stage 0 / Stage 1)

1) Install deps

```bash
pip install -r requirements.txt
```

2) Run Stage 0 sanity overfit

```bash
bash scripts/train_stage0.sh
```

### Stage 0 sanity: full run details

If you are not installing the package (or you are running in a container), use
`PYTHONPATH=src` and set a writable Hugging Face cache directory. This run
downloads the Qwen3 checkpoint, trains the connector on a tiny dummy set, saves
checkpoints, and runs the built-in sanity checks (token count, checkpoint
reload, deterministic generation).

From `qwen3-vlm/`:

```bash
export HF_HOME=/media/mark/SHARED/lumoOmni/.cache/huggingface
export HF_HUB_CACHE=/media/mark/SHARED/lumoOmni/.cache/huggingface/hub
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
PYTHONPATH=src python3 -m qwen3_vlm.train.stage0_sanity \
  --config configs/stage0_sanity.yaml
```

Long-running, logged run:

```bash
PYTHONPATH=src \
HF_HOME=/media/mark/SHARED/lumoOmni/.cache/huggingface \
HF_HUB_CACHE=/media/mark/SHARED/lumoOmni/.cache/huggingface/hub \
HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
nohup python3 -m qwen3_vlm.train.stage0_sanity \
  --config configs/stage0_sanity.yaml \
  > /media/mark/SHARED/lumoOmni/logs/stage0_sanity.log 2>&1 &
```

Outputs:

- checkpoints and tokenizer in `outputs/stage0_sanity/`
- log file (if using nohup) in `logs/stage0_sanity.log`
- cache in `/media/mark/SHARED/lumoOmni/.cache/huggingface/`

3) Download Stage 1 data (LLaVA-Pretrain metadata + images)

```bash
bash scripts/download_llava_pretrain.sh
```

The LLaVA `images.zip` extracts into subfolders like `data/llava_pretrain/00000/`.
Keep that layout and set `image_root: "data/llava_pretrain"` in configs.

4) Run Stage 1 alignment (LLaVA-Pretrain)

```bash
bash scripts/train_stage1_trial.sh
```

### Stage 1 trial: full run details

This trial uses a 5k subset with a 2% eval split to get speed/ETA.

From `qwen3-vlm/`:

```bash
export HF_HOME=/media/mark/SHARED/lumoOmni/.cache/huggingface
export HF_HUB_CACHE=/media/mark/SHARED/lumoOmni/.cache/huggingface/hub
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
PYTHONPATH=src python3 -m qwen3_vlm.train.stage1_align \
  --config configs/stage1_align_trial.yaml
```

Long-running, logged run:

```bash
PYTHONPATH=src \
HF_HOME=/media/mark/SHARED/lumoOmni/.cache/huggingface \
HF_HUB_CACHE=/media/mark/SHARED/lumoOmni/.cache/huggingface/hub \
HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
nohup python3 -m qwen3_vlm.train.stage1_align \
  --config configs/stage1_align_trial.yaml \
  > /media/mark/SHARED/lumoOmni/logs/stage1_align_trial.log 2>&1 &
```

Outputs:

- checkpoints in `outputs/stage1_align_trial/`
- log file (if using nohup) in `logs/stage1_align_trial.log`
- cache in `/media/mark/SHARED/lumoOmni/.cache/huggingface/`

For a full run on 558K:

```bash
bash scripts/train_stage1.sh
```

## Stage 1.1 hardening (stratified eval + golden set + sweep)

Stage 1.1 adds:
- stratified eval buckets by target length
- token-weighted eval loss + per-bucket loss
- a fixed golden set (64 samples) with caption dumps at checkpoints
- optional quick eval every N steps

Run the scaled 1-epoch trial (16k train / 2k eval):

```bash
bash scripts/train_stage1_p11.sh
```

Artifacts:
- split metadata in `outputs/stage1_align_p11/splits/`
- golden dumps in `outputs/stage1_align_p11/qual/`
- final eval metrics in `outputs/stage1_align_p11/metrics.json`

Sweep template (2k steps, full eval at steps 1000 + 2000):

```bash
bash scripts/train_stage1_p11_sweep.sh
```

To run the 6-way sweep, adjust `lr` and `model.vision_ln`/`train.train_vision_ln`
between runs (or copy the sweep config into per-run configs with unique `output_dir`).

## Stage 2/3 data (optional now)

```bash
bash scripts/download_llava_instruct.sh
bash scripts/download_coco2017.sh
bash scripts/download_mantis.sh
```

## Stage 2 inference web UI

This starts a Gradio UI for the latest Stage 2 checkpoint in
`outputs/stage2_sft_single_trial/` and exposes golden examples.

Build the web UI image (one-time):

```bash
echo '{sudo_passwd}' | sudo -S docker build -f Dockerfile -t lumo-run47-webui .
```

Host run (from `qwen3-vlm/`):

```bash
bash scripts/serve_stage2_trial.sh
```

Docker run (bind to `100.103.10.122:7860`):

```bash
echo '{sudo_passwd}' | sudo -S docker run --gpus all --ipc=host --rm -d \
  --name qwen3-vlm-stage2-webui \
  -p 100.103.10.122:7860:7860 \
  -v /media/mark/SHARED/lumoOmni:/workspace/lumoOmni \
  -v /media/mark/SHARED/lumoOmni/.cache/huggingface:/root/.cache/huggingface \
  lumo-run47-base bash -lc "\
    cd /workspace/lumoOmni/qwen3-vlm && \
    source /opt/lumo/venv/bin/activate && \
    pip install -e . >/tmp/pip_editable.log 2>&1 && \
    bash scripts/serve_stage2_trial.sh \
  "
```

Stop the container:

```bash
echo '{sudo_passwd}' | sudo -S docker stop qwen3-vlm-stage2-webui
```

## Notes

- Transformers must be >= 4.51.0 for Qwen3 support.
- Default configs use `google/siglip2-so400m-patch14-384`; swap to SigLIP1 if needed.
- `<image>` in text expands to `<im_start>` + `<image_patch>` x N + `<im_end>` at collate time.
- Stage 1 configs enable `attn_implementation: "flash_attention_2"` and TF32 for speed; set it to `"sdpa"` if FA2 is unavailable.

## LoRA + tokenizer consistency (hard requirement)

Stage 2 adds special tokens and resizes the LLM embeddings. To avoid silent mismatches:

- Always save/version the tokenizer, LoRA adapter, and resized embeddings (PEFT saves these in `lora_<step>`).
- Load order for inference/Stage 3:
  1) base model
  2) tokenizer + resize embeddings
  3) LoRA adapter (with embedding layers)
  4) connector checkpoint
