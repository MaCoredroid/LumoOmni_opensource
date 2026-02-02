# Running Python on this machine (container workflow)

This machine runs Python inside a Docker container. The host may not have `python` installed (by design).

## 1) Build the base container image (once)

`lumo-run47-base` is built from the Dockerfile in this repo. It now preinstalls
Stage‑3 UTI deps (`diffusers`, `timm`, `encodec`, `torchaudio`) and installs a
matching nightly `torch/torchvision/torchaudio` set inside the venv, so the audit
doesn’t need per-run installs.

```bash
cd /media/mark/SHARED/lumoOmni
echo '{sudo_passwd}' | sudo -S docker build -f Dockerfile -t lumo-run47-base .
```

Pinned base versions (as of 2026-01-29):
- torch `2.11.0.dev20260129+cu130`
- torchvision `0.25.0.dev20260129+cu130`
- torchaudio `2.11.0.dev20260129+cu130`
- transformers `4.56.2`
- diffusers `0.36.0`
- timm `1.0.24`
- encodec `0.1.1`
- soundfile `0.13.1`
- scipy `1.16.1`

If torch/torchaudio install fails due to CUDA wheel mismatches, override the
nightly index URL during build (match the CUDA version in the NGC image):

```bash
echo '{sudo_passwd}' | sudo -S docker build \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/nightly/cu130 \
  -f Dockerfile -t lumo-run47-base .
```

## 2) Start a container shell

Reuse cached models/datasets by mounting the host Hugging Face cache. If you already
downloaded models into `/media/mark/SHARED/lumoOmni/.cache/huggingface`, prefer that to
avoid re-downloading model shards (e.g., Qwen3-8B-Base).

```bash
echo '{sudo_passwd}' | sudo -S docker run --gpus all --ipc=host --rm -it \
  -v /media/mark/SHARED/lumoOmni:/workspace/lumoOmni \
  -v /media/mark/SHARED/lumoOmni/.cache/huggingface:/root/.cache/huggingface \
  lumo-run47-base bash
```

If your cache lives in your home directory instead, swap the volume:

```bash
-v $HOME/.cache/huggingface:/root/.cache/huggingface
```

Notes:
- If Docker works without sudo, drop `sudo`.
- Use `CUDA_VISIBLE_DEVICES=0` if you want to pin to a single GPU.

## 3) Activate the container Python env

Inside the container:

```bash
source /opt/lumo/venv/bin/activate
python -V
```

## 4) Run this repo's training scripts

```bash
cd /workspace/lumoOmni/qwen3-vlm
pip install -r requirements.txt
pip install -e .
bash scripts/train_stage0.sh
```

## 4a) Stage 1 trial run (container, logged)

This uses `configs/stage1_align_trial.yaml` (5k subset, deterministic split via seed, 3 epochs).
It writes logs to `qwen3-vlm/outputs/stage1_align_trial/train.log`.

```bash
echo '{sudo_passwd}' | sudo -S docker run --gpus all --ipc=host --rm -d \
  --name qwen3-vlm-stage1-trial \
  -v /media/mark/SHARED/lumoOmni:/workspace/lumoOmni \
  -v /media/mark/SHARED/lumoOmni/.cache/huggingface:/root/.cache/huggingface \
  lumo-run47-base bash -lc "\
    cd /workspace/lumoOmni/qwen3-vlm && \
    source /opt/lumo/venv/bin/activate && \
    pip install -e . >/tmp/pip_editable.log 2>&1 && \
    mkdir -p outputs/stage1_align_trial && \
    PYTHONUNBUFFERED=1 bash scripts/train_stage1_trial.sh 2>&1 | tee outputs/stage1_align_trial/train.log \
  "
```

Monitor:

```bash
tail -f /media/mark/SHARED/lumoOmni/qwen3-vlm/outputs/stage1_align_trial/train.log
```

Stop:

```bash
echo '{sudo_passwd}' | sudo -S docker stop qwen3-vlm-stage1-trial
```

## 4b) Stage 1.1 hardening run (container, logged)

This uses `configs/stage1_align_p11.yaml` (16k train / 2k eval, stratified buckets,
golden set dumps). Logs at `qwen3-vlm/outputs/stage1_align_p11/train.log`.

```bash
echo '{sudo_passwd}' | sudo -S docker run --gpus all --ipc=host --rm -d \
  --name qwen3-vlm-stage1-p11 \
  -v /media/mark/SHARED/lumoOmni:/workspace/lumoOmni \
  -v /media/mark/SHARED/lumoOmni/.cache/huggingface:/root/.cache/huggingface \
  lumo-run47-base bash -lc "\
    cd /workspace/lumoOmni/qwen3-vlm && \
    source /opt/lumo/venv/bin/activate && \
    pip install -e . >/tmp/pip_editable.log 2>&1 && \
    mkdir -p outputs/stage1_align_p11 && \
    PYTHONUNBUFFERED=1 bash scripts/train_stage1_p11.sh 2>&1 | tee outputs/stage1_align_p11/train.log \
  "
```

Monitor:

```bash
tail -f /media/mark/SHARED/lumoOmni/qwen3-vlm/outputs/stage1_align_p11/train.log
```

Stop:

```bash
echo '{sudo_passwd}' | sudo -S docker stop qwen3-vlm-stage1-p11
```

## 5) One-shot command (no interactive shell)

```bash
echo '{sudo_passwd}' | sudo -S docker run --gpus all --ipc=host --rm \
  -v /media/mark/SHARED/lumoOmni:/workspace/lumoOmni \
  -v /media/mark/SHARED/lumoOmni/.cache/huggingface:/root/.cache/huggingface \
  lumo-run47-base bash -lc "\
    cd /workspace/lumoOmni/qwen3-vlm && \
    source /opt/lumo/venv/bin/activate && \
    python -V \
  "
```

## 6) Troubleshooting

- If `docker: permission denied`, add your user to the docker group or use `sudo`.
- If CUDA is not visible in the container, verify the NVIDIA container runtime and `--gpus all` flag.
# LumoOmni_opensource
