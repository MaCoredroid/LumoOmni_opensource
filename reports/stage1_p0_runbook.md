# Stage 1 P0 Runbook (Spec + Results)

## Scope
- Verify Stage 1 connector uses image embeddings via ablation (teacher-forced loss).
- Measure truncation and label coverage with `max_seq_len=512`.

## Environment
- Host repo: `/media/mark/SHARED/lumoOmni`
- Container image: `lumo-run47-base`
- GPU: `--gpus all`
- Container workdir: `/workspace/lumoOmni/qwen3-vlm`
- Python env: `/opt/lumo/venv`

## Stage 1 Trial Config
- Config: `qwen3-vlm/configs/stage1_align_trial.yaml`
- Dataset: LLaVA-Pretrain (`data/llava_pretrain/blip_laion_cc_sbu_558k.json`)
- Image root: `data/llava_pretrain`
- Max seq len: 512
- Trial subset: 5,000 (eval ratio 2%)
- Checkpoint: `qwen3-vlm/outputs/stage1_align_trial/checkpoint_3675.pt`

## Spec: Check 1 - Ablation (Teacher-Forced Loss)
Compute loss on a fixed held-out set (64-256 items):
- Correct image
- Shuffled image (different image from the batch)
- Zeroed visual tokens
- Random visual tokens (Gaussian with same mean/std as projected embeddings)

Expected:
- `loss(correct) < loss(shuffled)`
- `loss(correct) < loss(zero/noise)`
- If not, likely placeholder mismatch, masking bug, or image tokens not injected.

## Spec: Check 2 - Truncation + Label Coverage
Log per epoch:
- % truncated samples
- % samples with `label_tokens == 0` (should be ~0)
- average `label_tokens` per sample

If truncation > ~1-2% in Stage 1:
- Increase `max_seq_len` (e.g., 768), or
- Shorten the prompt wrapper.

## Execution

### 1) Ablation in Stage 1 Container
Command (one-shot):
```bash
echo '{sudo_passwd}' | sudo -S docker run --gpus all --ipc=host --rm \
  -v /media/mark/SHARED/lumoOmni:/workspace/lumoOmni \
  -v /media/mark/SHARED/lumoOmni/.cache/huggingface:/root/.cache/huggingface \
  lumo-run47-base bash -lc "\
    cd /workspace/lumoOmni/qwen3-vlm && \
    source /opt/lumo/venv/bin/activate && \
    pip install -e . >/tmp/pip_editable.log 2>&1 && \
    PYTHONUNBUFFERED=1 python3 -m qwen3_vlm.eval_ablation \
      --config configs/stage1_align_trial.yaml \
      --num_samples 128 \
      --checkpoint outputs/stage1_align_trial/checkpoint_3675.pt \
  "
```

### 2) Truncation + Label Coverage
Stats computed inside the container on the Stage 1 trial subset.

## Results (Actual)

### Ablation Results
- Samples: 100 (eval subset size)
- Losses:
  - correct: 3.5150
  - shuffled: 4.4022
  - zero: 5.2885
  - noise: 5.6652
- Deltas vs correct:
  - shuffled: +0.8872
  - zero: +1.7735
  - noise: +2.1502

Conclusion: The connector is used; loss degrades clearly when visual embeddings are shuffled/zeroed/noised.

### Truncation + Label Coverage
- subset_train: 4,900
- subset_eval: 100
- truncated: 0.00%
- label_zero: 0.00%
- avg_label_tokens: 13.1

Conclusion: No truncation pressure at 512 for the Stage 1 trial configuration.

## Artifacts / Code Paths
- Ablation script: `qwen3-vlm/src/qwen3_vlm/eval_ablation.py`
- Collator stats: `qwen3-vlm/src/qwen3_vlm/data/collate.py`
- Per-epoch logging: `qwen3-vlm/src/qwen3_vlm/train/train_utils.py`
- Checkpoint: `qwen3-vlm/outputs/stage1_align_trial/checkpoint_3675.pt`

## Pass/Fail Summary
- Ablation test: PASS (correct loss lower than shuffled/zero/noise).
- Truncation/coverage: PASS (0% truncation, 0% label-zero).
