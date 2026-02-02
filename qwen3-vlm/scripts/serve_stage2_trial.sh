#!/usr/bin/env bash
set -euo pipefail

python3 -m qwen3_vlm.serve.webui \
  --config configs/stage2_sft_single_trial.yaml \
  --host 0.0.0.0 \
  --port 7860
