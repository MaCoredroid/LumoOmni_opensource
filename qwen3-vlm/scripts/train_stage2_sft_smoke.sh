#!/usr/bin/env bash
set -euo pipefail

python3 -m qwen3_vlm.train.stage2_sft --config configs/stage2_sft_single_smoke.yaml
