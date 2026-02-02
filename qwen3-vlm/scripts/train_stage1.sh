#!/usr/bin/env bash
set -euo pipefail

python3 -m qwen3_vlm.train.stage1_align --config configs/stage1_align.yaml
