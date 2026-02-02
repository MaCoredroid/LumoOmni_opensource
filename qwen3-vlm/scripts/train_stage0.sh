#!/usr/bin/env bash
set -euo pipefail

python3 -m qwen3_vlm.train.stage0_sanity --config configs/stage0_sanity.yaml
