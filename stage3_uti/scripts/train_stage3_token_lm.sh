#!/usr/bin/env bash
set -euo pipefail

python3 -m stage3_uti.train.stage3_token_lm --config stage3_uti/configs/stage3_token_lm.yaml
