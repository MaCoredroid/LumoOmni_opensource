#!/usr/bin/env bash
set -euo pipefail

docker run --rm --gpus all --name stage2_full_missing_checks \
  -v /media/mark/SHARED/lumoOmni:/workspace/lumoOmni \
  -w /workspace/lumoOmni \
  lumo-run47-base:latest \
  bash -lc '
    set -o pipefail
    source /opt/lumo/venv/bin/activate
    export PYTHONUNBUFFERED=1
    export UTI_SKIP_DIFFUSION=1
    for ds in clotho audiocaps wavcaps_as_100k llava_pretrain_100k; do
      echo "=== split_audit: ${ds} ==="
      python -m stage3_uti.stage2.split_audit \
        --dataset-name ${ds} \
        --eval-mod 1000 \
        --eval-cutoff 10 2>&1 | tee outputs/logs/${ds}_split_audit.log
      echo "=== retokenize_check: ${ds} ==="
      python -m stage3_uti.stage2.retokenize_check \
        --uti-config stage3_uti/configs/uti.yaml \
        --manifest-jsonl stage3_uti/data/manifests/${ds}.jsonl \
        --dataset-name ${ds} \
        --num-samples 512 2>&1 | tee outputs/logs/${ds}_retokenize.log
      echo "=== sequence_smoke: ${ds} ==="
      python -m stage3_uti.stage2.sequence_smoke \
        --uti-config stage3_uti/configs/uti.yaml \
        --dataset-name ${ds} \
        --num-samples 512 \
        --max-seq-len 2048 2>&1 | tee outputs/logs/${ds}_sequence_smoke.log
    done
  '
