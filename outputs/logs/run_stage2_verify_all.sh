#!/usr/bin/env bash
set -euo pipefail

docker run --rm --gpus all --name stage2_verify_all \
  -v /media/mark/SHARED/lumoOmni:/workspace/lumoOmni \
  -w /workspace/lumoOmni \
  lumo-run47-base:latest \
  bash -lc '
    set -o pipefail
    source /opt/lumo/venv/bin/activate
    export PYTHONUNBUFFERED=1
    export UTI_SKIP_DIFFUSION=1
    for ds in clotho audiocaps wavcaps_as_100k llava_pretrain_100k; do
      echo "=== manifest_audit: ${ds} ==="
      python -m stage3_uti.stage2.manifest_audit \
        --manifest-jsonl stage3_uti/data/manifests/${ds}.jsonl \
        --dataset-name ${ds} 2>&1 | tee outputs/logs/${ds}_manifest_audit.log
      echo "=== audit_tokenized: ${ds} ==="
      python -m stage3_uti.stage2.audit_tokenized \
        --uti-config stage3_uti/configs/uti.yaml \
        --dataset-name ${ds} \
        --compare-hashes stage3_uti/data/reports/${ds}_token_hashes.jsonl \
        --hashes-jsonl stage3_uti/data/reports/${ds}_audit_hashes.jsonl \
        --range-samples 1000 \
        --decode-samples 32 2>&1 | tee outputs/logs/${ds}_audit.log
    done
  '
