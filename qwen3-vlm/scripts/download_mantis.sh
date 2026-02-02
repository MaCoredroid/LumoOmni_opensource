#!/usr/bin/env bash
set -euo pipefail

data_dir="data/mantis_instruct"
cache_dir="data/.cache/huggingface"
mkdir -p "${data_dir}"
mkdir -p "${cache_dir}"

export HF_HOME="${cache_dir}"
export HF_DATASETS_CACHE="${cache_dir}/datasets"
export HF_HUB_CACHE="${cache_dir}/hub"
export MANTIS_CONFIG="${MANTIS_CONFIG:-llava_665k_multi}"

python - <<'PY'
import os
from datasets import load_dataset

config = os.environ.get("MANTIS_CONFIG", "llava_665k_multi")
out_dir = os.path.join("data/mantis_instruct", config)

print(f"Loading Mantis-Instruct config '{config}' with script revision (downloads images)...")

ds = load_dataset(
    "TIGER-Lab/Mantis-Instruct",
    config,
    split="train",
    revision="script",
    trust_remote_code=True,
)
print(ds)

ds.save_to_disk(out_dir)
print(f"Saved dataset to {out_dir}")
PY
