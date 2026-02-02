#!/usr/bin/env bash
set -euo pipefail

data_dir="data/llava_instruct"
cache_dir="data/.cache/huggingface"
mkdir -p "${data_dir}"
mkdir -p "${cache_dir}"

export HF_HOME="${cache_dir}"
export HF_DATASETS_CACHE="${cache_dir}/datasets"
export HF_HUB_CACHE="${cache_dir}/hub"

python - <<'PY'
from pathlib import Path
from huggingface_hub import hf_hub_download

repo_id = "liuhaotian/LLaVA-Instruct-150K"
repo_type = "dataset"

out_dir = Path("data/llava_instruct")
out_dir.mkdir(parents=True, exist_ok=True)

json_path = hf_hub_download(
    repo_id=repo_id,
    repo_type=repo_type,
    filename="llava_instruct_150k.json",
    local_dir=str(out_dir),
    local_dir_use_symlinks=False,
)
print(f"Downloaded {json_path}")
PY

cat <<'MSG'

COCO images are required for LLaVA-Instruct. Download with scripts/download_coco2017.sh
and place under data/coco/train2017/
MSG
