#!/usr/bin/env bash
set -euo pipefail

data_dir="data/llava_pretrain"
cache_dir="data/.cache/huggingface"
mkdir -p "${data_dir}"
mkdir -p "${cache_dir}"

export HF_HOME="${cache_dir}"
export HF_DATASETS_CACHE="${cache_dir}/datasets"
export HF_HUB_CACHE="${cache_dir}/hub"

python - <<'PY'
from pathlib import Path
from huggingface_hub import hf_hub_download

repo_id = "liuhaotian/LLaVA-Pretrain"
repo_type = "dataset"

out_dir = Path("data/llava_pretrain")
out_dir.mkdir(parents=True, exist_ok=True)

json_path = hf_hub_download(
    repo_id=repo_id,
    repo_type=repo_type,
    filename="blip_laion_cc_sbu_558k.json",
    local_dir=str(out_dir),
    local_dir_use_symlinks=False,
)
print(f"Downloaded {json_path}")

zip_path = out_dir / "images.zip"
if not zip_path.exists():
    zip_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            filename="images.zip",
            local_dir=str(out_dir),
            local_dir_use_symlinks=False,
        )
    )
    print(f"Downloaded {zip_path}")

images_dir = out_dir / "images"
if not images_dir.exists():
    import zipfile

    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    print(f"Extracted to {out_dir}")
PY
