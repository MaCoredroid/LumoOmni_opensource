#!/usr/bin/env bash
set -euo pipefail

data_dir="data/coco"
mkdir -p "${data_dir}"

zip_path="${data_dir}/train2017.zip"
url="http://images.cocodataset.org/zips/train2017.zip"

if [ ! -f "${zip_path}" ]; then
  echo "Downloading COCO train2017..."
  curl -L "${url}" -o "${zip_path}"
fi

if [ ! -d "${data_dir}/train2017" ]; then
  echo "Extracting COCO train2017..."
  python - <<'PY'
import zipfile
from pathlib import Path

zip_path = Path("data/coco/train2017.zip")
if not zip_path.exists():
    raise FileNotFoundError(zip_path)

out_dir = Path("data/coco")
with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(out_dir)
print("Extracted to data/coco/train2017")
PY
fi
