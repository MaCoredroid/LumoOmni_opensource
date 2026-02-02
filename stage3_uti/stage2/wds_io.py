import io
import json
import os
import tarfile
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import numpy as np


class TarShardWriter:
    def __init__(self, base_dir: Path, split: str, max_samples_per_shard: int = 1000) -> None:
        self.base_dir = Path(base_dir)
        self.split = str(split)
        self.max_samples_per_shard = int(max_samples_per_shard)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self._tar: Optional[tarfile.TarFile] = None
        self._shard_index = 0
        self._samples_in_shard = 0
        self._total_samples = 0

    def _open_new_shard(self) -> None:
        if self._tar is not None:
            self._tar.close()
        shard_name = f"shard-{self._shard_index:06d}.tar"
        shard_path = self.base_dir / shard_name
        self._tar = tarfile.open(shard_path, "w")
        self._shard_index += 1
        self._samples_in_shard = 0

    def _write_bytes(self, name: str, payload: bytes) -> None:
        if self._tar is None:
            raise RuntimeError("tar file not open")
        info = tarfile.TarInfo(name)
        info.size = len(payload)
        self._tar.addfile(info, io.BytesIO(payload))

    def write_sample(self, sample: Dict, arrays: Dict[str, np.ndarray]) -> str:
        if self._tar is None or self._samples_in_shard >= self.max_samples_per_shard:
            self._open_new_shard()

        key = f"{self._total_samples:08d}"
        sample = dict(sample)
        sample["__key__"] = key
        payload = json.dumps(sample, ensure_ascii=True).encode("utf-8")
        self._write_bytes(f"{key}.json", payload)

        for name, arr in arrays.items():
            if arr is None:
                continue
            arr = np.asarray(arr, dtype=np.int32)
            buf = io.BytesIO()
            np.save(buf, arr, allow_pickle=False)
            self._write_bytes(f"{key}.{name}.npy", buf.getvalue())

        self._samples_in_shard += 1
        self._total_samples += 1
        return key

    def close(self) -> None:
        if self._tar is not None:
            self._tar.close()
            self._tar = None

    @property
    def total_samples(self) -> int:
        return self._total_samples


def _split_tar_name(name: str) -> Optional[Tuple[str, str]]:
    base = os.path.basename(name)
    parts = base.split(".")
    if len(parts) < 2:
        return None
    key = parts[0]
    suffix = ".".join(parts[1:])
    return key, suffix


def iter_tar_samples(tar_path: Path) -> Iterator[Tuple[str, Dict]]:
    with tarfile.open(tar_path, "r") as tar:
        current_key = None
        current: Dict = {}
        for member in tar:
            if not member.isfile():
                continue
            info = _split_tar_name(member.name)
            if info is None:
                continue
            key, suffix = info
            if current_key is None:
                current_key = key
            if key != current_key:
                yield current_key, current
                current = {}
                current_key = key

            stream = tar.extractfile(member)
            if stream is None:
                continue
            data = stream.read()
            if suffix == "json":
                current["json"] = json.loads(data.decode("utf-8"))
            elif suffix.endswith(".npy"):
                arr_name = suffix[: -len(".npy")]
                current[arr_name] = np.load(io.BytesIO(data), allow_pickle=False)
        if current_key is not None:
            yield current_key, current
