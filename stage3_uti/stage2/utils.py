import hashlib
from typing import Iterable, Mapping, Sequence

import numpy as np


def stable_hash_u64(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def assign_split(id_str: str, eval_mod: int = 1000, eval_cutoff: int = 10) -> str:
    if eval_mod <= 0:
        raise ValueError("eval_mod must be positive")
    bucket = stable_hash_u64(id_str) % int(eval_mod)
    return "eval" if bucket < int(eval_cutoff) else "train"


def sha256_int32(tokens: Sequence[int]) -> str:
    arr = np.asarray(tokens, dtype=np.int32)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def percentile_stats(values: Iterable[int]) -> Mapping[str, int]:
    vals = [int(v) for v in values]
    if not vals:
        return {"count": 0, "p50": 0, "p90": 0, "p99": 0}
    arr = np.asarray(vals, dtype=np.int64)
    return {
        "count": int(arr.size),
        "p50": int(np.percentile(arr, 50)),
        "p90": int(np.percentile(arr, 90)),
        "p99": int(np.percentile(arr, 99)),
    }


def min_max(values: Iterable[int]) -> Mapping[str, int]:
    vals = [int(v) for v in values]
    if not vals:
        return {"min": 0, "max": 0}
    return {"min": int(min(vals)), "max": int(max(vals))}
