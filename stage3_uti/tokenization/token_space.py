import dataclasses
import datetime
import hashlib
import json
from typing import Any, Dict, Tuple


@dataclasses.dataclass(frozen=True)
class RangeSpec:
    start: int
    end: int
    size: int

    @staticmethod
    def from_start_size(start: int, size: int) -> "RangeSpec":
        if size <= 0:
            raise ValueError("range size must be positive")
        end = start + size - 1
        return RangeSpec(start=start, end=end, size=size)


class TokenSpace:
    def __init__(
        self,
        *,
        version: str,
        created_utc: str,
        base_text_model: str,
        text_vocab_size: int,
        special_tokens: Dict[str, int],
        ranges: Dict[str, Dict[str, int]],
        image_tokenizer: Dict[str, Any],
        audio_codec: Dict[str, Any],
    ) -> None:
        self.version = version
        self.created_utc = created_utc
        self.base_text_model = base_text_model
        self.text_vocab_size = int(text_vocab_size)
        self.special_tokens = dict(special_tokens)
        self.ranges = {k: dict(v) for k, v in ranges.items()}
        self.image_tokenizer = dict(image_tokenizer)
        self.audio_codec = dict(audio_codec)

    @property
    def vocab_size_total(self) -> int:
        max_end = max(r["end"] for r in self.ranges.values())
        return int(max_end) + 1

    def validate(self) -> None:
        # Range validity
        ranges = []
        for name, spec in self.ranges.items():
            start = int(spec["start"])
            end = int(spec["end"])
            size = int(spec["size"])
            if size != (end - start + 1):
                raise ValueError(f"range {name} size mismatch")
            ranges.append((start, end, name))

        ranges.sort(key=lambda x: x[0])
        for i, (start, end, name) in enumerate(ranges):
            if start < 0 or end < 0:
                raise ValueError(f"range {name} has negative bounds")
            if i > 0:
                prev_end = ranges[i - 1][1]
                if start <= prev_end:
                    raise ValueError(f"range overlap: {ranges[i-1][2]} and {name}")

        # Required ranges
        if "TEXT" not in self.ranges:
            raise ValueError("TEXT range missing")
        text_range = self.ranges["TEXT"]
        if int(text_range["start"]) != 0:
            raise ValueError("TEXT range must start at 0")
        if int(text_range["size"]) != self.text_vocab_size:
            raise ValueError("TEXT range size mismatch with text_vocab_size")

        if "SPECIAL" not in self.ranges:
            raise ValueError("SPECIAL range missing")

        # Special tokens inside SPECIAL range
        spec_range = self.ranges["SPECIAL"]
        spec_start = int(spec_range["start"])
        spec_end = int(spec_range["end"])
        for tok, tok_id in self.special_tokens.items():
            if not (spec_start <= int(tok_id) <= spec_end):
                raise ValueError(f"special token {tok} out of SPECIAL range")

        # Image range size vs tokenizer
        img_range = self.ranges.get("IMAGE")
        if img_range is None:
            raise ValueError("IMAGE range missing")
        img_size = int(img_range["size"])
        img_codebook = int(self.image_tokenizer.get("codebook_size", img_size))
        if img_size != img_codebook:
            raise ValueError("IMAGE range size mismatch with image tokenizer")

        # Audio ranges
        n_codebooks = int(self.audio_codec.get("n_codebooks", 0))
        codebook_size = int(self.audio_codec.get("codebook_size", 0))
        audio_ranges = [
            (name, spec)
            for name, spec in self.ranges.items()
            if name.startswith("AUDIO_CB")
        ]
        if n_codebooks and len(audio_ranges) != n_codebooks:
            raise ValueError("audio range count mismatch with n_codebooks")
        for name, spec in audio_ranges:
            if int(spec["size"]) != codebook_size:
                raise ValueError(f"{name} size mismatch with audio codebook_size")

    def to_json(self) -> Dict[str, Any]:
        payload = {
            "version": self.version,
            "created_utc": self.created_utc,
            "base_text_model": self.base_text_model,
            "text_vocab_size": self.text_vocab_size,
            "special_tokens": self.special_tokens,
            "ranges": self.ranges,
            "image_tokenizer": self.image_tokenizer,
            "audio_codec": self.audio_codec,
        }
        payload["vocab_size_total"] = self.vocab_size_total
        return payload

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, indent=2)

    @staticmethod
    def load_json(path: str) -> "TokenSpace":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return TokenSpace(
            version=payload["version"],
            created_utc=payload.get("created_utc", ""),
            base_text_model=payload.get("base_text_model", ""),
            text_vocab_size=int(payload["text_vocab_size"]),
            special_tokens=payload.get("special_tokens", {}),
            ranges=payload.get("ranges", {}),
            image_tokenizer=payload.get("image_tokenizer", {}),
            audio_codec=payload.get("audio_codec", {}),
        )

    def sha256(self) -> str:
        data = json.dumps(self.to_json(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(data).hexdigest()

    # Mapping helpers
    def img_to_global(self, img_token: int) -> int:
        rng = self.ranges["IMAGE"]
        return int(rng["start"]) + int(img_token)

    def img_from_global(self, global_id: int) -> int:
        rng = self.ranges["IMAGE"]
        start = int(rng["start"])
        end = int(rng["end"])
        if not (start <= int(global_id) <= end):
            raise ValueError("global_id not in IMAGE range")
        return int(global_id) - start

    def aud_to_global(self, codebook_idx: int, token: int) -> int:
        key = f"AUDIO_CB{int(codebook_idx)}"
        if key not in self.ranges:
            raise ValueError(f"missing audio range {key}")
        rng = self.ranges[key]
        return int(rng["start"]) + int(token)

    def aud_from_global(self, global_id: int) -> Tuple[int, int]:
        gid = int(global_id)
        for name, spec in self.ranges.items():
            if not name.startswith("AUDIO_CB"):
                continue
            start = int(spec["start"])
            end = int(spec["end"])
            if start <= gid <= end:
                codebook_idx = int(name.replace("AUDIO_CB", ""))
                return codebook_idx, gid - start
        raise ValueError("global_id not in any AUDIO range")


def _now_utc() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def build_token_space_from_config(
    cfg: Dict[str, Any],
    *,
    text_vocab_size: int,
    image_codebook_size: int,
    audio_codebook_size: int,
    n_codebooks: int,
) -> TokenSpace:
    token_cfg = cfg.get("token_space", {})
    n_special = int(token_cfg.get("n_special", 0))
    special_def = token_cfg.get("special_tokens", [])

    special_start = int(text_vocab_size)
    if n_special <= 0:
        n_special = len(special_def) if isinstance(special_def, list) else 0
    special_range = RangeSpec.from_start_size(special_start, n_special)

    img_start = special_range.end + 1
    img_range = RangeSpec.from_start_size(img_start, int(image_codebook_size))

    ranges: Dict[str, Dict[str, int]] = {
        "TEXT": dataclasses.asdict(RangeSpec.from_start_size(0, int(text_vocab_size))),
        "SPECIAL": dataclasses.asdict(special_range),
        "IMAGE": dataclasses.asdict(img_range),
    }

    audio_ranges = {}
    audio_start = img_range.end + 1
    for idx in range(int(n_codebooks)):
        rng = RangeSpec.from_start_size(audio_start, int(audio_codebook_size))
        audio_ranges[f"AUDIO_CB{idx}"] = dataclasses.asdict(rng)
        audio_start = rng.end + 1
    ranges.update(audio_ranges)

    special_tokens: Dict[str, int] = {}
    if isinstance(special_def, dict):
        special_tokens = {k: int(v) for k, v in special_def.items()}
    elif isinstance(special_def, list):
        for offset, tok in enumerate(special_def):
            special_tokens[str(tok)] = special_range.start + offset

    image_tokenizer = dict(cfg.get("image_tokenizer", {}))
    image_tokenizer.setdefault("codebook_size", int(image_codebook_size))
    audio_codec = dict(cfg.get("audio_codec", {}))
    audio_codec.setdefault("n_codebooks", int(n_codebooks))
    audio_codec.setdefault("codebook_size", int(audio_codebook_size))

    ts = TokenSpace(
        version="uti_v1",
        created_utc=_now_utc(),
        base_text_model=str(cfg.get("base_text_model", "")),
        text_vocab_size=int(text_vocab_size),
        special_tokens=special_tokens,
        ranges=ranges,
        image_tokenizer=image_tokenizer,
        audio_codec=audio_codec,
    )
    ts.validate()
    return ts
