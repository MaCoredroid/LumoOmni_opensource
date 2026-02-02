Below is a **full Stage 1 “Unified Token Interface (UTI)” specification** tailored to your Option‑B Type‑3 goal (**image + general audio + music**) and the resources you already have downloaded (Qwen3 base tokenizer, **SEED tokenizer v2**, **EnCodec**, and reference repos).

I’m writing this as a **research-grade contract**: if you implement it exactly, every subsequent stage (tokenized shards, pretraining, SFT, evaluation, checkpointing) becomes deterministic and debuggable.

---

# Stage 1 Spec: Unified Token Interface (UTI)

## 1) Scope and invariants

### Purpose

Create a single internal, **lossless-to-reconstruct** representation for:

* text → token IDs (text IDs are *not* lossy)
* image → discrete token IDs + metadata required for decoding
* audio → discrete token IDs + metadata required for decoding

and a unified token-space allocator so all modalities share one integer ID space.

### Invariants (must hold always)

1. **No token ID collisions** between text vocab, special control tokens, image tokens, and audio tokens.
2. Tokenization is **deterministic**:

   * `encode_*` called twice on same input yields identical tokens and meta.
3. Decode contracts are explicit:

   * `decode_image(tokens, meta)` must succeed without guessing resolution/grid.
   * `decode_audio(tokens, meta)` must succeed without guessing codec settings or codebook count.
4. Token-space is **versioned and serialized** (token_space.json) and saved with every model checkpoint.

---

# 2) Unified interface: file/module contract

Create: `qwen3_vlm/tokenization/unified_tokenizer.py` (or a clean equivalent path)

## 2.1 Public API (exact signatures)

```python
# unified_tokenizer.py

from typing import List, Dict, Tuple, Any, Optional
from PIL import Image
import numpy as np

class UnifiedTokenizer:
    def encode_text(self, text: str) -> List[int]:
        ...

    def encode_image(self, img: Image.Image | np.ndarray) -> Tuple[List[int], Dict[str, Any]]:
        ...

    def encode_audio(self, wav: np.ndarray, sample_rate: int) -> Tuple[List[int], Dict[str, Any]]:
        ...

    def decode_image(self, tokens: List[int], meta: Dict[str, Any]) -> Image.Image:
        ...

    def decode_audio(self, tokens: List[int], meta: Dict[str, Any]) -> Tuple[np.ndarray, int]:
        ...
```

### Required properties

* All returned `meta` must be **JSON serializable** (no tensors, no numpy scalars, no bytes).
* `encode_image` accepts PIL or ndarray; internally normalize to PIL RGB.
* `encode_audio` accepts float PCM `wav` as `np.ndarray`:

  * shape: `[T]` mono or `[C, T]` multi-channel
  * dtype: float32 (preferred) in range `[-1, 1]`

---

# 3) Token-space allocator (ID layout)

You need a deterministic allocator that builds the entire ID map from:

* base text tokenizer vocab size (Qwen3 tokenizer)
* chosen special/control tokens count
* image tokenizer codebook size (SEED)
* audio codec codebook sizes & number of codebooks (EnCodec)

Create: `qwen3_vlm/tokenization/token_space.py`

## 3.1 Vocabulary structure (recommended)

This is the cleanest approach for EnCodec multi-codebook and avoids needing “codebook marker tokens”:

### Regions (contiguous ranges)

1. **TEXT**: `[0, V_text - 1]`
2. **SPECIAL / CONTROL TOKENS**: `[V_text, V_text + V_special - 1]`
3. **IMAGE TOKENS**: `[V_text + V_special, ... + V_img - 1]`
4. **AUDIO TOKENS (per codebook)**:

   * `AUD_CB0` range of size `V_aud_cb`
   * `AUD_CB1` range of size `V_aud_cb`
   * ...
   * `AUD_CB{K-1}`

This gives you explicit `AUD_CBk` ranges and makes reshaping trivial.

### Special token set (minimum)

Define **fixed IDs** within the SPECIAL range for at least:

* modality boundaries:

  * `<|text_start|>`, `<|text_end|>`
  * `<|img_start|>`, `<|img_end|>`
  * `<|aud_start|>`, `<|aud_end|>`

* task selectors:

  * `<|gen_text|>`, `<|gen_img|>`, `<|gen_aud|>`

* optional safety/debug:

  * `<|pad_mm|>` (multimodal padding token, distinct from text pad)
  * `<|unk_mm|>`

**Recommendation:** Keep SPECIAL small (16–64 tokens). The important part is stability.

## 3.2 Token-space JSON schema (token_space.json)

Create a canonical schema that is saved into every checkpoint directory.

```json
{
  "version": "uti_v1",
  "created_utc": "2026-01-24T00:00:00Z",

  "base_text_model": "Qwen/Qwen3-8B-Base",
  "text_vocab_size": 151936,

  "special_tokens": {
    "<|text_start|>": 151936,
    "<|text_end|>": 151937,
    "<|img_start|>": 151938,
    "<|img_end|>": 151939,
    "<|aud_start|>": 151940,
    "<|aud_end|>": 151941,
    "<|gen_text|>": 151942,
    "<|gen_img|>": 151943,
    "<|gen_aud|>": 151944,
    "<|pad_mm|>": 151945
  },

  "ranges": {
    "TEXT": {"start": 0, "end": 151935, "size": 151936},

    "SPECIAL": {"start": 151936, "end": 151999, "size": 64},

    "IMAGE": {"start": 152000, "end": 168383, "size": 16384},

    "AUDIO_CB0": {"start": 168384, "end": 169407, "size": 1024},
    "AUDIO_CB1": {"start": 169408, "end": 170431, "size": 1024},
    "AUDIO_CB2": {"start": 170432, "end": 171455, "size": 1024},
    "AUDIO_CB3": {"start": 171456, "end": 172479, "size": 1024}
  },

  "image_tokenizer": {
    "name_or_path": "AILab-CVC/seed-tokenizer-2",
    "codebook_size": 16384,
    "notes": "token length depends on resolution; meta stores grid"
  },

  "audio_codec": {
    "name_or_path": "facebook/encodec_32khz",
    "sample_rate": 32000,
    "channels": 1,
    "n_codebooks": 4,
    "codebook_size": 1024,
    "clip_seconds": 10.0,
    "serialization": "flatten_by_time_interleaved_codebooks"
  }
}
```

### Required validation rules

Your allocator must enforce:

* Ranges are **non-overlapping**
* Each range is **contiguous**
* Range sizes match:

  * `V_img == image_tokenizer.codebook_size`
  * `V_aud_cb == audio_codec.codebook_size`
  * number of audio ranges == `n_codebooks`
* Special token IDs are inside SPECIAL range
* `vocab_size_total = max_id + 1`

## 3.3 Allocator API

```python
class TokenSpace:
    def __init__(...): ...

    def validate(self) -> None: ...
    def to_json(self) -> dict: ...
    def save_json(self, path: str) -> None: ...
    @staticmethod
    def load_json(path: str) -> "TokenSpace": ...

    # Mapping helpers
    def img_to_global(self, img_token: int) -> int: ...
    def img_from_global(self, global_id: int) -> int: ...

    def aud_to_global(self, codebook_idx: int, token: int) -> int: ...
    def aud_from_global(self, global_id: int) -> Tuple[int, int]: ...
```

---

# 4) Modality tokenization specs

## 4.1 Text tokenization (`encode_text`)

### Contract

* Use the Qwen3 HF tokenizer **without adding special tokens automatically**:

  * `add_special_tokens=False`
* No chat template in UTI; UTI is low-level.

### Why

You need explicit control over boundaries like `<|text_start|>` and `<|text_end|>` to create unified sequences reproducibly.

### Implementation rule

`encode_text(text)` returns only **text vocab IDs** (`< V_text`).

---

## 4.2 Image tokenization (`encode_image` / `decode_image`)

### Inputs

* `PIL.Image` or `np.ndarray`
* Always normalize to:

  * `RGB`
  * fixed resolution policy **in UTI config** (example: `256x256` or `384x384`)
  * deterministic resize method (e.g., bicubic)
  * deterministic crop policy (center-crop if needed)

### Output tokens

* Tokens must be **local codebook IDs** in `[0, V_img-1]` from SEED tokenizer.
* UTI must convert them to global IDs via `TokenSpace.img_to_global()` **only when you build sequences** (return local or global; prefer returning **global** from UTI to reduce caller mistakes).

### Required meta fields (image)

Meta must contain everything needed for decode without guessing:

```json
{
  "modality": "image",
  "tokenizer": "AILab-CVC/seed-tokenizer-2",
  "image_mode": "RGB",
  "orig_size": [H0, W0],
  "proc_size": [H, W],
  "grid": [Gh, Gw],          // if tokenizer is grid-based
  "n_tokens": 1024,          // equals Gh*Gw (or tokenizer-defined)
  "dtype": "uint8",
  "preprocess": {
    "resize": "bicubic",
    "crop": "center"
  }
}
```

### Determinism requirements

* Same input image bytes must yield identical tokens.
* Use fixed preprocessing parameters; no random crop/augment in UTI.

---

## 4.3 Audio tokenization (`encode_audio` / `decode_audio`)

This is the most important part for your “general audio + music from day one” decision.

### Inputs

`encode_audio(wav, sample_rate)` where `wav` is float PCM.

### Canonical preprocessing policy (lock now)

Define these in a config and write them into meta:

* `target_sample_rate`: choose **32k** or **48k** and never mix.

  * For music generation references, 32k is common; if you already standardized on 48k, keep it, but lock it.
* `channels`: choose **mono=1** for early experiments (recommended) to halve tokens/compute.
* `clip_seconds`: choose **10.0s** initial training window (good for music and general audio).
* `normalization`: clamp to `[-1,1]`, optional loudness normalization (but if you do it, do it always).

### EnCodec token shape + serialization

EnCodec typically returns codes shaped like:

* `codes`: `[B, n_codebooks, n_frames]` of integers in `[0, codebook_size-1]`

Your UTI must choose a canonical serialization into a single token list. Recommended:

**Flatten-by-time with interleaved codebooks:**
For each frame `t`:
`[cb0[t], cb1[t], cb2[t], cb3[t]]`

So total length:
`len(tokens) = n_frames * n_codebooks`

### Required meta fields (audio)

```json
{
  "modality": "audio",
  "codec": "facebook/encodec_32khz",
  "orig_sample_rate": 44100,
  "sample_rate": 32000,
  "channels": 1,
  "clip_seconds": 10.0,
  "n_codebooks": 4,
  "codebook_size": 1024,
  "n_frames": 500,
  "serialization": "flatten_by_time_interleaved_codebooks",
  "token_count": 2000,
  "preprocess": {
    "resample": "soxr|torchaudio|scipy",
    "mono": true,
    "trim_or_pad": "pad_end"
  }
}
```

### decode_audio contract

`decode_audio(tokens, meta)` must:

1. Validate `token_count == n_frames * n_codebooks`
2. Convert global token IDs → local `(codebook_idx, token)` values
3. Reshape back to `[1, n_codebooks, n_frames]`
4. Run EnCodec decoder
5. Return `(wav, sample_rate)` where `sample_rate == meta.sample_rate`

### Determinism requirements

* Encoding the same audio clip (same floating waveform values) yields identical tokens.
* Ensure resampling implementation is deterministic (don’t use random dithering).

---

# 5) What UTI returns: local vs global token IDs

To prevent silent bugs, choose one policy and enforce it.

## Recommended policy: UTI returns **global IDs**

* `encode_text`: returns **text IDs** (which are also global IDs because TEXT starts at 0)
* `encode_image`: returns **global IDs** in IMAGE range
* `encode_audio`: returns **global IDs** in AUDIO_CBk ranges

This makes downstream code (sequence assembly, masking, loss routing) much safer. You never risk “forgetting to offset.”

---

# 6) Acceptance tests (must be in repo and runnable)

Create: `qwen3_vlm/tests/test_uti.py` + `test_token_space.py` + small fixed assets.

## 6.1 Determinism tests

### Text

* `encode_text("hello")` twice → identical list

### Image

* Load a fixed PNG from `tests/assets/image0.png`
* Encode twice → identical tokens and identical meta (except for timestamps; meta must not include timestamps)

### Audio

* Load fixed WAV from `tests/assets/audio0.wav`
* Encode twice → identical tokens and meta

**Strictness:** byte-for-byte equal tokens.

## 6.2 Shape sanity tests

### Audio

Given `meta.n_frames` and `meta.n_codebooks`:

* `len(tokens) == n_frames * n_codebooks`
* `codebook_size` matches token-space range size for `AUDIO_CBk`

Also store:

* expected `n_codebooks` and `codebook_size` for your selected EnCodec weights
* assert they match what the codec reports at runtime

### Image

* `len(tokens) == meta.n_tokens`
* if meta has grid: `meta.n_tokens == grid_h * grid_w`

## 6.3 Decode smoke tests

These are “does it decode without guessing,” not “perfect reconstruction.”

### Image decode

* `img2 = decode_image(encode_image(img1))`
* Assert:

  * output is RGB
  * size equals `proc_size`
  * no exceptions

### Audio decode

* `(wav2, sr2) = decode_audio(encode_audio(wav1, sr1))`
* Assert:

  * `sr2 == meta.sample_rate`
  * waveform shape consistent with channels
  * duration approx equals `clip_seconds` (± small tolerance)

## 6.4 Token-space collision test

* Build token-space from actual tokenizer/codec introspection
* Validate:

  * no overlaps
  * `max_id + 1 == vocab_size_total`
  * special IDs are within SPECIAL range

---

# 7) Typical failure modes and hard mitigations

### Failure: Off-by-one offsets

**Symptom:** decode fails or produces garbage; model trains but generation nonsensical.
**Mitigation:**

* Return only global IDs from UTI.
* Add `token_space.validate()` and run it at startup.

### Failure: Collisions between ranges

**Symptom:** text decoder crashes; audio/image decode interprets text IDs as modality IDs.
**Mitigation:**

* Token-space allocator must compute offsets from `V_text` dynamically.
* Persist `token_space.json` and refuse to load a checkpoint if it doesn’t match the current token-space.

### Failure: Silent vocab mismatch across runs

**Symptom:** training cannot resume; embeddings don’t match.
**Mitigation:**

* Save `token_space.json` into every checkpoint directory.
* Save a short `token_space.sha256` and verify on load.
* Store `base_text_model`, `audio_codec_id`, `image_tokenizer_id` in token-space.

### Failure: Audio token length explosion

**Symptom:** sequences exceed context; training stalls.
**Mitigation:**

* Clip audio to fixed seconds in UTI (10s initial).
* Set channels to mono initially.
* Log token lengths at data-build time.

---

# 8) Minimal “UTI config” file (so you never hardcode decisions)

Create: `configs/uti.yaml`

```yaml
base_text_model: Qwen/Qwen3-8B-Base

image_tokenizer:
  name_or_path: AILab-CVC/seed-tokenizer-2
  resolution: 256
  resize: bicubic
  crop: center

audio_codec:
  name_or_path: facebook/encodec_32khz
  target_sample_rate: 32000
  channels: 1
  clip_seconds: 10.0
  serialization: flatten_by_time_interleaved_codebooks

token_space:
  n_special: 64
  special_tokens:
    - "<|text_start|>"
    - "<|text_end|>"
    - "<|img_start|>"
    - "<|img_end|>"
    - "<|aud_start|>"
    - "<|aud_end|>"
    - "<|gen_text|>"
    - "<|gen_img|>"
    - "<|gen_aud|>"
    - "<|pad_mm|>"
```

Your allocator reads this + introspects actual codebook sizes and produces `token_space.json`.

---

# 9) Integration with checkpoints (required behavior)

When you save any training checkpoint (Stage 3+):

* Write `token_space.json` into the checkpoint folder
* Also write:

  * `uti.yaml` (the exact config used)
  * `token_space.sha256`

On load:

* Refuse to load if `token_space.json` differs from current run unless you pass an explicit override (you want strictness by default).

---

# 10) Concrete next steps (implementation order)

1. Implement `TokenSpace` builder + `validate()` + JSON save/load.
2. Implement `UnifiedTokenizer` with:

   * Qwen3 tokenizer for text
   * SEED tokenizer encode/decode for images
   * EnCodec encode/decode for audio
3. Add deterministic preprocessing functions:

   * `preprocess_image(img) -> img_proc, meta_proc`
   * `preprocess_audio(wav, sr) -> wav_proc, sr_proc, meta_proc`
4. Add the test suite + fixed assets (`tests/assets/`)
5. Run tests in container and lock the `uti.yaml` choices

Once Stage 1 passes, proceed to **Stage 2 tokenized shards** without fear that token shapes or ID mappings will drift.

---

Optional: provide the exact local paths/IDs downloaded for:

* EnCodec (`facebook/encodec_32khz` vs `encodec_48khz`)
* SEED tokenizer (the folder name / revision)
  and provide a “drop-in” `token_space.json` template with the correct codebook sizes and the recommended SPECIAL token IDs so everything stays stable from the very first run.
