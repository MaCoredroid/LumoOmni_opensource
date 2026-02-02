import json
import wave
from pathlib import Path

import numpy as np
from PIL import Image

from stage3_uti.tokenization.token_space import build_token_space_from_config
from stage3_uti.tokenization.unified_tokenizer import (
    DummyAudioCodec,
    DummyImageTokenizer,
    UnifiedTokenizer,
)


ASSETS = Path(__file__).parent / "assets"


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels).T
        return audio, sr


def _build_tokenizer():
    image_tokenizer = DummyImageTokenizer(grid_size=16, codebook_size=256)
    audio_codec = DummyAudioCodec(sample_rate=16000, channels=1, n_codebooks=2, codebook_size=16)
    cfg = {
        "base_text_model": "dummy-text",
        "image_tokenizer": {"name_or_path": "dummy", "resolution": 16, "resize": "bicubic", "crop": "center"},
        "audio_codec": {
            "name_or_path": "dummy",
            "target_sample_rate": 16000,
            "channels": 1,
            "clip_seconds": 1.0,
            "serialization": "flatten_by_time_interleaved_codebooks",
        },
        "token_space": {
            "n_special": 8,
            "special_tokens": [
                "<|text_start|>",
                "<|text_end|>",
                "<|img_start|>",
                "<|img_end|>",
                "<|aud_start|>",
                "<|aud_end|>",
                "<|gen_text|>",
                "<|pad_mm|>",
            ],
        },
    }
    ts = build_token_space_from_config(
        cfg,
        text_vocab_size=128,
        image_codebook_size=image_tokenizer.codebook_size,
        audio_codebook_size=audio_codec.codebook_size,
        n_codebooks=audio_codec.n_codebooks,
    )

    class DummyTextTokenizer:
        def __len__(self):
            return 128

        def encode(self, text, add_special_tokens=False):
            return [min(127, ord(c)) for c in text][:32]

    return UnifiedTokenizer(
        text_tokenizer=DummyTextTokenizer(),
        image_tokenizer=image_tokenizer,
        audio_codec=audio_codec,
        token_space=ts,
        cfg=cfg,
    )


def test_image_determinism_and_decode():
    tokenizer = _build_tokenizer()
    img = Image.open(ASSETS / "image0.png")

    tokens1, meta1 = tokenizer.encode_image(img)
    tokens2, meta2 = tokenizer.encode_image(img)

    assert tokens1 == tokens2
    assert json.dumps(meta1, sort_keys=True) == json.dumps(meta2, sort_keys=True)
    assert meta1["n_tokens"] == len(tokens1)

    img2 = tokenizer.decode_image(tokens1, meta1)
    assert img2.mode == "RGB"
    assert img2.size == (meta1["proc_size"][1], meta1["proc_size"][0])


def test_audio_determinism_and_decode():
    tokenizer = _build_tokenizer()
    wav, sr = _read_wav(ASSETS / "audio0.wav")

    tokens1, meta1 = tokenizer.encode_audio(wav, sr)
    tokens2, meta2 = tokenizer.encode_audio(wav, sr)

    assert tokens1 == tokens2
    assert json.dumps(meta1, sort_keys=True) == json.dumps(meta2, sort_keys=True)
    assert len(tokens1) == meta1["n_frames"] * meta1["n_codebooks"]

    wav2, sr2 = tokenizer.decode_audio(tokens1, meta1)
    assert sr2 == meta1["sample_rate"]
    expected_len = int(meta1["sample_rate"] * meta1["clip_seconds"])
    assert wav2.shape[-1] == expected_len
