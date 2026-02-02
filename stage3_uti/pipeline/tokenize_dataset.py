import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from stage3_uti.pipeline.sequence_builder import SequenceBuilder
from stage3_uti.tokenization.unified_tokenizer import UnifiedTokenizer


def _load_wav(path: Path) -> Tuple[np.ndarray, int]:
    try:
        from scipy.io import wavfile

        sr, data = wavfile.read(path)
        if data.dtype == np.int16:
            wav = data.astype(np.float32) / 32767.0
        else:
            wav = data.astype(np.float32)
        if wav.ndim == 1:
            wav = wav[None, :]
        else:
            wav = wav.T
        return wav, int(sr)
    except Exception:
        import wave

        with wave.open(str(path), "rb") as wf:
            sr = wf.getframerate()
            n_channels = wf.getnchannels()
            frames = wf.readframes(wf.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
            if n_channels > 1:
                audio = audio.reshape(-1, n_channels).T
            else:
                audio = audio[None, :]
            return audio, int(sr)


def _encode_optional_image(uti, image_path: Optional[str]):
    if not image_path:
        return None, None
    img = Image.open(image_path).convert("RGB")
    tokens, meta = uti.encode_image(img)
    return tokens, meta


def _encode_optional_audio(uti, audio_path: Optional[str]):
    if not audio_path:
        return None, None
    wav, sr = _load_wav(Path(audio_path))
    tokens, meta = uti.encode_audio(wav, sr)
    return tokens, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uti-config", required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    args = parser.parse_args()

    uti = UnifiedTokenizer.from_config(args.uti_config)
    builder = SequenceBuilder(uti.token_space)

    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as src, output_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            text = item.get("text", "")
            task = item.get("task", "t2i")
            image_path = item.get("image_path")
            audio_path = item.get("audio_path")

            text_ids = uti.encode_text(text) if text else []
            image_ids, image_meta = _encode_optional_image(uti, image_path)
            audio_ids, audio_meta = _encode_optional_audio(uti, audio_path)

            if task == "t2i":
                seq = builder.build_t2i(text_ids, image_ids or [])
            elif task == "t2a":
                seq = builder.build_t2a(text_ids, audio_ids or [])
            elif task == "a2t":
                seq = builder.build_a2t(audio_ids or [], text_ids)
            elif task == "i2t":
                seq = builder.build_i2t(image_ids or [], text_ids)
            else:
                seq = builder.build_multi(
                    text_ids=text_ids,
                    image_ids=image_ids,
                    audio_ids=audio_ids,
                    gen=item.get("gen_token"),
                )

            record = {
                "id": item.get("id"),
                "task": task,
                "input_ids": seq,
                "length": len(seq),
                "text_meta": {"len": len(text_ids)},
                "image_meta": image_meta,
                "audio_meta": audio_meta,
            }
            dst.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
