import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from stage3_uti.tokenization.token_space import TokenSpace
from stage3_uti.tokenization.unified_tokenizer import EncodecAudioCodecAdapter, SeedImageTokenizerAdapter
from stage3_uti.utils.train_utils import _load_trainable_rows, _resize_and_init_embeddings, resolve_pad_id


def _load_model(
    checkpoint: Path,
    token_space: TokenSpace,
    device: torch.device,
    device_map: Optional[str],
    lora_path: Optional[Path],
):
    trainable_rows_path = checkpoint / "trainable_rows.pt"
    if not trainable_rows_path.exists():
        raise FileNotFoundError(trainable_rows_path)

    base_llm = ""
    meta_path = checkpoint / "trainable_rows.json"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        base_llm = str(meta.get("base_llm", ""))
    if not base_llm:
        payload = torch.load(trainable_rows_path, map_location="cpu")
        base_llm = str(payload.get("base_llm", ""))
    if not base_llm:
        raise ValueError("trainable_rows checkpoint missing base_llm metadata")

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    load_kwargs: Dict[str, object] = {"torch_dtype": dtype, "low_cpu_mem_usage": True}
    if device_map:
        load_kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(base_llm, **load_kwargs)
    _resize_and_init_embeddings(
        model,
        text_vocab_size=int(token_space.text_vocab_size),
        vocab_size_total=int(token_space.vocab_size_total),
        init_new_rows=True,
    )
    model.config.pad_token_id = resolve_pad_id(token_space)
    _load_trainable_rows(
        model,
        checkpoint,
        row_start=int(token_space.text_vocab_size),
        row_end=int(token_space.vocab_size_total),
    )

    if lora_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(lora_path))

    if not getattr(model, "hf_device_map", None):
        model.to(device)
    model.eval()
    return model, base_llm


def _first_label_span(labels: List[int]) -> Optional[Tuple[int, int]]:
    start = None
    end = None
    for idx, val in enumerate(labels):
        if val != -100 and start is None:
            start = idx
        if val != -100:
            end = idx
    if start is None or end is None:
        return None
    return start, end + 1


def _extract_between(input_ids: List[int], start_tok: int, end_tok: int) -> List[int]:
    try:
        start_idx = input_ids.index(start_tok)
        end_idx = input_ids.index(end_tok, start_idx + 1)
    except ValueError:
        return []
    if end_idx <= start_idx + 1:
        return []
    return input_ids[start_idx + 1 : end_idx]


def _write_wav(path: Path, wav: torch.Tensor | List[float] | np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import soundfile as sf

        sf.write(str(path), wav, sample_rate)
        return
    except Exception:
        pass
    try:
        from scipy.io import wavfile

        arr = wav
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
        arr = np.asarray(arr, dtype=np.float32)
        wavfile.write(str(path), sample_rate, arr.T if arr.ndim > 1 else arr)
        return
    except Exception:
        pass
    import wave

    arr = wav
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.T
    arr16 = (arr * 32767.0).clip(-32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1 if arr16.ndim == 1 else arr16.shape[1])
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(arr16.tobytes())


def _safe_name(value: object, fallback: int) -> str:
    text = str(value) if value is not None else ""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    return safe if safe else str(fallback)


def _grid_for_tokens(count: int) -> Optional[List[int]]:
    root = int(math.sqrt(count))
    if root * root == count:
        return [root, root]
    return None


def _decode_image_tokens(
    tokens: List[int],
    token_space: TokenSpace,
    image_tokenizer: SeedImageTokenizerAdapter,
    out_path: Path,
) -> None:
    local_tokens = [token_space.img_from_global(int(t)) for t in tokens]
    img = image_tokenizer.decode(np.array(local_tokens, dtype=np.int64), {})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def _decode_audio_tokens(
    tokens: List[int], token_space: TokenSpace, codec: EncodecAudioCodecAdapter
) -> Tuple[np.ndarray, int]:
    n_codebooks = int(token_space.audio_codec.get("n_codebooks", 1))
    n_frames = len(tokens) // n_codebooks
    usable = n_frames * n_codebooks
    tokens = tokens[:usable]
    codes = np.zeros((n_codebooks, n_frames), dtype=np.int64)
    idx = 0
    for t in range(n_frames):
        for cb in range(n_codebooks):
            cb_idx, tok = token_space.aud_from_global(int(tokens[idx]))
            if cb_idx != cb:
                raise ValueError("audio token codebook mismatch")
            codes[cb, t] = tok
            idx += 1
    # Encodec decode expects (nb_frames, batch, n_q, frame_len). We only have a flat stream,
    # so treat the entire sequence as a single frame.
    meta = {"scales": [1.0], "nb_frames": 1, "frame_len": int(n_frames)}
    wav = codec.decode(codes, meta)
    sample_rate = int(token_space.audio_codec.get("target_sample_rate", codec.sample_rate))
    return wav, sample_rate


def _load_jsonl(path: Path) -> List[Dict[str, object]]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            items.append(json.loads(line))
    return items


def _generate_task(
    *,
    task: str,
    items: List[Dict[str, object]],
    model,
    tokenizer,
    token_space: TokenSpace,
    audio_codec: EncodecAudioCodecAdapter,
    image_tokenizer: SeedImageTokenizerAdapter,
    output_dir: Path,
    tag: str,
    max_new_tokens_default: int,
    device: torch.device,
) -> Dict[str, object]:
    special = token_space.special_tokens
    text_start = int(special["<|text_start|>"])
    text_end = int(special["<|text_end|>"])
    aud_start = int(special["<|aud_start|>"])
    aud_end = int(special["<|aud_end|>"])
    img_start = int(special["<|img_start|>"])
    img_end = int(special["<|img_end|>"])
    pad_id = resolve_pad_id(token_space)

    if task in {"a2t", "i2t"}:
        eos_id = text_end
    elif task == "t2a":
        eos_id = aud_end
    else:
        eos_id = img_end

    summary = {"task": task, "total": 0, "decoded": 0, "failures": 0}
    out_jsonl = output_dir / f"{tag}_golden_{task}.jsonl"
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / f"{tag}_t2a_audio" if task == "t2a" else None
    image_dir = output_dir / f"{tag}_t2i_image" if task == "t2i" else None
    if audio_dir:
        audio_dir.mkdir(parents=True, exist_ok=True)
    if image_dir:
        image_dir.mkdir(parents=True, exist_ok=True)

    with out_jsonl.open("w", encoding="utf-8") as out_f:
        for item in items:
            input_ids = list(item["input_ids"])
            labels = list(item["labels"])
            span = _first_label_span(labels)
            if span is None:
                continue
            start, end = span
            prefix = input_ids[:start]
            target = input_ids[start:end]
            max_new = max(len(target), max_new_tokens_default)

            input_tensor = torch.tensor([prefix], dtype=torch.long, device=device)
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_tensor,
                    do_sample=False,
                    max_new_tokens=max_new,
                    eos_token_id=eos_id,
                    pad_token_id=pad_id,
                )
            gen = outputs[0].tolist()[len(prefix) :]

            eos_pos = None
            for idx, tok in enumerate(gen):
                if tok == eos_id:
                    eos_pos = idx
                    break
            if eos_pos is None:
                decoded_ok = False
                gen_trim = gen
            else:
                decoded_ok = True
                gen_trim = gen[: eos_pos + 1]

            record = {
                "id": item.get("id"),
                "source": item.get("source"),
                "task": task,
                "target_len": len(target),
                "generated_len": len(gen_trim),
                "decoded_ok": decoded_ok,
            }

            if task in {"a2t", "i2t"}:
                text_tokens = _extract_between(gen_trim, text_start, text_end)
                record["text"] = tokenizer.decode(text_tokens, skip_special_tokens=True)
            elif task == "t2a":
                audio_tokens = _extract_between(gen_trim, aud_start, aud_end)
                if decoded_ok and audio_tokens:
                    try:
                        wav, sr = _decode_audio_tokens(audio_tokens, token_space, audio_codec)
                        safe_id = _safe_name(item.get("id"), summary["total"])
                        wav_path = audio_dir / f"{safe_id}.wav"
                        _write_wav(wav_path, wav, sr)
                        record["audio_path"] = str(wav_path)
                        summary["decoded"] += 1
                    except Exception as exc:
                        record["decode_error"] = str(exc)
                        summary["failures"] += 1
                else:
                    summary["failures"] += 1
                record["audio_token_len"] = len(audio_tokens)
            else:
                image_tokens = _extract_between(gen_trim, img_start, img_end)
                if decoded_ok and image_tokens:
                    safe_id = _safe_name(item.get("id"), summary["total"])
                    img_path = image_dir / f"{safe_id}.png"
                    try:
                        _decode_image_tokens(image_tokens, token_space, image_tokenizer, img_path)
                        record["image_path"] = str(img_path)
                        summary["decoded"] += 1
                    except Exception as exc:
                        record["decode_error"] = str(exc)
                        summary["failures"] += 1
                else:
                    summary["failures"] += 1
                record["image_token_len"] = len(image_tokens)

            summary["total"] += 1
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate golden samples for Stage 4.")
    parser.add_argument("--token-space-json", required=True)
    parser.add_argument("--uti-config", required=False, default="")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--lora", default="")
    parser.add_argument("--golden-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device-map", default="")
    parser.add_argument("--max-new-text", type=int, default=256)
    parser.add_argument("--max-new-audio", type=int, default=1024)
    parser.add_argument("--max-new-image", type=int, default=1024)
    parser.add_argument(
        "--tasks",
        default="a2t,i2t,t2a,t2i",
        help="Comma-separated task list to run (default: a2t,i2t,t2a,t2i)",
    )
    args = parser.parse_args()

    token_space = TokenSpace.load_json(args.token_space_json)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_map = args.device_map or None
    lora_path = Path(args.lora) if args.lora else None

    model, base_llm = _load_model(
        Path(args.checkpoint),
        token_space,
        device,
        device_map,
        lora_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_llm, use_fast=True)

    audio_cfg = token_space.audio_codec or {}
    audio_name = audio_cfg.get("local_path") or audio_cfg.get("name_or_path")
    if not audio_name:
        raise ValueError("audio_codec name_or_path missing in token_space")
    audio_codec = EncodecAudioCodecAdapter(
        name_or_path=audio_name,
        device="cuda" if device.type == "cuda" else "cpu",
        target_bandwidth=audio_cfg.get("target_bandwidth"),
        n_codebooks=audio_cfg.get("n_codebooks"),
    )

    image_cfg = token_space.image_tokenizer or {}
    image_name = image_cfg.get("local_path") or image_cfg.get("name_or_path")
    if not image_name:
        raise ValueError("image_tokenizer name_or_path missing in token_space")
    image_tokenizer = SeedImageTokenizerAdapter(
        model_path=image_name,
        seed2_path=image_cfg.get("seed2_path"),
        diffusion_model_path=image_cfg.get("diffusion_model_path"),
        load_diffusion=bool(image_cfg.get("load_diffusion", True)),
        image_size=int(image_cfg.get("image_size", image_cfg.get("resolution", 224))),
        device="cuda" if device.type == "cuda" else "cpu",
        fp16=bool(image_cfg.get("fp16", True)),
    )

    golden_dir = Path(args.golden_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    task_map = {
        "a2t": args.max_new_text,
        "i2t": args.max_new_text,
        "t2a": args.max_new_audio,
        "t2i": args.max_new_image,
    }
    requested = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if not requested:
        raise ValueError("No tasks specified")
    unknown = [t for t in requested if t not in task_map]
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}")

    summaries = []
    for task in requested:
        max_new = task_map[task]
        golden_path = golden_dir / f"golden_{task}_64.jsonl"
        if not golden_path.exists():
            raise FileNotFoundError(golden_path)
        items = _load_jsonl(golden_path)
        summary = _generate_task(
            task=task,
            items=items,
            model=model,
            tokenizer=tokenizer,
            token_space=token_space,
            audio_codec=audio_codec,
            image_tokenizer=image_tokenizer,
            output_dir=out_dir,
            tag=out_dir.name,
            max_new_tokens_default=max_new,
            device=device,
        )
        summaries.append(summary)

    report_path = out_dir / "golden_summary.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump({"summaries": summaries}, f, indent=2)
    print(json.dumps({"summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()
