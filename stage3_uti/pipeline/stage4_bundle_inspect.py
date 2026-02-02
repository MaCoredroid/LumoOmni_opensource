import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set

from transformers import AutoTokenizer

from stage3_uti.pipeline.stage4_golden_generate import (
    _decode_audio_tokens,
    _decode_image_tokens,
    _extract_between,
    _safe_name,
)
from stage3_uti.tokenization.token_space import TokenSpace
from stage3_uti.tokenization.unified_tokenizer import EncodecAudioCodecAdapter, SeedImageTokenizerAdapter
from stage3_uti.utils.train_utils import resolve_pad_id


def _load_jsonl(path: Path) -> List[Dict[str, object]]:
    items: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            items.append(json.loads(line))
    return items


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _grid_for_tokens(count: int) -> List[int]:
    root = int(math.sqrt(count))
    if root * root == count:
        return [root, root]
    return [1, max(1, count)]


def _decode_image_grid(tokens: List[int], token_space: TokenSpace, out_path: Path) -> None:
    image_cfg = token_space.image_tokenizer or {}
    codebook_size = int(image_cfg.get("codebook_size", 8192))
    image_size = int(image_cfg.get("image_size", image_cfg.get("resolution", 224)))
    grid_h, grid_w = _grid_for_tokens(len(tokens))
    total = grid_h * grid_w
    arr = list(tokens)
    if len(arr) < total:
        arr = arr + [0] * (total - len(arr))
    elif len(arr) > total:
        arr = arr[:total]
    if codebook_size <= 1:
        norm = [0.0 for _ in arr]
    else:
        norm = [float(v) / float(codebook_size - 1) for v in arr]
    import numpy as np
    from PIL import Image

    grid_img = (np.array(norm).reshape(grid_h, grid_w) * 255.0).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(grid_img, mode="L").resize((image_size, image_size), Image.NEAREST)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(out_path)


def _bundle_name(item: Dict[str, object], *, use_source: bool) -> str:
    item_id = str(item.get("id"))
    source = str(item.get("source") or "")
    if use_source and source:
        if item_id.startswith(source) or item_id.startswith(f"{source}:") or item_id.startswith(f"{source}_"):
            return _safe_name(item_id, 0)
        return _safe_name(f"{source}_{item_id}", 0)
    return _safe_name(item_id, 0)


def _decode_label_text(
    labels: List[int], token_space: TokenSpace, tokenizer
) -> str:
    tokens = [int(t) for t in labels if isinstance(t, int) and t >= 0]
    if not tokens:
        return ""
    special = token_space.special_tokens
    text_start = int(special["<|text_start|>"])
    text_end = int(special["<|text_end|>"])
    if text_start in tokens and text_end in tokens:
        text_tokens = _extract_between(tokens, text_start, text_end)
    else:
        text_tokens = [t for t in tokens if t not in (text_start, text_end)]
    return tokenizer.decode(text_tokens, skip_special_tokens=True).strip()


def _label_name(label: str, fallback: str, max_len: int = 80) -> str:
    safe = _safe_name(label, 0)
    if not safe:
        safe = _safe_name(fallback, 0)
    if len(safe) > max_len:
        safe = safe[:max_len].rstrip("_-.")
    return safe


def _unique_name(name: str, seen: Dict[str, int]) -> str:
    if name not in seen:
        seen[name] = 1
        return name
    seen[name] += 1
    return f"{name}_{seen[name]}"


def _load_llava_pretrain_images(
    *,
    ids: Set[str],
    json_path: Path,
    images_root: Path,
) -> Dict[str, Path]:
    if not json_path.exists():
        return {}
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    data.sort(key=lambda x: str(x.get("image", "")))
    mapping: Dict[str, Path] = {}
    for full_id in ids:
        try:
            _, idx_str = full_id.split(":", 1)
            idx = int(idx_str)
        except Exception:
            continue
        if idx < 0 or idx >= len(data):
            continue
        image_name = data[idx].get("image")
        if not image_name:
            continue
        img_path = images_root / image_name
        if img_path.exists():
            mapping[full_id] = img_path
    return mapping


def _load_audio_manifest_paths(manifest_path: Path, ids: Set[str]) -> Dict[str, Path]:
    if not manifest_path.exists() or not ids:
        return {}
    mapping: Dict[str, Path] = {}
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                rec_id = str(rec.get("id", ""))
                if rec_id not in ids:
                    continue
                audio_path = (
                    rec.get("modalities", {})
                    .get("audio", {})
                    .get("path")
                )
                if audio_path:
                    mapping[rec_id] = Path(audio_path)
    except Exception:
        return mapping
    return mapping


def _maybe_rename(old_dir: Path, new_dir: Path) -> None:
    if old_dir == new_dir:
        return
    if old_dir.exists() and not new_dir.exists():
        new_dir.parent.mkdir(parents=True, exist_ok=True)
        old_dir.rename(new_dir)


def _build_prompt_map(
    items: List[Dict[str, object]], token_space: TokenSpace, tokenizer
) -> Dict[str, str]:
    special = token_space.special_tokens
    text_start = int(special["<|text_start|>"])
    text_end = int(special["<|text_end|>"])
    prompts: Dict[str, str] = {}
    for item in items:
        input_ids = list(item["input_ids"])
        text_tokens = _extract_between(input_ids, text_start, text_end)
        prompts[str(item.get("id"))] = tokenizer.decode(text_tokens, skip_special_tokens=True)
    return prompts


def _build_output_map(items: List[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for item in items:
        out[str(item.get("id"))] = item
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Bundle Stage 4 golden samples for inspection.")
    parser.add_argument("--token-space-json", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--golden-dir", required=True)
    parser.add_argument("--qual-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--tasks",
        default="a2t,i2t,t2a,t2i",
        help="Comma-separated tasks to bundle (default: a2t,i2t,t2a,t2i)",
    )
    parser.add_argument(
        "--no-decode",
        action="store_true",
        help="Skip audio/image decoding; write only text + ids.",
    )
    parser.add_argument(
        "--decode-missing-only",
        action="store_true",
        help="Only decode input audio/image if the file is missing.",
    )
    parser.add_argument(
        "--name-with-source",
        action="store_true",
        help="Prefix bundle folder names with source label when available.",
    )
    parser.add_argument(
        "--fast-image-grid",
        action="store_true",
        help="Decode images using a simple grid (no model).",
    )
    args = parser.parse_args()

    token_space = TokenSpace.load_json(args.token_space_json)
    special = token_space.special_tokens
    aud_start = int(special["<|aud_start|>"])
    aud_end = int(special["<|aud_end|>"])
    img_start = int(special["<|img_start|>"])
    img_end = int(special["<|img_end|>"])

    checkpoint = Path(args.checkpoint)
    meta_path = checkpoint / "trainable_rows.json"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    with meta_path.open("r", encoding="utf-8") as f:
        base_llm = json.load(f).get("base_llm")
    if not base_llm:
        raise ValueError("Missing base_llm in trainable_rows.json")

    tokenizer = AutoTokenizer.from_pretrained(base_llm, use_fast=True)
    _ = resolve_pad_id(token_space)

    audio_codec = None
    image_tokenizer = None
    device = "cuda"
    fp16 = True
    try:
        import torch

        if not torch.cuda.is_available():
            device = "cpu"
            fp16 = False
    except Exception:
        device = "cpu"
        fp16 = False
    if not args.no_decode:
        audio_cfg = token_space.audio_codec or {}
        audio_name = audio_cfg.get("local_path") or audio_cfg.get("name_or_path")
        if not audio_name:
            raise ValueError("audio_codec name_or_path missing in token_space")
        audio_codec = EncodecAudioCodecAdapter(
            name_or_path=audio_name,
            device=device,
            target_bandwidth=audio_cfg.get("target_bandwidth"),
            n_codebooks=audio_cfg.get("n_codebooks"),
        )

    if not args.no_decode and not args.fast_image_grid:
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
            device=device,
            fp16=bool(image_cfg.get("fp16", True)) and fp16,
        )

    golden_dir = Path(args.golden_dir)
    qual_dir = Path(args.qual_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load golden inputs + generation outputs
    requested = [t.strip() for t in args.tasks.split(",") if t.strip()]
    valid = {"a2t", "i2t", "t2a", "t2i"}
    unknown = [t for t in requested if t not in valid]
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}")

    golden_a2t = _load_jsonl(golden_dir / "golden_a2t_64.jsonl") if "a2t" in requested else []
    golden_i2t = _load_jsonl(golden_dir / "golden_i2t_64.jsonl") if "i2t" in requested else []
    golden_t2a = _load_jsonl(golden_dir / "golden_t2a_64.jsonl") if "t2a" in requested else []
    golden_t2i = _load_jsonl(golden_dir / "golden_t2i_64.jsonl") if "t2i" in requested else []

    out_a2t = (
        _build_output_map(_load_jsonl(qual_dir / "step_2000_golden_a2t.jsonl"))
        if "a2t" in requested
        else {}
    )
    out_i2t = (
        _build_output_map(_load_jsonl(qual_dir / "step_2000_golden_i2t.jsonl"))
        if "i2t" in requested
        else {}
    )
    out_t2a = (
        _build_output_map(_load_jsonl(qual_dir / "step_2000_golden_t2a.jsonl"))
        if "t2a" in requested
        else {}
    )
    out_t2i = (
        _build_output_map(_load_jsonl(qual_dir / "step_2000_golden_t2i.jsonl"))
        if "t2i" in requested
        else {}
    )

    llava_ref_map: Dict[str, Path] = {}
    if "t2i" in requested or "i2t" in requested:
        llava_ids: Set[str] = set()
        if "t2i" in requested:
            llava_ids.update({str(item.get("id")) for item in golden_t2i if item.get("source") == "llava_pretrain"})
        if "i2t" in requested:
            llava_ids.update({str(item.get("id")) for item in golden_i2t if item.get("source") == "llava_pretrain"})
        if llava_ids:
            llava_json = Path("data/llava_pretrain/blip_laion_cc_sbu_558k.json")
            llava_root = Path("data/llava_pretrain")
            llava_ref_map = _load_llava_pretrain_images(ids=llava_ids, json_path=llava_json, images_root=llava_root)

    t2a_ref_map: Dict[str, Path] = {}
    if "t2a" in requested:
        t2a_ids = {str(item.get("id")) for item in golden_t2a}
        if t2a_ids:
            t2a_ref_map.update(
                _load_audio_manifest_paths(Path("stage3_uti/data/manifests/clotho.jsonl"), t2a_ids)
            )
            t2a_ref_map.update(
                _load_audio_manifest_paths(Path("stage3_uti/data/manifests/audiocaps.jsonl"), t2a_ids)
            )
            t2a_ref_map.update(
                _load_audio_manifest_paths(Path("stage3_uti/data/manifests/wavcaps_as_100k.jsonl"), t2a_ids)
            )

    prompt_t2a = _build_prompt_map(golden_t2a, token_space, tokenizer) if "t2a" in requested else {}
    prompt_t2i = _build_prompt_map(golden_t2i, token_space, tokenizer) if "t2i" in requested else {}

    name_counts_t2a: Dict[str, int] = {}
    name_counts_t2i: Dict[str, int] = {}
    name_counts_a2t: Dict[str, int] = {}
    name_counts_i2t: Dict[str, int] = {}

    # T2A bundles: prompt + generated wav
    for item in golden_t2a:
        item_id = str(item.get("id"))
        prompt = prompt_t2a.get(item_id, "").strip()
        if not prompt:
            continue
        base_name = _label_name(prompt, item_id)
        if args.name_with_source and item.get("source"):
            base_name = _label_name(f"{item.get('source')}_{base_name}", item_id)
        bundle_name = _unique_name(base_name, name_counts_t2a)
        old_dir = out_dir / "t2a" / _safe_name(item_id, 0)
        bundle_dir = out_dir / "t2a" / bundle_name
        _maybe_rename(old_dir, bundle_dir)
        _write_text(bundle_dir / "prompt.txt", prompt)
        ref_audio = t2a_ref_map.get(item_id)
        if ref_audio is not None and ref_audio.exists():
            ref_target = bundle_dir / f"reference{ref_audio.suffix}"
            if not args.decode_missing_only or not ref_target.exists():
                try:
                    shutil.copy2(ref_audio, ref_target)
                except Exception:
                    pass
        rec = out_t2a.get(item_id, {})
        audio_path = rec.get("audio_path")
        if audio_path and not args.no_decode:
            src = Path(audio_path)
            if src.exists():
                shutil.copy2(src, bundle_dir / "generated.wav")
        _write_text(bundle_dir / "id.txt", item_id)
        _write_text(bundle_dir / "source.txt", str(item.get("source", "")))

    # T2I bundles: prompt + generated image
    for item in golden_t2i:
        item_id = str(item.get("id"))
        prompt = prompt_t2i.get(item_id, "").strip()
        if not prompt:
            continue
        base_name = _label_name(prompt, item_id)
        if args.name_with_source and item.get("source"):
            base_name = _label_name(f"{item.get('source')}_{base_name}", item_id)
        bundle_name = _unique_name(base_name, name_counts_t2i)
        old_dir = out_dir / "t2i" / _safe_name(item_id, 0)
        bundle_dir = out_dir / "t2i" / bundle_name
        _maybe_rename(old_dir, bundle_dir)
        _write_text(bundle_dir / "prompt.txt", prompt)
        ref_path = llava_ref_map.get(item_id)
        if ref_path is not None and ref_path.exists():
            if not args.decode_missing_only or not (bundle_dir / "reference.png").exists():
                try:
                    from PIL import Image

                    img = Image.open(ref_path).convert("RGB")
                    img.save(bundle_dir / "reference.png")
                except Exception:
                    try:
                        shutil.copy2(ref_path, bundle_dir / f"reference{ref_path.suffix}")
                    except Exception:
                        pass
        else:
            label_ids = list(item.get("labels", []))
            label_tokens = _extract_between(label_ids, img_start, img_end)
            need_ref_decode = args.fast_image_grid or (not args.no_decode and image_tokenizer is not None)
            if args.decode_missing_only and (bundle_dir / "reference.png").exists():
                need_ref_decode = False
            if label_tokens and need_ref_decode:
                try:
                    if args.fast_image_grid or image_tokenizer is None:
                        _decode_image_grid(label_tokens, token_space, bundle_dir / "reference.png")
                    else:
                        _decode_image_tokens(label_tokens, token_space, image_tokenizer, bundle_dir / "reference.png")
                except Exception:
                    pass
        rec = out_t2i.get(item_id, {})
        img_path = rec.get("image_path")
        if img_path and not args.no_decode:
            src = Path(img_path)
            if src.exists():
                shutil.copy2(src, bundle_dir / "generated.png")
        _write_text(bundle_dir / "id.txt", item_id)
        _write_text(bundle_dir / "source.txt", str(item.get("source", "")))

    # A2T bundles: input audio + generated text
    for item in golden_a2t:
        item_id = str(item.get("id"))
        label = _decode_label_text(list(item["labels"]), token_space, tokenizer)
        if not label:
            continue
        base_name = _label_name(label, item_id)
        if args.name_with_source and item.get("source"):
            base_name = _label_name(f"{item.get('source')}_{base_name}", item_id)
        bundle_name = _unique_name(base_name, name_counts_a2t)
        old_dir = out_dir / "a2t" / _safe_name(item_id, 0)
        bundle_dir = out_dir / "a2t" / bundle_name
        _maybe_rename(old_dir, bundle_dir)
        input_ids = list(item["input_ids"])
        audio_tokens = _extract_between(input_ids, aud_start, aud_end)
        need_decode = not args.no_decode and audio_codec is not None
        if args.decode_missing_only and (bundle_dir / "input.wav").exists():
            need_decode = False
        if audio_tokens and need_decode:
            try:
                wav, sr = _decode_audio_tokens(audio_tokens, token_space, audio_codec)
                from stage3_uti.pipeline.stage4_golden_generate import _write_wav

                _write_wav(bundle_dir / "input.wav", wav, sr)
            except Exception:
                pass
        rec = out_a2t.get(item_id, {})
        _write_text(bundle_dir / "generated.txt", str(rec.get("text", "")))
        _write_text(bundle_dir / "label.txt", label)
        _write_text(bundle_dir / "id.txt", item_id)
        _write_text(bundle_dir / "source.txt", str(item.get("source", "")))

    # I2T bundles: input image + generated text
    for item in golden_i2t:
        item_id = str(item.get("id"))
        label = _decode_label_text(list(item["labels"]), token_space, tokenizer)
        if not label:
            continue
        base_name = _label_name(label, item_id)
        if args.name_with_source and item.get("source"):
            base_name = _label_name(f"{item.get('source')}_{base_name}", item_id)
        bundle_name = _unique_name(base_name, name_counts_i2t)
        old_dir = out_dir / "i2t" / _safe_name(item_id, 0)
        bundle_dir = out_dir / "i2t" / bundle_name
        _maybe_rename(old_dir, bundle_dir)
        bundle_dir.mkdir(parents=True, exist_ok=True)
        ref_path = llava_ref_map.get(item_id)
        input_png = bundle_dir / "input.png"
        if ref_path is not None and ref_path.exists():
            if not args.decode_missing_only or not input_png.exists():
                try:
                    from PIL import Image

                    img = Image.open(ref_path).convert("RGB")
                    img.save(input_png)
                except Exception:
                    try:
                        shutil.copy2(ref_path, input_png)
                    except Exception:
                        pass
        else:
            input_ids = list(item["input_ids"])
            image_tokens = _extract_between(input_ids, img_start, img_end)
            need_decode = args.fast_image_grid or (not args.no_decode and image_tokenizer is not None)
            if args.decode_missing_only and input_png.exists():
                need_decode = False
            if image_tokens and need_decode:
                try:
                    if args.fast_image_grid or image_tokenizer is None:
                        _decode_image_grid(image_tokens, token_space, input_png)
                    else:
                        _decode_image_tokens(image_tokens, token_space, image_tokenizer, input_png)
                except Exception:
                    pass
        rec = out_i2t.get(item_id, {})
        _write_text(bundle_dir / "generated.txt", str(rec.get("text", "")))
        _write_text(bundle_dir / "label.txt", label)
        _write_text(bundle_dir / "id.txt", item_id)
        _write_text(bundle_dir / "source.txt", str(item.get("source", "")))

    print(f"Wrote bundles under {out_dir}")


if __name__ == "__main__":
    main()
