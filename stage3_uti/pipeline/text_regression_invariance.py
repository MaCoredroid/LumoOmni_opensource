import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from stage3_uti.tokenization.token_space import TokenSpace
from stage3_uti.utils.train_utils import _load_trainable_rows, _resize_and_init_embeddings, resolve_pad_id


PROMPTS = [
    "Hello world.",
    "The quick brown fox jumps over the lazy dog.",
    "Write a short list of three fruits:",
    "Question: What is the capital of France? Answer:",
    "In 2024, researchers studied multimodal learning.",
]


def _fingerprint(t: torch.Tensor) -> Dict[str, float]:
    s = float(t.sum().detach().cpu())
    a = float(t.abs().sum().detach().cpu())
    q = float((t * t).sum().detach().cpu())
    return {"sum": s, "abs_sum": a, "sq_sum": q}


def _load_base_model(base_llm: str, dtype: torch.dtype, device_map: str | None):
    kwargs = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if device_map:
        kwargs["device_map"] = device_map
    return AutoModelForCausalLM.from_pretrained(base_llm, **kwargs)


def _param_fingerprints(model) -> Dict[str, Dict[str, float]]:
    fps: Dict[str, Dict[str, float]] = {}
    emb = model.get_input_embeddings()
    emb_name = None
    lm_name = None
    for name, param in model.named_parameters():
        if emb is not None and param is emb.weight:
            emb_name = name
        lm_head = getattr(model, "lm_head", None)
        if lm_head is not None and param is lm_head.weight:
            lm_name = name
    for name, param in model.named_parameters():
        if name == emb_name or name == lm_name:
            continue
        fps[name] = _fingerprint(param)
    return fps


def _text_logits(model, tokenizer, device: torch.device) -> List[torch.Tensor]:
    outs = []
    model.eval()
    with torch.no_grad():
        for prompt in PROMPTS:
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            logits = model(input_ids=ids).logits.detach().cpu()
            outs.append(logits)
    return outs


def _fps_equal(a: Dict[str, float], b: Dict[str, float]) -> bool:
    return a["sum"] == b["sum"] and a["abs_sum"] == b["abs_sum"] and a["sq_sum"] == b["sq_sum"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-llm", required=True)
    parser.add_argument("--token-space", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    token_space = TokenSpace.load_json(args.token_space)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    device_map = "cuda" if device.type == "cuda" else None

    tokenizer = AutoTokenizer.from_pretrained(args.base_llm, use_fast=True)

    model = _load_base_model(args.base_llm, dtype, device_map)
    if not getattr(model, "hf_device_map", None):
        model.to(device)

    text_vocab = int(token_space.text_vocab_size)
    vocab_total = int(token_space.vocab_size_total)

    # Resize first so both base and patched use the same output shape
    _resize_and_init_embeddings(
        model,
        text_vocab_size=text_vocab,
        vocab_size_total=vocab_total,
        init_new_rows=True,
    )
    model.config.pad_token_id = resolve_pad_id(token_space)

    # Base fingerprints and logits before patch
    base_fps = _param_fingerprints(model)
    emb = model.get_input_embeddings().weight
    head = getattr(model, "lm_head", None)

    base_text_fp = _fingerprint(emb[:text_vocab])
    base_mod_fp = _fingerprint(emb[text_vocab:vocab_total])
    base_head_text_fp = _fingerprint(head.weight[:text_vocab]) if head is not None else None
    base_head_mod_fp = _fingerprint(head.weight[text_vocab:vocab_total]) if head is not None else None

    base_logits = _text_logits(model, tokenizer, device)

    # Patch rows
    _load_trainable_rows(
        model,
        Path(args.checkpoint),
        row_start=text_vocab,
        row_end=vocab_total,
    )

    # Fingerprints after patch
    patched_fps = _param_fingerprints(model)
    patched_text_fp = _fingerprint(emb[:text_vocab])
    patched_mod_fp = _fingerprint(emb[text_vocab:vocab_total])
    patched_head_text_fp = _fingerprint(head.weight[:text_vocab]) if head is not None else None
    patched_head_mod_fp = _fingerprint(head.weight[text_vocab:vocab_total]) if head is not None else None

    mismatched = [k for k, v in base_fps.items() if not _fps_equal(v, patched_fps.get(k, {}))]

    # Text regression (compare only text vocab logits)
    patched_logits = _text_logits(model, tokenizer, device)
    max_abs = 0.0
    mean_abs = 0.0
    count = 0
    for base_log, patch_log in zip(base_logits, patched_logits):
        base_log = base_log[..., :text_vocab]
        patch_log = patch_log[..., :text_vocab]
        diff = (base_log - patch_log).abs()
        max_abs = max(max_abs, float(diff.max()))
        mean_abs += float(diff.mean())
        count += 1

    report = {
        "checkpoint": args.checkpoint,
        "device": str(device),
        "text_regression": {
            "max_abs_diff": max_abs,
            "mean_abs_diff": mean_abs / max(count, 1),
            "prompts": len(PROMPTS),
        },
        "integrity_audit": {
            "text_rows_unchanged": _fps_equal(base_text_fp, patched_text_fp),
            "modality_rows_changed": not _fps_equal(base_mod_fp, patched_mod_fp),
            "lm_head_text_rows_unchanged": _fps_equal(base_head_text_fp, patched_head_text_fp) if base_head_text_fp else None,
            "lm_head_modality_rows_changed": not _fps_equal(base_head_mod_fp, patched_head_mod_fp) if base_head_mod_fp else None,
            "frozen_param_mismatch_count": len(mismatched),
            "frozen_param_mismatches": mismatched,
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
