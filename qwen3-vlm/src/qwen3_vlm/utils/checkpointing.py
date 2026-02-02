from pathlib import Path

import torch

from qwen3_vlm.models.llm_qwen3 import is_lora_model


def save_checkpoint(output_dir, step, model, tokenizer):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = output_dir / f"checkpoint_{step}.pt"
    state = {
        "resampler": model.resampler.state_dict(),
        "projector": model.projector.state_dict(),
    }
    if getattr(model, "vision_ln", None) is not None:
        state["vision_ln"] = model.vision_ln.state_dict()
    torch.save(state, ckpt_path)

    tok_dir = output_dir / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(tok_dir)

    if is_lora_model(model.llm):
        lora_dir = output_dir / f"lora_{step}"
        model.llm.save_pretrained(lora_dir, save_embedding_layers=True)
    return ckpt_path


def load_checkpoint(ckpt_path, model, map_location="cpu"):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    model.resampler.load_state_dict(ckpt["resampler"])
    model.projector.load_state_dict(ckpt["projector"])
    if "vision_ln" in ckpt and getattr(model, "vision_ln", None) is not None:
        model.vision_ln.load_state_dict(ckpt["vision_ln"])
    return ckpt


def lora_dir_for_checkpoint(ckpt_path):
    ckpt_path = Path(ckpt_path)
    try:
        step = int(ckpt_path.stem.split("_")[-1])
    except ValueError:
        return None
    lora_dir = ckpt_path.parent / f"lora_{step}"
    if lora_dir.exists():
        return lora_dir
    return None
