import argparse
import json
from pathlib import Path

import gradio as gr
import torch

from qwen3_vlm.data.collate import expand_image_tokens, load_image
from qwen3_vlm.data.llava_instruct import LlavaInstructDataset
from qwen3_vlm.train.train_utils import build_model
from qwen3_vlm.utils.checkpointing import lora_dir_for_checkpoint
from qwen3_vlm.utils.config import load_config
from qwen3_vlm.utils.device import resolve_device


def find_latest_checkpoint(output_dir):
    output_dir = Path(output_dir)
    checkpoints = list(output_dir.glob("checkpoint_*.pt"))
    if not checkpoints:
        return None

    def _step(path):
        try:
            return int(path.stem.split("_")[-1])
        except ValueError:
            return -1

    return max(checkpoints, key=_step)


def _load_golden_examples(cfg, image_token, limit=None):
    data_cfg = cfg.get("data", {})
    golden_path = data_cfg.get("golden_set_path")
    if not golden_path:
        return []
    golden_path = Path(golden_path)
    if not golden_path.exists():
        return []

    try:
        dataset = LlavaInstructDataset(
            json_path=data_cfg["json_path"],
            image_root=data_cfg["image_root"],
            image_token=image_token,
        )
    except Exception:
        return []

    examples = []
    with golden_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            meta = dataset.get_metadata(int(entry["id"]))
            if meta is None:
                continue
            messages = meta.get("messages") or []
            user_text = ""
            for msg in messages:
                if msg.get("role") == "user":
                    user_text = msg.get("content", "")
                    break
            user_text = user_text.replace(image_token, "").strip()
            image_rel = entry.get("image_relpath")
            if image_rel and hasattr(dataset, "image_root"):
                image_path = Path(dataset.image_root) / str(image_rel)
            else:
                image_path = Path(meta.get("image_path", ""))
            if not image_path.exists():
                continue
            examples.append([str(image_path), user_text])
            if limit and len(examples) >= limit:
                break
    return examples


def _prepare_prompt(tokenizer, tokens_cfg, num_image_tokens, text, system_text, max_seq_len):
    if not text:
        text = ""
    if tokens_cfg["image_token"] not in text:
        text = f"{tokens_cfg['image_token']}\n{text}".strip()

    if text.count(tokens_cfg["image_token"]) != 1:
        raise ValueError("Expected exactly one <image> token in the prompt.")

    messages = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": text})

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    expanded = expand_image_tokens(
        prompt,
        tokens_cfg["image_token"],
        tokens_cfg["im_start_token"],
        tokens_cfg["image_patch_token"],
        tokens_cfg["im_end_token"],
        num_image_tokens,
    )
    enc = tokenizer(
        expanded,
        add_special_tokens=False,
        padding=False,
        truncation=True,
        max_length=max_seq_len,
    )
    input_ids = torch.tensor([enc["input_ids"]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--lora", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--example_limit", type=int, default=64)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = cfg.get("train", {}).get("output_dir")
    checkpoint = args.checkpoint
    if checkpoint is None and output_dir:
        checkpoint = find_latest_checkpoint(output_dir)
    if checkpoint is None:
        raise FileNotFoundError("No checkpoint found; pass --checkpoint explicitly.")
    checkpoint = str(Path(checkpoint))

    lora_path = args.lora
    if lora_path is None:
        lora_path = lora_dir_for_checkpoint(checkpoint)
    if lora_path is not None:
        cfg.setdefault("model", {}).setdefault("lora", {})["enable"] = True
        cfg["model"]["lora"]["load_path"] = str(lora_path)

    tokenizer_path = args.tokenizer
    if tokenizer_path is None and output_dir:
        candidate = Path(output_dir) / "tokenizer"
        if candidate.exists():
            tokenizer_path = str(candidate)
    if tokenizer_path is not None:
        cfg.setdefault("model", {})["tokenizer_name"] = tokenizer_path

    cfg.setdefault("model", {})["connector_checkpoint"] = checkpoint

    device = resolve_device()
    model, tokenizer, image_processor = build_model(cfg)
    model.to(device)
    model.eval()

    tokens_cfg = cfg["tokens"]
    num_image_tokens = int(cfg["model"]["resampler"]["num_latents"])
    max_seq_len = int(cfg["train"]["max_seq_len"])
    precision = cfg["train"].get("precision", "bf16")
    use_autocast = device.type == "cuda" and precision in {"bf16", "fp16"}
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    examples = _load_golden_examples(cfg, tokens_cfg["image_token"], args.example_limit)

    def predict(image, text, system_text, max_new_tokens, do_sample, temperature, top_p, repetition_penalty):
        if image is None:
            return "Error: please upload an image."
        image = load_image(image)
        input_ids, attention_mask = _prepare_prompt(
            tokenizer,
            tokens_cfg,
            num_image_tokens,
            text,
            system_text,
            max_seq_len,
        )
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        pixel_values = image_processor(images=[image], return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(device)
        prompt_len = int(input_ids.size(1))

        gen_kwargs = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": bool(do_sample),
            "repetition_penalty": float(repetition_penalty),
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = float(temperature)
            gen_kwargs["top_p"] = float(top_p)

        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_autocast):
                gen_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_counts=[1],
                    **gen_kwargs,
                )

        seq = gen_ids[0]
        if seq.numel() > prompt_len:
            gen_slice = seq[prompt_len:]
        else:
            gen_slice = seq
        return tokenizer.decode(gen_slice, skip_special_tokens=True)

    with gr.Blocks() as demo:
        gr.Markdown("# Qwen3-VLM Stage 2 Inference")
        with gr.Row():
            image_input = gr.Image(type="pil", label="Image")
            text_input = gr.Textbox(lines=5, label="User prompt (no need to add <image>)")
        system_input = gr.Textbox(lines=2, label="System prompt (optional)")
        with gr.Row():
            max_new_tokens = gr.Slider(8, 512, value=128, step=1, label="Max new tokens")
            repetition_penalty = gr.Slider(1.0, 2.0, value=1.0, step=0.05, label="Repetition penalty")
        with gr.Row():
            do_sample = gr.Checkbox(value=False, label="Sampling")
            temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature")
            top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
        generate_btn = gr.Button("Generate")
        output = gr.Textbox(lines=8, label="Model output")

        generate_btn.click(
            predict,
            inputs=[
                image_input,
                text_input,
                system_input,
                max_new_tokens,
                do_sample,
                temperature,
                top_p,
                repetition_penalty,
            ],
            outputs=output,
        )

        if examples:
            gr.Examples(examples=examples, inputs=[image_input, text_input])

    demo.launch(server_name=args.host, server_port=args.port, show_error=True)


if __name__ == "__main__":
    main()
