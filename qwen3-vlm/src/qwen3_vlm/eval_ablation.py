import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from qwen3_vlm.data.collate import VLMDataCollator
from qwen3_vlm.train.train_utils import build_dataset, build_model, maybe_subset_and_split
from qwen3_vlm.utils.checkpointing import load_checkpoint, lora_dir_for_checkpoint
from qwen3_vlm.utils.config import load_config
from qwen3_vlm.utils.device import resolve_device
from qwen3_vlm.utils.seed import set_seed


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


def shuffle_pixel_values(pixel_values, image_counts):
    total_images = int(pixel_values.size(0))
    if total_images < 2:
        return pixel_values

    counts = [int(c) for c in image_counts]
    if counts and all(c == counts[0] for c in counts):
        group = counts[0]
        num_groups = len(counts)
        if num_groups < 2:
            return pixel_values
        perm = torch.arange(num_groups, device=pixel_values.device)
        perm = torch.roll(perm, shifts=1)
        indices = []
        for group_idx in perm.tolist():
            start = group_idx * group
            indices.extend(range(start, start + group))
        idx = torch.tensor(indices, device=pixel_values.device)
        return pixel_values.index_select(0, idx)

    perm = torch.arange(total_images, device=pixel_values.device)
    perm = torch.roll(perm, shifts=1)
    return pixel_values[perm]


def build_inputs_embeds_with_projected(model, input_ids, projected, image_counts):
    inputs_embeds = model.llm.get_input_embeddings()(input_ids)
    if projected is None or image_counts is None:
        return inputs_embeds

    image_idx = 0
    for batch_idx, num_images in enumerate(image_counts):
        num_images = int(num_images)
        if num_images == 0:
            continue
        num_tokens = num_images * model.num_image_tokens
        patch_positions = (input_ids[batch_idx] == model.image_patch_token_id).nonzero(
            as_tuple=False
        )
        patch_positions = patch_positions.flatten()
        if patch_positions.numel() != num_tokens:
            raise ValueError(
                f"image token count mismatch: expected {num_tokens}, got {patch_positions.numel()}"
            )

        img_tokens = projected[image_idx : image_idx + num_images].reshape(num_tokens, -1)
        inputs_embeds[batch_idx, patch_positions] = img_tokens.to(inputs_embeds.dtype)
        image_idx += num_images

    return inputs_embeds


def compute_loss_from_projected(model, input_ids, attention_mask, labels, projected, image_counts):
    inputs_embeds = build_inputs_embeds_with_projected(
        model=model,
        input_ids=input_ids,
        projected=projected,
        image_counts=image_counts,
    )
    outputs = model.llm(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        labels=labels,
    )
    return outputs.loss


def select_eval_subset(dataset, num_samples, seed):
    if num_samples <= 0 or len(dataset) <= num_samples:
        return dataset
    gen = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=gen).tolist()[:num_samples]
    return Subset(dataset, indices)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--num_samples", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else int(cfg["train"]["seed"])
    set_seed(seed, deterministic=cfg["train"].get("deterministic", False))

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg["train"].get("allow_tf32", True))
        torch.backends.cudnn.allow_tf32 = bool(cfg["train"].get("allow_tf32", True))
        torch.backends.cudnn.benchmark = bool(cfg["train"].get("cudnn_benchmark", True))
        torch.set_float32_matmul_precision(cfg["train"].get("matmul_precision", "high"))

    device = resolve_device()
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = find_latest_checkpoint(cfg["train"]["output_dir"])
    if ckpt_path is None:
        raise FileNotFoundError("No checkpoint found; pass --checkpoint explicitly.")

    lora_dir = lora_dir_for_checkpoint(ckpt_path)
    if lora_dir and cfg.get("model", {}).get("lora", {}).get("enable"):
        cfg["model"]["lora"]["load_path"] = str(lora_dir)

    model, tokenizer, image_processor = build_model(cfg)
    load_checkpoint(ckpt_path, model, map_location="cpu")

    model.to(device)
    model.eval()

    dataset = build_dataset(cfg, cfg["tokens"]["image_token"])
    _, eval_set, _, _ = maybe_subset_and_split(dataset, cfg, tokenizer=tokenizer)
    eval_base = eval_set if eval_set is not None else dataset
    eval_subset = select_eval_subset(eval_base, int(args.num_samples), seed)

    collator = VLMDataCollator(
        tokenizer=tokenizer,
        image_processor=image_processor,
        image_token=cfg["tokens"]["image_token"],
        image_patch_token=cfg["tokens"]["image_patch_token"],
        im_start_token=cfg["tokens"]["im_start_token"],
        im_end_token=cfg["tokens"]["im_end_token"],
        num_image_tokens=cfg["model"]["resampler"]["num_latents"],
        max_seq_len=cfg["train"]["max_seq_len"],
        pad_to_multiple_of=cfg["train"].get("pad_to_multiple_of", 8),
    )

    batch_size = int(args.batch_size) if args.batch_size else int(cfg["train"]["batch_size"])
    num_workers = int(cfg["train"].get("num_workers", 0))
    prefetch_factor = int(cfg["train"].get("prefetch_factor", 2))
    pin_memory = bool(cfg["train"].get("pin_memory", True)) and device.type == "cuda"
    persistent_workers = bool(cfg["train"].get("persistent_workers", True)) and num_workers > 0
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    dataloader = DataLoader(
        eval_subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        **loader_kwargs,
    )

    precision = cfg["train"]["precision"]
    use_autocast = device.type == "cuda" and precision in {"bf16", "fp16"}
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    loss_sums = {"correct": 0.0, "shuffled": 0.0, "zero": 0.0, "noise": 0.0}
    total_label_tokens = 0
    total_samples = 0
    skipped_batches = 0
    shuffle_fallback_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="ablation"):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            pixel_values = batch["pixel_values"]
            if pixel_values is not None:
                pixel_values = pixel_values.to(device, non_blocking=True)
            image_counts = batch["image_counts"]

            label_tokens = int((labels != -100).sum().item())
            if label_tokens == 0:
                skipped_batches += 1
                continue

            total_samples += int(input_ids.size(0))

            with torch.autocast(
                device_type="cuda",
                dtype=autocast_dtype,
                enabled=use_autocast,
            ):
                projected = None
                if pixel_values is not None:
                    projected = model._encode_images(pixel_values)

                loss_correct = compute_loss_from_projected(
                    model, input_ids, attention_mask, labels, projected, image_counts
                )

                if pixel_values is None:
                    projected_shuffled = None
                else:
                    if pixel_values.size(0) < 2:
                        shuffle_fallback_batches += 1
                        projected_shuffled = projected
                    else:
                        shuffled_pixels = shuffle_pixel_values(pixel_values, image_counts)
                        projected_shuffled = model._encode_images(shuffled_pixels)

                loss_shuffled = compute_loss_from_projected(
                    model, input_ids, attention_mask, labels, projected_shuffled, image_counts
                )

                if projected is None:
                    projected_zero = None
                    projected_noise = None
                else:
                    projected_zero = torch.zeros_like(projected)
                    mean = projected.float().mean()
                    std = projected.float().std(unbiased=False)
                    if float(std.item()) == 0.0:
                        projected_noise = projected_zero
                    else:
                        projected_noise = torch.randn_like(projected) * std.to(projected.dtype)
                        projected_noise = projected_noise + mean.to(projected.dtype)

                loss_zero = compute_loss_from_projected(
                    model, input_ids, attention_mask, labels, projected_zero, image_counts
                )
                loss_noise = compute_loss_from_projected(
                    model, input_ids, attention_mask, labels, projected_noise, image_counts
                )

            total_label_tokens += label_tokens
            loss_sums["correct"] += float(loss_correct.detach().cpu()) * label_tokens
            loss_sums["shuffled"] += float(loss_shuffled.detach().cpu()) * label_tokens
            loss_sums["zero"] += float(loss_zero.detach().cpu()) * label_tokens
            loss_sums["noise"] += float(loss_noise.detach().cpu()) * label_tokens

    if total_label_tokens == 0:
        raise RuntimeError("No labeled tokens found; cannot compute ablation loss.")

    avg_losses = {k: v / total_label_tokens for k, v in loss_sums.items()}
    print(
        "[ablation] samples="
        f"{total_samples} label_tokens={total_label_tokens} skipped_batches={skipped_batches}"
    )
    if shuffle_fallback_batches:
        print(f"[ablation] shuffle_fallback_batches={shuffle_fallback_batches}")
    for key in ("correct", "shuffled", "zero", "noise"):
        print(f"[ablation] loss_{key}={avg_losses[key]:.4f}")
    print(
        "[ablation] delta_shuffled="
        f"{avg_losses['shuffled'] - avg_losses['correct']:.4f} "
        "delta_zero="
        f"{avg_losses['zero'] - avg_losses['correct']:.4f} "
        "delta_noise="
        f"{avg_losses['noise'] - avg_losses['correct']:.4f}"
    )


if __name__ == "__main__":
    main()
