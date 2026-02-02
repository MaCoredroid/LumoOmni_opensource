import math
import time
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm

from qwen3_vlm.data.collate import VLMDataCollator
from qwen3_vlm.data.dummy import DummyVLMData
from qwen3_vlm.data.llava_pretrain import LlavaPretrainDataset
from qwen3_vlm.data.llava_instruct import LlavaInstructDataset
from qwen3_vlm.data.splits import build_stratified_split, select_candidate_indices
from qwen3_vlm.models.llm_qwen3 import build_llm, apply_lora, load_lora
from qwen3_vlm.models.projector import MLPProjector
from qwen3_vlm.models.resampler import PerceiverResampler
from qwen3_vlm.models.vision_siglip import build_vision_encoder
from qwen3_vlm.models.vlm import Qwen3VLM
from qwen3_vlm.utils.checkpointing import load_checkpoint, save_checkpoint
from qwen3_vlm.utils.device import resolve_device, resolve_dtype
from qwen3_vlm.utils.golden import ensure_golden_set, load_golden_set, run_golden_generation
from qwen3_vlm.utils.seed import set_seed


def build_dataset(cfg, image_token):
    data_cfg = cfg["data"]
    data_type = data_cfg["type"]
    if data_type == "dummy":
        return DummyVLMData(
            num_samples=data_cfg["num_samples"],
            image_size=data_cfg["image_size"],
            prompt=data_cfg["prompt"],
            image_token=image_token,
        )
    if data_type == "llava_pretrain":
        return LlavaPretrainDataset(
            json_path=data_cfg["json_path"],
            image_root=data_cfg["image_root"],
            prompt=data_cfg["prompt"],
            image_token=image_token,
        )
    if data_type == "llava_instruct":
        return LlavaInstructDataset(
            json_path=data_cfg["json_path"],
            image_root=data_cfg["image_root"],
            image_token=image_token,
        )
    raise ValueError(f"Unknown dataset type: {data_type}")


def _compute_split_sizes(num_samples, eval_ratio):
    eval_size = int(round(num_samples * eval_ratio))
    if eval_ratio > 0 and eval_size == 0 and num_samples > 1:
        eval_size = 1
    if eval_size >= num_samples and num_samples > 1:
        eval_size = num_samples - 1
    train_size = num_samples - eval_size
    return train_size, eval_size


def maybe_subset_and_split(dataset, cfg, tokenizer=None):
    data_cfg = cfg["data"]
    seed = cfg["train"]["seed"]
    full_len = len(dataset)

    max_samples = data_cfg.get("max_samples")
    candidate_indices = select_candidate_indices(full_len, seed, max_samples)
    subset_len = len(candidate_indices)

    train_ratio = data_cfg.get("train_ratio")
    eval_ratio = data_cfg.get("eval_ratio")
    split_mode = data_cfg.get("split_mode")

    if split_mode == "stratified_len":
        if tokenizer is None:
            raise ValueError("tokenizer is required for stratified_len split")
        train_set, eval_set, subset_len, bucket_counts = build_stratified_split(
            dataset, tokenizer, cfg, candidate_indices
        )
        full_train_len = full_len
        if eval_set is not None:
            full_train_len = max(full_len - len(eval_set), 0)
        if eval_set is not None and bucket_counts:
            eval_set.bucket_counts = bucket_counts
        return train_set, eval_set, full_len, full_train_len

    if subset_len < full_len:
        dataset = Subset(dataset, candidate_indices)

    if train_ratio is None and eval_ratio is None:
        return dataset, None, full_len, full_len

    if train_ratio is None:
        train_ratio = 1.0 - float(eval_ratio)
    if eval_ratio is None:
        eval_ratio = 1.0 - float(train_ratio)

    if eval_ratio <= 0:
        full_train_len = int(round(full_len * train_ratio))
        return dataset, None, full_len, full_train_len

    train_size, eval_size = _compute_split_sizes(subset_len, float(eval_ratio))
    gen = torch.Generator().manual_seed(seed)
    train_set, eval_set = random_split(dataset, [train_size, eval_size], generator=gen)

    full_train_len, _ = _compute_split_sizes(full_len, float(eval_ratio))
    return train_set, eval_set, full_len, full_train_len


def set_train_mode(model, cfg):
    model.train()
    if cfg["train"].get("train_lora", False):
        model.llm.train()
    else:
        model.llm.eval()
    model.vision.eval()
    if not cfg["train"].get("train_resampler", False):
        model.resampler.eval()
    if not cfg["train"].get("train_projector", False):
        model.projector.eval()
    if not cfg["train"].get("train_vision_ln", False) and model.vision_ln is not None:
        model.vision_ln.eval()


def _enable_lora_params(llm):
    for name, param in llm.named_parameters():
        if "lora_" in name:
            param.requires_grad = True


def evaluate(model, dataloader, device, use_autocast, autocast_dtype):
    if dataloader is None:
        return None
    model.eval()
    loss_token_sum = 0.0
    label_token_sum = 0
    examples_total = 0
    truncated_total = 0
    bucket_loss_tokens = {}
    bucket_label_tokens = {}
    bucket_sample_counts = {}
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            pixel_values = batch["pixel_values"]
            if pixel_values is not None:
                pixel_values = pixel_values.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_autocast):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    pixel_values=pixel_values,
                    image_counts=batch["image_counts"],
                )
                logits = outputs.logits

            if logits.size(1) < 2:
                continue
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_mask = shift_labels != -100
            if not bool(loss_mask.any().item()):
                continue
            vocab = shift_logits.size(-1)
            loss_per_token = F.cross_entropy(
                shift_logits.view(-1, vocab),
                shift_labels.view(-1),
                reduction="none",
            ).view(shift_labels.size())
            token_sums = (loss_per_token * loss_mask).sum(dim=1)
            token_counts = loss_mask.sum(dim=1)

            batch_label_tokens = int(token_counts.sum().item())
            loss_token_sum += float(token_sums.sum().item())
            label_token_sum += batch_label_tokens
            examples_total += int(input_ids.size(0))

            truncation_flags = batch.get("truncation_flags")
            if truncation_flags is not None:
                if isinstance(truncation_flags, torch.Tensor):
                    truncated_total += int(truncation_flags.sum().item())
                else:
                    truncated_total += int(sum(truncation_flags))

            buckets = batch.get("buckets")
            if buckets is not None:
                for i, bucket in enumerate(buckets):
                    if bucket is None:
                        continue
                    count = int(token_counts[i].item())
                    if count == 0:
                        continue
                    bucket_loss_tokens[bucket] = bucket_loss_tokens.get(bucket, 0.0) + float(
                        token_sums[i].item()
                    )
                    bucket_label_tokens[bucket] = bucket_label_tokens.get(bucket, 0) + count
                    bucket_sample_counts[bucket] = bucket_sample_counts.get(bucket, 0) + 1

    if label_token_sum == 0:
        return None

    eval_loss = loss_token_sum / label_token_sum
    bucket_losses = {}
    for bucket, loss_sum in bucket_loss_tokens.items():
        denom = bucket_label_tokens.get(bucket, 0)
        if denom:
            bucket_losses[bucket] = loss_sum / denom

    truncated_pct = 0.0
    if examples_total:
        truncated_pct = truncated_total / examples_total * 100

    return {
        "loss": eval_loss,
        "bucket_losses": bucket_losses,
        "bucket_counts": bucket_sample_counts,
        "label_tokens_total": label_token_sum,
        "examples_total": examples_total,
        "truncated_pct": truncated_pct,
    }


def build_model(cfg):
    model_cfg = cfg["model"]
    precision = cfg["train"]["precision"]
    dtype = resolve_dtype(precision)

    attn_impl = cfg["train"].get("attn_implementation")
    tokenizer_name = model_cfg.get("tokenizer_name")
    llm, tokenizer = build_llm(
        model_cfg["llm_name"],
        torch_dtype=dtype,
        attn_implementation=attn_impl,
        tokenizer_name=tokenizer_name,
    )
    vision, image_processor = build_vision_encoder(model_cfg["vision_name"], torch_dtype=dtype)

    resampler_cfg = model_cfg["resampler"]
    resampler = PerceiverResampler(
        input_dim=vision.config.hidden_size,
        num_latents=resampler_cfg["num_latents"],
        depth=resampler_cfg["depth"],
        num_heads=resampler_cfg["num_heads"],
        head_dim=resampler_cfg["head_dim"],
    )

    latent_dim = resampler.latent_dim
    projector_cfg = model_cfg["projector"]
    projector = MLPProjector(
        input_dim=latent_dim,
        output_dim=llm.config.hidden_size,
        mlp_ratio=projector_cfg.get("mlp_ratio", 4),
    )

    vision_ln = None
    if model_cfg.get("vision_ln", False):
        vision_ln = nn.LayerNorm(vision.config.hidden_size)

    token_cfg = cfg["tokens"]
    special_tokens = [
        token_cfg["image_token"],
        token_cfg["image_patch_token"],
        token_cfg["im_start_token"],
        token_cfg["im_end_token"],
    ]
    orig_vocab_size = len(tokenizer)
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm.resize_token_embeddings(len(tokenizer))
    if num_added > 0 and tokenizer.eos_token_id is not None:
        with torch.no_grad():
            eos_embed = llm.get_input_embeddings().weight[tokenizer.eos_token_id].detach().clone()
            for token in special_tokens:
                tok_id = tokenizer.convert_tokens_to_ids(token)
                if tok_id >= orig_vocab_size:
                    llm.get_input_embeddings().weight[tok_id].copy_(eos_embed)
    llm.config.pad_token_id = tokenizer.pad_token_id

    lora_cfg = model_cfg.get("lora", {})
    if lora_cfg.get("enable"):
        lora_path = lora_cfg.get("load_path")
        if lora_path:
            llm = load_lora(llm, lora_path)
        else:
            llm = apply_lora(llm, lora_cfg)

    if cfg["train"].get("gradient_checkpointing", False) and hasattr(
        llm, "gradient_checkpointing_enable"
    ):
        llm.gradient_checkpointing_enable()
    if "use_cache" in cfg["train"]:
        llm.config.use_cache = bool(cfg["train"]["use_cache"])

    image_patch_token_id = tokenizer.convert_tokens_to_ids(token_cfg["image_patch_token"])
    model = Qwen3VLM(
        llm=llm,
        vision=vision,
        resampler=resampler,
        projector=projector,
        image_patch_token_id=image_patch_token_id,
        num_image_tokens=resampler_cfg["num_latents"],
        vision_ln=vision_ln,
    )
    connector_ckpt = model_cfg.get("connector_checkpoint")
    if connector_ckpt:
        load_checkpoint(connector_ckpt, model, map_location="cpu")
    return model, tokenizer, image_processor


def configure_trainable(model, cfg):
    model.freeze_all()
    train_cfg = cfg["train"]
    if train_cfg.get("train_resampler", False):
        model.unfreeze_module(model.resampler)
    if train_cfg.get("train_projector", False):
        model.unfreeze_module(model.projector)
    if train_cfg.get("train_vision_ln", False) and model.vision_ln is not None:
        model.unfreeze_module(model.vision_ln)
    if train_cfg.get("train_lora", False):
        _enable_lora_params(model.llm)


def _add_param_group(groups, params, lr, weight_decay, seen):
    params = [p for p in params if p.requires_grad]
    if not params:
        return
    groups.append({"params": params, "lr": lr, "weight_decay": weight_decay})
    for param in params:
        seen.add(id(param))


def build_optimizer(model, cfg):
    train_cfg = cfg["train"]
    base_lr = float(train_cfg.get("lr", 0.0))
    base_wd = float(train_cfg.get("weight_decay", 0.0))
    connector_lr = float(train_cfg.get("lr_connector", base_lr))
    connector_wd = float(train_cfg.get("weight_decay_connector", base_wd))
    lora_lr = float(train_cfg.get("lr_lora", base_lr))
    lora_wd = float(train_cfg.get("weight_decay_lora", 0.0))

    groups = []
    seen = set()

    if train_cfg.get("train_resampler", False) or train_cfg.get("train_projector", False) or train_cfg.get(
        "train_vision_ln", False
    ):
        connector_params = []
        if train_cfg.get("train_resampler", False):
            connector_params.extend(list(model.resampler.parameters()))
        if train_cfg.get("train_projector", False):
            connector_params.extend(list(model.projector.parameters()))
        if train_cfg.get("train_vision_ln", False) and model.vision_ln is not None:
            connector_params.extend(list(model.vision_ln.parameters()))
        _add_param_group(groups, connector_params, connector_lr, connector_wd, seen)

    if train_cfg.get("train_lora", False):
        lora_params = [
            param
            for name, param in model.llm.named_parameters()
            if param.requires_grad and "lora_" in name
        ]
        _add_param_group(groups, lora_params, lora_lr, lora_wd, seen)

    remaining = [p for p in model.parameters() if p.requires_grad and id(p) not in seen]
    _add_param_group(groups, remaining, base_lr, base_wd, seen)

    if not groups:
        groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": base_lr, "weight_decay": base_wd}]

    return torch.optim.AdamW(groups)


def build_scheduler(optimizer, cfg, total_update_steps):
    train_cfg = cfg["train"]
    schedule = train_cfg.get("lr_scheduler")
    if not schedule:
        return None
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.0))
    warmup_steps = int(total_update_steps * warmup_ratio)

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        if total_update_steps <= warmup_steps:
            return 1.0
        progress = float(step - warmup_steps) / float(max(1, total_update_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        if schedule == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        if schedule == "linear":
            return 1.0 - progress
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(cfg):
    set_seed(cfg["train"]["seed"], deterministic=cfg["train"].get("deterministic", False))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg["train"].get("allow_tf32", True))
        torch.backends.cudnn.allow_tf32 = bool(cfg["train"].get("allow_tf32", True))
        torch.backends.cudnn.benchmark = bool(cfg["train"].get("cudnn_benchmark", True))
        torch.set_float32_matmul_precision(cfg["train"].get("matmul_precision", "high"))
    device = resolve_device()

    model, tokenizer, image_processor = build_model(cfg)
    configure_trainable(model, cfg)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    dataset = build_dataset(cfg, cfg["tokens"]["image_token"])
    golden_path = ensure_golden_set(dataset, tokenizer, cfg)
    golden_entries = load_golden_set(golden_path) if golden_path else []
    train_set, eval_set, full_len, full_train_len = maybe_subset_and_split(
        dataset, cfg, tokenizer=tokenizer
    )
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
        train_set,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        collate_fn=collator,
        **loader_kwargs,
    )
    eval_loader = None
    quick_eval_loader = None
    if eval_set is not None:
        eval_loader = DataLoader(
            eval_set,
            batch_size=cfg["train"]["batch_size"],
            shuffle=False,
            collate_fn=collator,
            **loader_kwargs,
        )
        quick_eval_size = int(cfg["train"].get("quick_eval_size", 0))
        if quick_eval_size > 0:
            gen = torch.Generator().manual_seed(cfg["train"]["seed"])
            indices = torch.randperm(len(eval_set), generator=gen).tolist()[:quick_eval_size]
            quick_eval_set = Subset(eval_set, indices)
            quick_eval_loader = DataLoader(
                quick_eval_set,
                batch_size=cfg["train"]["batch_size"],
                shuffle=False,
                collate_fn=collator,
                **loader_kwargs,
            )

    model.to(device)
    set_train_mode(model, cfg)

    optimizer = build_optimizer(model, cfg)

    precision = cfg["train"]["precision"]
    use_autocast = device.type == "cuda" and precision in {"bf16", "fp16"}
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(precision == "fp16" and device.type == "cuda"))

    output_dir = Path(cfg["train"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_len = len(train_set)
    eval_len = len(eval_set) if eval_set is not None else 0
    print(
        "[data] full_len="
        f"{full_len} subset_train={train_len} subset_eval={eval_len} "
        f"full_train={full_train_len}"
    )
    if eval_set is not None and hasattr(eval_set, "bucket_counts"):
        print(f"[data] eval_bucket_counts={eval_set.bucket_counts}")

    global_step = 0
    last_ckpt_path = None
    step_time_total = 0.0
    step_time_count = 0
    estimate_after = int(cfg["train"].get("estimate_after", 50))
    estimate_every = int(cfg["train"].get("estimate_every", cfg["train"]["log_every"]))
    eval_every = int(cfg["train"].get("eval_every", 0))
    full_eval_cfg = cfg["train"].get("full_eval_every")
    full_eval_on_epoch = full_eval_cfg is None
    if full_eval_cfg is not None:
        if isinstance(full_eval_cfg, str) and full_eval_cfg.lower() == "epoch":
            full_eval_on_epoch = True
            eval_every = 0
        else:
            full_eval_on_epoch = False
            eval_every = int(full_eval_cfg)
    quick_eval_steps = int(cfg["train"].get("quick_eval_steps", 0))
    full_steps_per_epoch = None
    if full_train_len > 0:
        full_steps_per_epoch = int(math.ceil(full_train_len / cfg["train"]["batch_size"]))
    steps_per_epoch = len(dataloader)
    use_tqdm = not bool(cfg["train"].get("tqdm_disable", False))
    progress_percent = float(cfg["train"].get("progress_percent", 0.1))
    progress_every = 0
    if progress_percent > 0 and steps_per_epoch > 0:
        progress_every = max(1, int(math.ceil(steps_per_epoch * progress_percent)))
    max_train_steps = int(cfg["train"].get("train_steps", 0))
    total_steps = max_train_steps if max_train_steps > 0 else cfg["train"]["num_epochs"] * steps_per_epoch
    total_update_steps = int(math.ceil(total_steps / cfg["train"]["gradient_accum"]))
    scheduler = build_scheduler(optimizer, cfg, total_update_steps)
    update_step = 0
    stop_training = False
    last_full_eval = None
    grad_clip = float(cfg["train"].get("grad_clip", 0.0))

    for epoch in range(cfg["train"]["num_epochs"]):
        epoch_samples = 0
        epoch_truncated = 0
        epoch_label_zero = 0
        epoch_label_tokens = 0
        epoch_iter = dataloader
        epoch_bar = None
        if use_tqdm:
            epoch_bar = tqdm(dataloader, desc=f"epoch {epoch}")
            epoch_iter = epoch_bar
        for step_idx, batch in enumerate(epoch_iter):
            step_start = time.perf_counter()
            label_token_counts = batch.get("label_token_counts")
            if label_token_counts is not None:
                if isinstance(label_token_counts, torch.Tensor):
                    batch_samples = int(label_token_counts.numel())
                    epoch_label_tokens += int(label_token_counts.sum().item())
                    epoch_label_zero += int((label_token_counts == 0).sum().item())
                else:
                    batch_samples = len(label_token_counts)
                    epoch_label_tokens += int(sum(label_token_counts))
                    epoch_label_zero += int(sum(1 for v in label_token_counts if v == 0))
                epoch_samples += batch_samples
                truncation_flags = batch.get("truncation_flags")
                if truncation_flags is not None:
                    if isinstance(truncation_flags, torch.Tensor):
                        epoch_truncated += int(truncation_flags.sum().item())
                    else:
                        epoch_truncated += int(sum(truncation_flags))
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            pixel_values = batch["pixel_values"]
            if pixel_values is not None:
                pixel_values = pixel_values.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_autocast):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    pixel_values=pixel_values,
                    image_counts=batch["image_counts"],
                )
                loss = outputs.loss / cfg["train"]["gradient_accum"]

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (global_step + 1) % cfg["train"]["gradient_accum"] == 0:
                if grad_clip > 0 and trainable_params:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                update_step += 1

            if global_step % cfg["train"]["log_every"] == 0:
                if epoch_bar is not None:
                    epoch_bar.set_postfix({"loss": float(loss.detach().cpu())})

            if progress_every and (step_idx + 1) % progress_every == 0:
                pct = (step_idx + 1) / steps_per_epoch * 100
                print(f"[progress] epoch={epoch} step={step_idx + 1}/{steps_per_epoch} ({pct:.0f}%)")

            step_time_total += time.perf_counter() - step_start
            step_time_count += 1
            if (
                full_steps_per_epoch is not None
                and step_time_count >= estimate_after
                and estimate_every > 0
                and global_step % estimate_every == 0
            ):
                avg_step = step_time_total / step_time_count
                full_epoch_sec = avg_step * full_steps_per_epoch
                full_total_sec = full_epoch_sec * cfg["train"]["num_epochs"]
                tqdm.write(
                    "[estimate] avg_step="
                    f"{avg_step:.3f}s full_epoch≈{full_epoch_sec/3600:.2f}h "
                    f"full_total≈{full_total_sec/3600:.2f}h"
                )

            if global_step % cfg["train"]["save_every"] == 0 and global_step > 0:
                last_ckpt_path = save_checkpoint(output_dir, global_step, model, tokenizer)
                if cfg["train"].get("run_golden_every") == "save" and golden_entries:
                    run_golden_generation(
                        model=model,
                        tokenizer=tokenizer,
                        image_processor=image_processor,
                        dataset=dataset,
                        golden_entries=golden_entries,
                        cfg=cfg,
                        device=device,
                        global_step=global_step,
                        output_dir=output_dir,
                    )
                    set_train_mode(model, cfg)

            if quick_eval_loader is not None and quick_eval_steps > 0 and global_step > 0:
                if global_step % quick_eval_steps == 0:
                    qmetrics = evaluate(
                        model=model,
                        dataloader=quick_eval_loader,
                        device=device,
                        use_autocast=use_autocast,
                        autocast_dtype=autocast_dtype,
                    )
                    if qmetrics is not None:
                        tqdm.write(
                            "[qeval] step="
                            f"{global_step} loss={qmetrics['loss']:.4f} "
                            f"tokens={qmetrics['label_tokens_total']} "
                            f"samples={qmetrics['examples_total']}"
                        )
                    set_train_mode(model, cfg)

            if eval_loader is not None and eval_every > 0 and global_step > 0:
                if global_step % eval_every == 0:
                    metrics = evaluate(
                        model=model,
                        dataloader=eval_loader,
                        device=device,
                        use_autocast=use_autocast,
                        autocast_dtype=autocast_dtype,
                    )
                    if metrics is not None:
                        last_full_eval = metrics
                        bucket_losses = metrics.get("bucket_losses", {})
                        bucket_msg = " ".join(
                            f"{k}={v:.4f}" for k, v in bucket_losses.items()
                        )
                        tqdm.write(
                            "[eval] step="
                            f"{global_step} loss={metrics['loss']:.4f} "
                            f"tokens={metrics['label_tokens_total']} "
                            f"samples={metrics['examples_total']} "
                            f"truncated={metrics['truncated_pct']:.2f}% "
                            f"{bucket_msg}"
                        )
                    set_train_mode(model, cfg)

            global_step += 1
            if max_train_steps and global_step >= max_train_steps:
                stop_training = True
                break

        if epoch_samples > 0:
            trunc_pct = epoch_truncated / epoch_samples * 100
            label_zero_pct = epoch_label_zero / epoch_samples * 100
            avg_label_tokens = epoch_label_tokens / epoch_samples
            msg = (
                f"[data] epoch={epoch} truncated={trunc_pct:.2f}% "
                f"label_zero={label_zero_pct:.2f}% avg_label_tokens={avg_label_tokens:.1f}"
            )
            if use_tqdm:
                tqdm.write(msg)
            else:
                print(msg)

        if eval_loader is not None and full_eval_on_epoch:
            metrics = evaluate(
                model=model,
                dataloader=eval_loader,
                device=device,
                use_autocast=use_autocast,
                autocast_dtype=autocast_dtype,
            )
            if metrics is not None:
                last_full_eval = metrics
                bucket_losses = metrics.get("bucket_losses", {})
                bucket_msg = " ".join(f"{k}={v:.4f}" for k, v in bucket_losses.items())
                tqdm.write(
                    "[eval] epoch="
                    f"{epoch} loss={metrics['loss']:.4f} "
                    f"tokens={metrics['label_tokens_total']} "
                    f"samples={metrics['examples_total']} "
                    f"truncated={metrics['truncated_pct']:.2f}% "
                    f"{bucket_msg}"
                )
            set_train_mode(model, cfg)

        if cfg["train"].get("run_golden_every") == "epoch" and golden_entries:
            run_golden_generation(
                model=model,
                tokenizer=tokenizer,
                image_processor=image_processor,
                dataset=dataset,
                golden_entries=golden_entries,
                cfg=cfg,
                device=device,
                global_step=global_step,
                output_dir=output_dir,
            )
            set_train_mode(model, cfg)

        if stop_training:
            break

    last_ckpt_path = save_checkpoint(output_dir, global_step, model, tokenizer)
    if cfg["train"].get("run_golden_every") == "end" and golden_entries:
        run_golden_generation(
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            dataset=dataset,
            golden_entries=golden_entries,
            cfg=cfg,
            device=device,
            global_step=global_step,
            output_dir=output_dir,
        )
        set_train_mode(model, cfg)

    if last_full_eval is not None:
        metrics_path = output_dir / "metrics.json"
        avg_step = step_time_total / step_time_count if step_time_count else 0.0
        payload = {
            "eval_loss": last_full_eval["loss"],
            "eval_label_tokens_total": last_full_eval["label_tokens_total"],
            "eval_examples_total": last_full_eval["examples_total"],
            "eval_truncated_pct": last_full_eval["truncated_pct"],
            "eval_bucket_losses": last_full_eval.get("bucket_losses", {}),
            "eval_bucket_counts": last_full_eval.get("bucket_counts", {}),
            "avg_step_sec": avg_step,
            "global_step": global_step,
        }
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    if cfg["train"].get("run_sanity_checks", False):
        run_sanity_checks(
            cfg=cfg,
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            collator=collator,
            checkpoint_path=last_ckpt_path,
            device=device,
        )


def run_sanity_checks(cfg, model, tokenizer, dataset, collator, checkpoint_path, device):
    sanity_cfg = cfg.get("sanity", {})
    num_samples = int(sanity_cfg.get("num_samples", 4))
    max_new_tokens = int(sanity_cfg.get("max_new_tokens", 16))

    if len(dataset) == 0 or num_samples <= 0:
        return

    image_patch_id = tokenizer.convert_tokens_to_ids(cfg["tokens"]["image_patch_token"])
    num_image_tokens = cfg["model"]["resampler"]["num_latents"]

    for idx in range(min(num_samples, len(dataset))):
        sample = dataset[idx]
        batch = collator([sample])
        expected = int(batch["image_counts"][0]) * num_image_tokens
        actual = int((batch["input_ids"][0] == image_patch_id).sum().item())
        if actual != expected:
            raise ValueError(
                f"sanity check failed: expected {expected} image tokens, got {actual}"
            )

    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model, map_location="cpu")
        model.to(device)

    sample = dataset[0]
    batch = collator([sample])
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    pixel_values = batch["pixel_values"]
    if pixel_values is not None:
        pixel_values = pixel_values.to(device)
    image_counts = batch["image_counts"]

    model.eval()
    with torch.no_grad():
        gen_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_counts=image_counts,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        gen_ids_2 = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_counts=image_counts,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    if not torch.equal(gen_ids, gen_ids_2):
        raise RuntimeError("sanity check failed: non-deterministic generation")

    decoded = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    print(f"[sanity] sample generation: {decoded[0]}")
