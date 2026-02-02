import json
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from stage3_uti.data.tokenized_jsonl import TokenizedJsonlDataset, collate_tokenized
from stage3_uti.tokenization.token_space import TokenSpace


def load_token_space(path: str) -> TokenSpace:
    return TokenSpace.load_json(path)


def resolve_pad_id(token_space: TokenSpace) -> int:
    special = token_space.special_tokens
    if "<|pad_mm|>" in special:
        return int(special["<|pad_mm|>"])
    # fallback to 0 if not provided
    return 0


def _token_space_equal(a: TokenSpace, b: TokenSpace) -> bool:
    return json.dumps(a.to_json(), sort_keys=True) == json.dumps(b.to_json(), sort_keys=True)


def _assert_token_space_match(path: Path, token_space: TokenSpace, context: str) -> None:
    if not path.exists():
        raise ValueError(f"{context} missing token_space.json: {path}")
    other = load_token_space(str(path))
    if not _token_space_equal(other, token_space):
        raise ValueError(f"token_space mismatch for {context}: {path}")


def _init_new_rows(weight: torch.Tensor, start: int, mean: torch.Tensor, std: torch.Tensor) -> None:
    if start >= weight.size(0):
        return
    noise = torch.randn_like(weight[start:]) * std
    weight[start:] = mean + noise


def _resize_and_init_embeddings(
    model,
    *,
    text_vocab_size: int,
    vocab_size_total: int,
    init_scale: float = 0.02,
    init_new_rows: bool = True,
) -> None:
    input_emb = model.get_input_embeddings()
    if input_emb is None:
        raise ValueError("model has no input embeddings")
    if input_emb.weight.size(0) != text_vocab_size:
        model.resize_token_embeddings(text_vocab_size)

    if vocab_size_total > text_vocab_size:
        model.resize_token_embeddings(vocab_size_total)

    emb_weight = model.get_input_embeddings().weight
    if init_new_rows:
        with torch.no_grad():
            base = emb_weight[:text_vocab_size]
            mean = base.mean(dim=0)
            std = base.std(dim=0).clamp_min(1e-6) * init_scale
            _init_new_rows(emb_weight, text_vocab_size, mean, std)

    out_emb = model.get_output_embeddings()
    if out_emb is None:
        return
    out_weight = out_emb.weight
    if init_new_rows:
        with torch.no_grad():
            base = out_weight[:text_vocab_size]
            mean = base.mean(dim=0)
            std = base.std(dim=0).clamp_min(1e-6) * init_scale
            _init_new_rows(out_weight, text_vocab_size, mean, std)


def _apply_row_grad_mask(weight: torch.Tensor, train_start: int) -> None:
    if train_start <= 0:
        return
    mask = torch.ones_like(weight)
    mask[:train_start] = 0

    def _hook(grad: torch.Tensor) -> torch.Tensor:
        return grad * mask

    weight.register_hook(_hook)


def build_model(
    base_llm: str,
    text_vocab_size: int,
    vocab_size_total: int,
    pad_id: int,
    precision: str,
    attn_impl: Optional[str] = None,
    device_map: Optional[str] = None,
    low_cpu_mem_usage: bool = False,
):
    dtype = torch.float32
    if precision == "bf16":
        dtype = torch.bfloat16
    elif precision == "fp16":
        dtype = torch.float16
    load_kwargs = {"torch_dtype": dtype}
    if attn_impl:
        load_kwargs["attn_implementation"] = attn_impl
    if low_cpu_mem_usage:
        load_kwargs["low_cpu_mem_usage"] = True
    if device_map:
        load_kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(base_llm, **load_kwargs)
    _resize_and_init_embeddings(
        model,
        text_vocab_size=text_vocab_size,
        vocab_size_total=vocab_size_total,
        init_new_rows=True,
    )
    model.config.pad_token_id = pad_id
    return model


def build_dataloader(cfg: Dict[str, object], pad_id: int):
    data_cfg = cfg.get("data", {})
    dataset = TokenizedJsonlDataset(data_cfg["jsonl_path"])
    max_seq_len = cfg.get("train", {}).get("max_seq_len")
    trunc_side = cfg.get("train", {}).get("truncation_side", "left")
    batch_size = int(cfg.get("train", {}).get("batch_size", 1))
    num_workers = int(data_cfg.get("num_workers", 0))

    def _collate(batch):
        return collate_tokenized(
            batch,
            pad_id=pad_id,
            max_seq_len=max_seq_len,
            truncation_side=trunc_side,
        )

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=_collate)


def _serialize_token_space(token_space: Optional[object]) -> Optional[Dict[str, object]]:
    if token_space is None:
        return None
    if isinstance(token_space, dict):
        return token_space
    if hasattr(token_space, "to_dict"):
        return token_space.to_dict()
    if hasattr(token_space, "to_json"):
        return token_space.to_json()
    return None


def _save_trainable_rows(
    model,
    ckpt_dir: Path,
    *,
    row_start: int,
    row_end: int,
    base_llm: str,
) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    emb = model.get_input_embeddings()
    if emb is None:
        raise ValueError("model has no input embeddings")
    emb_rows = emb.weight.detach()[row_start:row_end].to("cpu")
    lm_head = getattr(model, "lm_head", None)
    head_rows = None
    if lm_head is not None and lm_head.weight is not None:
        head_rows = lm_head.weight.detach()[row_start:row_end].to("cpu")
    payload = {
        "row_start": int(row_start),
        "row_end": int(row_end),
        "base_llm": base_llm,
        "embeddings": emb_rows,
        "lm_head": head_rows,
        "dtype": str(emb_rows.dtype),
    }
    torch.save(payload, ckpt_dir / "trainable_rows.pt")
    meta = {
        "row_start": int(row_start),
        "row_end": int(row_end),
        "base_llm": base_llm,
        "has_lm_head": head_rows is not None,
    }
    with (ckpt_dir / "trainable_rows.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _load_trainable_rows(
    model,
    ckpt_dir: Path,
    *,
    row_start: int,
    row_end: int,
) -> None:
    payload = torch.load(ckpt_dir / "trainable_rows.pt", map_location="cpu")
    saved_start = int(payload.get("row_start", -1))
    saved_end = int(payload.get("row_end", -1))
    if saved_start != int(row_start) or saved_end != int(row_end):
        raise ValueError(
            f"trainable row range mismatch: expected {row_start}:{row_end} "
            f"got {saved_start}:{saved_end}"
        )
    emb_rows = payload.get("embeddings")
    if emb_rows is None:
        raise ValueError("trainable_rows.pt missing embeddings")
    emb = model.get_input_embeddings()
    if emb is None:
        raise ValueError("model has no input embeddings")
    with torch.no_grad():
        emb.weight.data[row_start:row_end].copy_(emb_rows.to(device=emb.weight.device, dtype=emb.weight.dtype))
        lm_head = getattr(model, "lm_head", None)
        head_rows = payload.get("lm_head")
        if lm_head is not None and head_rows is not None:
            lm_head.weight.data[row_start:row_end].copy_(
                head_rows.to(device=lm_head.weight.device, dtype=lm_head.weight.dtype)
            )


def save_checkpoint(
    model,
    output_dir: Path,
    step: int,
    token_space: Optional[object] = None,
    *,
    save_trainable_only: bool = False,
    row_start: int = 0,
    row_end: int = 0,
    base_llm: str = "",
) -> Path:
    ckpt_dir = output_dir / f"checkpoint_{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if save_trainable_only:
        _save_trainable_rows(
            model,
            ckpt_dir,
            row_start=row_start,
            row_end=row_end,
            base_llm=base_llm,
        )
    else:
        model.save_pretrained(ckpt_dir)
    token_space_payload = _serialize_token_space(token_space)
    if token_space_payload is not None:
        with (ckpt_dir / "token_space.json").open("w", encoding="utf-8") as f:
            json.dump(token_space_payload, f, indent=2)
    return ckpt_dir


def _load_ablation_examples(
    jsonl_path: str, task: str, count: int, seed: int
) -> list[dict]:
    rng = random.Random(seed)
    seen = 0
    reservoir: list[dict] = []
    path = Path(jsonl_path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("task") != task:
                continue
            seen += 1
            if len(reservoir) < count:
                reservoir.append(obj)
            else:
                j = rng.randrange(seen)
                if j < count:
                    reservoir[j] = obj
    if len(reservoir) < count:
        raise RuntimeError(f"Only found {len(reservoir)} rows for task={task}, need {count}")
    return reservoir


def _span_indices(input_ids: list[int], start_tok: int, end_tok: int) -> Optional[Tuple[int, int]]:
    try:
        start_idx = input_ids.index(start_tok)
        end_idx = input_ids.index(end_tok, start_idx + 1)
    except ValueError:
        return None
    return start_idx, end_idx


def _build_ranges(token_space: TokenSpace, modality: str) -> list[Tuple[int, int]]:
    if modality == "image":
        spec = token_space.ranges["IMAGE"]
        return [(int(spec["start"]), int(spec["end"]))]
    ranges = []
    for name, spec in token_space.ranges.items():
        if name.startswith("AUDIO_CB"):
            ranges.append((int(spec["start"]), int(spec["end"])))
    if not ranges:
        raise ValueError("No audio ranges found in token space")
    return ranges


def _sample_from_ranges(rng: random.Random, ranges: list[Tuple[int, int]]) -> int:
    total = sum(end - start + 1 for start, end in ranges)
    r = rng.randrange(total)
    for start, end in ranges:
        size = end - start + 1
        if r < size:
            return start + r
        r -= size
    return ranges[-1][1]


def _apply_ablation(
    input_ids: list[int],
    *,
    start_idx: int,
    end_idx: int,
    mode: str,
    rng: random.Random,
    ranges: list[Tuple[int, int]],
) -> list[int]:
    if mode == "correct":
        return list(input_ids)
    out = list(input_ids)
    span = out[start_idx + 1 : end_idx]
    if not span:
        return out
    if mode == "shuffle":
        rng.shuffle(span)
        out[start_idx + 1 : end_idx] = span
        return out
    if mode == "noise":
        out[start_idx + 1 : end_idx] = [_sample_from_ranges(rng, ranges) for _ in span]
        return out
    if mode == "const":
        const_tok = ranges[0][0]
        out[start_idx + 1 : end_idx] = [const_tok for _ in span]
        return out
    raise ValueError(f"Unknown mode: {mode}")


def _eval_ablation(
    model,
    examples: list[dict],
    *,
    token_space: TokenSpace,
    modality: str,
    seed: int,
    device: torch.device,
) -> Dict[str, float]:
    rng = random.Random(seed)
    modes = ["correct", "shuffle", "noise"]
    losses: Dict[str, list[float]] = {m: [] for m in modes}
    start_tok = int(
        token_space.special_tokens["<|img_start|>"]
        if modality == "image"
        else token_space.special_tokens["<|aud_start|>"]
    )
    end_tok = int(
        token_space.special_tokens["<|img_end|>"]
        if modality == "image"
        else token_space.special_tokens["<|aud_end|>"]
    )
    ranges = _build_ranges(token_space, modality)

    with torch.no_grad():
        for ex in examples:
            input_ids = ex["input_ids"]
            labels = ex["labels"]
            span = _span_indices(input_ids, start_tok, end_tok)
            if span is None:
                continue
            start_idx, end_idx = span
            for mode in modes:
                mod_ids = _apply_ablation(
                    input_ids,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    mode=mode,
                    rng=rng,
                    ranges=ranges,
                )
                input_tensor = torch.tensor([mod_ids], dtype=torch.long, device=device)
                label_tensor = torch.tensor([labels], dtype=torch.long, device=device)
                outputs = model(input_ids=input_tensor, labels=label_tensor)
                losses[mode].append(float(outputs.loss.detach().cpu()))

    if not losses["correct"]:
        raise RuntimeError("No valid examples found for ablation")

    summary: Dict[str, float] = {}
    for mode in modes:
        summary[f"loss_{mode}_mean"] = sum(losses[mode]) / len(losses[mode])
    summary["win_rate_shuffle"] = sum(
        1 for lc, ls in zip(losses["correct"], losses["shuffle"]) if lc < ls
    ) / len(losses["correct"])
    summary["win_rate_noise"] = sum(
        1 for lc, ln in zip(losses["correct"], losses["noise"]) if lc < ln
    ) / len(losses["correct"])
    summary["n_examples"] = float(len(losses["correct"]))
    return summary


def train(cfg: Dict[str, object]) -> None:
    train_cfg = cfg.get("train", {})
    output_dir = Path(train_cfg.get("output_dir", "outputs/stage3_token_lm"))
    output_dir.mkdir(parents=True, exist_ok=True)

    token_space_path = cfg.get("uti", {}).get("token_space_json")
    if not token_space_path:
        raise ValueError("uti.token_space_json required")
    token_space = load_token_space(token_space_path)
    token_space.validate()
    text_vocab_size = int(token_space.text_vocab_size)
    vocab_size = int(token_space.vocab_size_total)
    pad_id = resolve_pad_id(token_space)

    base_llm = cfg.get("model", {}).get("base_llm")
    if not base_llm:
        raise ValueError("model.base_llm required")

    precision = str(train_cfg.get("precision", "bf16"))
    attn_impl = train_cfg.get("attn_implementation")

    tokenizer = AutoTokenizer.from_pretrained(base_llm, use_fast=True)
    if len(tokenizer) != text_vocab_size:
        raise ValueError(
            f"token_space text_vocab_size ({text_vocab_size}) != len(tokenizer) ({len(tokenizer)})"
        )

    resume_from = train_cfg.get("resume_from")
    save_trainable_only = bool(train_cfg.get("save_trainable_only", False))
    if resume_from:
        resume_path = Path(resume_from)
        _assert_token_space_match(resume_path / "token_space.json", token_space, "resume checkpoint")

    existing_token_space = output_dir / "token_space.json"
    if existing_token_space.exists():
        _assert_token_space_match(existing_token_space, token_space, "output_dir")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_map = None
    if device.type == "cuda" and train_cfg.get("device_map"):
        device_map = str(train_cfg.get("device_map"))
    low_cpu_mem_usage = bool(train_cfg.get("low_cpu_mem_usage", False))
    if resume_from:
        dtype = torch.float32
        if precision == "bf16":
            dtype = torch.bfloat16
        elif precision == "fp16":
            dtype = torch.float16
        load_kwargs = {"torch_dtype": dtype}
        if attn_impl:
            load_kwargs["attn_implementation"] = attn_impl
        if low_cpu_mem_usage:
            load_kwargs["low_cpu_mem_usage"] = True
        if device_map:
            load_kwargs["device_map"] = device_map
        resume_path = Path(resume_from)
        if (resume_path / "trainable_rows.pt").exists():
            model = build_model(
                base_llm,
                text_vocab_size,
                vocab_size,
                pad_id,
                precision,
                attn_impl=attn_impl,
                device_map=device_map,
                low_cpu_mem_usage=low_cpu_mem_usage,
            )
            _load_trainable_rows(
                model,
                resume_path,
                row_start=text_vocab_size,
                row_end=vocab_size,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(str(resume_from), **load_kwargs)
            _resize_and_init_embeddings(
                model,
                text_vocab_size=text_vocab_size,
                vocab_size_total=vocab_size,
                init_new_rows=False,
            )
            model.config.pad_token_id = pad_id
    else:
        model = build_model(
            base_llm,
            text_vocab_size,
            vocab_size,
            pad_id,
            precision,
            attn_impl=attn_impl,
            device_map=device_map,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )
    if not getattr(model, "hf_device_map", None):
        model.to(device)

    if train_cfg.get("freeze_base", False):
        for param in model.parameters():
            param.requires_grad = False
        model.get_input_embeddings().weight.requires_grad = True
        if hasattr(model, "lm_head") and model.lm_head is not None:
            model.lm_head.weight.requires_grad = True
        _apply_row_grad_mask(model.get_input_embeddings().weight, text_vocab_size)
        if hasattr(model, "lm_head") and model.lm_head is not None:
            _apply_row_grad_mask(model.lm_head.weight, text_vocab_size)

    dataloader = build_dataloader(cfg, pad_id)

    lr = float(train_cfg.get("lr", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    lr_embed = float(train_cfg.get("lr_embed", lr))
    lr_head = float(train_cfg.get("lr_head", lr))
    params = []
    emb = model.get_input_embeddings()
    if emb is not None and emb.weight.requires_grad:
        params.append({"params": emb.weight, "lr": lr_embed, "weight_decay": weight_decay})
    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None and lm_head.weight.requires_grad:
        params.append({"params": lm_head.weight, "lr": lr_head, "weight_decay": weight_decay})
    if not params:
        params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    gradient_accum = int(train_cfg.get("gradient_accum", 1))
    num_epochs = int(train_cfg.get("num_epochs", 1))
    save_every = int(train_cfg.get("save_every", 1000))
    log_every = int(train_cfg.get("log_every", 50))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    log_embed_stats = bool(train_cfg.get("log_embed_stats", False))

    ablation_every = int(train_cfg.get("eval_ablation_every", 0))
    ablation_jsonl = train_cfg.get("eval_ablation_jsonl")
    ablation_audio_n = int(train_cfg.get("eval_ablation_audio_n", 0))
    ablation_image_n = int(train_cfg.get("eval_ablation_image_n", 0))
    ablation_seed = int(train_cfg.get("eval_ablation_seed", 123))
    ablation_log_path = train_cfg.get("eval_ablation_log")
    ablation_audio_examples = []
    ablation_image_examples = []
    if ablation_every and ablation_jsonl:
        if ablation_audio_n and ablation_image_n:
            ablation_audio_examples = _load_ablation_examples(
                str(ablation_jsonl), "a2t", ablation_audio_n, ablation_seed
            )
            ablation_image_examples = _load_ablation_examples(
                str(ablation_jsonl), "i2t", ablation_image_n, ablation_seed
            )

    use_autocast = device.type == "cuda" and precision in {"bf16", "fp16"}
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    global_step = 0
    task_sums: Dict[str, float] = {}
    task_counts: Dict[str, int] = {}
    model.train()
    for epoch in range(num_epochs):
        for batch in dataloader:
            input_ids = torch.tensor(batch["input_ids"], dtype=torch.long, device=device)
            attention_mask = torch.tensor(batch["attention_mask"], dtype=torch.long, device=device)
            labels = torch.tensor(batch["labels"], dtype=torch.long, device=device)

            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_autocast):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss_unscaled = outputs.loss
                loss = loss_unscaled / gradient_accum

            loss.backward()

            tasks = batch.get("task") if isinstance(batch, dict) else None
            if tasks and len(tasks) == 1:
                task = str(tasks[0] or "unknown")
                task_sums[task] = task_sums.get(task, 0.0) + float(loss_unscaled.detach().cpu())
                task_counts[task] = task_counts.get(task, 0) + 1

            if (global_step + 1) % gradient_accum == 0:
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], grad_clip
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if global_step % log_every == 0:
                extra = ""
                if task_counts:
                    parts = []
                    for key in sorted(task_counts.keys()):
                        if task_counts[key]:
                            avg = task_sums[key] / task_counts[key]
                            parts.append(f"{key}_avg={avg:.4f}")
                    if parts:
                        extra += " " + " ".join(parts)
                if log_embed_stats:
                    with torch.no_grad():
                        emb = model.get_input_embeddings()
                        emb_rows = emb.weight[text_vocab_size:vocab_size]
                        norms = torch.linalg.norm(emb_rows, dim=1)
                        emb_mean = float(norms.mean().detach().cpu())
                        emb_std = float(norms.std().detach().cpu())
                        extra += f" emb_norm_mean={emb_mean:.4f} emb_norm_std={emb_std:.4f}"
                        lm_head = getattr(model, "lm_head", None)
                        if lm_head is not None and lm_head.weight is not None:
                            head_rows = lm_head.weight[text_vocab_size:vocab_size]
                            head_norms = torch.linalg.norm(head_rows, dim=1)
                            head_mean = float(head_norms.mean().detach().cpu())
                            head_std = float(head_norms.std().detach().cpu())
                            extra += f" head_norm_mean={head_mean:.4f} head_norm_std={head_std:.4f}"
                loss_raw = float(loss_unscaled.detach().cpu())
                loss_scaled = float(loss.detach().cpu())
                print(
                    f"[train] epoch={epoch} step={global_step} loss={loss_raw:.4f} "
                    f"loss_scaled={loss_scaled:.4f}{extra}"
                )

            if global_step > 0 and global_step % save_every == 0:
                save_checkpoint(
                    model,
                    output_dir,
                    global_step,
                    token_space=token_space,
                    save_trainable_only=save_trainable_only,
                    row_start=text_vocab_size,
                    row_end=vocab_size,
                    base_llm=base_llm,
                )

            if (
                ablation_every
                and ablation_audio_examples
                and ablation_image_examples
                and global_step > 0
                and global_step % ablation_every == 0
            ):
                model.eval()
                audio_metrics = _eval_ablation(
                    model,
                    ablation_audio_examples,
                    token_space=token_space,
                    modality="audio",
                    seed=ablation_seed,
                    device=device,
                )
                image_metrics = _eval_ablation(
                    model,
                    ablation_image_examples,
                    token_space=token_space,
                    modality="image",
                    seed=ablation_seed,
                    device=device,
                )
                record = {
                    "step": global_step,
                    "audio": audio_metrics,
                    "image": image_metrics,
                }
                print(f"[ablation] step={global_step} audio={audio_metrics} image={image_metrics}")
                if ablation_log_path:
                    log_path = Path(str(ablation_log_path))
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    with log_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(record) + "\n")
                model.train()

            global_step += 1

    save_checkpoint(
        model,
        output_dir,
        global_step,
        token_space=token_space,
        save_trainable_only=save_trainable_only,
        row_start=text_vocab_size,
        row_end=vocab_size,
        base_llm=base_llm,
    )

    # persist token_space.json into output for reproducibility
    token_space_payload = _serialize_token_space(token_space)
    if token_space_payload is not None:
        with (output_dir / "token_space.json").open("w", encoding="utf-8") as f:
            json.dump(token_space_payload, f, indent=2)
