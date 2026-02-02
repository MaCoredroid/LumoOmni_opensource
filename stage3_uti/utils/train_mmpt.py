import json
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from stage3_uti.data.tokenized_jsonl import TokenizedJsonlDataset, collate_tokenized
from stage3_uti.tokenization.token_space import TokenSpace
from stage3_uti.utils.train_utils import (
    _apply_row_grad_mask,
    _assert_token_space_match,
    _load_trainable_rows,
    _resize_and_init_embeddings,
    build_model,
    load_token_space,
    resolve_pad_id,
    save_checkpoint,
)


def apply_lora(model, lora_cfg: Dict[str, object]):
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=int(lora_cfg.get("r", 8)),
        lora_alpha=int(lora_cfg.get("alpha", 16)),
        lora_dropout=float(lora_cfg.get("dropout", 0.0)),
        bias=str(lora_cfg.get("bias", "none")),
        target_modules=list(lora_cfg.get("target_modules", [])),
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)


def load_lora(model, lora_path: str):
    from peft import PeftModel

    return PeftModel.from_pretrained(model, lora_path)


def _enable_lora_params(model) -> None:
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True


def _build_dataloader(cfg: Dict[str, object], pad_id: int) -> DataLoader:
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


def _compute_token_loss_sums(logits: torch.Tensor, labels: torch.Tensor):
    if logits.size(1) < 2:
        return None, None
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss_mask = shift_labels != -100
    if not bool(loss_mask.any().item()):
        return None, None
    vocab = shift_logits.size(-1)
    loss_per_token = F.cross_entropy(
        shift_logits.view(-1, vocab),
        shift_labels.view(-1),
        reduction="none",
    ).view(shift_labels.size())
    token_sums = (loss_per_token * loss_mask).sum(dim=1)
    token_counts = loss_mask.sum(dim=1)
    return token_sums, token_counts


def _check_dataset_token_space(jsonl_path: str, token_space_sha: str) -> None:
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            sha = obj.get("token_space_sha256")
            if sha and str(sha) != str(token_space_sha):
                raise ValueError(
                    f"token_space_sha256 mismatch in dataset {jsonl_path}: {sha} != {token_space_sha}"
                )
            break


def _save_lora_if_needed(model, output_dir: Path, step: int) -> None:
    if not hasattr(model, "peft_config"):
        return
    lora_dir = output_dir / f"lora_{step}"
    lora_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(lora_dir)


def train_mmpt(cfg: Dict[str, object]) -> None:
    train_cfg = cfg.get("train", {})
    output_dir = Path(train_cfg.get("output_dir", "outputs/stage4_mmpt"))
    output_dir.mkdir(parents=True, exist_ok=True)

    token_space_path = cfg.get("uti", {}).get("token_space_json")
    if not token_space_path:
        raise ValueError("uti.token_space_json required")
    token_space = load_token_space(token_space_path)
    token_space.validate()
    token_space_sha = token_space.sha256()
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
    save_trainable_only = bool(train_cfg.get("save_trainable_only", True))
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

    lora_cfg = cfg.get("model", {}).get("lora", {}) or {}
    lora_enable = bool(lora_cfg.get("enable", False))
    if lora_enable:
        lora_path = lora_cfg.get("load_path")
        if lora_path:
            model = load_lora(model, str(lora_path))
        else:
            model = apply_lora(model, lora_cfg)

    if not getattr(model, "hf_device_map", None):
        model.to(device)

    if train_cfg.get("freeze_base", True):
        for param in model.parameters():
            param.requires_grad = False
        model.get_input_embeddings().weight.requires_grad = True
        if hasattr(model, "lm_head") and model.lm_head is not None:
            model.lm_head.weight.requires_grad = True
        _apply_row_grad_mask(model.get_input_embeddings().weight, text_vocab_size)
        if hasattr(model, "lm_head") and model.lm_head is not None:
            _apply_row_grad_mask(model.lm_head.weight, text_vocab_size)
        if lora_enable:
            _enable_lora_params(model)

    data_cfg = cfg.get("data", {})
    _check_dataset_token_space(data_cfg["jsonl_path"], token_space_sha)
    dataloader = _build_dataloader(cfg, pad_id)

    lr = float(train_cfg.get("lr", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    lr_embed = float(train_cfg.get("lr_embed", lr))
    lr_head = float(train_cfg.get("lr_head", lr))
    lr_lora = float(train_cfg.get("lr_lora", lr))
    wd_lora = float(train_cfg.get("weight_decay_lora", 0.0))

    params = []
    seen = set()

    def _add_group(named_params, group_lr, group_wd):
        group = []
        for param in named_params:
            if param is None or not param.requires_grad:
                continue
            pid = id(param)
            if pid in seen:
                continue
            seen.add(pid)
            group.append(param)
        if group:
            params.append({"params": group, "lr": group_lr, "weight_decay": group_wd})

    emb = model.get_input_embeddings()
    if emb is not None and emb.weight.requires_grad:
        _add_group([emb.weight], lr_embed, weight_decay)
    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None and lm_head.weight.requires_grad:
        _add_group([lm_head.weight], lr_head, weight_decay)
    if lora_enable:
        lora_params = [p for n, p in model.named_parameters() if "lora_" in n]
        _add_group(lora_params, lr_lora, wd_lora)

    if not params:
        params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    gradient_accum = int(train_cfg.get("gradient_accum", 1))
    train_steps = int(train_cfg.get("train_steps", 2000))
    save_every = int(train_cfg.get("save_every", 500))
    log_every = int(train_cfg.get("log_every", 50))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    log_embed_stats = bool(train_cfg.get("log_embed_stats", False))
    enforce_no_trunc = bool(train_cfg.get("enforce_no_truncation", True))

    use_autocast = device.type == "cuda" and precision in {"bf16", "fp16"}
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    global_step = 0
    micro_step = 0
    loss_token_sum = 0.0
    token_sum = 0
    task_loss_tokens: Dict[str, float] = {}
    task_token_sums: Dict[str, int] = {}

    model.train()
    while global_step < train_steps:
        for batch in dataloader:
            input_ids = torch.tensor(batch["input_ids"], dtype=torch.long, device=device)
            attention_mask = torch.tensor(batch["attention_mask"], dtype=torch.long, device=device)
            labels = torch.tensor(batch["labels"], dtype=torch.long, device=device)
            trunc_flags = batch.get("truncation_flags")
            if enforce_no_trunc and trunc_flags and any(trunc_flags):
                raise ValueError("truncation detected in batch; increase max_seq_len or drop samples")

            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_autocast):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / gradient_accum

            loss.backward()
            micro_step += 1

            with torch.no_grad():
                token_sums, token_counts = _compute_token_loss_sums(outputs.logits.detach(), labels)
                if token_sums is not None and token_counts is not None:
                    tasks = batch.get("task") or []
                    for idx, task in enumerate(tasks):
                        count = int(token_counts[idx].item())
                        if count <= 0:
                            continue
                        token_sum += count
                        loss_token_sum += float(token_sums[idx].item())
                        task_name = str(task or "unknown")
                        task_loss_tokens[task_name] = task_loss_tokens.get(task_name, 0.0) + float(
                            token_sums[idx].item()
                        )
                        task_token_sums[task_name] = task_token_sums.get(task_name, 0) + count

            if micro_step % gradient_accum == 0:
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], grad_clip
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % log_every == 0:
                    extra = ""
                    if token_sum > 0:
                        loss_mean = loss_token_sum / token_sum
                        extra += f" loss_mean={loss_mean:.4f}"
                    if task_token_sums:
                        parts = []
                        for key in sorted(task_token_sums.keys()):
                            denom = task_token_sums[key]
                            if denom:
                                avg = task_loss_tokens[key] / denom
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
                    print(f"[train] step={global_step}{extra}")

                if global_step % save_every == 0:
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
                    if lora_enable:
                        _save_lora_if_needed(model, output_dir, global_step)

                if global_step >= train_steps:
                    break

        if global_step >= train_steps:
            break

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
    if lora_enable:
        _save_lora_if_needed(model, output_dir, global_step)

    token_space_payload = token_space.to_json()
    with (output_dir / "token_space.json").open("w", encoding="utf-8") as f:
        json.dump(token_space_payload, f, indent=2)
