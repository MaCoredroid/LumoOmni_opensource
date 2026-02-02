import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, Subset

from qwen3_vlm.data.collate import expand_image_tokens


def select_candidate_indices(full_len, seed, max_samples):
    if max_samples is None:
        return list(range(full_len))
    max_samples = int(max_samples)
    if max_samples <= 0 or max_samples >= full_len:
        return list(range(full_len))
    gen = torch.Generator().manual_seed(seed)
    return torch.randperm(full_len, generator=gen).tolist()[:max_samples]


def _resolve_base(dataset, idx):
    if isinstance(dataset, Subset):
        return _resolve_base(dataset.dataset, dataset.indices[idx])
    return dataset, idx


def _extract_sample_metadata(dataset, idx, image_token):
    base_dataset, base_idx = _resolve_base(dataset, idx)
    if hasattr(base_dataset, "get_metadata"):
        meta = base_dataset.get_metadata(base_idx)
        if meta is None:
            return None
        meta = dict(meta)
        if meta.get("messages"):
            return meta
        if meta.get("prompt") and image_token not in meta["prompt"]:
            meta["prompt"] = f"{meta['prompt']} {image_token}"
        return meta
    sample = base_dataset[base_idx]
    if "messages" in sample:
        images = sample.get("images") or []
        image_path = None
        if images:
            image_path = images[0] if isinstance(images[0], str) else None
        return {
            "id": base_idx,
            "image_relpath": sample.get("image_relpath") or image_path,
            "image_path": image_path,
            "messages": sample.get("messages"),
            "answer": sample.get("answer"),
        }
    prompt = sample.get("prompt")
    answer = sample.get("answer")
    if not prompt or not answer:
        return None
    if image_token not in prompt:
        prompt = f"{prompt} {image_token}"
    images = sample.get("images") or []
    image_path = None
    if images:
        image_path = images[0] if isinstance(images[0], str) else None
    return {
        "id": base_idx,
        "image_relpath": image_path,
        "image_path": image_path,
        "prompt": prompt,
        "answer": answer,
    }


def _compute_label_len(answer, tokenizer, tokens_cfg, num_image_tokens):
    expanded = expand_image_tokens(
        answer,
        tokens_cfg["image_token"],
        tokens_cfg["im_start_token"],
        tokens_cfg["image_patch_token"],
        tokens_cfg["im_end_token"],
        num_image_tokens,
    )
    enc = tokenizer(expanded, add_special_tokens=False, padding=False, truncation=False)
    return len(enc["input_ids"])


def _compute_label_len_from_messages(messages, tokenizer, tokens_cfg, num_image_tokens):
    total = 0
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content") or ""
        expanded = expand_image_tokens(
            content,
            tokens_cfg["image_token"],
            tokens_cfg["im_start_token"],
            tokens_cfg["image_patch_token"],
            tokens_cfg["im_end_token"],
            num_image_tokens,
        )
        enc = tokenizer(expanded, add_special_tokens=False, padding=False, truncation=False)
        total += len(enc["input_ids"])
    return total


def _bucket_names(len_buckets):
    if len(len_buckets) == 4:
        return ["short", "medium", "long"]
    return [f"bucket_{i}" for i in range(len(len_buckets) - 1)]


def _assign_bucket(label_len, len_buckets, names):
    for idx in range(len(len_buckets) - 1):
        upper = len_buckets[idx + 1]
        if label_len <= upper:
            return names[idx]
    return names[-1]


def collect_label_metadata(dataset, indices, tokenizer, cfg):
    tokens_cfg = cfg["tokens"]
    num_image_tokens = int(cfg["model"]["resampler"]["num_latents"])
    len_buckets = cfg["data"].get("len_buckets", [0, 10, 20, 9999])
    bucket_names = _bucket_names(len_buckets)

    metadata = []
    skipped_missing = 0
    for idx in indices:
        meta = _extract_sample_metadata(dataset, idx, tokens_cfg["image_token"])
        if meta is None:
            skipped_missing += 1
            continue
        image_path = meta.get("image_path")
        if image_path:
            if not Path(image_path).exists():
                skipped_missing += 1
                continue
        label_len = meta.get("label_len")
        if label_len is None and meta.get("messages"):
            label_len = _compute_label_len_from_messages(
                meta["messages"], tokenizer, tokens_cfg, num_image_tokens
            )
        if label_len is None:
            answer = meta.get("answer")
            if not answer:
                skipped_missing += 1
                continue
            label_len = _compute_label_len(answer, tokenizer, tokens_cfg, num_image_tokens)
        bucket = _assign_bucket(label_len, len_buckets, bucket_names)
        meta.update({"label_len": label_len, "bucket": bucket})
        metadata.append(meta)

    return metadata, skipped_missing, bucket_names


def stratified_sample(metadata, sample_size, seed, bucket_names):
    if sample_size <= 0 or not metadata:
        return []
    total_available = len(metadata)
    if sample_size > total_available:
        sample_size = total_available

    buckets = {name: [] for name in bucket_names}
    for meta in metadata:
        buckets.setdefault(meta["bucket"], []).append(meta)

    base = sample_size // len(bucket_names)
    remainder = sample_size % len(bucket_names)
    desired = {name: base for name in bucket_names}
    if remainder:
        desired[bucket_names[-1]] += remainder

    gen = torch.Generator().manual_seed(seed)
    selected = []
    selected_ids = set()
    carry = 0
    permuted = {}
    for name in bucket_names:
        bucket_list = buckets.get(name, [])
        if bucket_list:
            perm = torch.randperm(len(bucket_list), generator=gen).tolist()
            permuted[name] = [bucket_list[i] for i in perm]
        else:
            permuted[name] = []

    for name in bucket_names:
        desired_count = desired[name] + carry
        bucket_list = permuted.get(name, [])
        pick = min(desired_count, len(bucket_list))
        if pick:
            chosen = bucket_list[:pick]
            selected.extend(chosen)
            selected_ids.update(item["id"] for item in chosen)
        carry = desired_count - pick if pick < desired_count else 0

    if carry:
        for name in bucket_names:
            for item in permuted.get(name, []):
                if item["id"] in selected_ids:
                    continue
                selected.append(item)
                selected_ids.add(item["id"])
                carry -= 1
                if carry == 0:
                    break
            if carry == 0:
                break

    return selected[:sample_size]


class AnnotatedDataset(Dataset):
    def __init__(self, dataset, indices, metadata):
        self.dataset = dataset
        self.indices = indices
        self.metadata = metadata

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample = dict(self.dataset[self.indices[idx]])
        meta = self.metadata[idx]
        sample["bucket"] = meta.get("bucket")
        sample["label_len"] = meta.get("label_len")
        sample["id"] = meta.get("id", -1)
        sample["image_relpath"] = meta.get("image_relpath") or meta.get("image_path")
        return sample


def build_stratified_split(dataset, tokenizer, cfg, candidate_indices):
    data_cfg = cfg["data"]
    seed = int(cfg["train"]["seed"])
    eval_size = int(data_cfg.get("eval_size", 0))
    if eval_size <= 0:
        return Subset(dataset, candidate_indices), None, len(candidate_indices), {}

    split_dir = Path(cfg["train"]["output_dir"]) / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    split_path = split_dir / f"split_seed{seed}_eval{eval_size}.json"
    if split_path.exists():
        with split_path.open("r", encoding="utf-8") as f:
            saved = json.load(f)
        records = saved.get("eval", [])
        eval_indices = [rec.get("id") for rec in records]
        candidate_set = set(candidate_indices)
        valid = True
        eval_meta = []
        for rec in records:
            rec_id = rec.get("id")
            if rec_id is None or rec_id not in candidate_set:
                valid = False
                break
            meta = _extract_sample_metadata(dataset, rec_id, cfg["tokens"]["image_token"])
            if meta is None:
                valid = False
                break
            image_path = meta.get("image_path")
            if image_path and not Path(image_path).exists():
                valid = False
                break
            eval_meta.append(
                {
                    "id": rec_id,
                    "bucket": rec.get("bucket"),
                    "label_len": rec.get("label_len"),
                    "image_relpath": meta.get("image_relpath") or rec.get("image_path"),
                }
            )
        if valid and eval_indices:
            valid_indices = []
            for idx in candidate_indices:
                meta = _extract_sample_metadata(dataset, idx, cfg["tokens"]["image_token"])
                if meta is None:
                    continue
                image_path = meta.get("image_path")
                if image_path and not Path(image_path).exists():
                    continue
                valid_indices.append(idx)
            train_indices = [idx for idx in valid_indices if idx not in set(eval_indices)]
            bucket_counts = {}
            for item in eval_meta:
                bucket_counts[item["bucket"]] = bucket_counts.get(item["bucket"], 0) + 1
            eval_set = AnnotatedDataset(dataset, eval_indices, eval_meta)
            eval_set.bucket_counts = bucket_counts
            eval_set.eval_size = len(eval_indices)
            train_set = Subset(dataset, train_indices)
            return train_set, eval_set, len(valid_indices), bucket_counts

    metadata, skipped_missing, bucket_names = collect_label_metadata(
        dataset, candidate_indices, tokenizer, cfg
    )
    if skipped_missing:
        print(f"[split] skipped_missing={skipped_missing}")

    selected = stratified_sample(metadata, eval_size, seed, bucket_names)
    eval_indices = [item["id"] for item in selected]
    eval_meta = selected
    eval_id_set = set(eval_indices)
    valid_indices = [item["id"] for item in metadata]
    train_indices = [idx for idx in valid_indices if idx not in eval_id_set]

    bucket_counts = {name: 0 for name in bucket_names}
    for item in eval_meta:
        bucket_counts[item["bucket"]] = bucket_counts.get(item["bucket"], 0) + 1

    eval_set = AnnotatedDataset(dataset, eval_indices, eval_meta)
    eval_set.bucket_counts = bucket_counts
    eval_set.eval_size = len(eval_indices)
    train_set = Subset(dataset, train_indices)

    records = []
    for item in eval_meta:
        records.append(
            {
                "id": item["id"],
                "bucket": item["bucket"],
                "label_len": item["label_len"],
                "image_path": item.get("image_relpath") or item.get("image_path"),
            }
        )
    with split_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": seed,
                "eval_size": len(eval_indices),
                "len_buckets": data_cfg.get("len_buckets", [0, 10, 20, 9999]),
                "eval": records,
                "bucket_counts": bucket_counts,
            },
            f,
            indent=2,
        )

    return train_set, eval_set, len(valid_indices), bucket_counts
