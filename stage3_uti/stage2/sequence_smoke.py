import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from stage3_uti.pipeline.sequence_builder import SequenceBuilder, build_supervised_labels, find_span
from stage3_uti.stage2.utils import percentile_stats
from stage3_uti.stage2.wds_io import iter_tar_samples
from stage3_uti.tokenization.unified_tokenizer import UnifiedTokenizer


class Reservoir:
    def __init__(self, size: int) -> None:
        self.size = int(size)
        self.items: List[Any] = []
        self.seen = 0

    def add(self, item: Any) -> None:
        if self.size <= 0:
            return
        self.seen += 1
        if len(self.items) < self.size:
            self.items.append(item)
            return
        idx = np.random.randint(0, self.seen)
        if idx < self.size:
            self.items[idx] = item


def _pad_to_max(seqs: List[List[int]], pad_id: int, max_len: int) -> List[List[int]]:
    padded = []
    for seq in seqs:
        if len(seq) < max_len:
            seq = seq + [pad_id] * (max_len - len(seq))
        padded.append(seq)
    return padded


def _truncate(seq: List[int], max_len: Optional[int]) -> List[int]:
    if not max_len or len(seq) <= max_len:
        return seq
    # keep tail (text is at end for a2t/i2t)
    return seq[-max_len:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequence builder + dataloader smoke for tokenized shards.")
    parser.add_argument("--uti-config", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--data-dir", default="stage3_uti/data")
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--output-report")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    dataset = args.dataset_name
    tokenized_root = data_dir / "tokenized" / dataset
    reports_dir = data_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    report_path = (
        Path(args.output_report)
        if args.output_report
        else reports_dir / f"{dataset}_sequence_smoke.json"
    )

    uti = UnifiedTokenizer.from_config(args.uti_config)
    builder = SequenceBuilder(uti.token_space)
    special = uti.token_space.special_tokens
    text_start = int(special.get("<|text_start|>"))
    text_end = int(special.get("<|text_end|>"))
    pad_id = int(special.get("<|pad_mm|>", 0))

    reservoir = Reservoir(args.num_samples)
    shards = sorted((tokenized_root / "train").glob("shard-*.tar")) + sorted(
        (tokenized_root / "eval").glob("shard-*.tar")
    )
    for shard in shards:
        for _, sample in iter_tar_samples(shard):
            reservoir.add(sample)

    seq_lengths: Dict[str, List[int]] = {}
    label_lengths: Dict[str, List[int]] = {}
    failures: Dict[str, int] = {}
    trunc_count = 0
    label_zero = 0

    sequences: List[List[int]] = []
    labels_list: List[List[int]] = []

    for sample in reservoir.items:
        meta = sample.get("json") or {}
        task = str(meta.get("task") or "unknown")
        audio = sample.get("audio")
        image = sample.get("image")
        text_out = sample.get("text_out")

        audio_ids = audio.tolist() if isinstance(audio, np.ndarray) else []
        image_ids = image.tolist() if isinstance(image, np.ndarray) else []
        text_ids = text_out.tolist() if isinstance(text_out, np.ndarray) else []

        if task == "a2t":
            seq = builder.build_a2t(audio_ids, text_ids)
        elif task == "i2t":
            seq = builder.build_i2t(image_ids, text_ids)
        else:
            failures["unsupported_task"] = failures.get("unsupported_task", 0) + 1
            continue

        seq_lengths.setdefault(task, []).append(len(seq))
        seq = _truncate(seq, args.max_seq_len)
        if args.max_seq_len and len(seq) == args.max_seq_len:
            trunc_count += 1

        span = find_span(seq, text_start, text_end)
        if span is None:
            failures["missing_text_span"] = failures.get("missing_text_span", 0) + 1
            continue
        # supervise text tokens + <text_end>
        label_span = slice(span.start + 1, span.stop)
        labels = build_supervised_labels(seq, label_span)
        label_len = sum(1 for v in labels if v != -100)
        label_lengths.setdefault(task, []).append(label_len)
        if label_len == 0:
            label_zero += 1

        sequences.append(seq)
        labels_list.append(labels)

    # Dataloader smoke: single forward pass on tiny LM
    smoke_ok = False
    smoke_batch = 0
    smoke_device = "cpu"
    try:
        if sequences:
            # Use a small batch for the forward pass to avoid OOM; stats already
            # cover the full reservoir samples above.
            batch_size = min(8, len(sequences))
            smoke_batch = batch_size
            seq_batch = sequences[:batch_size]
            label_batch = labels_list[:batch_size]
            max_len = max(len(s) for s in seq_batch)
            input_ids = torch.tensor(_pad_to_max(seq_batch, pad_id, max_len), dtype=torch.long)
            labels = torch.tensor(_pad_to_max(label_batch, -100, max_len), dtype=torch.long)
            vocab_size = int(uti.token_space.vocab_size_total)
            hidden = 256
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            smoke_device = device.type
            emb = torch.nn.Embedding(vocab_size, hidden).to(device)
            head = torch.nn.Linear(hidden, vocab_size).to(device)
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            logits = head(emb(input_ids))
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )
            _ = float(loss.detach().cpu())
            smoke_ok = True
    except Exception:
        failures["smoke_forward_failed"] = failures.get("smoke_forward_failed", 0) + 1

    report = {
        "dataset": dataset,
        "samples_checked": len(reservoir.items),
        "sequence_counts": {k: len(v) for k, v in seq_lengths.items()},
        "sequence_lengths": {k: percentile_stats(v) for k, v in seq_lengths.items()},
        "label_lengths": {k: percentile_stats(v) for k, v in label_lengths.items()},
        "label_zero": label_zero,
        "trunc_count": trunc_count,
        "max_seq_len": args.max_seq_len,
        "smoke_batch": smoke_batch,
        "smoke_device": smoke_device,
        "smoke_forward_ok": smoke_ok,
        "failures": failures,
    }

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
