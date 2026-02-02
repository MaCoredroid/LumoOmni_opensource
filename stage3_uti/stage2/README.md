# Stage 2 Tokenized Dataset Pipeline

This folder implements the Stage 2 contract from `stage3.2spec.md`:

- raw manifest auditing
- deterministic splits
- WebDataset shard creation
- range/shape + decode audits

## Typical flow

```bash
# 0) build raw manifests (example for Clotho)
python3 -m stage3_uti.stage2.build_manifests clotho \
  --data-root data/Clotho \
  --audio-root data/Clotho/extracted \
  --out stage3_uti/data/manifests/clotho.jsonl

# 1) audit a manifest
python3 -m stage3_uti.stage2.manifest_audit \
  --manifest-jsonl stage3_uti/data/manifests/clotho.jsonl \
  --dataset-name clotho

# 2) tokenize → WebDataset shards
python3 -m stage3_uti.stage2.tokenize_wds \
  --uti-config stage3_uti/configs/uti.yaml \
  --manifest-jsonl stage3_uti/data/manifests/clotho.jsonl \
  --dataset-name clotho

# 3) audit tokenized shards
python3 -m stage3_uti.stage2.audit_tokenized \
  --uti-config stage3_uti/configs/uti.yaml \
  --dataset-name clotho
```

Outputs land under `stage3_uti/data/`:

- `manifests/` raw JSONL
- `tokenized/<dataset>/<split>/shard-*.tar`
- `splits/<dataset>/{train,eval}_ids.txt`
- `reports/*_manifest_stats.json`, `*_token_stats.json`, `*_audit.json`
- `errors/tokenize_errors.jsonl`
