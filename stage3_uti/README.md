# Stage 3.1 UTI (Unified Token Interface)

This folder is **self-contained** and does not modify Stage 1/2 code.
It contains:

- `tokenization/`: TokenSpace + UnifiedTokenizer + SEED/EnCodec adapters
- `pipeline/`: Stage 3 token-sequence scaffolding (UTI-driven)
- `configs/`: UTI and Stage 3 config templates
- `tests/`: deterministic UTI tests + fixed assets

## Quick entry points

- Build token space JSON:

```bash
python3 -m stage3_uti.pipeline.build_token_space \
  --uti-config stage3_uti/configs/uti.yaml \
  --out outputs/stage3_token_lm/token_space.json
```

- Stage 2 tokenized shards (manifest → WebDataset):

```bash
python3 -m stage3_uti.stage2.tokenize_wds \
  --uti-config stage3_uti/configs/uti.yaml \
  --manifest-jsonl stage3_uti/data/manifests/clotho.jsonl \
  --dataset-name clotho
```

- Stage 2 audits:

```bash
python3 -m stage3_uti.stage2.manifest_audit \
  --manifest-jsonl stage3_uti/data/manifests/clotho.jsonl \
  --dataset-name clotho

python3 -m stage3_uti.stage2.audit_tokenized \
  --uti-config stage3_uti/configs/uti.yaml \
  --dataset-name clotho
```

- Use `stage3_uti/pipeline/sequence_builder.py` to assemble modality sequences.

## Training (Stage 3 token LM)

Training should be run **inside the CUDA Docker container** (see top-level `readme.md`).
Once inside the container, run:

```bash
cd /workspace/lumoOmni
source /opt/lumo/venv/bin/activate
python3 -m stage3_uti.train.stage3_token_lm --config stage3_uti/configs/stage3_token_lm.yaml
```

## UTI audit (verification)

This runs the end‑to‑end UTI checks (determinism, shapes, decode smoke) and writes
an artifact bundle (`token_space.json`, token hashes, recon.wav/png, report.json).

```bash
python3 -m stage3_uti.tokenization.uti_audit \
  --config stage3_uti/configs/uti.yaml \
  --token-space-out outputs/uti_audit/token_space.json \
  --smoke-assets stage3_uti/tests/assets \
  --outdir outputs/uti_audit \
  --decode-mode diffusion
```
