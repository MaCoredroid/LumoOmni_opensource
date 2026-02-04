# Reproducibility

This project treats reproducibility as a first‑class feature:

- deterministic audits where possible
- stable token space hashes
- explicit artifacts/logs for each run

Below are common “entry points” for reproducing core checks.

---

## Track A — VLM

### 1) Connector ablation evaluation

Run the ablation script against a trained checkpoint:

- correct image
- shuffled image
- zero/noise visual tokens

Outputs:
- per‑condition losses
- deltas vs correct

### 2) Truncation + label coverage

During training, log:
- truncation rate
- label_zero rate
- avg label token counts

---

## Track B — UTI + token LM

### 1) UTI audit (deterministic)

Run the audit in deterministic mode and write:
- `token_space.json`
- `report.json`
- reconstructed `recon.png` / `recon.wav`

Also run with:
- baseline regression report for gating

### 2) Token LM smoke training

Run a short training job that confirms:
- vocab resize works
- checkpoints can be saved/resumed
- short generation decodes without crash

---

## Artifacts you should always keep

- `token_space.json` + hash
- audit reports (`report.json`, `baseline_report.json`)
- train/eval split metadata
- full training logs (`train.log`)
- qualitative “golden sets” (for VLM captioning or token decoding)

---

## Suggested repo structure for published artifacts

```
outputs/
  stage1_.../
    checkpoints/
    metrics/
    qual/
  uti_audit/
    token_space.json
    token_space.sha256
    report.json
    baseline_report.json
    recon.png
    recon.wav
  stage3_token_lm_.../
    checkpoint_.../
    eval_....json
```
