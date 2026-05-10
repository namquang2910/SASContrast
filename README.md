# Subject-Aware Contrastive Learning for Physiological Stress Detection

Official implementation of **Subject-Aware Self-Supervised Contrastive Learning (SAS-CL)**,
a contrastive pre-training framework that learns both subject-invariant and
subject-specific representations from wearable physiological signals.


---

## Overview

Wearable sensors produce highly subject-dependent physiological signals. Standard
contrastive learning treats all samples identically, which either discards
subject-specific structure or conflates it with noise. SAS-CL addresses this by
training two parallel projection heads under a unified contrastive objective:

- **Invariant head** — pulls together representations of the same *state* regardless of subject identity.
- **Specific head** — pulls together representations of the same *subject* to capture personal baseline shifts.

The combined embedding is transferred to a downstream linear probe for stress detection.

---

## Requirements

```
python >= 3.9
torch >= 2.0
numpy
scipy
neurokit2
scikit-learn
pandas
pyarrow          # for parquet
```

Install all dependencies:

```bash
pip install torch numpy scipy neurokit2 scikit-learn pandas pyarrow
```

---

## Quick Start

### 1 — Prepare data

Place processed dataset directories under `./data/`:

```
./data/
├── WESAD/wesad_10_05_no_standardize/   # per-subject parquet files
├── SWELL/SWELL_1280_320_3label/
└── ...
```

### 2 — Create LOSO folds (required before training)

The training pipeline reads pre-computed fold CSV files from `./datasets/process_dataset/<DATASET>_LOSO/`.
These must be created **once** before running any experiments:

```bash
# Open and run the fold-creation notebook
jupyter notebook datasets/process_dataset/create_folds.ipynb
```

Each fold CSV encodes a train / val / test subject split for one Leave-One-Subject-Out round.
Without these files `single_train.py` will error because `split_path` will be empty.

### 3 — Pre-train + fine-tune (WESAD)

```bash
python single_train.py \
    --config_path configs/pretrain_wesad.json \
    --model_type  sas_pretrain \
    --dataset     WESADDataset
```

### 4 — Pre-train + fine-tune (SWELL)

```bash
python single_train.py \
    --config_path configs/pretrain_swell.json \
    --model_type  sas_pretrain \
    --dataset     SWELLDataset
```

### 5 — Baseline (SimCLR-style contrastive)

```bash
python single_train.py \
    --config_path configs/pretrain_wesad.json \
    --model_type  contrastive \
    --dataset     WESADDataset
```

### 6 — Resume fine-tuning from a saved checkpoint

```bash
python single_train.py \
    --config_path    configs/pretrain_wesad.json \
    --model_type     sas_pretrain \
    --dataset        WESADDataset \
    --resume_finetune 2 \
    --model_path     ./save/pretrain/.../encoder_best_.pt
```

---

## Configuration

Each JSON config contains three top-level sections:

| Section | Purpose |
|---------|---------|
| `pretrain_args` | Dataset, model architecture, augmentations, optimiser for pre-training |
| `finetune_args` | Dataset, model architecture, loss, optimiser for fine-tuning |
| `logging_args`  | Output directories, checkpoint frequency |

Key model arguments (`pretrain_args.model_args`):

| Field | Values | Description |
|-------|--------|-------------|
| `base_encoder` | `"sas"`, `"cnn"` | Encoder backbone |
| `model_type` | `"sas_pretrain"`, `"contrastive"`, `"subject_specific"`, `"subject_invariant"` | Pre-training objective |
| `lambda_inv` | float | Weight for invariant NCE loss |
| `lambda_spec` | float | Weight for subject-specific NCE loss |

See [`configs/config_options.py`](configs/config_options.py) for a full reference of every field
and its accepted values.

---

## Supported Model Types

| `--model_type` | Description |
|----------------|-------------|
| `sas_pretrain` | **SAS-CL** (this work): dual-head invariant + specific pre-training |
| `contrastive` | SimCLR-style instance contrastive baseline |
| `subject_specific` | Subject-conditioned contrastive (subject pairs as positives) |
| `subject_invariant` | Adversarial GRL model discouraging subject identity |

---

## Evaluation Protocol

Evaluation follows a **Leave-One-Subject-Out (LOSO)** cross-validation scheme:

1. Create fold CSV files via `datasets/process_dataset/create_folds.ipynb` (one-time step).
2. Pre-train on all subjects (no labels used).
3. For each fold: fine-tune a linear probe on `train` split, evaluate on `test` split.
4. Report mean ± std macro-F1 and accuracy across folds and random seeds.

Results are saved to `./save/results.json` (per-seed) and `./save/summary_results.csv`
(cross-seed summary).

---

## Distributed Training

Multi-GPU training via PyTorch DDP is supported automatically. Launch with `torchrun`:

```bash
torchrun --nproc_per_node=4 single_train.py \
    --config_path configs/pretrain_wesad.json \
    --model_type  sas_pretrain \
    --dataset     WESADDataset
```

## Reference
> **Based on** the contrastive learning framework for EDA by K. Matton et al.:
> [github.com/kmatton/contrastive-learning-for-eda](https://github.com/kmatton/contrastive-learning-for-eda/tree/main).
> The augmentation pipeline, dataset wrappers, and InfoNCE loss in this repository
> are adapted from that work.
---

## License

This repository is released for research purposes. See `LICENSE` for details.
