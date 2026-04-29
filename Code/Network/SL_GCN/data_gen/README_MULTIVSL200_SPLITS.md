# MultiVSL200 Split Generation Guide

## Overview

This guide explains how to generate train/val/test splits for the MultiVSL200 dataset with signer-independent split (8:1:1 ratio).

## Step-by-step Execution

### Step 1: Generate Split CSV Files (signer allocation)

From `Code/Network/SL_GCN/data/MultiVSL200/`, run:

```bash
python create_splits_by_signer.py
```

**Output**: Creates `splits/` directory with:

- `train_labels.csv` (4515 samples, 22 signers)
- `val_labels.csv` (791 samples, 3 signers)
- `test_labels.csv` (593 samples, 3 signers)

### Step 2: Generate Dataset .pkl and .npy Files

From `Code/Network/SL_GCN/data_gen/`, run:

```bash
python generate_multivsl200_splits.py \
  --data_dir ../data/MultiVSL200 \
  --split_dir ../data/MultiVSL200/splits \
  --out_dir ../data/MultiVSL200 \
  --config 46_all
```

**Output**: Writes to `Code/Network/SL_GCN/data/MultiVSL200/`:

- `train_label.pkl`, `train_data_joint.npy` (4515 samples, 46 joints)
- `val_label.pkl`, `val_data_joint.npy` (791 samples, 46 joints)
- `test_label.pkl`, `test_data_joint.npy` (593 samples, 46 joints)

### One-line Execution (All at once)

```bash
cd Code/Network/SL_GCN/data/MultiVSL200 && \
python create_splits_by_signer.py && \
cd ../../data_gen && \
python generate_multivsl200_splits.py --data_dir ../data/MultiVSL200 --split_dir ../data/MultiVSL200/splits --out_dir ../data/MultiVSL200 --config 46_all
```

## Files Location

```
Code/Network/SL_GCN/
├── data/
│   └── MultiVSL200/
│       ├── *.npy                    (raw skeleton files)
│       ├── create_splits_by_signer.py
│       └── splits/                  (generated)
│           ├── train_labels.csv
│           ├── val_labels.csv
│           ├── test_labels.csv
│           ├── train_files.txt
│           ├── val_files.txt
│           └── test_files.txt
└── data_gen/
    ├── generate_multivsl200_splits.py
    └── README.md (this file)
```

## Signer Allocation (8:1:1 split, seed=42)

- **Train (22 signers)**: 20,18,12,23,31,07,06,13,16,10,27,14,21,17,28,02,15,03,19,30,05,24
- **Val (3 signers)**: 08,09,26
- **Test (3 signers)**: 01,04,22

## Notes

- Do NOT commit `.npy` and `.pkl` files to git (see `.gitignore`)
- Each teammate must run these scripts locally after cloning
- Use `--config 27_cvpr` to select 27 key joint points from 133 total
- Processing time: ~10-30 minutes depending on hardware
