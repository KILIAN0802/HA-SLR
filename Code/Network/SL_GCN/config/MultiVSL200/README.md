# MultiVSL200 train setup

This folder contains a minimal config for training HA-SLR on the attached `MultiVSL200` raw `.npy` folder.

## 1. Convert raw files into HA-SLR dataset files

Run the converter from `Code/Network/SL_GCN/data_gen` to auto-split the raw folder by the label prefix in the filename:

```bash
python data_gen/convert_folder_to_dataset.py \
  --input-dir ./data/MultiVSL200 \
  --output-dir ./data/MultiVSL200/46_cvpr \
  --modality joint \
  --window-size 150 \
  --auto-split \
  --split-ratios 0.8 0.1 0.1
```

This writes:
- `train_data_joint.npy`, `val_data_joint.npy`, `test_data_joint.npy`
- `train_label.pkl`, `val_label.pkl`, `test_label.pkl`

## 2. Train

Use the provided config:

```bash
python main_base.py --config config/MultiVSL200/train_joint_multivsl200.yaml
```

## Notes

- The repo has no official 46-node graph for this dataset, so `graph.multivsl200_46.Graph` uses a simple chain graph as a safe default.
- `random_mirror` is disabled because there is no left-right joint mapping for the 46-keypoint layout.
- `num_class` is set to 200 because the dataset name suggests 200 classes; adjust it if your labels differ.
