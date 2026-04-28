# Full Run Guide: MultiVSL200 + HA-SLR (46 joints) + WandB

This guide explains the full pipeline from scratch:
- environment setup
- data conversion from raw `.npy`
- training and testing
- resume training
- logging to WandB
- loading/restoring checkpoints from WandB

All commands below are written for Windows PowerShell.

## 0) Prerequisites

- Repository already cloned.
- Raw dataset folder exists:
  - `Code/Network/SL_GCN/data/MultiVSL200/*.npy`
- File names contain class id prefix like `01_...npy`, `02_...npy`, ...

## 1) Go to project root

```powershell
cd "D:\20252\Lab\viết báo\References\HA-GCN\HA-SLR-GCN"
```

## 2) Create and activate virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## 3) Install dependencies

The repo uses a `requirements` file (without extension), plus `numpy`.

```powershell
pip install -r requirements
pip install numpy
```

## 4) Move to training folder

```powershell
cd "Code\Network\SL_GCN"
```

## 5) Convert raw MultiVSL200 files to HA-SLR format

This command creates:
- `train_data_joint.npy`, `val_data_joint.npy`, `test_data_joint.npy`
- `train_label.pkl`, `val_label.pkl`, `test_label.pkl`

Output directory:
- `./data/MultiVSL200/46_cvpr`

```powershell
python data_gen/convert_folder_to_dataset.py `
  --input-dir ./data/MultiVSL200 `
  --output-dir ./data/MultiVSL200/46_cvpr `
  --modality joint `
  --window-size 150 `
  --auto-split `
  --split-ratios 0.8 0.1 0.1
```

## 6) Quick sanity check (optional but recommended)

```powershell
@'
import numpy as np, pickle
x = np.load('./data/MultiVSL200/46_cvpr/train_data_joint.npy', mmap_mode='r')
print('train_data_joint shape:', x.shape)  # expected: (N, C, T, V, M)
with open('./data/MultiVSL200/46_cvpr/train_label.pkl', 'rb') as f:
  names, labels = pickle.load(f)
print('num samples:', len(names), 'num labels:', len(labels), 'label range:', int(min(labels)), int(max(labels)))
'@ | python
```

Expected key points:
- `V = 46`
- data shape format is `(N, C, T, V, M)`

## 7) Train (without WandB)

Config file:
- `config/MultiVSL200/train_joint_multivsl200.yaml`

```powershell
python main_base.py --config config/MultiVSL200/train_joint_multivsl200.yaml
```

Outputs are saved under `work_dir/...`:
- `checkpoints/`
- `runs/` (TensorBoard)
- `scores/`

## 8) Train with WandB logging

### 8.1 Login WandB

Option A: environment variable

```powershell
$env:WANDB_API_KEY="<your_wandb_api_key>"
```

Option B: interactive login

```powershell
wandb login
```

### 8.2 Start training with WandB enabled

```powershell
python main_base.py --config config/MultiVSL200/train_joint_multivsl200.yaml `
  --use-wandb True `
  --wandb-project "HA-SLR-MultiVSL200" `
  --wandb-entity "<your_entity_or_team>" `
  --wandb-run-name "multivsl200_joint_run01" `
  --wandb-group "multivsl200_46"
```

Useful optional WandB flags supported by this repo:
- `--wandb-mode online|offline|disabled`
- `--wandb-id <run_id>`
- `--wandb-resume auto|allow|must|never`

## 9) Resume local training from checkpoint

### 9.1 Resume from explicit checkpoint path

```powershell
python main_base.py --config config/MultiVSL200/train_joint_multivsl200.yaml `
  --resume-checkpoint "./work_dir/.../checkpoints/<experiment>_latest.pt"
```

### 9.2 Auto-resume latest checkpoint for same experiment name

```powershell
python main_base.py --config config/MultiVSL200/train_joint_multivsl200.yaml --auto-resume True
```

## 10) Resume WandB run (same run id)

If you want metrics to continue in the same WandB run:

```powershell
python main_base.py --config config/MultiVSL200/train_joint_multivsl200.yaml `
  --use-wandb True `
  --wandb-project "HA-SLR-MultiVSL200" `
  --wandb-id "<existing_run_id>" `
  --wandb-resume must `
  --resume-checkpoint "./work_dir/.../checkpoints/<experiment>_latest.pt"
```

Notes:
- `--wandb-id` identifies the previous WandB run.
- `--wandb-resume must` forces resume to that run id.
- You still need model weights locally unless you download them from artifacts (next section).

## 11) Load checkpoint from WandB artifacts

This repo logs checkpoints as WandB artifacts when WandB is enabled.

### 11.1 Download with Python API (recommended)

```powershell
@'
import wandb
run = wandb.init(project="HA-SLR-MultiVSL200", job_type="download", mode="online")
# Example artifact name format: "<entity>/<project>/<artifact_name>:latest"
artifact = run.use_artifact("<entity>/HA-SLR-MultiVSL200/<artifact_name>:latest", type="checkpoint")
path = artifact.download()
print('Downloaded to:', path)
run.finish()
'@ | python
```

After download, point `--resume-checkpoint` or `--weights` to the `.pt` file.

### 11.2 (Alternative) Download via UI

- Open your WandB run.
- Go to `Artifacts`.
- Download `*_latest`, `*_best_acc`, or `*_best_loss` checkpoint.
- Place it in your workspace and use the path with `--resume-checkpoint`.

## 12) Test / evaluation

For quick evaluation using a saved checkpoint:

```powershell
python main_base.py --config config/MultiVSL200/train_joint_multivsl200.yaml `
  --phase test `
  --weights "./work_dir/.../checkpoints/<your_checkpoint>.pt"
```


