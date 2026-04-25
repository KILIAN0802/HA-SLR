# HA-SLR-GCN

This repository contains the HA-SLR-GCN implementation for sign language recognition.

## 1. Data preparation

Download the processed keypoints data from [Google Drive](https://drive.google.com/drive/folders/1LM6gpmtgrcUvXdDkyKGVTQuDtqfViMKz?usp=drive_link).

After extracting the archive, place the data folder here:

```text
Code/Network/SL_GCN/data/
```

The expected subfolders are already referenced by the configs, for example:

```text
Code/Network/SL_GCN/data/sign_autsl/27_cvpr/
Code/Network/SL_GCN/data/sign_include/27_cvpr/
```

## 2. Requirements

Install the Python packages below before running the scripts:

```text
torch
torchvision
torchaudio
tqdm
tensorboard
pyyaml
pandas
wandb
```

If you are on Colab or a fresh environment, the simplest install command is:

```bash
pip install -r requirements
```

## 3. Where to run

Run all commands from:

```bash
cd Code/Network/SL_GCN
```

If you are using Colab, make sure the working directory points to `Code/Network/SL_GCN` before launching `main_base.py`.

## 4. Training

Use one of the training configs below.

### AUTSL

```bash
python main_base.py --config config/sign_cvpr_A_hands/AUTSL/train_joint_autsl.yaml
python main_base.py --config config/sign_cvpr_A_hands/AUTSL/train_bone_autsl.yaml
python main_base.py --config config/sign_cvpr_A_hands/AUTSL/train_joint_motion_autsl.yaml
python main_base.py --config config/sign_cvpr_A_hands/AUTSL/train_bone_motion_autsl.yaml
```

### INCLUDE

```bash
python main_base.py --config config/sign_cvpr_A_hands/INCLUDE/train_joint_include.yaml
python main_base.py --config config/sign_cvpr_A_hands/INCLUDE/train_bone_include.yaml
python main_base.py --config config/sign_cvpr_A_hands/INCLUDE/train_joint_motion_include.yaml
python main_base.py --config config/sign_cvpr_A_hands/INCLUDE/train_bone_motion_include.yaml
```

## 5. Testing

Use the matching test config for the dataset and modality you trained.

### AUTSL

```bash
python main_base.py --config config/sign_cvpr_A_hands/AUTSL/test_joint_autsl.yaml
python main_base.py --config config/sign_cvpr_A_hands/AUTSL/test_bone_autsl.yaml
python main_base.py --config config/sign_cvpr_A_hands/AUTSL/test_joint_motion_autsl.yaml
python main_base.py --config config/sign_cvpr_A_hands/AUTSL/test_bone_motion_autsl.yaml
```

### INCLUDE

```bash
python main_base.py --config config/sign_cvpr_A_hands/INCLUDE/test_joint_include.yaml
python main_base.py --config config/sign_cvpr_A_hands/INCLUDE/test_bone_include.yaml
python main_base.py --config config/sign_cvpr_A_hands/INCLUDE/test_joint_motion_include.yaml
python main_base.py --config config/sign_cvpr_A_hands/INCLUDE/test_bone_motion_include.yaml
```

## 6. Resume training

The code saves the latest checkpoint automatically in the run directory.

You can resume in 2 ways:

### Manual resume

Pass an explicit checkpoint path:

```bash
python main_base.py --config config/sign_cvpr_A_hands/INCLUDE/train_joint_include.yaml --resume-checkpoint "./work_dir/.../checkpoints/<experiment>_latest.pt"
```

### Auto resume

Let the code search for the newest `*_latest.pt` of the same `Experiment_name`:

```bash
python main_base.py --config config/sign_cvpr_A_hands/INCLUDE/train_joint_include.yaml --auto-resume True
```

When resuming, training continues from the saved epoch instead of restarting from 0.

## 7. Optional WandB tracking

WandB logging is optional and can be enabled with a single flag.

### Minimal setup

Set your WandB API key once in the environment:

```bash
export WANDB_API_KEY="your_key_here"
```

Then run training with WandB enabled:

```bash
python main_base.py --config config/sign_cvpr_A_hands/INCLUDE/train_joint_include.yaml --use-wandb True
```

The code will automatically:

- log training and validation metrics by epoch
- save the latest, best-acc, and best-loss checkpoints as WandB artifacts
- use `Experiment_name` as the default project/group when you do not pass extra WandB flags

## 8. Outputs

Each run creates a timestamped folder under `work_dir/` that contains:

- `checkpoints/` for model weights
- `runs/` for TensorBoard logs
- `scores/` for prediction score pickles
- `log.txt` and `config.yaml` for reproducibility

## 9. Common notes

- Keep the config file aligned with the dataset split you are using.
- If you run on Colab, mount Google Drive first if you want to keep logs and checkpoints after the session ends.
- If you only want evaluation, set `phase: test` in the config or use a test config directly.
