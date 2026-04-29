# MultiVSL200 Full Pipeline: Từ Raw Data đến 4-Stream Training Data

## Overview

Quy trình hoàn chỉnh để chuẩn bị MultiVSL200 cho training HA-SLR models:

```
Raw .npy files (46 joints)
         ↓
[Phase 1] Create Splits (create_splits_by_signer.py)
         ↓
train_labels.csv, val_labels.csv, test_labels.csv
         ↓
[Phase 2] Generate Joint Data (generate_multivsl200_splits.py)
         ↓
*_data_joint.npy, *_label.pkl
         ↓
[Phase 3] Optional: Convert to 27-joint format (generate_multivsl200_splits.py --config 46_to_27)
         ↓
*_data_joint.npy (27-joint HA-SLR format)
         ↓
[Phase 4] Generate Multi-Stream Data (generate_multivsl200_4streams.py)
         ↓
*_data_bone.npy, *_data_joint_motion.npy, *_data_bone_motion.npy
         ↓
Ready for training with 4-stream models ✓
```

---

## Phase 1: Create Splits by Signer

**File**: `Code/Network/SL_GCN/data_gen/create_splits_by_signer.py`

**Purpose**: Allocate 28 signers into 8:1:1 split (train:val:test) with signer independence

**Status**: ✓ COMPLETE

```bash
cd Code/Network/SL_GCN/data_gen

python create_splits_by_signer.py \
  --data_dir ../data/MultiVSL200 \
  --out_dir ../data/MultiVSL200/splits

# Output:
# - splits/train_labels.csv (4515 samples, 22 signers)
# - splits/val_labels.csv   (791 samples, 3 signers)
# - splits/test_labels.csv  (593 samples, 3 signers)
```

**Key Features**:

- Regex to extract signer IDs from filenames
- Shuffles signers with fixed seed (42) for reproducibility
- No signer appears in multiple splits
- CSV format: `sample_name,label_id`

---

## Phase 2A: Generate Joint Data (46-joint version)

**File**: `Code/Network/SL_GCN/data_gen/generate_multivsl200_splits.py`

**Purpose**: Convert CSV labels + raw .npy files → standardized dataset format

**Config**: `46_all` (default)

```bash
cd Code/Network/SL_GCN/data_gen

python generate_multivsl200_splits.py \
  --data_dir ../data/MultiVSL200 \
  --split_dir ../data/MultiVSL200/splits \
  --out_dir ../data/MultiVSL200 \
  --config 46_all

# Output:
# - train_label.pkl (4515 samples)
# - train_data_joint.npy (4515, 3, 150, 46, 1)
# - val_label.pkl (791 samples)
# - val_data_joint.npy (791, 3, 150, 46, 1)
# - test_label.pkl (593 samples)
# - test_data_joint.npy (593, 3, 150, 46, 1)
```

**Data Processing**:

1. Read CSV → extract sample_names and labels
2. Load .npy file → reshape to (num_frames, 46, 3)
3. Select joints (46_all: use all 46 joints)
4. Pad/truncate frames to 150
5. Transpose to (N, 3, 150, 46, 1) format
6. Save label pickle + data npy

**Advantages**: Keeps full information from MultiVSL200's 46 joints

---

## Phase 2B: Generate Joint Data (27-joint HA-SLR format)

**File**: Same as Phase 2A, different config

**Purpose**: Generate HA-SLR compatible 27-joint format with dummy elbows

**Config**: `46_to_27`

**Why use 46_to_27?**

- ✓ Direct compatibility with HA-SLR pre-trained models
- ✓ Fair comparison with AUTSL/INCLUDE experiments
- ✓ Leverages HA-SLR's bone graph for hand connections
- ✗ Loses information (46→27 joints)
- ⚠ Elbow values are dummy (copied from shoulders) due to MultiVSL200 structure

```bash
cd Code/Network/SL_GCN/data_gen

python generate_multivsl200_splits.py \
  --data_dir ../data/MultiVSL200 \
  --split_dir ../data/MultiVSL200/splits \
  --out_dir ../data/MultiVSL200_27 \
  --config 46_to_27

# Output (27-joint format):
# - train_label.pkl (4515 samples) ← SAME as 46_all
# - train_data_joint.npy (4515, 3, 150, 27, 1) ← NEW shape!
# - val_label.pkl (791 samples)
# - val_data_joint.npy (791, 3, 150, 27, 1)
# - test_label.pkl (593 samples)
# - test_data_joint.npy (593, 3, 150, 27, 1)
```

**Joint Layout** (46_to_27 mapping):

```
HA-SLR 27-point indices ← MultiVSL200 46-point indices
0: Nose               ← 42
1: L-Shoulder        ← 43
2: R-Shoulder        ← 44
3: L-Elbow (dummy)   ← 43 (same as L-Shoulder!)
4: R-Elbow (dummy)   ← 44 (same as R-Shoulder!)
5: L-Wrist           ← 0
6: R-Wrist           ← 21
7-16: L-Hand         ← 0,4,5,8,9,12,13,16,17,20
17-26: R-Hand        ← 21,25,26,29,30,33,34,37,38,41
```

**Rationale for dummy elbows**:

- MultiVSL200 lacks elbow joints (only has 46 pre-selected points)
- HA-SLR requires elbows for bone chain (shoulder→elbow→wrist)
- Quick fix: Copy shoulder position → acts as dummy elbow
- Result: bone[wrist] = wrist_pos - shoulder_pos (longer arm bone)

---

## Phase 3: Choose Which Config to Use

### Decision Matrix:

| Goal          | Config   | Data Shape     | Train Time | Compatibility              | Notes                          |
| ------------- | -------- | -------------- | ---------- | -------------------------- | ------------------------------ |
| Keep all info | 46_all   | (N,3,150,46,1) | Longer     | None (need 46-joint model) | Full skeleton from MultiVSL200 |
| HA-SLR compat | 46_to_27 | (N,3,150,27,1) | Shorter    | ✓ HA-SLR models            | Dummy elbows, fair comparison  |

### Recommendation:

**Use `46_to_27` if:**

- ✓ Want to use pre-trained HA-SLR models on MultiVSL200
- ✓ Need fair comparison with AUTSL/INCLUDE results
- ✓ Want to compare "how well HA-SLR generalizes to MultiVSL200"

**Use `46_all` if:**

- ✓ Training new model architecture designed for 46 joints
- ✓ Want to maximize information from MultiVSL200
- ✓ Comparing purely on MultiVSL200 (no cross-dataset)

**For now**: We'll proceed with **both** versions (they don't conflict)

---

## Phase 4: Generate 4-Stream Data

### 4 Stream Types:

| Stream           | Calculation                 | Purpose                               |
| ---------------- | --------------------------- | ------------------------------------- |
| **Joint**        | Position data               | Absolute skeleton pose                |
| **Bone**         | Position[v2] - Position[v1] | Edge vectors between connected joints |
| **Joint Motion** | Position[t+1] - Position[t] | Temporal velocity of joints           |
| **Bone Motion**  | Bone[t+1] - Bone[t]         | Temporal velocity of bones            |

### For 46-joint version:

```bash
cd Code/Network/SL_GCN/data_gen

python generate_multivsl200_4streams.py \
  --data_dir ../data/MultiVSL200 \
  --config 46_all

# This reads:
# - ../data/MultiVSL200/train_data_joint.npy (46 joints)
# - ../data/MultiVSL200/val_data_joint.npy
# - ../data/MultiVSL200/test_data_joint.npy
#
# Output:
# - train_data_bone.npy, train_data_joint_motion.npy, train_data_bone_motion.npy
# - val_data_bone.npy, val_data_joint_motion.npy, val_data_bone_motion.npy
# - test_data_bone.npy, test_data_joint_motion.npy, test_data_bone_motion.npy
#
# Total: 12 new files (4 streams - 1 joint that already exists, × 3 splits)
```

### For 27-joint version:

```bash
cd Code/Network/SL_GCN/data_gen

python generate_multivsl200_4streams.py \
  --data_dir ../data/MultiVSL200_27 \
  --config 46_to_27

# This reads:
# - ../data/MultiVSL200_27/train_data_joint.npy (27 joints)
# - ../data/MultiVSL200_27/val_data_joint.npy
# - ../data/MultiVSL200_27/test_data_joint.npy
#
# Output:
# - Bone data using BONE_PAIRS_27 (which includes dummy elbow handling)
# - Joint motion & Bone motion streams
```

### Bone Pairs for Each Config:

**BONE_PAIRS_46** (26 pairs):

```python
(0,1), (0,2),                      # Nose → Shoulders
(1,3), (3,5), (2,4), (4,6),        # Shoulder → Elbow → Wrist
(5,7), (7,9), (6,8), (8,10),       # Elbows → Wrists
(1,11), (2,12),                    # Shoulders → Hips
(11,13), (13,15), (12,14), (14,16), # Hips → Knees
(15,17), (17,19), (16,18), (18,20), # Knees → Ankles
(1,2), (11,12)                     # Spine connections
```

**BONE_PAIRS_27** (27 pairs - HA-SLR standard):

```python
(5,6), (5,7),                      # Shoulder connections
(6,8), (8,10), (7,9), (9,11),      # Arm chain (3,4 are dummy elbows!)
(12,13), (12,14), (12,16), ...     # Left hand
(22,23), (22,24), (22,26), ...     # Right hand
(10,12), (11,22)                   # Wrist → Hand root
```

**Note on dummy elbows in BONE_PAIRS_27**:

- Indices 3, 4 (elbows) map to indices 1, 2 (shoulders) in 46→27 conversion
- Bone[3] = joint[3] - joint[5] = shoulder[1] - joint[5] = valid calculation
- No IndexError because index 3 still refers to valid data (just copied from shoulder)

---

## Complete Workflow Example

```bash
# Step 1: Navigate to data_gen
cd Code/Network/SL_GCN/data_gen

# Step 2: Phase 1 - Create splits (if not already done)
echo "=== Phase 1: Creating splits ==="
python create_splits_by_signer.py \
  --data_dir ../data/MultiVSL200 \
  --out_dir ../data/MultiVSL200/splits

# Step 3a: Phase 2A - Generate 46-joint data
echo "=== Phase 2A: Generating 46-joint data ==="
python generate_multivsl200_splits.py \
  --data_dir ../data/MultiVSL200 \
  --split_dir ../data/MultiVSL200/splits \
  --out_dir ../data/MultiVSL200 \
  --config 46_all

# Step 3b: Phase 2B - Generate 27-joint data (optional)
echo "=== Phase 2B: Generating 27-joint HA-SLR compatible data ==="
python generate_multivsl200_splits.py \
  --data_dir ../data/MultiVSL200 \
  --split_dir ../data/MultiVSL200/splits \
  --out_dir ../data/MultiVSL200_27 \
  --config 46_to_27

# Step 4a: Phase 4 - Generate 4-stream data (46-joint)
echo "=== Phase 4A: Generating 4-stream data (46-joint) ==="
python generate_multivsl200_4streams.py \
  --data_dir ../data/MultiVSL200 \
  --config 46_all

# Step 4b: Phase 4 - Generate 4-stream data (27-joint)
echo "=== Phase 4B: Generating 4-stream data (27-joint) ==="
python generate_multivsl200_4streams.py \
  --data_dir ../data/MultiVSL200_27 \
  --config 46_to_27

echo "✓ All phases complete!"
```

---

## Output Structure

### 46-joint version:

```
MultiVSL200/
├── train_label.pkl
├── train_data_joint.npy (4515, 3, 150, 46, 1)
├── train_data_bone.npy (4515, 3, 150, 46, 1)
├── train_data_joint_motion.npy (4515, 3, 150, 46, 1)
├── train_data_bone_motion.npy (4515, 3, 150, 46, 1)
├── val_label.pkl
├── val_data_joint.npy (791, 3, 150, 46, 1)
├── val_data_bone.npy (791, 3, 150, 46, 1)
├── val_data_joint_motion.npy (791, 3, 150, 46, 1)
├── val_data_bone_motion.npy (791, 3, 150, 46, 1)
├── test_label.pkl
├── test_data_joint.npy (593, 3, 150, 46, 1)
├── test_data_bone.npy (593, 3, 150, 46, 1)
├── test_data_joint_motion.npy (593, 3, 150, 46, 1)
└── test_data_bone_motion.npy (593, 3, 150, 46, 1)
```

### 27-joint version:

```
MultiVSL200_27/
├── train_label.pkl
├── train_data_joint.npy (4515, 3, 150, 27, 1) ← Different shape!
├── train_data_bone.npy (4515, 3, 150, 27, 1)
├── train_data_joint_motion.npy (4515, 3, 150, 27, 1)
├── train_data_bone_motion.npy (4515, 3, 150, 27, 1)
├── (val & test similarly)
```

---

## Training Configuration

### For 46-joint training:

```yaml
# config/sign_cvpr_A_hands/MultiVSL200_46/train_joint_multivsl200_46.yaml
device: [0]
batch_size: 32
num_worker: 4
num_class: 28
num_epoch: 50
train_feeder_args:
  data_path: ./data/sign_multivsl200/46_all/train_data_joint.npy
  label_path: ./data/sign_multivsl200/46_all/train_label.pkl
  num_person: 1
  num_frame: 150
  num_point: 46
  random_choose: False
  random_shift: True
  window_size: 150
```

### For 27-joint training:

```yaml
# config/sign_cvpr_A_hands/MultiVSL200_27/train_joint_multivsl200_27.yaml
device: [0]
batch_size: 32
num_worker: 4
num_class: 28
num_epoch: 50
train_feeder_args:
  data_path: ./data/sign_multivsl200/46_to_27/train_data_joint.npy
  label_path: ./data/sign_multivsl200/46_to_27/train_label.pkl
  num_person: 1
  num_frame: 150
  num_point: 27 # ← Key difference
  random_choose: False
  random_shift: True
  window_size: 150
```

---

## Timeline & Execution Time (Estimated)

| Phase     | Script                           | 46_all            | 46_to_27                                         |
| --------- | -------------------------------- | ----------------- | ------------------------------------------------ |
| 1         | create_splits_by_signer.py       | <1 min            | -                                                |
| 2         | generate_multivsl200_splits.py   | ~100 sec          | ~100 sec                                         |
| 4         | generate_multivsl200_4streams.py | ~600 sec (10 min) | ~300 sec (5 min)                                 |
| **Total** |                                  | ~710 sec (12 min) | **For 27-joint: ~400 sec (7 min) from Phase 2B** |

---

## Troubleshooting

### "IndexError: index 46 is out of bounds for axis 1 with size 46"

**Solution**: Using wrong config for your data. Check that you're using matching config:

- `--config 46_all` for 46-joint data
- `--config 46_to_27` for data pre-processed with 46_to_27 config

### "IndexError: index 28 is out of bounds for axis 3 with size 27"

**Status**: ✅ FIXED in latest version

This was a critical bug in BONE_PAIRS_27 indexing. It has been **FIXED**.

**Solution**: Update your code:

```bash
git pull origin 46kpt
```

Then re-run:

```bash
python generate_multivsl200_4streams.py --config 46_to_27
```

**Details**: See [BUGFIX_BONE_PAIRS_27_INDEX_MISMATCH.md](BUGFIX_BONE_PAIRS_27_INDEX_MISMATCH.md)

### "FileNotFoundError: train_data_joint.npy"

**Solution**: Ensure Phase 2 completed successfully. Check file exists:

```bash
ls -la ../data/MultiVSL200/*_data_joint.npy
```

### "Dummy elbow values make training worse"

**Solution**: This is expected if your model uses absolute joint positions. Consider:

1. **Option 1**: Use 46_all config (keep full 46 joints, train new model)
2. **Option 2**: Use bone-based representations (which handle dummy elbows better)
3. **Option 3**: Improve dummy elbow strategy (e.g., interpolate between shoulder and wrist)

---

## References

- [Elbow Gap Analysis](ELBOW_GAP_AND_JOINT_MAPPING.md)
- [MultiVSL200 Splits Guide](README_MULTIVSL200_SPLITS.md)
- [Original Multi-Stream Script](generate_multivsl200_4streams.py)

**Last Updated**: 2026-04-30
