# MultiVSL200 Preprocessing: Elbow Gap Fix & 4-Stream Data Generation

## 📋 Quick Navigation

**Just want to run?** → Start with [WORKFLOW_EXAMPLES.sh](#)  
**Need full details?** → Read [FULL_PIPELINE_GUIDE.md](#)  
**Want technical depth?** → See [ELBOW_GAP_AND_JOINT_MAPPING.md](#)  
**What changed?** → Check [CHANGES_SUMMARY.md](#)

---

## 🎯 What's New?

### Problem Solved: "The Elbow Gap"

MultiVSL200 has **46 joints**, but **lacks elbows**. HA-SLR models need elbows for bone calculation.

**Quick fix**: Use dummy elbows (copied from shoulders) when mapping 46→27 joints.

### New Feature: Dual Config Support

Now you can generate data in **two formats**:

| Format       | Joints | Use Case                     | Status         |
| ------------ | ------ | ---------------------------- | -------------- |
| **46_all**   | 46     | Full MultiVSL200 information | ✅ Recommended |
| **46_to_27** | 27     | HA-SLR model compatibility   | ✅ Optional    |

---

## 🚀 Quick Start (5 minutes)

### Prerequisites

- ✓ Raw MultiVSL200 .npy files in `Code/Network/SL_GCN/data/MultiVSL200/`
- ✓ Split CSV files created (see `create_splits_by_signer.py`)

### Run Full Pipeline

```bash
cd Code/Network/SL_GCN/data_gen

# Run all phases (automatic with examples)
bash WORKFLOW_EXAMPLES.sh

# Or run manually:
python generate_multivsl200_splits.py --config 46_all
python generate_multivsl200_4streams.py --config 46_all
```

### Expected Output (46-joint)

```
MultiVSL200/
├── train_data_joint.npy (4515, 3, 150, 46, 1)
├── train_data_bone.npy ← NEW
├── train_data_joint_motion.npy ← NEW
├── train_data_bone_motion.npy ← NEW
└── (val & test splits)
```

---

## 📚 Documentation Files

### 1. **WORKFLOW_EXAMPLES.sh**

**Type**: Executable Script  
**Purpose**: Copy-paste ready commands for complete pipeline  
**Best for**: Users who just want to run and see results

**Contains**:

- Phase-by-phase commands with explanations
- Expected output descriptions
- Verification checks
- Troubleshooting commands

**Usage**:

```bash
bash WORKFLOW_EXAMPLES.sh
```

---

### 2. **FULL_PIPELINE_GUIDE.md**

**Type**: Comprehensive Guide  
**Purpose**: Complete end-to-end workflow documentation  
**Best for**: Understanding all phases and making decisions

**Sections**:

- Phase 1-4 detailed explanations
- Decision matrix: "Use 46_all or 46_to_27?"
- 4-stream type descriptions
- Training configuration examples
- Timeline & resource estimates
- Troubleshooting Q&A

**Key Decision**:

- Use **46_all** if: keeping full info, no pre-trained model dependency
- Use **46_to_27** if: HA-SLR compatibility, comparing with AUTSL/INCLUDE

---

### 3. **ELBOW_GAP_AND_JOINT_MAPPING.md**

**Type**: Technical Reference  
**Purpose**: Deep dive into the elbow gap problem and solution  
**Best for**: Understanding why dummy elbows work

**Covers**:

- Why elbows are needed (bone calculation)
- Dummy elbow strategy (copy from shoulder)
- Joint mapping details
- Bone pair specifications
- Implementation insights
- Advanced troubleshooting

**Key Insight**:

```python
# With dummy elbows:
bone[wrist] = joint[wrist] - joint[elbow]
            = joint[wrist] - joint[shoulder]  # ← longer arm bone, still valid!
```

---

### 4. **CHANGES_SUMMARY.md**

**Type**: Change Log  
**Purpose**: Summary of all modifications made  
**Best for**: Developers and version control

**Lists**:

- Files modified/created
- Specific changes per file
- New configs added
- Testing checklist
- Backward compatibility info

---

## 🔧 Files Modified

### Updated Files

#### 1. **generate_multivsl200_splits.py**

```diff
+ Added '46_to_27' mapping to selected_joints dict
+ Updated docstring with 46_to_27 usage examples
+ Enhanced --config argument help text
```

**New feature**: `--config 46_to_27`

```bash
python generate_multivsl200_splits.py --config 46_to_27 --out_dir ../data/MultiVSL200_27
# Output: (N, 3, 150, 27, 1) with dummy elbows
```

#### 2. **generate_multivsl200_4streams.py**

```diff
+ Enhanced bone pair definitions for both configs
+ Added MultiStream46Generator(config) parameter
+ Updated docstring with 46_to_27 examples
+ Added --config argument to CLI
```

**New feature**: `--config 46_to_27`

```bash
python generate_multivsl200_4streams.py --config 46_to_27 --data_dir ../data/MultiVSL200_27
# Output: 12 files with shape (N, 3, 150, 27, 1)
```

### New Files Created

| File                                   | Type       | Purpose                          |
| -------------------------------------- | ---------- | -------------------------------- |
| WORKFLOW_EXAMPLES.sh                   | Script     | Runnable examples                |
| FULL_PIPELINE_GUIDE.md                 | Guide      | Complete documentation           |
| ELBOW_GAP_AND_JOINT_MAPPING.md         | Reference  | Technical depth                  |
| CHANGES_SUMMARY.md                     | ChangeLog  | What was changed                 |
| BUGFIX_BONE_PAIRS_27_INDEX_MISMATCH.md | ⚠ Critical | Bug fix (BONE_PAIRS_27 indexing) |
| PREPROCESSING_README.md                | This file  | Navigation guide                 |

---

## 💡 Common Workflows

### Workflow 1: Generate 46-joint data only (Recommended for most users)

```bash
cd Code/Network/SL_GCN/data_gen

python generate_multivsl200_splits.py --config 46_all
python generate_multivsl200_4streams.py --config 46_all

# Time: ~12 minutes
# Output: 46-joint 4-stream datasets in MultiVSL200/
```

### Workflow 2: Generate both 46-joint and 27-joint versions

```bash
cd Code/Network/SL_GCN/data_gen

# 46-joint
python generate_multivsl200_splits.py --config 46_all
python generate_multivsl200_4streams.py --config 46_all

# 27-joint (HA-SLR compatible)
python generate_multivsl200_splits.py --config 46_to_27 --out_dir ../data/MultiVSL200_27
python generate_multivsl200_4streams.py --config 46_to_27 --data_dir ../data/MultiVSL200_27

# Time: ~20 minutes total
# Outputs: Both 46-joint and 27-joint datasets
```

### Workflow 3: Re-run specific phase

```bash
# Re-generate just multi-stream data
python generate_multivsl200_4streams.py --config 46_all

# Re-generate just joint data
python generate_multivsl200_splits.py --config 46_all

# No need to re-run splits or earlier phases
```

---

## 📊 Output Comparison

### 46-joint (46_all) vs 27-joint (46_to_27)

| Aspect                    | 46_all             | 46_to_27           |
| ------------------------- | ------------------ | ------------------ |
| **Joint count**           | 46                 | 27                 |
| **Data shape**            | (N, 3, 150, 46, 1) | (N, 3, 150, 27, 1) |
| **Skeleton completeness** | ✓ Full             | ⚠ Missing elbows   |
| **HA-SLR compatibility**  | ✗ No               | ✓ Yes              |
| **Training speed**        | Slower             | Faster             |
| **Information loss**      | None               | Minimal\*          |
| **Recommended**           | ✅ Yes             | Optional           |

\* Dummy elbows still allow valid bone calculation

---

## ⚡ Performance

### Execution Time

- **Phase 1** (Splits): <1 min
- **Phase 2A** (46-joint): ~100 sec
- **Phase 2B** (27-joint): ~100 sec
- **Phase 4A** (46-joint multi-stream): ~600 sec (10 min)
- **Phase 4B** (27-joint multi-stream): ~300 sec (5 min)

### Resource Usage

- **CPU**: Multi-threaded (uses ~4-8 cores)
- **Memory**: ~2-3 GB peak
- **Disk**: ~4 GB total (3 GB raw + 1 GB processed)

---

## ❓ FAQ

### Q: Should I use 46_all or 46_to_27?

**A**: Use **46_all** unless you specifically need HA-SLR model compatibility. 46_all keeps full information.

### Q: What are dummy elbows?

**A**: Elbow positions copied from shoulder positions. Enables bone calculation without MultiVSL200's missing elbows.

### Q: Do dummy elbows hurt accuracy?

**A**: Unlikely if using bone-based features (which handle relative positions). Might hurt absolute coordinate models.

### Q: Can I use pre-trained HA-SLR models on MultiVSL200?

**A**: Yes! Generate 27-joint data with `--config 46_to_27`, then use it with HA-SLR models.

### Q: How long does full pipeline take?

**A**: ~12 min for 46_all only, ~20 min for both configs.

### Q: What if I only want 27-joint data?

**A**: Skip Phase 2A, jump directly to Phase 2B (46_to_27).

---

## 🛠 Troubleshooting

### Error: "IndexError: index 46 is out of bounds"

```bash
# Solution: Check your config matches data
# If using 46-joint data, use:
python generate_multivsl200_4streams.py --config 46_all

# If using 27-joint data, use:
python generate_multivsl200_4streams.py --config 46_to_27
```

### Error: "FileNotFoundError: train_labels.csv"

```bash
# Solution: Run Phase 1 first
python create_splits_by_signer.py
```

### Error: "MemoryError" during phase 4

```bash
# Solution: Not enough disk space. Check:
df -h

# Or process one split at a time if needed
```

**For more troubleshooting**: See FULL_PIPELINE_GUIDE.md section 10

---

## 📖 Reading Order

**Beginner**:

1. This file (README)
2. WORKFLOW_EXAMPLES.sh (run it!)
3. FULL_PIPELINE_GUIDE.md (for context)

**Intermediate**:

1. FULL_PIPELINE_GUIDE.md (understand phases)
2. ELBOW_GAP_AND_JOINT_MAPPING.md (understand problem)
3. CHANGES_SUMMARY.md (what changed)

**Advanced**:

1. ELBOW_GAP_AND_JOINT_MAPPING.md (technical details)
2. BUGFIX_BONE_PAIRS_27_INDEX_MISMATCH.md (critical bug fix ⚠)
3. CHANGES_SUMMARY.md (code changes)
4. Source code: `generate_multivsl200_*.py`

---

## ✅ Validation Checklist

After running pipeline, verify:

- [ ] All 12 files exist in MultiVSL200/ (46_all) or MultiVSL200_27/ (46_to_27)
- [ ] Joint data shape: (4515, 3, 150, 46, 1) or (4515, 3, 150, 27, 1)
- [ ] Bone data shape matches joint data
- [ ] Motion data shape matches joint data
- [ ] Last frame of motion data is all zeros (velocity=0)
- [ ] Label pickle files: 4515 entries for train

```bash
# Verification script
python << 'EOF'
import numpy as np
import pickle
import os

data_dir = "../data/MultiVSL200"  # or MultiVSL200_27

# Check files exist
required_files = ['train_data_joint.npy', 'train_data_bone.npy',
                  'train_data_joint_motion.npy', 'train_data_bone_motion.npy']
for f in required_files:
    path = os.path.join(data_dir, f)
    if os.path.exists(path):
        d = np.load(path)
        print(f"✓ {f}: {d.shape}")
    else:
        print(f"✗ {f}: NOT FOUND")

# Check label pickle
with open(os.path.join(data_dir, 'train_label.pkl'), 'rb') as f:
    names, labels = pickle.load(f)
    print(f"✓ train_label.pkl: {len(names)} samples")

# Check motion data (last frame should be 0)
motion = np.load(os.path.join(data_dir, 'train_data_joint_motion.npy'))
last_frame = motion[:, :, -1, :, :]
if np.allclose(last_frame, 0):
    print("✓ Motion data: Last frame is zero (correct)")
else:
    print("✗ Motion data: Last frame is NOT zero (check!)")
EOF
```

---

## 🔗 Related Files

- Split generation: `create_splits_by_signer.py`
- 4-stream generation: `generate_multivsl200_4streams.py`
- Training configs: `config/sign_cvpr_A_hands/MultiVSL200/`
- Training script: `main_base.py`

---

## 📞 Support

**For issues with:**

- Pipeline steps → See FULL_PIPELINE_GUIDE.md
- Elbow gap problem → See ELBOW_GAP_AND_JOINT_MAPPING.md
- What changed → See CHANGES_SUMMARY.md
- How to run → See WORKFLOW_EXAMPLES.sh

---

**Last Updated**: 2026-04-30  
**Status**: ✅ Complete & Ready to Use  
**Version**: v1.0
