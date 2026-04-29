#!/bin/bash
# MultiVSL200 Complete Workflow - Executable Examples
# Copy and paste these commands directly to terminal

# ============================================================================
# SETUP
# ============================================================================

cd "Code/Network/SL_GCN/data_gen"
echo "Current directory: $(pwd)"
echo ""

# ============================================================================
# PHASE 1: CREATE SPLITS (if not done before)
# ============================================================================

echo "==========================================="
echo "PHASE 1: CREATE SPLITS BY SIGNER"
echo "==========================================="
echo ""
echo "Creating 8:1:1 signer-independent splits..."
echo "Expected output:"
echo "  - splits/train_labels.csv (4515 samples)"
echo "  - splits/val_labels.csv (791 samples)"
echo "  - splits/test_labels.csv (593 samples)"
echo ""

python create_splits_by_signer.py \
  --data_dir ../data/MultiVSL200 \
  --out_dir ../data/MultiVSL200/splits

echo "✓ Phase 1 complete!"
echo ""

# ============================================================================
# PHASE 2A: GENERATE 46-JOINT DATA (Original MultiVSL200)
# ============================================================================

echo "==========================================="
echo "PHASE 2A: GENERATE 46-JOINT DATA"
echo "==========================================="
echo ""
echo "Generating 46-joint format (keep full skeleton)..."
echo "Expected output shapes:"
echo "  - train: (4515, 3, 150, 46, 1)"
echo "  - val: (791, 3, 150, 46, 1)"
echo "  - test: (593, 3, 150, 46, 1)"
echo ""

python generate_multivsl200_splits.py \
  --data_dir ../data/MultiVSL200 \
  --split_dir ../data/MultiVSL200/splits \
  --out_dir ../data/MultiVSL200 \
  --config 46_all

echo "✓ Phase 2A complete!"
echo ""

# ============================================================================
# PHASE 2B: GENERATE 27-JOINT DATA (HA-SLR compatible)
# ============================================================================

echo "==========================================="
echo "PHASE 2B: GENERATE 27-JOINT DATA (OPTIONAL)"
echo "==========================================="
echo ""
echo "Generating 27-joint format (HA-SLR compatible with dummy elbows)..."
echo "Expected output shapes:"
echo "  - train: (4515, 3, 150, 27, 1)"
echo "  - val: (791, 3, 150, 27, 1)"
echo "  - test: (593, 3, 150, 27, 1)"
echo ""
echo "NOTE: This is OPTIONAL. Only run if you want HA-SLR compatibility."
echo "Press Ctrl+C to skip, or Enter to continue..."
read -p ""

python generate_multivsl200_splits.py \
  --data_dir ../data/MultiVSL200 \
  --split_dir ../data/MultiVSL200/splits \
  --out_dir ../data/MultiVSL200_27 \
  --config 46_to_27

echo "✓ Phase 2B complete!"
echo ""

# ============================================================================
# PHASE 4A: GENERATE 4-STREAM DATA (46-JOINT)
# ============================================================================

echo "==========================================="
echo "PHASE 4A: GENERATE 4-STREAM DATA (46-JOINT)"
echo "==========================================="
echo ""
echo "Generating 4 streams: Joint, Bone, Joint Motion, Bone Motion"
echo "This may take ~10 minutes..."
echo ""
echo "Output files:"
echo "  - *_data_bone.npy"
echo "  - *_data_joint_motion.npy"
echo "  - *_data_bone_motion.npy"
echo ""

python generate_multivsl200_4streams.py \
  --data_dir ../data/MultiVSL200 \
  --config 46_all

echo "✓ Phase 4A complete!"
echo ""

# ============================================================================
# PHASE 4B: GENERATE 4-STREAM DATA (27-JOINT)
# ============================================================================

echo "==========================================="
echo "PHASE 4B: GENERATE 4-STREAM DATA (27-JOINT) (OPTIONAL)"
echo "==========================================="
echo ""
echo "Generating 4 streams for 27-joint format..."
echo "This may take ~5 minutes..."
echo ""
echo "NOTE: This is OPTIONAL. Only run if you ran Phase 2B."
echo "Press Ctrl+C to skip, or Enter to continue..."
read -p ""

python generate_multivsl200_4streams.py \
  --data_dir ../data/MultiVSL200_27 \
  --config 46_to_27

echo "✓ Phase 4B complete!"
echo ""

# ============================================================================
# VERIFICATION
# ============================================================================

echo "==========================================="
echo "VERIFICATION"
echo "==========================================="
echo ""
echo "Checking generated files..."
echo ""

echo "46-joint files:"
ls -lh ../data/MultiVSL200/*_data_*.npy | awk '{print $9, "-", $5}'
echo ""

if [ -d "../data/MultiVSL200_27" ]; then
  echo "27-joint files:"
  ls -lh ../data/MultiVSL200_27/*_data_*.npy | awk '{print $9, "-", $5}'
  echo ""
fi

# Count files
count_46=$(find ../data/MultiVSL200 -name "*_data_*.npy" | wc -l)
count_27=$(find ../data/MultiVSL200_27 -name "*_data_*.npy" 2>/dev/null | wc -l)

echo "Files generated:"
echo "  46-joint: $count_46 files (expected: 12)"
echo "  27-joint: $count_27 files (expected: 12, if Phase 2B run)"
echo ""

# ============================================================================
# SUMMARY
# ============================================================================

echo "==========================================="
echo "✓ COMPLETE!"
echo "==========================================="
echo ""
echo "You now have:"
echo ""
echo "46-joint datasets (in ../data/MultiVSL200/):"
echo "  ✓ train_data_joint.npy (4515, 3, 150, 46, 1)"
echo "  ✓ train_data_bone.npy"
echo "  ✓ train_data_joint_motion.npy"
echo "  ✓ train_data_bone_motion.npy"
echo "  (+ val and test splits)"
echo ""

if [ -d "../data/MultiVSL200_27" ]; then
  echo "27-joint datasets (in ../data/MultiVSL200_27/):"
  echo "  ✓ train_data_joint.npy (4515, 3, 150, 27, 1) [HA-SLR compatible]"
  echo "  ✓ train_data_bone.npy"
  echo "  ✓ train_data_joint_motion.npy"
  echo "  ✓ train_data_bone_motion.npy"
  echo "  (+ val and test splits)"
  echo ""
fi

echo "Ready for training! 🚀"
echo ""
echo "Next steps:"
echo "  1. Create training configs in config/sign_cvpr_A_hands/MultiVSL200/"
echo "  2. Run main_base.py with your chosen config"
echo "  3. Monitor training in work_dir/"
echo ""

# ============================================================================
# INDIVIDUAL COMMAND EXAMPLES
# ============================================================================

cat << 'EOF'

============================================================================
INDIVIDUAL COMMAND EXAMPLES
============================================================================

# If you need to re-run individual phases:

# Phase 1 only:
python create_splits_by_signer.py \
  --data_dir ../data/MultiVSL200 \
  --out_dir ../data/MultiVSL200/splits

# Phase 2A only (46-joint):
python generate_multivsl200_splits.py \
  --data_dir ../data/MultiVSL200 \
  --split_dir ../data/MultiVSL200/splits \
  --out_dir ../data/MultiVSL200 \
  --config 46_all

# Phase 2B only (27-joint):
python generate_multivsl200_splits.py \
  --data_dir ../data/MultiVSL200 \
  --split_dir ../data/MultiVSL200/splits \
  --out_dir ../data/MultiVSL200_27 \
  --config 46_to_27

# Phase 4A only (46-joint multi-stream):
python generate_multivsl200_4streams.py \
  --data_dir ../data/MultiVSL200 \
  --config 46_all

# Phase 4B only (27-joint multi-stream):
python generate_multivsl200_4streams.py \
  --data_dir ../data/MultiVSL200_27 \
  --config 46_to_27

# Check output shapes:
python << 'PYEOF'
import numpy as np
d = np.load("../data/MultiVSL200/train_data_joint.npy")
print(f"46-joint shape: {d.shape}")

d = np.load("../data/MultiVSL200_27/train_data_joint.npy")
print(f"27-joint shape: {d.shape}")
PYEOF

============================================================================
TROUBLESHOOTING
============================================================================

# Problem: "FileNotFoundError: train_labels.csv"
# Solution: Run Phase 1 first

# Problem: "IndexError: index 46 is out of bounds"
# Solution: Check config matches data. Use 46_all for 46-joint data, 46_to_27 for 27-joint

# Problem: "MemoryError"
# Solution: Processing on SSD. Ensure enough disk space. Check: df -h

# Problem: Missing output files
# Solution: Check Phase X completed successfully. Look for "✓ Complete!" message

============================================================================
PERFORMANCE NOTES
============================================================================

Timeline (approximate):
- Phase 1: <1 minute
- Phase 2A (46_all): 100 seconds
- Phase 2B (46_to_27): 100 seconds
- Phase 4A (46_all, multi-stream): 600 seconds (10 min)
- Phase 4B (46_to_27, multi-stream): 300 seconds (5 min)
- Total (both configs): ~15 minutes

Memory usage:
- Joint data: ~100 MB per stream per split
- Bone data: ~100 MB per stream per split
- Motion data: ~100 MB per stream per split
- Total per config: ~3-4 GB

Disk space needed:
- MultiVSL200/ (46-joint): ~1.5 GB
- MultiVSL200_27/ (27-joint): ~1.2 GB
- Raw .npy files: ~3 GB

============================================================================
EOF

echo ""
