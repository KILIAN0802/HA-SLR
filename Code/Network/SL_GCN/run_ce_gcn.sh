#!/bin/bash

# Tự động dừng script nếu có bước nào bị lỗi
set -e

PROJECT_NAME="HA-SLR-CEGCN"

echo "=========================================================="
echo "GIAI ĐOẠN 1: HUẤN LUYỆN LUỒNG GỐC JOINT (SEMANTIC ANCHOR)"
echo "=========================================================="
python -u main_base.py \
  --config config/MultiVSL200/train_joint.yaml \
  --use-wandb True \
  --wandb-project "$PROJECT_NAME" \
  --wandb-run-name MultiVSL200_Joint_Baseline

echo "=========================================================="
echo "GIAI ĐOẠN 2 & 3: CLONE & EVOLVE CHO LUỒNG BONE"
echo "=========================================================="
python -u main_base.py \
  --config config/MultiVSL200/train_bone.yaml \
  --clone_auto True \
  --evolve_mode True \
  --use-wandb True \
  --wandb-project "$PROJECT_NAME" \
  --wandb-run-name MultiVSL200_Bone_Evolve

echo "=========================================================="
echo "GIAI ĐOẠN 2 & 3: CLONE & EVOLVE CHO LUỒNG JOINT MOTION"
echo "=========================================================="
python -u main_base.py \
  --config config/MultiVSL200/train_joint_motion.yaml \
  --clone_auto True \
  --evolve_mode True \
  --use-wandb True \
  --wandb-project "$PROJECT_NAME" \
  --wandb-run-name MultiVSL200_JointMotion_Evolve

echo "=========================================================="
echo "GIAI ĐOẠN 2 & 3: CLONE & EVOLVE CHO LUỒNG BONE MOTION"
echo "=========================================================="
python -u main_base.py \
  --config config/MultiVSL200/train_bone_motion.yaml \
  --clone_auto True \
  --evolve_mode True \
  --use-wandb True \
  --wandb-project "$PROJECT_NAME" \
  --wandb-run-name MultiVSL200_BoneMotion_Evolve

echo "=========================================================="
echo "QUÁ TRÌNH HUẤN LUYỆN LIÊN TIẾP 4 LUỒNG CE-GCN ĐÃ HOÀN TẤT!"
echo "=========================================================="
