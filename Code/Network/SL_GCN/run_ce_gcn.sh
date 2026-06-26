#!/bin/bash

# Tự động dừng script nếu có bước nào bị lỗi
set -e

# GPU ID sử dụng cho toàn bộ pipeline (Sử dụng GPU 1)
export GPU_ID=1
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Cấu hình W&B ở chế độ online để đẩy kết quả lên server
export WANDB_MODE=online

# Đường dẫn tới Python của môi trường ảo haslr_env
PYTHON_BIN="/mnt/nvme0/home/utbt_sv1/miniconda3/envs/haslr_env/bin/python"

PROJECT_NAME="HA-SLR-GCN"
DATA_DIR="data/data/MultiVSL200/processed/27_direct"

echo "=========================================================="
echo "GIAI ĐOẠN 0: BỎ QUA TRÍCH XUẤT LUỒNG (ĐÃ CÓ SẴN TRONG MULTIVSL200)"
echo "=========================================================="
# Đã được sinh trực tiếp bằng preprocess_multivsl200.py, bỏ qua bước này.

echo "=========================================================="
echo "GIAI ĐOẠN 1: HUẤN LUYỆN LUỒNG GỐC JOINT (SEMANTIC ANCHOR)"
echo "=========================================================="
$PYTHON_BIN -u main_base.py \
  --config config/MultiVSL200/train_joint.yaml \
  --use-wandb True \
  --wandb-project "$PROJECT_NAME" \
  --wandb-run-name MultiVSL200_Joint_Evolve-1

echo "=========================================================="
echo "GIAI ĐOẠN 2 & 3: CLONE & EVOLVE CHO LUỒNG BONE"
echo "=========================================================="
$PYTHON_BIN -u main_base.py \
  --config config/MultiVSL200/train_bone.yaml \
  --clone_auto True \
  --evolve_mode True \
  --use-wandb True \
  --wandb-project "$PROJECT_NAME" \
  --wandb-run-name MultiVSL200_Bone_Evolve-1

echo "=========================================================="
echo "GIAI ĐOẠN 2 & 3: CLONE & EVOLVE CHO LUỒNG JOINT MOTION"
echo "=========================================================="
$PYTHON_BIN -u main_base.py \
  --config config/MultiVSL200/train_joint_motion.yaml \
  --clone_auto True \
  --evolve_mode True \
  --use-wandb True \
  --wandb-project "$PROJECT_NAME" \
  --wandb-run-name MultiVSL200_JointMotion_Evolve-1

echo "=========================================================="
echo "GIAI ĐOẠN 2 & 3: CLONE & EVOLVE CHO LUỒNG BONE MOTION"
echo "=========================================================="
$PYTHON_BIN -u main_base.py \
  --config config/MultiVSL200/train_bone_motion.yaml \
  --clone_auto True \
  --evolve_mode True \
  --use-wandb True \
  --wandb-project "$PROJECT_NAME" \
  --wandb-run-name MultiVSL200_BoneMotion_Evolve-1

echo "=========================================================="
echo "GIAI ĐOẠN 4: TỰ ĐỘNG TÌM CHECKPOINT TỐT NHẤT VÀ HUẤN LUYỆN ADAPTIVE FUSION GATE"
echo "=========================================================="

# Hàm helper tìm checkpoint mới nhất có độ chính xác tốt nhất
find_latest_checkpoint() {
    local stream_dir=$1
    local cp=$(find "$stream_dir" -name "*_best_acc_*.pt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -f2- -d" ")
    
    if [ -z "$cp" ]; then
        cp=$(find "$stream_dir" -name "*_best_acc_*.pt" -type f 2>/dev/null | xargs ls -t 2>/dev/null | head -1)
    fi
    echo "$cp"
}

JOINT_CKPT=$(find_latest_checkpoint "work_dir/MultiVSL200/Joint")
BONE_CKPT=$(find_latest_checkpoint "work_dir/MultiVSL200/Bone")
JM_CKPT=$(find_latest_checkpoint "work_dir/MultiVSL200/Joint_Motion")
BM_CKPT=$(find_latest_checkpoint "work_dir/MultiVSL200/Bone_Motion")

echo "-> Checkpoint Joint tìm thấy: $JOINT_CKPT"
echo "-> Checkpoint Bone tìm thấy: $BONE_CKPT"
echo "-> Checkpoint Joint Motion tìm thấy: $JM_CKPT"
echo "-> Checkpoint Bone Motion tìm thấy: $BM_CKPT"

# Kiểm tra sự tồn tại của đầy đủ 4 checkpoint
if [ -z "$JOINT_CKPT" ] || [ -z "$BONE_CKPT" ] || [ -z "$JM_CKPT" ] || [ -z "$BM_CKPT" ]; then
    echo "CẢNH BÁO: Không tìm thấy đầy đủ 4 checkpoint thực tế trong thư mục work_dir!"
    echo "Huấn luyện Fusion Gate bằng các trọng số mặc định được khai báo trong train_fusion_gate.py..."
    $PYTHON_BIN -u train_fusion_gate.py \
      --data_path "$DATA_DIR/val_data_joint.npy" \
      --label_path "$DATA_DIR/val_label.pkl" \
      --save_path "work_dir/fusion_gate_best.pt" \
      --epochs 20 \
      --lr 1e-3
else
    echo "Đang huấn luyện AdaptiveFusionGate bằng các checkpoint vừa tìm thấy..."
    $PYTHON_BIN -u train_fusion_gate.py \
      --data_path "$DATA_DIR/val_data_joint.npy" \
      --label_path "$DATA_DIR/val_label.pkl" \
      --w_joint "$JOINT_CKPT" \
      --w_bone "$BONE_CKPT" \
      --w_jm "$JM_CKPT" \
      --w_bm "$BM_CKPT" \
      --save_path "work_dir/fusion_gate_best.pt" \
      --epochs 20 \
      --lr 1e-3
fi

echo "=========================================================="
echo "GIAI ĐOẠN 5: ĐÁNH GIÁ TOÀN BỘ HỆ THỐNG VÀ XUẤT BẢNG KẾT QUẢ"
echo "=========================================================="
$PYTHON_BIN -u evaluate_all.py --config ensemble/test_ensemble.yaml

echo "=========================================================="
echo "QUÁ TRÌNH HUẤN LUYỆN CE-GCN VÀ FUSION GATE ĐÃ HOÀN TẤT!"
echo "=========================================================="
