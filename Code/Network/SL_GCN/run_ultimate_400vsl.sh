#!/bin/bash

# Tự động dừng script nếu có bước nào bị lỗi
set -e


# Cấu hình đường dẫn
VIDEO_DIR="data/data/400VSL/400VSLcropped"
RAW_NPY_DIR="data/data/400VSL/raw_npy"
PROCESSED_DIR="data/data/400VSL/processed/27_direct"

echo "=========================================================="
echo "KHỞI ĐỘNG ULTIMATE PIPELINE: TỪ VIDEO RAW ĐẾN MODEL FUSION"
echo "=========================================================="

# BƯỚC 1: TRÍCH XUẤT SKELETON TỪ VIDEO (VIDEO RAW -> NPY)
# Đây là bước tốn thời gian nhất (MediaPipe Holistic)
echo ">>> BƯỚC 1: Trích xuất tọa độ khớp từ video..."
if [ ! -d "$RAW_NPY_DIR" ] || [ -z "$(ls -A $RAW_NPY_DIR)" ]; then
    python Preprocess/demo.py \
        --input_dir "$VIDEO_DIR" \
        --output_dir "$RAW_NPY_DIR" \
        --max_frame 150
else
    echo "Thư mục raw_npy đã có dữ liệu. Bỏ qua bước trích xuất video để tiết kiệm thời gian."
fi

# BƯỚC 2: TỔ CHỨC DỮ LIỆU VÀ TẠO CÁC LUỒNG BONE/MOTION
# Bước này chia tập Train/Val/Test và tổng hợp các file .npy thành 1 file lớn cho GCN
echo ">>> BƯỚC 2: Tổ chức dữ liệu và tạo các luồng Bone, Motion..."
python prepare_400vsl_pipeline.py

# BƯỚC 3: HUẤN LUYỆN 4 LUỒNG (FOUR STREAMS)
echo "=========================================================="
echo ">>> BƯỚC 3: Huấn luyện đồng thời 4 luồng đặc trưng..."
echo "=========================================================="

# 3.1 Luồng Joint
echo "[3.1/4] Training JOINT stream..."
python -u main_base.py \
  --config config/400VSL/train_joint.yaml 

# 3.2 Luồng Bone (Sử dụng Clone & Evolve từ Joint để nhanh hơn)
echo "[3.2/4] Training BONE stream..."
python -u main_base.py \
  --config config/400VSL/train_bone.yaml \
  --clone_auto True \
  --evolve_mode True 

# 3.3 Luồng Joint Motion
echo "[3.3/4] Training JOINT MOTION stream..."
python -u main_base.py \
  --config config/400VSL/train_joint_motion.yaml \
  --clone_auto True \
  --evolve_mode True 

# 3.4 Luồng Bone Motion
echo "[3.4/4] Training BONE MOTION stream..."
python -u main_base.py \
  --config config/400VSL/train_bone_motion.yaml \
  --clone_auto True \
  --evolve_mode True 

# BƯỚC 4: ADAPTIVE FUSION (KẾT HỢP KẾT QUẢ)
echo "=========================================================="
echo ">>> BƯỚC 4: Huấn luyện Adaptive Fusion Gate để tối ưu hóa kết quả..."
echo "=========================================================="

# Tìm checkpoint tốt nhất
find_best() {
    find "$1" -name "*_best_acc_*.pt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -f2- -d" "
}

JOINT_CKPT=$(find_best "work_dir/400VSL/Joint")
BONE_CKPT=$(find_best "work_dir/400VSL/Bone")
JM_CKPT=$(find_best "work_dir/400VSL/Joint_Motion")
BM_CKPT=$(find_best "work_dir/400VSL/Bone_Motion")

if [ -n "$JOINT_CKPT" ]; then
    python -u train_fusion_gate.py \
      --data_path "$PROCESSED_DIR/val_data_joint.npy" \
      --label_path "$PROCESSED_DIR/val_label.pkl" \
      --w_joint "$JOINT_CKPT" \
      --w_bone "$BONE_CKPT" \
      --w_jm "$JM_CKPT" \
      --w_bm "$BM_CKPT" \
      --save_path "work_dir/400VSL/fusion_gate_final.pt" \
      --epochs 50
else
    echo "LỖI: Không tìm thấy checkpoint để thực hiện Fusion!"
fi

echo "=========================================================="
echo "PIPELINE HOÀN TẤT THÀNH CÔNG!"
echo "Model cuối cùng lưu tại: work_dir/400VSL/fusion_gate_final.pt"
echo "=========================================================="
