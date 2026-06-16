#!/bin/bash

# Tự động dừng script nếu có bước nào bị lỗi
set -e

# Chỉ định GPU chạy (Ví dụ: 0, 1, 2...). Nếu GPU 0 bị đầy, hãy đổi thành số GPU trống khác.
export CUDA_VISIBLE_DEVICES=1

# Lấy ngày tháng năm và giờ phút giây để tạo thư mục lưu log riêng biệt
LOG_DATE=$(date +%d-%m-%Y)
echo "LOGS WILL BE SAVED TO: log/${LOG_DATE}"
mkdir -p log/${LOG_DATE}

# Cấu hình đường dẫn dữ liệu
VIDEO_DIR="data/data/400VSL/400VSLcropped"
RAW_NPY_DIR="data/data/400VSL/raw_npy"
PROCESSED_DIR="data/data/400VSL/processed/27_direct"

# # BƯỚC 1: TRÍCH XUẤT SKELETON TỪ VIDEO (VIDEO RAW -> NPY)
# # Đây là bước tốn thời gian nhất (MediaPipe Holistic)
# echo ">>> BƯỚC 1: Trích xuất tọa độ khớp từ video..."
# if [ ! -d "$RAW_NPY_DIR" ] || [ -z "$(ls -A $RAW_NPY_DIR)" ]; then
#     python Preprocess/demo.py \
#         --input_dir "$VIDEO_DIR" \
#         --output_dir "$RAW_NPY_DIR" \
#         --max_frame 150
# else
#     echo "Thư mục raw_npy đã có dữ liệu. Bỏ qua bước trích xuất video để tiết kiệm thời gian."
# fi

# # BƯỚC 2: TỔ CHỨC DỮ LIỆU VÀ TẠO CÁC LUỒNG BONE/MOTION
# # Bước này chia tập Train/Val/Test và tổng hợp các file .npy thành 1 file lớn cho GCN
# echo ">>> BƯỚC 2: Tổ chức dữ liệu và tạo các luồng Bone, Motion..."
# python prepare_400vsl_pipeline.py

# BƯỚC 3: HUẤN LUYỆN TIẾP CÁC LUỒNG TUẦN TỰ (FOUR STREAMS - AUTO RESUME, CLONE DISABLED)
# Chạy tuần tự giúp tránh lỗi CUDA Out of Memory trên GPU dùng chung
echo "=========================================================="
echo ">>> BƯỚC 3: Huấn luyện tiếp tục các luồng đặc trưng (Tuần tự)..."
echo "=========================================================="

# 3.1 Luồng Joint
echo "[3.1/4] Training JOINT stream..."
python -u main_base.py \
  --config config/400VSL/train_joint.yaml \
  --auto-resume True \
  --clone_auto False > log/${LOG_DATE}/train_joint.log 2>&1

# 3.2 Luồng Bone (Sử dụng Evolve nhưng tắt Clone)
echo "[3.2/4] Training BONE stream..."
python -u main_base.py \
  --config config/400VSL/train_bone.yaml \
  --auto-resume True \
  --clone_auto False \
  --evolve_mode True > log/${LOG_DATE}/train_bone.log 2>&1

# 3.3 Luồng Joint Motion
echo "[3.3/4] Training JOINT MOTION stream..."
python -u main_base.py \
  --config config/400VSL/train_joint_motion.yaml \
  --auto-resume True \
  --clone_auto False \
  --evolve_mode True > log/${LOG_DATE}/train_joint_motion.log 2>&1

# 3.4 Luồng Bone Motion
echo "[3.4/4] Training BONE MOTION stream..."
python -u main_base.py \
  --config config/400VSL/train_bone_motion.yaml \
  --auto-resume True \
  --clone_auto False \
  --evolve_mode True > log/${LOG_DATE}/train_bone_motion.log 2>&1

# BƯỚC 4: ADAPTIVE FUSION (KẾT HỢP KẾT QUẢ VÀ HUẤN LUYỆN GATING NETWORK)
echo "=========================================================="
echo ">>> BƯỚC 4: Huấn luyện Adaptive Fusion Gate..."
echo "=========================================================="

# Tìm checkpoint tốt nhất
find_best() {
    find "$1" -name "*_best_acc_*.pt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -f2- -d" "
}

JOINT_CKPT=$(find_best "work_dir/400VSL-no_clone/Joint")
BONE_CKPT=$(find_best "work_dir/400VSL-no_clone/Bone")
JM_CKPT=$(find_best "work_dir/400VSL-no_clone/Joint_Motion")
BM_CKPT=$(find_best "work_dir/400VSL-no_clone/Bone_Motion")

if [ -n "$JOINT_CKPT" ]; then
    echo "Starting Adaptive Fusion Gate training..."
    python -u train_fusion_gate.py \
      --data_path "$PROCESSED_DIR/val_data_joint.npy" \
      --label_path "$PROCESSED_DIR/val_label.pkl" \
      --w_joint "$JOINT_CKPT" \
      --w_bone "$BONE_CKPT" \
      --w_jm "$JM_CKPT" \
      --w_bm "$BM_CKPT" \
      --save_path "work_dir/400VSL-no_clone/fusion_gate_final.pt" \
      --epochs 50 > log/${LOG_DATE}/train_fusion.log 2>&1
else
    echo "LỖI: Không tìm thấy checkpoint để thực hiện Fusion!"
fi

# BƯỚC 5: ĐÁNH GIÁ ĐA LUỒNG (STATIC & DYNAMIC INFERENCE)
echo "=========================================================="
echo ">>> BƯỚC 5: Chạy đánh giá đa luồng (Static & Dynamic)..."
echo "=========================================================="
python -u evaluate_all.py \
  --config config/400VSL/test_ensemble.yaml > log/${LOG_DATE}/evaluation.log 2>&1

echo "=========================================================="
echo "PIPELINE HOÀN TẤT THÀNH CÔNG!"
echo " - Log huấn luyện các luồng và Fusion Gate lưu tại: log/${LOG_DATE}/"
echo " - Báo cáo đánh giá (Static & Dynamic) lưu tại: log/${LOG_DATE}/evaluation.log"
echo "=========================================================="
