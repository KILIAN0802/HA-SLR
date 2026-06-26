#!/bin/bash
# ==============================================================================
# train_and_evaluate_multivsl200.sh
# ==============================================================================
# Script tự động chạy trọn gói từ huấn luyện đến suy luận/ablation ensemble
# trên tập dữ liệu MultiVSL200 (tắt Mixup/JDMA và tắt Clone & Evolve).
#
# Cách chạy bằng nohup:
#   nohup ./train_and_evaluate_multivsl200.sh > train_pipeline.log 2>&1 &
# ==============================================================================

# Thoát ngay lập tức nếu gặp lỗi
set -e

# Cấu hình GPU ID sử dụng cho toàn bộ pipeline (Có thể thay đổi thành 0, 1, 2, 3 tùy ý)
export GPU_ID=1
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Đường dẫn tới Python của môi trường ảo haslr_env
PYTHON_BIN="/mnt/nvme0/home/utbt_sv1/miniconda3/envs/haslr_env/bin/python"

echo "=============================================================================="
echo "BẮT ĐẦU PIPELINE HUẤN LUYỆN & SUY LUẬN MULTIVSL200"
echo "Thời gian: $(date)"
echo "Sử dụng GPU: $GPU_ID"
echo "=============================================================================="

# ------------------------------------------------------------------------------
# BƯỚC 1: HUẤN LUYỆN 4 LUỒNG DỮ LIỆU
# ------------------------------------------------------------------------------

echo -e "\n>>> [1/5] Đang huấn luyện luồng Joint (Khớp)..."
$PYTHON_BIN main_base.py --config config/MultiVSL200/train_joint.yaml --use-wandb True --wandb-project HA-SLR-GCN --wandb-run-name "MultiVSL200-Joint-Baseline"

echo -e "\n>>> [2/5] Đang huấn luyện luồng Bone (Xương)..."
$PYTHON_BIN main_base.py --config config/MultiVSL200/train_bone.yaml --use-wandb True --wandb-project HA-SLR-GCN --wandb-run-name "MultiVSL200-Bone-Baseline"

echo -e "\n>>> [3/5] Đang huấn luyện luồng Joint Motion (Chuyển động Khớp)..."
$PYTHON_BIN main_base.py --config config/MultiVSL200/train_joint_motion.yaml --use-wandb True --wandb-project HA-SLR-GCN --wandb-run-name "MultiVSL200-Joint_Motion-Baseline"

echo -e "\n>>> [4/5] Đang huấn luyện luồng Bone Motion (Chuyển động Xương)..."
$PYTHON_BIN main_base.py --config config/MultiVSL200/train_bone_motion.yaml --use-wandb True --wandb-project HA-SLR-GCN --wandb-run-name "MultiVSL200-Bone_Motion-Baseline"

# ------------------------------------------------------------------------------
# BƯỚC 2: CẬP NHẬT CẤU HÌNH ENSEMBLE VỚI CHECKPOINT MỚI NHẤT
# ------------------------------------------------------------------------------
echo -e "\n>>> [5/5] Đang tìm kiếm checkpoint tốt nhất và cập nhật YAML cấu hình..."

$PYTHON_BIN -c "
import glob
import os
import yaml

def get_latest_checkpoint(pattern):
    files = glob.glob(pattern, recursive=True)
    if not files:
        return None
    # Lấy file có thời gian sửa đổi mới nhất
    return max(files, key=os.path.getmtime)

joint_ckpt = get_latest_checkpoint('work_dir/MultiVSL200/Joint/**/Joint_best_acc_*.pt')
bone_ckpt = get_latest_checkpoint('work_dir/MultiVSL200/Bone/**/Bone_best_acc_*.pt')
jm_ckpt = get_latest_checkpoint('work_dir/MultiVSL200/Joint_Motion/**/Joint_Motion_best_acc_*.pt')
bm_ckpt = get_latest_checkpoint('work_dir/MultiVSL200/Bone_Motion/**/Bone_Motion_best_acc_*.pt')

print('Checkpoints tìm được:')
print(f'  - Joint: {joint_ckpt}')
print(f'  - Bone: {bone_ckpt}')
print(f'  - Joint Motion: {jm_ckpt}')
print(f'  - Bone Motion: {bm_ckpt}')

yaml_path = 'ensemble/test_ensemble.yaml'
if not os.path.exists(yaml_path):
    print(f'Lỗi: Không tìm thấy file {yaml_path}')
    exit(1)

with open(yaml_path, 'r') as f:
    config = yaml.safe_load(f)

if joint_ckpt:
    config['joint_weights'] = joint_ckpt
if bone_ckpt:
    config['bone_weights'] = bone_ckpt
if jm_ckpt:
    config['joint_motion_weights'] = jm_ckpt
if bm_ckpt:
    config['bone_motion_weights'] = bm_ckpt

with open(yaml_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
print(f'Đã cập nhật thành công {yaml_path}!')
"

# ------------------------------------------------------------------------------
# BƯỚC 3: CHẠY ABLATION STUDY ENSEMBLE
# ------------------------------------------------------------------------------
echo -e "\n>>> ĐANG CHẠY ABLATION STUDY ENSEMBLE..."
$PYTHON_BIN ablation_ensemble.py --config ensemble/test_ensemble.yaml

echo -e "\n=============================================================================="
echo "HOÀN THÀNH PIPELINE!"
echo "Thời gian kết thúc: $(date)"
echo "=============================================================================="
