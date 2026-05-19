# Hướng Dẫn Sử Dụng Kiến Trúc CE-GCN (Clone & Evolve) trên HA-SLR-GCN

Tài liệu này hướng dẫn bạn cách thiết lập dữ liệu, cấu hình WandB, và huấn luyện mô hình qua 3 giai đoạn của mạng đa luồng động học.

---

## BƯỚC 1: ĐẶT DỮ LIỆU VÀ SINH LUỒNG (OFFLINE STREAMS)

**1. Vị trí đặt file .npy gốc:**
Bạn cần đảm bảo các tệp dữ liệu `Joint` (tọa độ gốc) được đặt ĐÚNG tại đường dẫn sau (tính từ thư mục gốc của project):
```text
Code/Network/SL_GCN/data/MultiVSL200/
 ├── train_data_joint.npy
 ├── test_data_joint.npy
 ├── train_label.pkl
 └── test_label.pkl
```

**2. Sinh 3 luồng Động học (Bone, Joint Motion, Bone Motion):**
Mở terminal/cmd, di chuyển vào thư mục `Code/Network/SL_GCN` và chạy kịch bản sinh dữ liệu ngoại tuyến:
```bash
cd Code/Network/SL_GCN
python generate_derived_streams.py
```
*(Nếu thành công, trong thư mục `data/MultiVSL200/` sẽ xuất hiện thêm 6 file `.npy` tương ứng cho bone, joint_motion, và bone_motion).*

---

## BƯỚC 2: CẤU HÌNH WANDB (WEIGHTS & BIASES)

Để hệ thống ghi nhận chính xác biểu đồ Loss/Accuracy lên WandB, bạn cần làm 2 việc:
1. **Đăng nhập WandB trên máy trạm:**
   Gõ lệnh sau vào terminal của máy trạm và dán API Key của bạn vào:
   ```bash
   wandb login
   ```
2. **Kích hoạt WandB trong câu lệnh chạy (hoặc trong file YAML):**
   Bạn chỉ cần thêm cờ `--use-wandb True` và đặt tên Project qua cờ `--wandb-project Tên_Project`.

---

## BƯỚC 3: HUẤN LUYỆN GIAI ĐOẠN 1 (SEMANTIC ANCHOR - LUỒNG JOINT)

Đây là bước huấn luyện Baseline. Chúng ta sẽ train luồng Joint cho đến khi hội tụ (đạt Top-1 cao nhất) để làm điểm tựa (Anchor).

```bash
CUDA_VISIBLE_DEVICES=0 python main_base.py \
  --config config/MultiVSL200/train_joint.yaml \
  --use-wandb True \
  --wandb-project HA-SLR-CEGCN \
  --wandb-run-name MultiVSL200_Joint_Baseline
```
*Lưu ý: Bạn có thể bật tính năng JDMA (Mixup) bằng cách thêm `use_jdma: True` vào mục `train_feeder_args` trong file `train_joint.yaml`.*

Sau khi chạy xong, trọng số tốt nhất sẽ tự động được lưu tại: 
`work_dir/MultiVSL200/Joint/baseline/checkpoints/..._best_acc_...pt`

---

## BƯỚC 4: HUẤN LUYỆN GIAI ĐOẠN 2 & 3 (CLONE & EVOLVE)

Để huấn luyện 3 luồng còn lại (Bone, Joint Motion, Bone Motion), bạn cần tạo 3 file config riêng biệt (bằng cách copy file `train_joint.yaml` và sửa lại đường dẫn data).

Ví dụ, tạo file `config/MultiVSL200/train_bone.yaml` với nội dung sửa đổi như sau:
```yaml
Experiment_name: MultiVSL200/Bone/baseline
train_feeder_args:
  data_path: data/MultiVSL200/train_data_bone.npy # Đã sửa thành bone
  label_path: data/MultiVSL200/train_label.pkl
  use_jdma: True # Bật JDMA nếu muốn
val_feeder_args:
  data_path: data/MultiVSL200/test_data_bone.npy  # Đã sửa thành bone
  label_path: data/MultiVSL200/test_label.pkl
```

**Lệnh chạy huấn luyện tiến hóa (Evolve) cho luồng Bone:**
Bây giờ, bạn thêm cờ `--clone_auto True` (để tự nạp trọng số Joint) và `--evolve_mode True` (để giảm tốc độ học và số epoch).

```bash
CUDA_VISIBLE_DEVICES=1 python main_base.py \
  --config config/MultiVSL200/train_bone.yaml \
  --clone_auto True \
  --evolve_mode True \
  --use-wandb True \
  --wandb-project HA-SLR-CEGCN \
  --wandb-run-name MultiVSL200_Bone_Evolve
```
*(Bạn làm tương tự cho `train_joint_motion.yaml` và `train_bone_motion.yaml` nhé).*

---

## BƯỚC 5: SUY LUẬN ĐA LUỒNG TRỰC TUYẾN (ONLINE MULTI-STREAM)

Sau khi bạn đã huấn luyện xong cả 4 luồng và có đủ 4 file `.pt`. Bạn mở file `online_inference_pipeline.py` và sửa biến `weight_paths` (ở dòng 85) thành đường dẫn thực tế đến 4 file checkpoints của bạn:

```python
weight_paths = [
    'work_dir/MultiVSL200/Joint/baseline/checkpoints/...pt',
    'work_dir/MultiVSL200/Bone/baseline/checkpoints/...pt',
    'work_dir/MultiVSL200/Joint_Motion/baseline/checkpoints/...pt',
    'work_dir/MultiVSL200/Bone_Motion/baseline/checkpoints/...pt'
]
```

Cuối cùng, để thử dự đoán 1 video thực tế, bạn chỉ cần chạy:
```bash
python online_inference_pipeline.py
```
