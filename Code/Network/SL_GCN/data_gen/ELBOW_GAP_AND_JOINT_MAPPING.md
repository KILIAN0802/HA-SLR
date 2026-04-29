# MultiVSL200 vs HA-SLR: Elbow Gap và Joint Mapping

## Vấn đề 1: "Khoảng trống khuỷu tay" (The Elbow Gap)

### Cấu trúc skeleton trong hai dataset:

| Layout              | Joints    | Cấu trúc                                   | Vấn đề                             |
| ------------------- | --------- | ------------------------------------------ | ---------------------------------- |
| **MultiVSL200**     | 46 joints | Pre-selected skeleton (no elbow)           | Thiếu elbow (Node 3, 4)            |
| **HA-SLR 27-point** | 27 joints | Nose, Shoulders, **Elbows**, Wrists, Hands | Yêu cầu elbow cho bone calculation |

### HA-SLR 27-point layout:

```
Index  Component
0      Nose
1      L-Shoulder
2      R-Shoulder
3      L-Elbow          ← HA-SLR cần để tính bone arm
4      R-Elbow          ← HA-SLR cần để tính bone arm
5      L-Wrist
6      R-Wrist
7-16   L-Hand (10 joints)
17-26  R-Hand (10 joints)
```

### Vì sao cần Elbow?

Trong `gen_bone_data.py`, bone được tính như:

```python
bone[v2] = joint[v2] - joint[v1]  # Sự chênh lệch giữa 2 joints liên tiếp

# Ví dụ: Để tính bone cánh tay
bone[wrist] = joint[wrist] - joint[elbow]  # Cần elbow để có vector xương cánh tay
```

Nếu thiếu elbow, bone calculation sẽ:

- **Lỗi IndexError** nếu code cứng elbow indices
- **Sai lệch hình học** nếu bỏ qua elbow

---

## Giải pháp: Dummy Elbow Values

### Chiến lược nhanh (Quick Fix)

Khi MultiVSL200 thiếu elbow, ta có 3 tùy chọn:

| Option                  | Cách                             | Ưu điểm                                | Nhược điểm                                    |
| ----------------------- | -------------------------------- | -------------------------------------- | --------------------------------------------- |
| **1. Copy từ Shoulder** | `elbow = shoulder`               | Giữ nguyên shape, không lỗi IndexError | Vector elbow→wrist = shoulder→wrist (kéo dài) |
| **2. Copy từ Wrist**    | `elbow = wrist`                  | Trực quan (elbow ở gần wrist)          | Vector elbow→wrist = 0 (thiếu thông tin)      |
| **3. Interpolate**      | `elbow = (shoulder + wrist) / 2` | Trung bình hợp lý                      | Phức tạp hơn, ít được dùng                    |

### Ưu tiên: **Option 1 (Copy từ Shoulder)**

- Lý do: Elbow thường gần shoulder hơn wrist trong mô hình skeleton
- Motion hợp lý: Wrist chuyển động so với shoulder (qua elbow)

---

## Mapping 46_to_27 trong generate_multivsl200_splits.py

### Định nghĩa mapping:

```python
'46_to_27': np.array([
    42, 43, 44,          # 0-2: Nose, L-Shoulder, R-Shoulder
    43, 44,              # 3-4: L-Elbow(dummy=L-Sho), R-Elbow(dummy=R-Sho)
    0, 21,               # 5-6: L-Wrist, R-Wrist
    0, 4, 5, 8, 9, 12, 13, 16, 17, 20,      # 7-16: L-Hand 10 joints
    21, 25, 26, 29, 30, 33, 34, 37, 38, 41  # 17-26: R-Hand 10 joints
], dtype=int)
```

### Cách sử dụng:

```bash
# Generate 46-joint version (mặc định)
python generate_multivsl200_splits.py \
  --data_dir ../data/MultiVSL200 \
  --split_dir ../data/MultiVSL200/splits \
  --out_dir ../data/MultiVSL200 \
  --config 46_all

# Generate 27-joint version với dummy elbow (cho HA-SLR compatibility)
python generate_multivsl200_splits.py \
  --data_dir ../data/MultiVSL200 \
  --split_dir ../data/MultiVSL200/splits \
  --out_dir ../data/MultiVSL200 \
  --config 46_to_27
```

### Output:

- `46_all`: `(N, 3, 150, 46, 1)` - Full 46-joint version
- `46_to_27`: `(N, 3, 150, 27, 1)` - HA-SLR compatible with dummy elbows

---

## Bone Pairs cho mỗi config

### Cho 46-joint (46_all):

```python
BONE_PAIRS_46 = [
    # Body connections
    (0, 1), (0, 2),           # Nose → Shoulders
    (1, 3), (3, 5), (2, 4), (4, 6),  # Shoulder → Elbow → Wrist
    (5, 7), (7, 9), (6, 8), (8, 10), # Elbows → Wrists
    # ... (26 cặp tổng cộng)
]
```

### Cho 27-joint (46_to_27):

```python
# Sử dụng bone pairs chuẩn HA-SLR
# Chú ý: Indices 3, 4 (elbow) sẽ là dummy values (= shoulder indices)
paris['sign/27_cvpr'] = [
    (5, 6), (5, 7),
    (6, 8), (8, 10), (7, 9), (9, 11),  # Arm chain
    # ... Hand connections
]
```

---

## Implementation trong Multi-Stream Generation

### File: generate_multivsl200_4streams.py

Khi sinh bone data:

```python
# Cho 46-joint
for v1, v2 in BONE_PAIRS_46:
    bone_fp[:, :, :, v2, :] = joint_data[:, :, :, v2, :] - joint_data[:, :, :, v1, :]

# Cho 27-joint
# Elbow indices (3, 4) sẽ sử dụng shoulder indices (1, 2) từ mapping
# bone[3] = joint[3] - joint[5] = shoulder[3] - joint[5] ← Valid (vì 3 = shoulder index)
```

**Kết quả**: Shape không thay đổi, lỗi IndexError được tránh ✓

---

## Checklist triển khai

- [ ] Thêm mapping `46_to_27` vào `selected_joints` ✓
- [ ] Cập nhật docstring trong `generate_multivsl200_splits.py` ✓
- [ ] Cập nhật help text `--config` argument ✓
- [ ] Test 46_all config:
  ```bash
  python generate_multivsl200_splits.py --config 46_all
  # Output: (4515, 3, 150, 46, 1), (791, 3, 150, 46, 1), (593, 3, 150, 46, 1)
  ```
- [ ] Test 46_to_27 config:
  ```bash
  python generate_multivsl200_splits.py --config 46_to_27
  # Output: (4515, 3, 150, 27, 1), (791, 3, 150, 27, 1), (593, 3, 150, 27, 1)
  ```
- [ ] Generate multi-stream data:
  ```bash
  python generate_multivsl200_4streams.py --data_dir ../data/MultiVSL200
  # Sinh 4 streams × 3 splits = 12 files
  ```

---

## Tham khảo

**MultiVSL200 46-joint structure** (từ vsl_graph.py):

- Indices 0-9: Body + Nose
- Indices 10-21: L-Hand (12 joints)
- Indices 22-33: R-Hand (12 joints)
- Indices 34-45: (TBD)

**HA-SLR 27-joint structure** (từ sign_27_cvpr.py):

- Indices 0-6: Body (Nose, Shoulders, Elbows, Wrists)
- Indices 7-16: L-Hand (10 joints)
- Indices 17-26: R-Hand (10 joints)

---

## Troubleshooting

### Q: Khi sinh bone data với 46_to_27, tại sao bone[3] (L-Elbow) có giá trị?

**A**: Vì mapping sẽ copy index 43 (L-Shoulder từ 46-joint space) vào index 1 (L-Shoulder của 27-joint space). Khi tính bone, `bone[3] = joint[3] - joint[5]` sẽ là `shoulder_value - wrist_value`, từ đó tạo ra valid bone vector.

### Q: Dummy elbow có ảnh hưởng đến độ chính xác mô hình không?

**A**:

- **Nếu mô hình dùng elbow tọa độ trực tiếp**: Dummy values sẽ gây lỗi (không có elbow position)
- **Nếu mô hình chỉ dùng bone vectors (khuyến khích)**: Dummy values vẫn tạo được valid bone vectors (shoulder→wrist), ít ảnh hưởng
- **Khuyên cáo**: Kiểm tra xem HA-SLR graph có dựa trực tiếp vào absolute elbow coordinates không

### Q: Nên dùng 46_all hay 46_to_27?

**A**:

- **46_all**: Nếu muốn giữ toàn bộ thông tin MultiVSL200, train mô hình 46-joint riêng
- **46_to_27**: Nếu muốn dùng HA-SLR pre-trained models hoặc so sánh fair with AUTSL/INCLUDE

---

**Last Updated**: 2026-04-30
