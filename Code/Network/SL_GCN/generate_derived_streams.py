import os
import numpy as np
from tqdm import tqdm
import glob

def generate_derived_streams(data_dir):
    """
    Sinh các luồng dữ liệu Động học (Bone, Joint Motion, Bone Motion) từ Joint data.
    Tự động quét và xử lý tất cả các tập (train, val, test) có trong data_dir.
    """
    joint_files = glob.glob(os.path.join(data_dir, "*_data_joint.npy"))
    
    if not joint_files:
        print(f"Lỗi: Không tìm thấy file *_data_joint.npy nào trong thư mục {data_dir}")
        return

    for joint_file in joint_files:
        # Tách tên tập (train, val, hoặc test)
        split = os.path.basename(joint_file).replace('_data_joint.npy', '')
        print(f"\n=====================================")
        print(f"Đang xử lý tập dữ liệu: {split.upper()}")
        print(f"=====================================")
        
        print(f"Đang tải {joint_file} ...")
        joint_data = np.load(joint_file)
        N, C, T, V, M = joint_data.shape
        
        # 1. Định nghĩa kết nối xương (Bone connections) cho 27 keypoints
        inward_ori_index = [
            (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (9, 11),
            (12, 13), (12, 14), (12, 16), (12, 18), (12, 20),
            (14, 15), (16, 17), (18, 19), (20, 21),
            (22, 23), (22, 24), (22, 26), (22, 28), (22, 30),
            (24, 25), (26, 27), (28, 29), (30, 31),
            (10, 12), (11, 22)
        ]
        
        bone_conn = np.zeros(V, dtype=int)
        for (i, j) in inward_ori_index:
            bone_conn[j - 5] = i - 5
        bone_conn[0] = 0

        print("Đang tạo luồng Bone...")
        # Chuẩn hóa độ dài (L2 Normalization) dọc theo trục tọa độ (C=3, axis 1) để biến thành Vector Đơn Vị Chỉ Hướng Xương
        bone_data = joint_data - joint_data[:, :, :, bone_conn, :]
        bone_data = bone_data / (np.linalg.norm(bone_data, axis=1, keepdims=True) + 1e-6)
            
        print("Đang tạo luồng Joint Motion...")
        joint_motion_data = np.zeros_like(joint_data)
        joint_motion_data[:, :, :-1, :, :] = joint_data[:, :, 1:, :, :] - joint_data[:, :, :-1, :, :]
        # Cơ chế sao chép biên (Border Replication) để bảo toàn dòng chảy thời gian
        joint_motion_data[:, :, -1, :, :] = joint_motion_data[:, :, -2, :, :]

        print("Đang tạo luồng Bone Motion...")
        bone_motion_data = np.zeros_like(bone_data)
        bone_motion_data[:, :, :-1, :, :] = bone_data[:, :, 1:, :, :] - bone_data[:, :, :-1, :, :]
        # Cơ chế sao chép biên (Border Replication) để bảo toàn dòng chảy thời gian
        bone_motion_data[:, :, -1, :, :] = bone_motion_data[:, :, -2, :, :]

        # 2. Lưu các luồng dữ liệu mới
        bone_file = os.path.join(data_dir, f'{split}_data_bone.npy')
        joint_motion_file = os.path.join(data_dir, f'{split}_data_joint_motion.npy')
        bone_motion_file = os.path.join(data_dir, f'{split}_data_bone_motion.npy')

        print(f"Đang lưu: {bone_file}")
        np.save(bone_file, bone_data)
        print(f"Đang lưu: {joint_motion_file}")
        np.save(joint_motion_file, joint_motion_data)
        print(f"Đang lưu: {bone_motion_file}")
        np.save(bone_motion_file, bone_motion_data)
        
        print(f"Hoàn thành xử lý cho tập {split.upper()}!\n")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/MultiVSL200/', help='Thư mục chứa file *_data_joint.npy')
    args = parser.parse_args()
    
    data_directory = args.data_dir
    
    if not os.path.exists(data_directory):
        print(f"Lỗi: Thư mục {data_directory} không tồn tại!")
        print("Vui lòng truyền đúng đường dẫn bằng lệnh: python generate_derived_streams.py --data_dir đường/dẫn/đến/thư/mục")
    else:
        print(f"Thư mục dữ liệu đang dùng: {data_directory}")
        generate_derived_streams(data_directory)
        print("Quá trình trích xuất Offline Kinematic Streams cho tất cả các tập đã HOÀN TẤT.")
