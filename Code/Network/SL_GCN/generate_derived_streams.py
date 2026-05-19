import os
import numpy as np
from tqdm import tqdm

def generate_derived_streams(data_dir, split="train"):
    """
    Sinh các luồng dữ liệu Động học (Bone, Joint Motion, Bone Motion) từ Joint data.
    data_dir: Thư mục chứa file npy gốc.
    split: 'train' hoặc 'test'.
    """
    joint_file = os.path.join(data_dir, f'{split}_data_joint.npy')
    
    if not os.path.exists(joint_file):
        print(f"Error: Not found {joint_file}")
        return

    print(f"Loading {joint_file} ...")
    # Tải dữ liệu joint (N, C, T, V, M)
    # Ví dụ: N mẫu, C=3 (x,y,z), T khung hình, V=27 khớp, M=1 người
    joint_data = np.load(joint_file)
    N, C, T, V, M = joint_data.shape
    
    # 1. Định nghĩa kết nối xương (Bone connections) cho 27 keypoints
    # Dựa theo inward_ori_index của file graph/sign_27.py (trừ đi 5 để về index 0-26)
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
    
    # Root node (nose) point to itself
    bone_conn[0] = 0

    print("Generating Bone Stream...")
    bone_data = np.zeros_like(joint_data)
    for v in tqdm(range(V), desc="Bone"):
        u = bone_conn[v]
        bone_data[:, :, :, v, :] = joint_data[:, :, :, v, :] - joint_data[:, :, :, u, :]
        
    print("Generating Joint Motion Stream...")
    joint_motion_data = np.zeros_like(joint_data)
    for t in tqdm(range(T - 1), desc="Joint Motion"):
        joint_motion_data[:, :, t, :, :] = joint_data[:, :, t + 1, :, :] - joint_data[:, :, t, :, :]
    # Frame cuối cùng bằng 0
    joint_motion_data[:, :, T - 1, :, :] = 0

    print("Generating Bone Motion Stream...")
    bone_motion_data = np.zeros_like(bone_data)
    for t in tqdm(range(T - 1), desc="Bone Motion"):
        bone_motion_data[:, :, t, :, :] = bone_data[:, :, t + 1, :, :] - bone_data[:, :, t, :, :]
    # Frame cuối cùng bằng 0
    bone_motion_data[:, :, T - 1, :, :] = 0

    # 2. Lưu các luồng dữ liệu mới
    bone_file = os.path.join(data_dir, f'{split}_data_bone.npy')
    joint_motion_file = os.path.join(data_dir, f'{split}_data_joint_motion.npy')
    bone_motion_file = os.path.join(data_dir, f'{split}_data_bone_motion.npy')

    print(f"Saving to {bone_file}")
    np.save(bone_file, bone_data)
    
    print(f"Saving to {joint_motion_file}")
    np.save(joint_motion_file, joint_motion_data)
    
    print(f"Saving to {bone_motion_file}")
    np.save(bone_motion_file, bone_motion_data)
    
    print(f"Done for {split} split!\n")

if __name__ == '__main__':
    data_directory = "data/MultiVSL200/"
    
    # Đảm bảo đường dẫn hiện tại là thư mục Code/Network/SL_GCN
    if not os.path.exists(data_directory):
        print(f"Vui lòng chạy script tại thư mục chứa {data_directory}")
    else:
        generate_derived_streams(data_directory, split="train")
        generate_derived_streams(data_directory, split="test")
        print("Quá trình trích xuất Offline Kinematic Streams hoàn tất.")
