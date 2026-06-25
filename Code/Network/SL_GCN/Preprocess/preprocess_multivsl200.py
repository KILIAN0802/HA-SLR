import os
import pickle
import numpy as np
from tqdm import tqdm

def main():
    raw_npy_dir = "/mnt/nvme2/users/utbt_sv1/data/MultiVSL200/raw_npy"
    raw_label_csv = "/mnt/nvme2/users/utbt_sv1/data/MultiVSL200/raw_npy/labels.csv"
    output_base_dir = "data/data/MultiVSL200"

    splits_dir = os.path.join(output_base_dir, "splits")
    train_npy_dir = os.path.join(output_base_dir, "train_npy")
    val_npy_dir = os.path.join(output_base_dir, "val_npy")
    test_npy_dir = os.path.join(output_base_dir, "test_npy")
    processed_dir = os.path.join(output_base_dir, "processed/27_direct")

    # Create directories if they don't exist
    for d in [splits_dir, train_npy_dir, val_npy_dir, test_npy_dir, processed_dir]:
        os.makedirs(d, exist_ok=True)

    print("Step 1: Reading and splitting labels...")
    with open(raw_label_csv, "r", encoding="utf-8") as f:
        lines = f.readlines()

    train_lines = []
    val_lines = []
    test_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        filename = parts[0]
        try:
            signer_id = int(filename[:2])
            if 1 <= signer_id <= 24:
                train_lines.append(line)
            elif 25 <= signer_id <= 27:
                val_lines.append(line)
            elif 28 <= signer_id <= 31:
                test_lines.append(line)
            else:
                train_lines.append(line)
        except Exception:
            train_lines.append(line)

    # Write split label files
    with open(os.path.join(splits_dir, "train_labels.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(train_lines) + "\n")
    with open(os.path.join(splits_dir, "val_labels.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(val_lines) + "\n")
    with open(os.path.join(splits_dir, "test_labels.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(test_lines) + "\n")

    print(f"Split done: {len(train_lines)} train, {len(val_lines)} val, {len(test_lines)} test")

    print("\nStep 2: Creating symlinks for raw npy files...")
    splits_mapping = [
        ("train", train_lines, train_npy_dir),
        ("val", val_lines, val_npy_dir),
        ("test", test_lines, test_npy_dir)
    ]

    for part_name, lines, dest_dir in splits_mapping:
        print(f"Creating symlinks for {part_name}...")
        for line in tqdm(lines):
            filename = line.split(",")[0]
            # Check potential file paths
            potential_files = [filename + ".npy", filename + "_color.mp4.npy"]
            found_src = None
            found_name = None
            for pf in potential_files:
                p_path = os.path.join(raw_npy_dir, pf)
                if os.path.exists(p_path):
                    found_src = p_path
                    found_name = pf
                    break
            
            if found_src is None:
                print(f"Warning: File {filename} not found in {raw_npy_dir}!")
                continue

            dst_path = os.path.join(dest_dir, found_name)
            if os.path.exists(dst_path) or os.path.islink(dst_path):
                os.remove(dst_path)
            os.symlink(found_src, dst_path)

    print("\nStep 3: Compiling Joint datasets and labels...")
    max_frame = 150
    num_joints = 27
    num_channels = 3
    max_body_true = 1

    for part_name, lines, data_dir in splits_mapping:
        print(f"Compiling {part_name} data...")
        data_paths = []
        labels = []
        sample_names = []

        for line in lines:
            filename = line.split(",")[0]
            label = int(line.split(",")[1])

            potential_files = [filename + ".npy", filename + "_color.mp4.npy"]
            for pf in potential_files:
                p_path = os.path.join(data_dir, pf)
                if os.path.exists(p_path):
                    data_paths.append(p_path)
                    labels.append(label)
                    sample_names.append(filename)
                    break

        num_samples = len(data_paths)
        fp = np.zeros((num_samples, max_frame, num_joints, num_channels, max_body_true), dtype=np.float32)

        for i, path in enumerate(tqdm(data_paths)):
            skel = np.load(path)
            # Shape is (150, 27, 3) or similar.
            # Just in case, crop or pad
            if skel.shape[0] < max_frame:
                L = skel.shape[0]
                fp[i, :L, :, :, 0] = skel
                rest = max_frame - L
                num = int(np.ceil(rest / L))
                pad = np.concatenate([skel for _ in range(num)], 0)[:rest]
                fp[i, L:, :, :, 0] = pad
            else:
                fp[i, :, :, :, 0] = skel[:max_frame, :, :]

        # Dump labels
        with open(os.path.join(processed_dir, f"{part_name}_label.pkl"), "wb") as f:
            pickle.dump((sample_names, labels), f)

        # Transpose to (N, C, T, V, M)
        fp = np.transpose(fp, [0, 3, 1, 2, 4])
        np.save(os.path.join(processed_dir, f"{part_name}_data_joint.npy"), fp)
        print(f"Compiled {part_name}_data_joint.npy with shape {fp.shape}")

    print("\nStep 4: Generating offline derived kinematic streams (Bone, Joint Motion, Bone Motion)...")
    joint_files = [
        ("train", os.path.join(processed_dir, "train_data_joint.npy")),
        ("val", os.path.join(processed_dir, "val_data_joint.npy")),
        ("test", os.path.join(processed_dir, "test_data_joint.npy"))
    ]

    inward_ori_index = [
        (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (9, 11),
        (12, 13), (12, 14), (12, 16), (12, 18), (12, 20),
        (14, 15), (16, 17), (18, 19), (20, 21),
        (22, 23), (22, 24), (22, 26), (22, 28), (22, 30),
        (24, 25), (26, 27), (28, 29), (30, 31),
        (10, 12), (11, 22)
    ]
    
    bone_conn = np.zeros(num_joints, dtype=int)
    for (i, j) in inward_ori_index:
        bone_conn[j - 5] = i - 5
    bone_conn[0] = 0

    for split, joint_file in joint_files:
        print(f"Processing streams for {split}...")
        joint_data = np.load(joint_file)
        N, C, T, V, M = joint_data.shape

        # 1. Bone data
        print("Generating Bone stream...")
        bone_data = joint_data - joint_data[:, :, :, bone_conn, :]
        norm = np.linalg.norm(bone_data, axis=1, keepdims=True)
        bone_data = np.where(norm > 1e-6, bone_data / norm, 0.0)

        # 2. Joint Motion data
        print("Generating Joint Motion stream...")
        joint_motion_data = np.zeros_like(joint_data)
        joint_motion_data[:, :, :-1, :, :] = joint_data[:, :, 1:, :, :] - joint_data[:, :, :-1, :, :]
        joint_motion_data[:, :, T - 1, :, :] = joint_motion_data[:, :, T - 2, :, :]

        # 3. Bone Motion data
        print("Generating Bone Motion stream...")
        bone_motion_data = np.zeros_like(bone_data)
        bone_motion_data[:, :, :-1, :, :] = bone_data[:, :, 1:, :, :] - bone_data[:, :, :-1, :, :]
        bone_motion_data[:, :, T - 1, :, :] = bone_motion_data[:, :, T - 2, :, :]

        # Save files
        np.save(os.path.join(processed_dir, f"{split}_data_bone.npy"), bone_data)
        np.save(os.path.join(processed_dir, f"{split}_data_joint_motion.npy"), joint_motion_data)
        np.save(os.path.join(processed_dir, f"{split}_data_bone_motion.npy"), bone_motion_data)
        print(f"Successfully generated derived streams for {split}!")

    print("\nAll preprocessing and data generation completed successfully!")

if __name__ == "__main__":
    main()
