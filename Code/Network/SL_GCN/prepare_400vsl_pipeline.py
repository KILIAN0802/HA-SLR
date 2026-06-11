import os
import shutil
import pandas as pd
import subprocess
from tqdm import tqdm

# Cấu hình đường dẫn
ROOT_DIR = "/mnt/nvme2/users/utbt_sv1/HASLR/HA-SLR/Code/Network/SL_GCN"
DATA_ROOT = os.path.join(ROOT_DIR, "data/data/400VSL")
VIDEO_DIR = os.path.join(DATA_ROOT, "400VSLcropped")
RAW_NPY_DIR = os.path.join(DATA_ROOT, "raw_npy")
SPLIT_DIR = os.path.join(DATA_ROOT, "splits")
PROCESSED_DIR = os.path.join(DATA_ROOT, "processed/27_direct")

def run_command(cmd, cwd=ROOT_DIR):
    print(f"Running: {cmd}")
    process = subprocess.Popen(cmd, shell=True, cwd=cwd)
    process.wait()

def prepare_dirs():
    for d in [RAW_NPY_DIR, SPLIT_DIR, PROCESSED_DIR]:
        os.makedirs(d, exist_ok=True)
    
    # Tạo các thư mục con cho npy
    for part in ['train', 'val', 'test']:
        os.makedirs(os.path.join(DATA_ROOT, f"{part}_npy"), exist_ok=True)

def extract_skeletons():
    print("--- Bước 1: Trích xuất tọa độ khớp từ Video ---")
    # Sử dụng demo.py để trích xuất 27 điểm khớp
    cmd = f"python Preprocess/demo.py --input_dir {VIDEO_DIR} --output_dir {RAW_NPY_DIR}"
    run_command(cmd)

def split_npy_files():
    print("--- Bước 2: Chia file npy vào các thư mục train/val/test ---")
    csv_files = {
        'train': os.path.join(VIDEO_DIR, 'train.csv'),
        'val': os.path.join(VIDEO_DIR, 'val.csv'),
        'test': os.path.join(VIDEO_DIR, 'test.csv')
    }
    
    for part, csv_path in csv_files.items():
        print(f"Processing {part} set...")
        df = pd.read_csv(csv_path)
        # Format: sample_id,label_id
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            video_id = row['sample_id']
            # Chuyển thành string và format đủ 6 chữ số nếu cần
            video_id_str = f"{int(video_id):06d}" 
            src = os.path.join(RAW_NPY_DIR, f"{video_id_str}.npy")
            dst = os.path.join(DATA_ROOT, f"{part}_npy", f"{video_id}.npy")
            if os.path.exists(src):
                if not os.path.exists(dst):
                    os.symlink(src, dst)
            else:
                print(f"Warning: {src} not found!")

def prepare_csv_labels():
    print("--- Bước 3: Chuẩn bị file CSV nhãn cho sign_gendata.py ---")
    csv_files = {
        'train': os.path.join(VIDEO_DIR, 'train.csv'),
        'val': os.path.join(VIDEO_DIR, 'val.csv'),
        'test': os.path.join(VIDEO_DIR, 'test.csv')
    }
    
    for part, csv_path in csv_files.items():
        dst_csv = os.path.join(SPLIT_DIR, f"{part}_labels.csv")
        df = pd.read_csv(csv_path)
        # Chuyển sample_id thành string 6 chữ số để khớp với file npy
        df['sample_id'] = df['sample_id'].apply(lambda x: f"{int(x):06d}")
        # Lưu lại không có header để sign_gendata.py đọc được
        df.to_csv(dst_csv, header=False, index=False)

def generate_joint_data():
    print("--- Bước 4: Tạo dữ liệu Joint (Tổng hợp) ---")
    cmd = (f"python data_gen/sign_gendata.py "
           f"--data_path {DATA_ROOT} "
           f"--label_dir {SPLIT_DIR} "
           f"--out_folder {os.path.join(DATA_ROOT, 'processed')} "
           f"--train_dir train_npy --val_dir val_npy --test_dir test_npy")
    run_command(cmd)

def generate_derived_streams():
    print("--- Bước 5: Tạo dữ liệu Bone và Motion ---")
    parts = ['train', 'val', 'test']
    
    for part in parts:
        joint_file = os.path.join(PROCESSED_DIR, f"{part}_data_joint.npy")
        bone_file = os.path.join(PROCESSED_DIR, f"{part}_data_bone.npy")
        
        # 1. Bone
        cmd_bone = f"python data_gen/gen_bone_data.py --data_path {joint_file} --out_path {bone_file} --tag sign/27"
        run_command(cmd_bone)
        
        # 2. Joint Motion
        joint_motion_file = os.path.join(PROCESSED_DIR, f"{part}_data_joint_motion.npy")
        cmd_jm = f"python data_gen/gen_motion_data.py --data_path {joint_file} --out_path {joint_motion_file}"
        run_command(cmd_jm)
        
        # 3. Bone Motion
        bone_motion_file = os.path.join(PROCESSED_DIR, f"{part}_data_bone_motion.npy")
        cmd_bm = f"python data_gen/gen_motion_data.py --data_path {bone_file} --out_path {bone_motion_file}"
        run_command(cmd_bm)

if __name__ == "__main__":
    prepare_dirs()
    # extract_skeletons() # User should run this manually first or I can keep it
    split_npy_files()
    prepare_csv_labels()
    generate_joint_data()
    generate_derived_streams()
    print("\nHOÀN THÀNH TIỀN XỬ LÝ DỮ LIỆU 400VSL!")
