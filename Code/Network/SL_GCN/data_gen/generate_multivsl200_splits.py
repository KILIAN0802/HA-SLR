#!/usr/bin/env python3
"""
Generate MultiVSL200 train/val/test splits (.pkl + .npy) from CSV labels.
This script:
  1. Reads *_labels.csv files from MultiVSL200/splits/
  2. Loads corresponding .npy skeleton files
  3. Extracts selected joints and pads/truncates to 150 frames
  4. Writes *_label.pkl and *_data_joint.npy for each split

Usage:
  # Generate 46-joint version (MultiVSL200 native)
    python generate_multivsl200_splits.py --data_dir ../data/MultiVSL200 --split_dir ../data/MultiVSL200/splits --out_dir ../data/MultiVSL200 --config 46_all

  # Generate 27-joint version (compatible with HA-SLR models, with dummy elbow values)
    python generate_multivsl200_splits.py --data_dir ../data/MultiVSL200 --split_dir ../data/MultiVSL200/splits --out_dir ../data/MultiVSL200_27 --config 46_to_27

Note: 
  - MultiVSL200 .npy files have 46 joints (already pre-selected).
  - Use config='46_all' to generate all 46 joints (default)
  - Use config='46_to_27' to generate 27 joints for HA-SLR compatibility
    (Elbow points are dummy values copied from Shoulder due to MultiVSL200 structure)
  - For other datasets with 133 joints, use config='27_cvpr'.
  - Ensure MultiVSL200/splits/ contains train_labels.csv, val_labels.csv, test_labels.csv
    and raw .npy files are in the MultiVSL200/ directory.
"""

import argparse
import os
import pickle
import sys
import numpy as np
from tqdm import tqdm

ROOT = os.path.dirname(__file__)
DEFAULT_DATA_DIR = os.path.normpath(os.path.join(ROOT, '..', 'data', 'MultiVSL200'))
DEFAULT_SPLIT_DIR = os.path.join(DEFAULT_DATA_DIR, 'splits')

# Joint selection configs (same as sign_gendata.py)
selected_joints = {
    '27_cvpr': np.concatenate(([0, 3, 4, 5, 6, 7, 8],
                               [91, 95, 96, 99, 100, 103, 104, 107, 108, 111],
                               [112, 116, 117, 120, 121, 124, 125, 128, 129, 132]), axis=0),
    # MultiVSL200 has only 46 joints (already selected skeleton points)
    '46_all': np.arange(46),  # Use all 46 joints
    
    # Mapping từ 46 điểm MultiVSL200 sang 27 điểm HA-SLR
    # HA-SLR 27-point layout: 0:Nose, 1:L-Sho, 2:R-Sho, 3:L-Elb(dummy), 4:R-Elb(dummy), 
    #                         5:L-Wri, 6:R-Wri, 7-16:L-Hand, 17-26:R-Hand
    # MultiVSL200 46-point layout (estimates from original skeleton):
    #   0-1: Nose, 2-9: Body, 10-21: L-Hand(12 joints), 22-33: R-Hand(12 joints), 34-45: ???
    # 
    # Chiến lược: Sử dụng các điểm có sẵn trong 46-joint format, với dummy values cho khuỷu tay
    # (Elbow được lấy từ Shoulder vì MultiVSL200 thiếu điểm này)
    '46_to_27': np.array([
        42, 43, 44,          # 0-2: Nose, L-Shoulder, R-Shoulder
        43, 44,              # 3-4: L-Elbow(dummy=L-Sho), R-Elbow(dummy=R-Sho)
        0, 21,               # 5-6: L-Wrist, R-Wrist (từ hand indices)
        0, 4, 5, 8, 9, 12, 13, 16, 17, 20,      # 7-16: L-Hand 10 joints
        21, 25, 26, 29, 30, 33, 34, 37, 38, 41  # 17-26: R-Hand 10 joints
    ], dtype=int),
}

MAX_BODY = 1
MAX_FRAME = 150
NUM_CHANNELS = 3


def gendata(data_dir, label_csv, out_dir, part='train', config='27_cvpr'):
    """
    Generate dataset .pkl and .npy files from CSV label file.
    
    Args:
        data_dir: directory containing .npy skeleton files
        label_csv: path to CSV with format "sample_name,label_id"
        out_dir: output directory for .pkl and .npy
        part: 'train', 'val', or 'test'
        config: joint config name (e.g., '27_cvpr')
    """
    if config not in selected_joints:
        raise ValueError(f"Unknown config '{config}'. Available: {list(selected_joints.keys())}")
    
    selected = selected_joints[config]
    num_joints = len(selected)
    
    # Read CSV
    sample_names = []
    labels = []
    data_paths = []
    
    print(f"Reading {label_csv}...")
    with open(label_csv, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            sample_name = parts[0]
            # Label can be text (signer ID) or number; keep as-is
            label_str = parts[1].strip() if len(parts) > 1 else '0'
            # Try to convert to int, otherwise keep as string
            try:
                label = int(label_str)
            except ValueError:
                label = label_str
            
            sample_names.append(sample_name)
            labels.append(label)
            data_paths.append(os.path.join(data_dir, sample_name + '.npy'))
    
    print(f"  Found {len(sample_names)} samples")
    
    # Initialize data array
    fp = np.zeros((len(data_paths), MAX_FRAME, num_joints, NUM_CHANNELS, MAX_BODY), dtype=np.float32)
    print(f"  Allocating array shape: {fp.shape}")
    
    # Load and process each sample
    skipped = 0
    for i, data_path in enumerate(tqdm(data_paths, desc=f"Processing {part} samples")):
        if not os.path.exists(data_path):
            print(f"  WARNING: Missing {data_path}")
            skipped += 1
            continue
        
        # Load skeleton (shape: [num_frames, 133, 3])
        skel = np.load(data_path)
        
        # Select joints
        skel = skel[:, selected, :]  # [num_frames, num_joints, 3]
        
        # Pad or truncate to MAX_FRAME
        if skel.shape[0] < MAX_FRAME:
            L = skel.shape[0]
            fp[i, :L, :, :, 0] = skel
            
            # Pad by repeating
            rest = MAX_FRAME - L
            num_repeats = int(np.ceil(rest / L))
            pad = np.concatenate([skel for _ in range(num_repeats)], axis=0)[:rest]
            fp[i, L:, :, :, 0] = pad
        else:
            # Truncate to first MAX_FRAME frames
            fp[i, :, :, :, 0] = skel[:MAX_FRAME, :, :]
    
    if skipped > 0:
        print(f"  Skipped {skipped} missing files")
    
    # Save label pickle
    label_pkl = os.path.join(out_dir, f'{part}_label.pkl')
    with open(label_pkl, 'wb') as f:
        pickle.dump((sample_names, labels), f)
    print(f"  Saved {label_pkl}")
    
    # Transpose to [N, C, T, J, B] and save .npy
    fp = np.transpose(fp, [0, 3, 1, 2, 4])
    print(f"  Final array shape: {fp.shape}")
    
    data_npy = os.path.join(out_dir, f'{part}_data_joint.npy')
    np.save(data_npy, fp)
    print(f"  Saved {data_npy}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate MultiVSL200 train/val/test splits.'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default=DEFAULT_DATA_DIR,
        help='Directory containing raw .npy skeleton files'
    )
    parser.add_argument(
        '--split_dir',
        type=str,
        default=DEFAULT_SPLIT_DIR,
        help='Directory containing train/val/test_labels.csv'
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default=DEFAULT_DATA_DIR,
        help='Output directory for .pkl and .npy files'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='46_all',
        help='Joint config to use: 46_all (all MultiVSL200 joints), 46_to_27 (27-joint HA-SLR format with dummy elbows), or 27_cvpr (OpenPose 133-joint subset)'
    )
    
    args = parser.parse_args()
    
    # Ensure directories exist
    if not os.path.isdir(args.data_dir):
        print(f"ERROR: data_dir '{args.data_dir}' not found")
        sys.exit(1)
    
    if not os.path.isdir(args.split_dir):
        print(f"ERROR: split_dir '{args.split_dir}' not found")
        sys.exit(1)
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Process each split
    for part in ['train', 'val', 'test']:
        label_csv = os.path.join(args.split_dir, f'{part}_labels.csv')
        if not os.path.isfile(label_csv):
            print(f"ERROR: {label_csv} not found")
            sys.exit(1)
        
        print(f"\n=== Processing {part.upper()} ===")
        gendata(args.data_dir, label_csv, args.out_dir, part=part, config=args.config)
    
    print("\n✓ All splits generated successfully!")


if __name__ == '__main__':
    main()
