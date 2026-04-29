#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MultiVSL200 Multi-Stream Data Generation (4 streams)
Tạo 4 luồng dữ liệu từ Joint data:
  1. Joint (đã có từ generate_multivsl200_splits.py)
  2. Bone (calculated from Joint)
  3. Joint Motion (temporal diff of Joint)
  4. Bone Motion (temporal diff of Bone)

Supports 2 configs:
  - 46_all: Use all 46 joints from MultiVSL200 (default)
  - 46_to_27: Use 27-joint HA-SLR format with dummy elbows (from 46→27 mapping)

Usage:
  # Generate 4 streams for 46-joint version
  python generate_multivsl200_4streams.py \\
    --data_dir ../data/MultiVSL200 \\
    --config 46_all

  # Generate 4 streams for 27-joint version (HA-SLR compatible)
  python generate_multivsl200_4streams.py \\
    --data_dir ../data/MultiVSL200 \\
    --config 46_to_27

Output (46_all):
  MultiVSL200/
    ├── *_data_joint.npy           (N, 3, 150, 46, 1) [existing]
    ├── *_data_bone.npy            ← NEW
    ├── *_data_joint_motion.npy    ← NEW
    ├── *_data_bone_motion.npy     ← NEW

Output (46_to_27):
  MultiVSL200/
    ├── *_data_joint.npy           (N, 3, 150, 27, 1) [from 46_to_27 config, with dummy elbows]
    ├── *_data_bone.npy            ← NEW
    ├── *_data_joint_motion.npy    ← NEW
    ├── *_data_bone_motion.npy     ← NEW
"""

import os
import sys
import argparse
import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm


# Joint pairs for 46-point skeleton (MultiVSL200)
# These define which joints form bones (edges in the skeleton graph)
BONE_PAIRS_46 = [
    # Body connections (estimated based on common skeleton structure)
    # Nose to shoulders
    (0, 1), (0, 2),
    # Shoulders to elbows
    (1, 3), (3, 5), (2, 4), (4, 6),
    # Elbows to wrists
    (5, 7), (7, 9), (6, 8), (8, 10),
    # Shoulders to hips
    (1, 11), (2, 12),
    # Hips to knees
    (11, 13), (13, 15), (12, 14), (14, 16),
    # Knees to ankles
    (15, 17), (17, 19), (16, 18), (18, 20),
    # Spine connections (if available)
    (1, 2), (11, 12),
]

# For 27-point HA-SLR skeleton with dummy elbows from MultiVSL200 46→27 mapping
# Note: Indices 3, 4 (elbows) are dummy values from shoulders (indices 1, 2)
BONE_PAIRS_27 = [
    # Body chain
    (5, 6), (5, 7),                    # Shoulder connections
    (6, 8), (8, 10), (7, 9), (9, 11), # Arm chain (note: 3,4 are dummy elbows)
    
    # Left hand
    (12, 13), (12, 14), (12, 16), (12, 18), (12, 20),
    (14, 15), (16, 17), (18, 19), (20, 21),
    
    # Right hand
    (22, 23), (22, 24), (22, 26), (22, 28), (22, 30),
    (24, 25), (26, 27), (28, 29), (30, 31),
    
    # Wrist to hand root
    (10, 12), (11, 22),
]

# For reference: Full 27-point sign language skeleton (sign_27_cvpr)
_BONE_PAIRS_27_CVPR = [
    (5, 6), (5, 7),
    (6, 8), (8, 10), (7, 9), (9, 11),
    (12, 13), (12, 14), (12, 16), (12, 18), (12, 20),
    (14, 15), (16, 17), (18, 19), (20, 21),
    (22, 23), (22, 24), (22, 26), (22, 28), (22, 30),
    (24, 25), (26, 27), (28, 29), (30, 31),
    (10, 12), (11, 22)
]


class MultiStream46Generator:
    """Generate 4-stream data for 46-point or 27-point skeleton"""
    
    def __init__(self, data_dir, config='46_all', num_joints=None, bone_pairs=None):
        """
        Initialize multi-stream generator
        
        Args:
            data_dir: Directory containing *_data_joint.npy files
            config: '46_all' (default) or '46_to_27'
            num_joints: Override number of joints (auto-detected from config if None)
            bone_pairs: Override bone pairs (auto-selected from config if None)
        """
        self.data_dir = data_dir
        self.config = config
        self.splits = ['train', 'val', 'test']
        
        # Auto-configure based on config name
        if config == '46_all':
            self.num_joints = num_joints or 46
            self.bone_pairs = bone_pairs or BONE_PAIRS_46
            self.config_desc = "46-joint full skeleton"
        elif config == '46_to_27':
            self.num_joints = num_joints or 27
            self.bone_pairs = bone_pairs or BONE_PAIRS_27
            self.config_desc = "27-joint HA-SLR format (with dummy elbows)"
        else:
            raise ValueError(f"Unknown config: {config}. Use '46_all' or '46_to_27'")
    
    
    def generate_bone_data(self, joint_data, output_path):
        """
        Calculate bone data from joint data
        Bone = Joint[v2] - Joint[v1]
        
        Args:
            joint_data: (N, C, T, V, M) shaped array
            output_path: path to save bone data
        
        Returns:
            bone_data: (N, C, T, V, M) shaped array
        """
        N, C, T, V, M = joint_data.shape
        
        # Create output memmap
        bone_fp = open_memmap(
            output_path,
            dtype='float32',
            mode='w+',
            shape=(N, C, T, V, M)
        )
        
        # Initialize with joint data
        bone_fp[:, :, :, :, :] = joint_data
        
        # Calculate bones: difference between connected joints
        print(f"  Calculating bones from {len(self.bone_pairs)} pairs...")
        for v1, v2 in tqdm(self.bone_pairs, desc="Bone pairs"):
            # bone[v2] = joint[v2] - joint[v1]
            bone_fp[:, :, :, v2, :] = joint_data[:, :, :, v2, :] - joint_data[:, :, :, v1, :]
        
        bone_fp.flush()
        del bone_fp
        print(f"  ✓ Saved: {output_path}")
        return np.load(output_path)
    
    def generate_motion_data(self, data, output_path, data_type='joint'):
        """
        Calculate motion data (temporal difference)
        Motion[t] = Data[t+1] - Data[t]
        
        Args:
            data: (N, C, T, V, M) shaped array
            output_path: path to save motion data
            data_type: 'joint' or 'bone' (for logging)
        
        Returns:
            motion_data: (N, C, T, V, M) shaped array
        """
        N, C, T, V, M = data.shape
        
        # Create output memmap
        motion_fp = open_memmap(
            output_path,
            dtype='float32',
            mode='w+',
            shape=(N, C, T, V, M)
        )
        
        # Calculate temporal differences
        print(f"  Calculating {data_type} motion (temporal diff)...")
        for t in tqdm(range(T - 1), desc="Frames"):
            motion_fp[:, :, t, :, :] = data[:, :, t + 1, :, :] - data[:, :, t, :, :]
        
        # Last frame: zero motion
        motion_fp[:, :, T - 1, :, :] = 0
        
        motion_fp.flush()
        del motion_fp
        print(f"  ✓ Saved: {output_path}")
        return np.load(output_path)
    
    def process_split(self, split_name):
        """Process one train/val/test split to generate all 4 streams"""
        print(f"\n{'='*70}")
        print(f"Processing {split_name.upper()} split")
        print(f"{'='*70}")
        
        # Paths
        joint_path = os.path.join(self.data_dir, f'{split_name}_data_joint.npy')
        bone_path = os.path.join(self.data_dir, f'{split_name}_data_bone.npy')
        joint_motion_path = os.path.join(self.data_dir, f'{split_name}_data_joint_motion.npy')
        bone_motion_path = os.path.join(self.data_dir, f'{split_name}_data_bone_motion.npy')
        
        # 1. Load joint data
        print(f"[1/4] Loading joint data: {joint_path}")
        if not os.path.exists(joint_path):
            print(f"  ✗ File not found: {joint_path}")
            return False
        
        joint_data = np.load(joint_path)
        print(f"  Shape: {joint_data.shape}")
        
        # 2. Generate bone data
        print(f"\n[2/4] Generating bone data: {bone_path}")
        if os.path.exists(bone_path):
            print(f"  ⚠ File already exists, skipping...")
            bone_data = np.load(bone_path)
        else:
            bone_data = self.generate_bone_data(joint_data, bone_path)
        
        # 3. Generate joint motion data
        print(f"\n[3/4] Generating joint motion data: {joint_motion_path}")
        if os.path.exists(joint_motion_path):
            print(f"  ⚠ File already exists, skipping...")
        else:
            self.generate_motion_data(joint_data, joint_motion_path, data_type='joint')
        
        # 4. Generate bone motion data
        print(f"\n[4/4] Generating bone motion data: {bone_motion_path}")
        if os.path.exists(bone_motion_path):
            print(f"  ⚠ File already exists, skipping...")
        else:
            self.generate_motion_data(bone_data, bone_motion_path, data_type='bone')
        
        print(f"\n✓ {split_name.upper()} split complete!")
        return True
    
    def run(self):
        """Generate all 4 streams for all splits"""
        print("\n" + "="*70)
        print("MultiVSL200 Multi-Stream Data Generator (4 streams)")
        print("="*70)
        print(f"Data directory: {self.data_dir}")
        print(f"Config: {self.config} ({self.config_desc})")
        print(f"Number of joints: {self.num_joints}")
        print(f"Number of bone pairs: {len(self.bone_pairs)}")
        
        success_count = 0
        for split in self.splits:
            if self.process_split(split):
                success_count += 1
        
        print("\n" + "="*70)
        print(f"✓ COMPLETE: {success_count}/{len(self.splits)} splits processed")
        print("="*70)
        print("\nGenerated 4-stream files:")
        print("  Stream 1: *_data_joint.npy")
        print("  Stream 2: *_data_bone.npy")
        print("  Stream 3: *_data_joint_motion.npy")
        print("  Stream 4: *_data_bone_motion.npy")
        print("\nReady for multi-stream training!")
        print(f"Use config --config {self.config} in training script.")


def main():
    parser = argparse.ArgumentParser(
        description='Generate 4-stream data for MultiVSL200 (Joint, Bone, Joint Motion, Bone Motion)'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='../data/MultiVSL200',
        help='Directory containing *_data_joint.npy files'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='46_all',
        choices=['46_all', '46_to_27'],
        help='Configuration: 46_all (default, all 46 joints) or 46_to_27 (27-joint HA-SLR format with dummy elbows)'
    )
    parser.add_argument(
        '--num_joints',
        type=int,
        default=None,
        help='Number of joints in skeleton (auto-detected from config if not specified)'
    )
    
    args = parser.parse_args()
    
    # Validate data directory
    if not os.path.exists(args.data_dir):
        print(f"✗ Error: Directory not found: {args.data_dir}")
        sys.exit(1)
    
    # Generate all 4 streams
    generator = MultiStream46Generator(args.data_dir, config=args.config, num_joints=args.num_joints)
    generator.run()


if __name__ == '__main__':
    main()
