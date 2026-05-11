import os
import argparse
import numpy as np
from numpy.lib.format import open_memmap
import pdb


paris = {
    'ntu/xview': (
        (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
        (13, 1),
        (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),
        (25, 12)
    ),
    'ntu/xsub': (
        (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
        (13, 1),
        (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),
        (25, 12)
    ),
    'ntu120/xsetup': (
        (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
        (13, 1),
        (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),
        (25, 12)
    ),
    'ntu120/xsub': (
        (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
        (13, 1),
        (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),
        (25, 12)
    ),
    'sign/27': ((5, 6), (5, 7),
                (6, 8), (8, 10), (7, 9), (9, 11), 
                (12,13),(12,14),(12,16),(12,18),(12,20),
                (14,15),(16,17),(18,19),(20,21),
                (22,23),(22,24),(22,26),(22,28),(22,30),
                (24,25),(26,27),(28,29),(30,31),
                (10,12),(11,22)
    ),

    # bone: (v1, v2)  骨头bone的两端关节点v1、v2  
    # 5-31 (-5) → 0-26 共计27个关节点
    # 'sign/27_2' the same as 'sign/27' 
    'sign/27_2': 
                # (鼻子，肩膀)
                ((5, 6), (5, 7),  

                # 肩膀 - 手肘 - 手腕
                (6, 8), (8, 10), (7, 9), (9, 11),     
                
                # 12-21 (-5) 7-16  左手
                (12,13),(12,14),(12,16),(12,18),(12,20),
                (14,15),(16,17),(18,19),(20,21),

                # 22-31 (-5) 17-26  左手
                (22,23),(22,24),(22,26),(22,28),(22,30),
                (24,25),(26,27),(28,29),(30,31),

                # (手腕, 手掌)
                (10,12),(11,22)
    ),

    'sign/27_cvpr': 
                # (鼻子，眼睛)
                ((5, 6), (5, 7),  

                # (鼻子，肩膀)
                (5, 8), (5, 9), 

                # 肩膀 - 手肘
                (8, 10), (9, 11),
                
                # 12-21 (-5) 7-16  左手
                (12,13),(12,14),(12,16),(12,18),(12,20),
                (14,15),(16,17),(18,19),(20,21),

                # 22-31 (-5) 17-26  右手
                (22,23),(22,24),(22,26),(22,28),(22,30),
                (24,25),(26,27),(28,29),(30,31),

                # (手肘, 手掌)
                (10,12),(11,22)
    ),

    'sign/hands': (     # bias :  12 (data/sign/hands/xxx_data_joint.npy)  或 5 (data/sign/27/xxx_data_joint.npy) 
                # 12-21 (-5) 7-16  左手
                (12,13),(12,14),(12,16),(12,18),(12,20),
                (14,15),(16,17),(18,19),(20,21),

                # 22-31 (-5) 17-26  左手
                (22,23),(22,24),(22,26),(22,28),(22,30),
                (24,25),(26,27),(28,29),(30,31),
    ),

    'sign/body_27':  (     # bias: 5  (data/sign/27/xxx_data_joint.npy 、data/sign/hands/xxx_data_joint.npy)
                # (鼻子 0 -> 5 -> 0，肩膀 5\6 -> 6\7 -> 1\2)
                (5, 6), (5, 7),  

                # 肩膀 5\6 -> 6\7 -> 1\2 
                # - 手肘 7\8 -> 8\9 -> 3\4
                # - 手腕 9\10 -> 10\11 -> 5\6
                (6, 8), (8, 10), (7, 9), (9, 11)
    )

}

all_splits = {
    'train', 'val', 'test'    # autsl / gsl
    # 'train', 'test'  # include
}

# datasets = {
#     'sign/27_2'
# }

from tqdm import tqdm

def process_file(joint_path, bone_path, tag, bias):
    print(f"Processing: {joint_path}")
    data = np.load(joint_path)
    N, C, T, V, M = data.shape  
    
    fp_sp = open_memmap(
        bone_path,
        dtype='float32',
        mode='w+',
        shape=(N, 3, T, V, M))

    fp_sp[:, :C, :, :, :] = data
    for v1, v2 in tqdm(paris[tag]):
        v1 -= bias
        v2 -= bias
        if v1 < V and v2 < V:
            fp_sp[:, :, :, v2, :] = data[:, :, :, v2, :] - data[:, :, :, v1, :]
        else:
            print(f"Index out of range: {v1}, {v2} for V={V}")

    print(f"Saved bone data to {bone_path}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Bone Data Converter.')
    parser.add_argument('--datasets', default='sign_gsl/27_cvpr')
    parser.add_argument('--tag', default='sign/27')  
    parser.add_argument('--data_path', type=str, default=None, help='File or directory containing joint data')
    parser.add_argument('--out_path', type=str, default=None, help='File or directory to save bone data')
    arg = parser.parse_args()

    tag = arg.tag 
    bias = 5 # Mặc định cho các bộ sign/27

    # Trường hợp 1: data_path là file lẻ
    if arg.data_path and os.path.isfile(arg.data_path):
        src_path = arg.data_path
        dst_path = arg.out_path if arg.out_path else src_path.replace('joint.npy', 'bone.npy')
        process_file(src_path, dst_path, tag, bias)
    
    # Trường hợp 2: data_path là thư mục
    else:
        data_base = arg.data_path if arg.data_path else '../data/{}'.format(arg.datasets)
        out_base = arg.out_path if arg.out_path else data_base

        for splits in all_splits:
            joint_path = os.path.join(data_base, '{}_data_joint.npy'.format(splits))
            bone_path = os.path.join(out_base, '{}_data_bone.npy'.format(splits))
            
            if not os.path.exists(joint_path):
                continue

            process_file(joint_path, bone_path, tag, bias)


