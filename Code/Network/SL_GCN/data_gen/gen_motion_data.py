import argparse
from tqdm import tqdm
import os
import numpy as np
from numpy.lib.format import open_memmap

all_splits = {
    'train', 'val', 'test'  # autsl
    # 'train', 'test'  # include
}

# datasets = {
#     'sign/27_2'
# }

parts = {
    'joint', 'bone'
}

def process_file(src_path, dst_path):
    print(f"Processing: {src_path}")
    data = np.load(src_path)
    N, C, T, V, M = data.shape
    
    fp_sp = open_memmap(
        dst_path,
        dtype='float32',
        mode='w+',
        shape=(N, C, T, V, M))
    
    for t in tqdm(range(T - 1)):
        fp_sp[:, :, t, :, :] = data[:, :, t + 1, :, :] - data[:, :, t, :, :]
    fp_sp[:, :, T - 1, :, :] = 0
    
    print(f"Saved motion data to {dst_path}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Motion Data Converter.')
    parser.add_argument('--datasets', default='sign_gsl/27_cvpr')
    parser.add_argument('--data_path', type=str, default=None, help='File or directory containing joint/bone data')
    parser.add_argument('--out_path', type=str, default=None, help='File or directory to save motion data')
    arg = parser.parse_args()

    # Trường hợp 1: data_path là một file cụ thể
    if arg.data_path and os.path.isfile(arg.data_path):
        src_path = arg.data_path
        dst_path = arg.out_path if arg.out_path else src_path.replace('.npy', '_motion.npy')
        process_file(src_path, dst_path)
    
    # Trường hợp 2: data_path là thư mục hoặc dùng mặc định
    else:
        data_base = arg.data_path if arg.data_path else '../data/{}'.format(arg.datasets)
        out_base = arg.out_path if arg.out_path else data_base

        for splits in all_splits:
            for part in parts:
                src_path = os.path.join(data_base, '{}_data_{}.npy'.format(splits, part))
                dst_path = os.path.join(out_base, '{}_data_{}_motion.npy'.format(splits, part))
                
                if not os.path.exists(src_path):
                    continue

                process_file(src_path, dst_path)


