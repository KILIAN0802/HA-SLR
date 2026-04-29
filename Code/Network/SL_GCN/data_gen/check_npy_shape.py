import numpy as np, os, glob
base = r'../data/MultiVSL200'
files = glob.glob(os.path.join(base, '*.npy'))[:3]
for f in files:
    d = np.load(f)
    print(f'{os.path.basename(f)}: shape={d.shape}')
