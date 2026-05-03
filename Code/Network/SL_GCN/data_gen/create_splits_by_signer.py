import argparse
import os
import re
import random
import sys
from collections import defaultdict

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = os.path.dirname(__file__)
# Always read raw files from data/MultiVSL200 and write outputs to data/MultiVSL200/splits
FILES_DIR = os.path.normpath(os.path.join(ROOT, '..', 'data', 'MultiVSL200'))
OUT_DIR = os.path.join(FILES_DIR, 'splits')
SEED = 42

# CLI: allow custom signer counts (useful to set train/val/test signer counts)
parser = argparse.ArgumentParser(description='Create signer-based splits for MultiVSL200')
parser.add_argument('--train-signers', type=int, default=None, help='Number of signers for train split')
parser.add_argument('--val-signers', type=int, default=None, help='Number of signers for val split')
parser.add_argument('--test-signers', type=int, default=None, help='Number of signers for test split')
parser.add_argument('--seed', type=int, default=SEED, help='Random seed for signer shuffling')
args = parser.parse_args()
SEED = args.seed

os.makedirs(OUT_DIR, exist_ok=True)

# gather .npy files
files = [f for f in os.listdir(FILES_DIR) if f.endswith('.npy')]
# map signer -> samples
signer_map = defaultdict(list)
for fn in files:
    name = fn[:-4]
    m = re.search(r'signer(\d{1,3})', name)
    if not m:
        continue
    signer = m.group(1)
    signer_map[signer].append(name)

signers = sorted(signer_map.keys(), key=lambda s: int(s))
random.Random(SEED).shuffle(signers)

n = len(signers)

# If user provided explicit signer counts, use them (with basic validation)
if args.train_signers is not None or args.val_signers is not None or args.test_signers is not None:
    # default to 0 for unspecified counts
    t = args.train_signers if args.train_signers is not None else 0
    v = args.val_signers if args.val_signers is not None else 0
    te = args.test_signers if args.test_signers is not None else 0
    if t + v + te != n:
        print(f"Warning: requested signer counts sum {t+v+te} != total signers {n}.")
        print("Adjusting by giving remainder to train split.")
        # give leftover to train
        leftover = n - (t + v + te)
        t = max(0, t + leftover)
    n_train, n_val, n_test = t, v, te
else:
    # allocate 80/10/10 by signers using rounding to get balanced small counts
    n_train = round(n * 0.8)
    n_val = round(n * 0.1)
    n_test = n - n_train - n_val

    # correct for edge cases if rounding pushed counts out of bounds
    if n_test < 0:
        n_test = 0
    if n_train < 0:
        n_train = max(0, n - n_val - n_test)

train_signers = signers[:n_train]
val_signers = signers[n_train:n_train+n_val]
test_signers = signers[n_train+n_val:]

splits = {
    'train': train_signers,
    'val': val_signers,
    'test': test_signers,
}

# helper to parse label from filename prefix (e.g., '23_Name_...')
label_re = re.compile(r'ord1_(\d+)')

for part, s_list in splits.items():
    out_csv = os.path.join(OUT_DIR, f'{part}_labels.csv')
    out_list = os.path.join(OUT_DIR, f'{part}_files.txt')
    samples = []
    for s in s_list:
        samples.extend(signer_map[s])
    samples = sorted(samples)
    with open(out_csv, 'w', encoding='utf-8') as fo, open(out_list, 'w', encoding='utf-8') as fl:
        for name in samples:
            m = label_re.search(name)
            label = m.group(1) if m else ''
            fo.write(f'{name},{label}\n')
            fl.write(f'{name}.npy\n')
    print(f'Wrote {len(samples)} entries to {out_csv} and {out_list}')

print('\nSigner allocation:')
print('train:', train_signers)
print('val  :', val_signers)
print('test :', test_signers)
