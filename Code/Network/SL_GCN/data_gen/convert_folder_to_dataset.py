import argparse
import os
import re
import pickle
from glob import glob
from collections import defaultdict
import random
import numpy as np


def infer_and_fix_shape(x):
    # Accept common shapes and return (C, T, V, 1)
    x = np.array(x)
    if x.ndim == 3:
        # possibilities: (T, V, C) or (C, T, V)
        T, V, last = x.shape
        if last in (2, 3):
            # (T, V, C) -> (C, T, V)
            x = x.transpose(2, 0, 1)
        else:
            # assume (C, T, V)
            x = x
    elif x.ndim == 2:
        # (T, V) -> add channel dim
        T, V = x.shape
        x = x.reshape(1, T, V)
    elif x.ndim == 4:
        # maybe already (C, T, V, M) or (N, T, V, C) unusual
        # try to detect channel dim
        C, T, V, M = x.shape
        if C in (2, 3):
            pass
        else:
            # fallback: take first sample
            x = x[0]
    else:
        raise ValueError(f"Unsupported sample ndim={x.ndim}")

    # ensure now (C, T, V)
    if x.ndim != 3:
        raise ValueError("After conversion sample is not 3D (C,T,V)")

    C, T, V = x.shape
    # add M dim
    x = x.reshape((C, T, V, 1))
    return x


def pad_or_truncate(x, target_T, pad_value=0):
    C, T, V, M = x.shape
    if T == target_T:
        return x
    if T < target_T:
        pad = np.ones((C, target_T - T, V, M), dtype=x.dtype) * pad_value
        return np.concatenate([x, pad], axis=1)
    else:
        return x[:, :target_T, :, :]


def compute_motion(x):
    # x: (C, T, V, M) -> motion same shape
    # temporal diff along T
    C, T, V, M = x.shape
    motion = np.zeros_like(x)
    if T > 1:
        motion[:, 1:, :, :] = x[:, 1:, :, :] - x[:, :-1, :, :]
    return motion


def compute_bone(x, parent_index):
    # x: (C, T, V, M), parent_index: list length V with parent idx or -1
    C, T, V, M = x.shape
    bone = np.zeros_like(x)
    for v in range(V):
        p = parent_index[v]
        if p >= 0:
            bone[:, :, v, :] = x[:, :, v, :] - x[:, :, p, :]
        else:
            bone[:, :, v, :] = 0
    return bone


def parse_label_from_name(name, regex=None, zero_based=True):
    # default: take leading number before underscore, e.g. '01_Foo...'
    base = os.path.basename(name)
    if regex:
        m = re.search(regex, base)
        if m:
            label = int(m.group(1))
            return label - (0 if zero_based else 1)
    m = re.match(r"(\d+)_", base)
    if m:
        label = int(m.group(1))
        return label - 1 if m and (label != 0) else label
    # fallback: 0
    return 0


def main(args):
    files = sorted(glob(os.path.join(args.input_dir, "*.npy")))
    if len(files) == 0:
        raise FileNotFoundError("No .npy files found in input_dir")

    samples = []
    lengths = []
    names = []
    labels = []

    for fp in files:
        arr = np.load(fp)
        try:
            arr = infer_and_fix_shape(arr)
        except Exception as e:
            print(f"Skipping {fp}: {e}")
            continue
        samples.append(arr)
        lengths.append(arr.shape[1])
        names.append(os.path.relpath(fp, args.input_dir))
        labels.append(parse_label_from_name(fp, regex=args.label_regex, zero_based=True))

    if args.auto_split:
        grouped = defaultdict(list)
        for idx, label in enumerate(labels):
            grouped[int(label)].append(idx)

        rng = random.Random(args.seed)
        split_indices = {"train": [], "val": [], "test": []}
        train_ratio, val_ratio, test_ratio = args.split_ratios
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("split-ratios must sum to 1.0")

        for label, idxs in grouped.items():
            idxs = list(idxs)
            rng.shuffle(idxs)
            n_total = len(idxs)
            n_train = int(round(n_total * train_ratio))
            n_val = int(round(n_total * val_ratio))
            if n_train + n_val > n_total:
                n_val = max(0, n_total - n_train)
            n_test = n_total - n_train - n_val
            split_indices["train"].extend(idxs[:n_train])
            split_indices["val"].extend(idxs[n_train:n_train + n_val])
            split_indices["test"].extend(idxs[n_train + n_val:n_train + n_val + n_test])

        for split_name in ["train", "val", "test"]:
            split_idx = split_indices[split_name]
            split_samples = [samples[i] for i in split_idx]
            split_lengths = [lengths[i] for i in split_idx]
            split_names = [names[i] for i in split_idx]
            split_labels = [labels[i] for i in split_idx]
            save_split(split_name, split_samples, split_lengths, split_names, split_labels, args)
        return

    save_split(args.split, samples, lengths, names, labels, args)


def save_split(split_name, samples, lengths, names, labels, args):
    if len(samples) == 0:
        raise ValueError(f"No valid samples to save for split '{split_name}'")

    # determine target_T
    if args.window_size > 0:
        target_T = args.window_size
    else:
        target_T = max(lengths)

    # pad/truncate
    fixed = [pad_or_truncate(s, target_T) for s in samples]

    data = np.stack(fixed, axis=0)  # N, C, T, V, M

    os.makedirs(args.output_dir, exist_ok=True)

    # modality handling
    if args.modality == 'joint':
        out_data = data
        np.save(os.path.join(args.output_dir, f"{split_name}_data_joint.npy"), out_data)
    elif args.modality == 'motion':
        out_data = np.stack([compute_motion(s) for s in fixed], axis=0)
        np.save(os.path.join(args.output_dir, f"{split_name}_data_joint_motion.npy"), out_data)
    elif args.modality == 'bone':
        # user can provide parent map via file or default chain
        # default: linear chain parents (v -> v-1)
        V = data.shape[3]
        parent = [-1] + [i - 1 for i in range(1, V)]
        out_data = np.stack([compute_bone(s, parent) for s in fixed], axis=0)
        np.save(os.path.join(args.output_dir, f"{split_name}_data_bone.npy"), out_data)
    else:
        raise ValueError('Unknown modality')

    # save label.pkl (sample_name, label)
    label_path = os.path.join(args.output_dir, f"{split_name}_label.pkl")
    with open(label_path, 'wb') as f:
        pickle.dump((names, np.array(labels, dtype=int)), f)

    print(f'Saved {split_name} split to {args.output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--split', default='train', help='train/val/test')
    parser.add_argument('--modality', default='joint', choices=['joint', 'motion', 'bone'])
    parser.add_argument('--window-size', type=int, default=-1, help='pad/truncate to this many frames; -1 -> use max')
    parser.add_argument('--label-regex', type=str, default=None, help='regex with one capturing group for label')
    parser.add_argument('--auto-split', action='store_true', help='split one folder into train/val/test by label')
    parser.add_argument('--split-ratios', type=float, nargs=3, default=[0.8, 0.1, 0.1], help='train val test ratios')
    parser.add_argument('--seed', type=int, default=42, help='random seed for auto split')
    args = parser.parse_args()
    main(args)
