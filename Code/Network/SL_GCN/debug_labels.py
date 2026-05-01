#!/usr/bin/env python
"""
Debug script to inspect training and validation labels
"""
import pickle
import numpy as np

def inspect_labels(label_path, name):
    print(f"\n{'='*60}")
    print(f"Inspecting {name}: {label_path}")
    print(f"{'='*60}")
    
    try:
        with open(label_path, 'rb') as f:
            sample_name, labels = pickle.load(f, encoding='latin1')
    except:
        with open(label_path) as f:
            sample_name, labels = pickle.load(f)
    
    print(f"Total samples: {len(labels)}")
    print(f"Label type: {type(labels)}")
    if isinstance(labels, np.ndarray):
        print(f"Label dtype: {labels.dtype}")
        print(f"Label shape: {labels.shape}")
    
    # Convert to array for analysis
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    
    print(f"\nLabel statistics:")
    print(f"  Unique values: {len(np.unique(labels))}")
    print(f"  Min: {np.min(labels)}")
    print(f"  Max: {np.max(labels)}")
    
    # Check for empty/invalid labels
    invalid_count = 0
    empty_count = 0
    for i, label in enumerate(labels[:50]):  # Check first 50
        if isinstance(label, str):
            if label.strip() == '':
                empty_count += 1
            print(f"  Sample {i}: '{label}' (type: {type(label).__name__})")
        else:
            print(f"  Sample {i}: {label} (type: {type(label).__name__})")
    
    print(f"\nEmpty string labels in first 50: {empty_count}")
    
    # Check distribution
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\nTop 10 label distribution:")
    for lbl, cnt in sorted(zip(unique, counts), key=lambda x: -x[1])[:10]:
        print(f"  Class {lbl}: {cnt} samples")

# Inspect training labels
inspect_labels(
    './data/sign_include/27_cvpr/train_label.pkl',
    'TRAIN LABELS'
)

# Inspect validation labels
inspect_labels(
    './data/sign_include/27_cvpr/test_label.pkl',
    'VALIDATION LABELS'
)
