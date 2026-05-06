"""
Fully Connected Fusion Ensemble.

Học cách kết hợp 4 stream scores bằng 1 lớp fully connected.

Cách dùng:
  python ensemble_fully_connected_fusion.py
  
Sẽ train lớp FC trên test scores, eval lên test (để kiểm chứng).
Sau khi verify, thay bằng val scores để train đúng cách.
"""

import pickle
import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


SCRIPT_DIR = os.path.dirname(__file__)
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))


def load_scores(score_paths):
    """Load 4 score pkl files."""
    scores_list = []
    for i, path in enumerate(score_paths):
        if not os.path.exists(path):
            print(f"[ERROR] File not found: {path}")
            return None
        print(f"Loading score {i+1}/4: {path}")
        with open(path, 'rb') as f:
            scores = list(pickle.load(f).items())
            scores_list.append(scores)
    return scores_list


def get_score_from_pkl_list(scores_list, idx):
    """Extract scores for sample idx from all 4 streams."""
    s1 = scores_list[0][idx][1]
    s2 = scores_list[1][idx][1]
    s3 = scores_list[2][idx][1]
    s4 = scores_list[3][idx][1]
    return s1, s2, s3, s4


def prepare_fc_input(scores_list):
    """
    Prepare input for FC layer: concatenate 4 streams.
    Returns: (n_samples, 4*num_classes) tensor
    """
    n_samples = len(scores_list[0])
    
    # Infer num_classes from first sample
    s1 = scores_list[0][0][1]
    num_classes = len(s1)
    
    X = np.zeros((n_samples, 4 * num_classes), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.int64)
    
    for i in range(n_samples):
        name, true_label = (scores_list[0][i][0], None)  # Get name from first stream
        # Find true label from first stream item (assume it's same order)
        s1, s2, s3, s4 = get_score_from_pkl_list(scores_list, i)
        X[i, :num_classes] = s1
        X[i, num_classes:2*num_classes] = s2
        X[i, 2*num_classes:3*num_classes] = s3
        X[i, 3*num_classes:4*num_classes] = s4
    
    return X


def prepare_fc_input_with_labels(scores_list, labels):
    """
    Prepare input for FC layer with labels.
    """
    n_samples = len(labels[0])
    
    # Infer num_classes from first sample
    s1 = scores_list[0][0][1]
    num_classes = len(s1)
    
    X = np.zeros((n_samples, 4 * num_classes), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.int64)
    
    for i in range(n_samples):
        name, true_label = labels[:, i]
        s1, s2, s3, s4 = get_score_from_pkl_list(scores_list, i)
        X[i, :num_classes] = s1
        X[i, num_classes:2*num_classes] = s2
        X[i, 2*num_classes:3*num_classes] = s3
        X[i, 3*num_classes:4*num_classes] = s4
        y[i] = int(true_label)
    
    return X, y


class SimpleFC(nn.Module):
    """Simple FC fusion: 4*C -> C"""
    def __init__(self, num_classes):
        super().__init__()
        input_dim = 4 * num_classes
        self.fc = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)


def train_fc(train_X, train_y, num_epochs=50, lr=0.01, batch_size=32):
    """Train FC layer."""
    num_classes = train_X.shape[1] // 4
    model = SimpleFC(num_classes).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    X_tensor = torch.from_numpy(train_X).cuda()
    y_tensor = torch.from_numpy(train_y).long().cuda()
    
    print(f"\nTraining FC layer for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        # Shuffle
        perm = np.random.permutation(len(train_X))
        X_perm = X_tensor[perm]
        y_perm = y_tensor[perm]
        
        total_loss = 0.0
        n_batches = 0
        for i in range(0, len(train_X), batch_size):
            X_batch = X_perm[i:i+batch_size]
            y_batch = y_perm[i:i+batch_size]
            
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    model.eval()
    return model


def evaluate_fc(model, test_X, test_y):
    """Evaluate FC layer."""
    X_tensor = torch.from_numpy(test_X).cuda()
    y_tensor = torch.from_numpy(test_y).long().cuda()
    
    with torch.no_grad():
        logits = model(X_tensor)
        logits_cpu = logits.cpu().numpy()
    
    right_num = 0
    right_num_5 = 0
    
    for i in range(len(test_X)):
        pred = np.argmax(logits_cpu[i])
        true_label = test_y[i]
        right_num += int(pred == true_label)
        
        rank_5 = logits_cpu[i].argsort()[-5:]
        right_num_5 += int(true_label in rank_5)
    
    top1 = right_num / len(test_X)
    top5 = right_num_5 / len(test_X)
    return top1, top5


def weighted_average_ensemble(scores_list, alpha=[1.0, 0.9, 0.5, 0.5]):
    """Weighted average ensemble baseline."""
    n_samples = len(scores_list[0])
    num_classes = len(scores_list[0][0][1])
    
    result = np.zeros((n_samples, num_classes), dtype=np.float32)
    for i in range(n_samples):
        s1, s2, s3, s4 = get_score_from_pkl_list(scores_list, i)
        score = (s1*alpha[0] + s2*alpha[1] + s3*alpha[2] + s4*alpha[3]) / np.array(alpha).sum()
        result[i] = score
    
    return result


def evaluate_weighted_avg(scores_weighted, test_y):
    """Evaluate weighted average ensemble."""
    right_num = 0
    right_num_5 = 0
    
    for i in range(len(test_y)):
        pred = np.argmax(scores_weighted[i])
        true_label = test_y[i]
        right_num += int(pred == true_label)
        
        rank_5 = scores_weighted[i].argsort()[-5:]
        right_num_5 += int(true_label in rank_5)
    
    top1 = right_num / len(test_y)
    top5 = right_num_5 / len(test_y)
    return top1, top5


def main():
    print("=" * 80)
    print("FULLY CONNECTED FUSION ENSEMBLE (Option 1: 80/20 Split)")
    print("=" * 80)
    
    labels_path = os.path.join(BASE_DIR, 'data', 'MultiVSL200_27', 'test_label.pkl')
    score_paths = [
        os.path.join(BASE_DIR, 'work_dir', 'MultiVSL200', 'Joint', 'multivsl200_joint_27_cvpr_hand_aware_sl_lgcn_baseline', 'bs64_f100_lr1e-08_trainlr0.1_warmup20_test', '2026-05-04_07-04-02', 'scores', 'multivsl200_joint_27_cvpr_hand_aware_sl_lgcn_baseline_best_acc_test_score.pkl'),
        os.path.join(BASE_DIR, 'work_dir', 'MultiVSL200', 'Bone', 'multivsl200_bone_27_cvpr_hand_aware_sl_lgcn_baseline', 'bs64_f100_lr1e-08_trainlr0.1_warmup20_test', '2026-05-04_07-06-42', 'scores', 'multivsl200_bone_27_cvpr_hand_aware_sl_lgcn_baseline_best_acc_test_score.pkl'),
        os.path.join(BASE_DIR, 'work_dir', 'MultiVSL200', 'Joint-Motion', 'multivsl200_joint_motion_27_cvpr_hand_aware_sl_lgcn_baseline', 'bs64_f100_lr1e-08_trainlr0.1_warmup20_test', '2026-05-04_07-08-21', 'scores', 'multivsl200_joint_motion_27_cvpr_hand_aware_sl_lgcn_baseline_best_acc_test_score.pkl'),
        os.path.join(BASE_DIR, 'work_dir', 'MultiVSL200', 'Bone-Motion', 'multivsl200_bone_motion_27_cvpr_hand_aware_sl_lgcn_baseline', 'bs64_f100_lr1e-08_trainlr0.1_warmup20_test', '2026-05-04_07-10-07', 'scores', 'multivsl200_bone_motion_27_cvpr_hand_aware_sl_lgcn_baseline_best_acc_test_score.pkl'),
    ]
    
    if not os.path.exists(labels_path):
        print(f"[ERROR] Label file not found: {labels_path}")
        return
    
    with open(labels_path, 'rb') as f:
        sample_names, labels_raw = pickle.load(f)
        labels = np.array([sample_names, labels_raw])
    
    print(f"Loaded labels: {labels.shape}")
    
    scores_list = load_scores(score_paths)
    if scores_list is None:
        print("[ERROR] Failed to load scores")
        return
    
    # Prepare all data
    print("\nPreparing data...")
    full_X, full_y = prepare_fc_input_with_labels(scores_list, labels)
    print(f"Full data shape: {full_X.shape}")
    print(f"Total samples: {len(full_y)}")
    
    # Split 80/20
    n_total = len(full_y)
    n_train_fc = int(0.8 * n_total)
    
    print(f"\nSplitting data: 80% train FC ({n_train_fc}), 20% eval ({n_total - n_train_fc})")
    
    indices = np.random.permutation(n_total)
    train_idx = indices[:n_train_fc]
    eval_idx = indices[n_train_fc:]
    
    train_X = full_X[train_idx]
    train_y = full_y[train_idx]
    eval_X = full_X[eval_idx]
    eval_y = full_y[eval_idx]
    
    # Train FC
    model = train_fc(train_X, train_y, num_epochs=50, lr=0.01, batch_size=32)
    
    # Evaluate FC on holdout 20%
    print("\nEvaluating FC Fusion on holdout 20%...")
    fc_top1, fc_top5 = evaluate_fc(model, eval_X, eval_y)
    
    # Compare with weighted average on same 20%
    print("\nComputing Weighted Average baseline on same 20%...")
    scores_weighted = np.zeros((len(eval_idx), full_X.shape[1] // 4), dtype=np.float32)
    for i, idx in enumerate(eval_idx):
        s1, s2, s3, s4 = get_score_from_pkl_list(scores_list, idx)
        alpha = [1.0, 0.9, 0.5, 0.5]
        score = (s1*alpha[0] + s2*alpha[1] + s3*alpha[2] + s4*alpha[3]) / np.array(alpha).sum()
        scores_weighted[i] = score
    
    wa_top1, wa_top5 = evaluate_weighted_avg(scores_weighted, eval_y)
    
    # Results
    print("\n" + "=" * 80)
    print("RESULTS (Evaluation on 20% holdout set):")
    print("=" * 80)
    print(f"{'Method':<30} {'Top1':<15} {'Top5':<15}")
    print("-" * 60)
    print(f"{'Weighted Average':<30} {wa_top1:.6f} {wa_top5:.6f}")
    print(f"{'Fully Connected Fusion':<30} {fc_top1:.6f} {fc_top5:.6f}")
    print("-" * 60)
    improvement_top1 = (fc_top1 - wa_top1) * 100
    improvement_top5 = (fc_top5 - wa_top5) * 100
    print(f"Improvement (FC vs WA):        +{improvement_top1:.2f}% {'' if improvement_top1 >= 0 else '!':<7} +{improvement_top5:.2f}% {'':>7}")
    print("=" * 80)
    
    # Save model
    torch.save(model.state_dict(), 'fc_fusion_model.pth')
    print("\n✓ Model saved to: fc_fusion_model.pth")


if __name__ == '__main__':
    main()
