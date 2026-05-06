"""
Tìm alpha tốt nhất trên val bằng cách thử vài bộ alpha gần mặc định.

Chạy từ thư mục Code/Network/SL_GCN:
    python ensemble\ensemble_search_alpha.py
"""

import pickle
import os
import numpy as np


def load_scores(path_list):
    """Load 4 score pkl files."""
    scores_list = []
    for i, path in enumerate(path_list):
        if path is None:
            print(f"[ERROR] Path {i} is None")
            return None
        if not os.path.exists(path):
            print(f"[ERROR] File not found: {path}")
            return None
        print(f"Loading score {i+1}/4: {path}")
        with open(path, 'rb') as f:
            scores = list(pickle.load(f).items())
            scores_list.append(scores)
    return scores_list


def ensemble_with_alpha(scores_list, labels, alpha):
    """Tính top1/top5 với bộ alpha cụ thể."""
    n_samples = len(labels[0])
    right_num = 0
    right_num_5 = 0
    
    for i in range(n_samples):
        # Extract sample name and true label
        name, true_label = labels[:, i]
        
        # Extract scores from 4 streams
        _, s1 = scores_list[0][i]
        _, s2 = scores_list[1][i]
        _, s3 = scores_list[2][i]
        _, s4 = scores_list[3][i]
        
        # Weighted ensemble
        score = (s1*alpha[0] + s2*alpha[1] + s3*alpha[2] + s4*alpha[3]) / np.array(alpha).sum()
        
        # Top1 accuracy
        pred = np.argmax(score)
        right_num += int(pred == int(true_label))
        
        # Top5 accuracy
        rank_5 = score.argsort()[-5:]
        right_num_5 += int(int(true_label) in rank_5)
    
    top1 = right_num / n_samples
    top5 = right_num_5 / n_samples
    return top1, top5


def main():
    labels_path = './data/MultiVSL200_27/val_label.pkl'
    score_paths = [
        './work_dir/MultiVSL200/Joint/multivsl200_joint_27_cvpr_hand_aware_sl_lgcn_baseline/bs64_f100_lr1e-08_trainlr0.1_warmup20_test/2026-05-04_07-04-02/scores/multivsl200_joint_27_cvpr_hand_aware_sl_lgcn_baseline_best_acc_test_score.pkl',
        './work_dir/MultiVSL200/Bone/multivsl200_bone_27_cvpr_hand_aware_sl_lgcn_baseline/bs64_f100_lr1e-08_trainlr0.1_warmup20_test/2026-05-04_07-06-42/scores/multivsl200_bone_27_cvpr_hand_aware_sl_lgcn_baseline_best_acc_test_score.pkl',
        './work_dir/MultiVSL200/Joint-Motion/multivsl200_joint_motion_27_cvpr_hand_aware_sl_lgcn_baseline/bs64_f100_lr1e-08_trainlr0.1_warmup20_test/2026-05-04_07-08-21/scores/multivsl200_joint_motion_27_cvpr_hand_aware_sl_lgcn_baseline_best_acc_test_score.pkl',
        './work_dir/MultiVSL200/Bone-Motion/multivsl200_bone_motion_27_cvpr_hand_aware_sl_lgcn_baseline/bs64_f100_lr1e-08_trainlr0.1_warmup20_test/2026-05-04_07-1₀-₀7/scores/multivsl2₀₀_bone_motion_₂₇_cvpr_hand_aware_sl_lgcn_baseline_best_acc_test_score.pkl',
    ]

    if not os.path.exists(labels_path):
        print(f"[ERROR] Label file not found: {labels_path}")
        return

    with open(labels_path, 'rb') as f:
        sample_names, labels = pickle.load(f)
        labels = np.array([sample_names, labels])

    print(f"Loaded labels: {labels.shape}")

    scores_list = load_scores(score_paths)
    if scores_list is None:
        print("[ERROR] Failed to load scores")
        return

    # Simple local search around the default alpha.
    candidate_alphas = [
        [1.0, 0.9, 0.5, 0.5],
        [1.1, 0.9, 0.5, 0.5],
        [1.2, 0.9, 0.5, 0.5],
        [1.0, 1.0, 0.5, 0.5],
        [1.0, 1.1, 0.5, 0.5],
        [1.0, 1.2, 0.5, 0.5],
        [1.0, 0.9, 0.4, 0.4],
        [1.0, 0.9, 0.6, 0.6],
        [1.0, 0.9, 0.7, 0.7],
        [1.2, 1.0, 0.5, 0.5],
    ]
    
    print("\n" + "=" * 80)
    print("ALPHA SEARCH RESULTS (on VAL set)")
    print("=" * 80)
    print(f"{'Alpha':<30} {'Top1 Acc':<12} {'Top5 Acc':<12}")
    print("-" * 80)
    
    results = []
    for alpha in candidate_alphas:
        top1, top5 = ensemble_with_alpha(scores_list, labels, alpha)
        results.append((alpha, top1, top5))
        alpha_str = f"[{alpha[0]}, {alpha[1]}, {alpha[2]}, {alpha[3]}]"
        print(f"{alpha_str:<30} {top1:.6f}      {top5:.6f}")
    
    # Find best alpha
    best_idx = np.argmax([r[1] for r in results])
    best_alpha, best_top1, best_top5 = results[best_idx]
    
    print("=" * 80)
    print(f"\n🏆 BEST ALPHA (on val):")
    print(f"   alpha = {best_alpha}")
    print(f"   top1 = {best_top1:.6f}")
    print(f"   top5 = {best_top5:.6f}")
    print("\n📝 Copy this into ensemble_wo_val_final_test.py:")
    print(f"   alpha = {best_alpha}")
    print("\n" + "=" * 80)
    
    # Save results to log file
    log_file = 'alpha_search_log.txt'
    with open(log_file, 'w') as f:
        f.write("ALPHA SEARCH RESULTS (on VAL set)\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Alpha':<30} {'Top1 Acc':<12} {'Top5 Acc':<12}\n")
        f.write("-" * 80 + "\n")
        for alpha, top1, top5 in results:
            alpha_str = f"[{alpha[0]}, {alpha[1]}, {alpha[2]}, {alpha[3]}]"
            f.write(f"{alpha_str:<30} {top1:.6f}      {top5:.6f}\n")
        f.write("=" * 80 + "\n")
        f.write(f"\n🏆 BEST ALPHA:\n")
        f.write(f"   alpha = {best_alpha}\n")
        f.write(f"   top1 = {best_top1:.6f}\n")
        f.write(f"   top5 = {best_top5:.6f}\n")
    
    print(f"\n✓ Results saved to: {log_file}")


if __name__ == '__main__':
    main()
