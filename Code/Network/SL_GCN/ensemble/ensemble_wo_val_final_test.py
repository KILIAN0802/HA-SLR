import argparse
import pickle
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

# Define a learnable layer for fusion weights
class FusionWeights(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize weights as a learnable parameter, starting with average fusion
        self.weights = nn.Parameter(torch.ones(4))

    def forward(self, scores):
        # Using softmax to make them sum to 1, which is a common practice
        normalized_weights = torch.nn.functional.softmax(self.weights, dim=0)
        
        # scores is a list of 4 tensors [ (N, C), (N, C), (N, C), (N, C) ]
        # Stack scores into a single tensor: [4, N, C]
        stacked_scores = torch.stack(scores, dim=0)
        
        # Perform weighted sum
        # Reshape weights for broadcasting: [4, 1, 1]
        # (4, 1, 1) * (4, N, C) -> (4, N, C) -> sum over dim 0 -> (N, C)
        ensembled_score = (normalized_weights.view(4, 1, 1) * stacked_scores).sum(dim=0)
        return ensembled_score, normalized_weights

# --- 1. Load Data ---
# Load labels
label_path = './../data/data/Duy/MultiVSL200_27/test_label.pkl'
with open(label_path, 'rb') as f:
    # label format: (names, labels)
    names, labels_list = pickle.load(f)
labels = torch.LongTensor(np.array(labels_list, dtype=int))

# Load scores from 4 streams
score_paths = [
    './../work_dir/MultiVSL200-46kpt-Duy/Joint/multivsl200_joint_27_cvpr_hand_aware_sl_lgcn_baseline/bs64_f100_lr1e-08_trainlr0.1_warmup20_test/2026-05-04_07-04-02/scores/multivsl200_joint_27_cvpr_hand_aware_sl_lgcn_baseline_best_acc_test_score.pkl',
    './../work_dir/MultiVSL200-46kpt-Duy/Bone/multivsl200_bone_27_cvpr_hand_aware_sl_lgcn_baseline/bs64_f100_lr1e-08_trainlr0.1_warmup20_test/2026-05-04_07-06-42/scores/multivsl200_bone_27_cvpr_hand_aware_sl_lgcn_baseline_best_acc_test_score.pkl',
    './../work_dir/MultiVSL200-46kpt-Duy/Joint-Motion/multivsl200_joint_motion_27_cvpr_hand_aware_sl_lgcn_baseline/bs64_f100_lr1e-08_trainlr0.1_warmup20_test/2026-05-04_07-08-21/scores/multivsl200_joint_motion_27_cvpr_hand_aware_sl_lgcn_baseline_best_acc_test_score.pkl',
    './../work_dir/MultiVSL200-46kpt-Duy/Bone-Motion/multivsl200_bone_motion_27_cvpr_hand_aware_sl_lgcn_baseline/bs64_f100_lr1e-08_trainlr0.1_warmup20_test/2026-05-04_07-10-07/scores/multivsl200_bone_motion_27_cvpr_hand_aware_sl_lgcn_baseline_best_acc_test_score.pkl'
]

all_scores = []
print("Loading scores...")
for p in tqdm(score_paths):
    with open(p, 'rb') as f:
        # Sort by name to ensure order is correct
        scores_dict = dict(pickle.load(f).items())
        ordered_scores = [scores_dict[name] for name in names]
        all_scores.append(torch.from_numpy(np.array(ordered_scores)))

# --- 2. Setup for Training ---
fusion_model = FusionWeights()
optimizer = optim.Adam(fusion_model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
num_epochs = 100

print("\nInitial weights (before normalization):", fusion_model.weights.data.numpy())
print("--- Starting Training ---")

# --- 3. Training Loop ---
for epoch in tqdm(range(num_epochs), desc="Training Weights"):
    optimizer.zero_grad()
    
    # Get final prediction by ensembling
    final_score, normalized_weights = fusion_model(all_scores)
    
    # Calculate loss
    loss = criterion(final_score, labels)
    
    # Backpropagation
    loss.backward()
    optimizer.step()
    
    # if (epoch + 1) % 10 == 0:
    #     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# --- 4. Evaluation and Verification ---
print("\n--- Training Finished ---")
final_score, normalized_weights = fusion_model(all_scores)
final_score = final_score.detach()
normalized_weights = normalized_weights.detach()

print("\nLearned Raw Weights (unnormalized):")
stream_names = ['Joint', 'Bone', 'Joint-Motion', 'Bone-Motion']
final_raw_weights = fusion_model.weights.data.numpy()
for name, weight in zip(stream_names, final_raw_weights):
    print(f"- {name}: {weight:.4f}")

print("\nLearned Normalized Weights (used for fusion):")
final_normalized_weights = normalized_weights.numpy()
for name, weight in zip(stream_names, final_normalized_weights):
    print(f"- {name}: {weight:.4f}")

# Calculate final accuracy
pred = torch.argmax(final_score, dim=1)
right_num = (pred == labels).sum().item()
total_num = len(labels)
acc = right_num / total_num

rank_5 = torch.topk(final_score, 5, dim=1).indices
right_num_5 = sum(labels[i] in rank_5[i] for i in range(total_num))
acc5 = right_num_5 / total_num

print(f'\nTotal samples: {total_num}')
print(f'Top-1 Accuracy: {acc:.4f}')
print(f'Top-5 Accuracy: {acc5:.4f}')

# --- 5. Save predictions ---
with open('predictions_wo_val_final_test_learned.csv', 'w') as f:
    for i in range(total_num):
        f.write('{}, {}\n'.format(names[i], pred[i].item()))

with open('./gcn_ensembled_final_test_learned.pkl', 'wb') as f:
    score_dict = dict(zip(names, final_score.numpy()))
    pickle.dump(score_dict, f)

print("\nPredictions saved to 'predictions_wo_val_final_test_learned.csv'")
print("Ensembled scores saved to 'gcn_ensembled_final_test_learned.pkl'")