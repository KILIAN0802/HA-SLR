#!/usr/bin/env python
"""
Check model predictions on validation set
Add this to main_base.py eval() method to debug
"""

# Add this code snippet in the eval() method after computing predictions
# to print actual vs predicted labels

print("\n" + "="*60)
print("PREDICTION DEBUG INFO")
print("="*60)

# Show first 10 predictions
print("\nFirst 10 predictions (should vary if model is learning):")
for i in range(min(10, len(predict))):
    print(f"  Sample {i}: Predicted={predict[i]}, True={true[i]}, Match={predict[i]==true[i]}")

# Check prediction distribution
unique_preds, pred_counts = np.unique(predict, return_counts=True)
print(f"\nPrediction distribution (total {len(predict)} samples):")
for pred, cnt in zip(unique_preds, pred_counts):
    print(f"  Class {pred}: {cnt} samples ({cnt/len(predict)*100:.1f}%)")

# Check true label distribution
unique_true, true_counts = np.unique(true, return_counts=True)
print(f"\nTrue label distribution:")
for true_lbl, cnt in zip(unique_true, true_counts):
    print(f"  Class {true_lbl}: {cnt} samples ({cnt/len(true)*100:.1f}%)")

# Check if model is predicting same class for everything
if len(unique_preds) == 1:
    print(f"\n⚠️  WARNING: Model predicting ONLY class {unique_preds[0]} for all samples!")
    print("    This suggests a label encoding or data mismatch issue.")

print("="*60 + "\n")
