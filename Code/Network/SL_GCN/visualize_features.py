import os
import sys
import yaml
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add current path to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def load_class_names(csv_path):
    # Initialize with default names
    class_names = [f"Class {i}" for i in range(200)]
    if os.path.exists(csv_path):
        import csv
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)  # skip header
                for row in reader:
                    if len(row) >= 2:
                        idx = int(row[0].strip()) - 1  # 1-indexed in CSV -> 0-indexed in code
                        name = row[1].strip()
                        if 0 <= idx < 200:
                            class_names[idx] = name
            print(f"-> Successfully loaded {len(class_names)} Vietnamese class names from {csv_path}")
        except Exception as e:
            print(f"Warning: Failed to load class names from CSV: {e}")
    else:
        print(f"Warning: Lookup table not found at {csv_path}. Using default class names.")
    return class_names

def extract_features_and_predictions(model, data_loader, device='cuda', limit_batches=None):
    model.to(device)
    model.eval()
    
    all_features = []
    all_labels = []
    all_predictions = []
    
    spatial_attentions = []
    temporal_attentions = []
    
    # Temporary lists to store batch attention weights
    batch_spatial_att = []
    batch_temporal_att = []
    
    # Registers hooks to extract spatial and temporal attention weights dynamically
    hooks = []
    
    def make_sa_hook(layer_name):
        def hook(module, inp, out):
            # out shape: [N * M, 1, V]
            val = torch.sigmoid(out).detach().cpu().numpy()
            batch_spatial_att.append((layer_name, val))
        return hook

    def make_ta_hook(layer_name):
        def hook(module, inp, out):
            # out shape: [N * M, 1, T]
            val = torch.sigmoid(out).detach().cpu().numpy()
            batch_temporal_att.append((layer_name, val))
        return hook

    # Register hooks on the attention modules in TCN_GCN_unit blocks
    for name, module in model.named_modules():
        if name.endswith('conv_sa'):
            hooks.append(module.register_forward_hook(make_sa_hook(name)))
        elif name.endswith('conv_ta'):
            hooks.append(module.register_forward_hook(make_ta_hook(name)))

    print("-> Extracting features, predictions, and attention maps from the model...")
    with torch.no_grad():
        for batch_idx, (data, label, index) in enumerate(tqdm(data_loader, desc="Inference")):
            if limit_batches is not None and batch_idx >= limit_batches:
                break
                
            data = data.float().to(device)
            N, C, T, V, M = data.size()
            
            # Clear batch-level attention caches
            batch_spatial_att.clear()
            batch_temporal_att.clear()
            
            # Replicate the forward pass up to the feature layer to get 256-D features
            x = data.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
            x = model.data_bn(x)
            x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

            x = model.l1(x, keep_prob=1.0)
            x = model.l2(x, keep_prob=1.0)
            x = model.l3(x, keep_prob=1.0)
            x = model.l4(x, keep_prob=1.0)
            x = model.l5(x, keep_prob=1.0)
            x = model.l6(x, keep_prob=1.0)
            x = model.l7(x, keep_prob=1.0)
            x = model.l8(x, keep_prob=1.0)
            x = model.l9(x, keep_prob=1.0)
            x = model.l10(x, keep_prob=1.0)
            
            c_new = x.size(1)
            x = x.reshape(N, M, c_new, -1)
            features = x.mean(3).mean(1).cpu().numpy()
            
            # Forward pass output for logits and predictions
            logits = model.fc(x.mean(3).mean(1))
            if isinstance(logits, tuple):
                logits = logits[0]
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            
            all_features.append(features)
            all_labels.append(label.numpy())
            all_predictions.append(preds)
            
            # Save the captured attention maps for the first batch to visualize
            if batch_idx == 0:
                spatial_attentions = list(batch_spatial_att)
                temporal_attentions = list(batch_temporal_att)
                
    # Remove all hooks
    for h in hooks:
        h.remove()
        
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    
    return all_features, all_labels, all_predictions, spatial_attentions, temporal_attentions

def plot_tsne(features, labels, class_names, save_path='visualizations/tsne_features.png', num_classes=10):
    from sklearn.manifold import TSNE
    import seaborn as sns
    
    print("\n[t-SNE] Computing 2D embeddings...")
    unique_classes = np.unique(labels)
    
    # If there are too many classes, select a subset to make the plot clean and readable
    if len(unique_classes) > num_classes:
        # Select the top num_classes classes with the most samples
        class_counts = {c: np.sum(labels == c) for c in unique_classes}
        sorted_classes = sorted(class_counts.keys(), key=lambda x: class_counts[x], reverse=True)
        selected_classes = sorted_classes[:num_classes]
        print(f"[t-SNE] Selecting top {num_classes} classes for cleaner visualization: {selected_classes}")
    else:
        selected_classes = unique_classes
        
    mask = np.isin(labels, selected_classes)
    feat_subset = features[mask]
    label_subset = labels[mask]
    
    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    embeds = tsne.fit_transform(feat_subset)
    
    # Plot using a clean, modern academic aesthetic
    plt.figure(figsize=(10, 8), dpi=300)
    sns.set_theme(style="whitegrid")
    
    palette = sns.color_palette("husl", len(selected_classes))
    
    for i, cls in enumerate(selected_classes):
        cls_mask = label_subset == cls
        cls_name = class_names[cls] if cls < len(class_names) else f"Class {cls}"
        plt.scatter(
            embeds[cls_mask, 0], 
            embeds[cls_mask, 1], 
            label=cls_name,
            alpha=0.85, 
            s=40,
            edgecolors='white',
            linewidths=0.5,
            color=palette[i]
        )
        
    plt.title("t-SNE Visualization of GCN Feature Representations", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Dimension 1", fontsize=12)
    plt.ylabel("Dimension 2", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=10)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[t-SNE] Plot saved successfully to: {save_path}")

def plot_confusion_matrix(y_true, y_pred, class_names, save_path_full='visualizations/confusion_matrix_full.png', save_path_subset='visualizations/confusion_matrix_subset.png'):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    print("\n[Confusion Matrix] Computing confusion matrices...")
    
    # 1. Plot Full 200x200 Confusion Matrix
    cm_full = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10), dpi=300)
    sns.heatmap(cm_full, cmap='Blues', cbar=True, xticklabels=False, yticklabels=False)
    plt.title("Full 200x200 Confusion Matrix", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Predicted Labels", fontsize=12)
    plt.ylabel("True Labels", fontsize=12)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path_full), exist_ok=True)
    plt.savefig(save_path_full, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Confusion Matrix] Full matrix saved to: {save_path_full}")
    
    # 2. Plot Subset Confusion Matrix (12 Classes for high readability in paper)
    unique_classes = np.unique(y_true)
    num_subset = 12
    # Select first 12 unique classes present in the test set
    selected_classes = list(unique_classes[:num_subset])
    
    # Filter true and predicted labels to only include these selected classes
    mask = np.isin(y_true, selected_classes) & np.isin(y_pred, selected_classes)
    y_true_sub = y_true[mask]
    y_pred_sub = y_pred[mask]
    
    if len(y_true_sub) > 0:
        cm_sub = confusion_matrix(y_true_sub, y_pred_sub, labels=selected_classes)
        sub_class_names = [class_names[c] for c in selected_classes]
        
        plt.figure(figsize=(10, 8), dpi=300)
        # Use Oranges or Greens for contrast in papers
        sns.heatmap(
            cm_sub, 
            annot=True, 
            fmt='d', 
            cmap='Oranges', 
            xticklabels=sub_class_names, 
            yticklabels=sub_class_names,
            square=True,
            cbar=True,
            annot_kws={"size": 10}
        )
        plt.title("Confusion Matrix (Subset of 12 Classes)", fontsize=13, fontweight='bold', pad=15)
        plt.xlabel("Predicted Labels", fontsize=11)
        plt.ylabel("True Labels", fontsize=11)
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        
        plt.savefig(save_path_subset, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[Confusion Matrix] Subset matrix saved to: {save_path_subset}")
    else:
        print("[Confusion Matrix] Warning: No samples found for the selected subset of classes.")

def plot_spatial_attention(spatial_atts, save_path='visualizations/spatial_attention.png'):
    plt.figure(figsize=(12, 6), dpi=300)
    
    # Choose 3 layers to show progression (e.g. early, middle, late layer)
    target_indices = [0, len(spatial_atts)//2, len(spatial_atts)-1]
    
    # 27 joints grouped by type for coloring:
    # J0 - J6: Body & Face (Silver/Gray)
    # J7 - J16: Left Hand (Coral/Orange)
    # J17 - J26: Right Hand (Emerald Green)
    joint_colors = []
    for j in range(27):
        if j <= 6:
            joint_colors.append('#bdc3c7')  # Silver/Gray for body/face
        elif j <= 16:
            joint_colors.append('#ff7f50')  # Coral for left hand
        else:
            joint_colors.append('#2ecc71')  # Emerald for right hand
            
    # For labeling the x-axis:
    joint_labels = [f"J{i}" for i in range(27)]
    # Add descriptive markers for key joints
    joint_labels[0] = "Nose (J0)"
    joint_labels[1] = "L-Shld (J1)"
    joint_labels[2] = "R-Shld (J2)"
    joint_labels[5] = "L-Wrist (J5)"
    joint_labels[6] = "R-Wrist (J6)"
    joint_labels[7] = "L-Hand (J7)"
    joint_labels[17] = "R-Hand (J17)"

    for idx, t_idx in enumerate(target_indices):
        if t_idx >= len(spatial_atts):
            continue
        layer_name, att_val = spatial_atts[t_idx]
        mean_att = np.mean(att_val, axis=(0, 1))  # Mean over batch and channel -> [V]
        
        # Normalize to 0-1 for plotting consistency
        mean_att = (mean_att - mean_att.min()) / (mean_att.max() - mean_att.min() + 1e-8)
        
        plt.subplot(1, 3, idx + 1)
        bars = plt.bar(range(27), mean_att, color=joint_colors, edgecolor='black', linewidth=0.5)
        
        plt.title(f"Block: {layer_name.split('.')[0].upper()}", fontsize=11, fontweight='semibold')
        plt.xlabel("Joint Indices", fontsize=9)
        if idx == 0:
            plt.ylabel("Normalized Attention Weight", fontsize=10)
            
        plt.xticks([0, 1, 2, 5, 6, 7, 17, 26], [0, 1, 2, 5, 6, 7, 17, 26], fontsize=8)
        plt.ylim(0, 1.15)
        plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Add custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#bdc3c7', edgecolor='black', label='Body / Face (J0-J6)'),
        Patch(facecolor='#ff7f50', edgecolor='black', label='Left Hand (J7-J16)'),
        Patch(facecolor='#2ecc71', edgecolor='black', label='Right Hand (J17-J26)')
    ]
    plt.figlegend(handles=legend_elements, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.98), fontsize=10)
    plt.suptitle("Spatial Attention Distribution across Skeleton Joint Nodes", fontsize=14, fontweight='bold', y=1.06)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Attention] Spatial attention map saved to: {save_path}")

def plot_temporal_attention(temporal_atts, save_path='visualizations/temporal_attention.png'):
    # temporal_atts is a list of (layer_name, np_array)
    # np_array has shape [N*M, 1, T]
    # We will average over batch to find active frames
    
    plt.figure(figsize=(10, 4), dpi=300)
    
    # We will plot the temporal attention for early, middle, and late layers
    target_indices = [0, len(temporal_atts)//2, len(temporal_atts)-1]
    colors = ['#3498db', '#e74c3c', '#9b59b6']
    
    for idx, t_idx in enumerate(target_indices):
        if t_idx >= len(temporal_atts):
            continue
        layer_name, att_val = temporal_atts[t_idx]
        mean_att = np.mean(att_val, axis=(0, 1))  # Mean over batch and channel -> [T]
        
        # Scale frames to percentage of gesture duration (0% - 100%) since sequence length shrinks
        time_percent = np.linspace(0, 100, len(mean_att))
        
        plt.plot(time_percent, mean_att, label=f"Block {layer_name.split('.')[0].upper()}", color=colors[idx], linewidth=2.0)
        plt.fill_between(time_percent, mean_att, alpha=0.15, color=colors[idx])
        
    plt.title("Temporal Attention Distribution over Gesture Time Duration", fontsize=12, fontweight='bold', pad=15)
    plt.xlabel("Gesture Timeline (%)", fontsize=10)
    plt.ylabel("Attention Weight", fontsize=10)
    plt.xlim(0, 100)
    plt.ylim(0, 1.05)
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Attention] Temporal attention map saved to: {save_path}")

def main():
    config_path = 'ensemble/test_ensemble.yaml'
    print(f"Reading configuration from: {config_path}")
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}.")
        return
        
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    # We will visualize the 'Joint' stream by default
    model_name = config['joint_model']
    model_args = config['joint_model_args']
    weights_path = config['joint_weights']
    feeder_name = config['joint_feeder']
    feeder_args = config['joint_test_feeder_args']
    
    # Load Vietnamese class names from lookup table
    lookuptable_path = 'data/lookuptable.csv'
    class_names = load_class_names(lookuptable_path)
    
    # Load model
    print(f"\nInitializing model {model_name}...")
    Model = import_class(model_name)
    model = Model(**model_args)
    
    # Load weights
    if not os.path.exists(weights_path):
        # Fallback to check evaluate_all approved checkpoints
        print(f"Warning: Checkpoint at {weights_path} not found. Attempting to look in work_dir...")
        from evaluate_all import APPROVED_CHECKPOINTS
        weights_path = APPROVED_CHECKPOINTS.get('joint', weights_path)
        if not os.path.exists(weights_path):
            print(f"Error: Pretrained weights not found at {weights_path}. Please update weights path in test_ensemble.yaml.")
            return

    print(f"Loading pretrained weights from: {weights_path}")
    ckpt = torch.load(weights_path, map_location='cpu')
    state_dict = ckpt['model_state_dict'] if (isinstance(ckpt, dict) and 'model_state_dict' in ckpt) else ckpt
    cleaned_state_dict = {k.replace('module.', '').replace('processor.model.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned_state_dict, strict=False)
    
    # Load dataset
    print(f"\nLoading test dataset with feeder {feeder_name}...")
    Feeder = import_class(feeder_name)
    dataset = Feeder(**feeder_args)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2
    )
    
    # Extract features, predictions, and attention weights
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running extraction on device: {device}")
    
    # Limit to 30 batches for fast execution, can set to None to run on the entire test dataset
    features, labels, predictions, spatial_atts, temporal_atts = extract_features_and_predictions(
        model, data_loader, device=device, limit_batches=30
    )
    
    print(f"\nExtracted Features shape: {features.shape}")
    print(f"Extracted Labels (True) shape: {labels.shape}")
    print(f"Extracted Predictions shape: {predictions.shape}")
    
    # Calculate Overall Accuracy on this subset
    subset_acc = np.mean(predictions == labels) * 100
    print(f"Subset Accuracy: {subset_acc:.2f}%")
    
    # Create Visualizations folder
    os.makedirs('visualizations', exist_ok=True)
    
    # 1. t-SNE Plot
    plot_tsne(features, labels, class_names=class_names, num_classes=10)
    
    # 2. Confusion Matrices (Full and Subset)
    plot_confusion_matrix(labels, predictions, class_names=class_names)
    
    # 3. Spatial Attention Plot
    if spatial_atts:
        plot_spatial_attention(spatial_atts)
    else:
        print("No spatial attention weights captured.")
        
    # 4. Temporal Attention Plot
    if temporal_atts:
        plot_temporal_attention(temporal_atts)
    else:
        print("No temporal attention weights captured.")
        
    print("\n" + "="*60)
    print(" ALL VISUALIZATIONS HAVE BEEN GENERATED SUCCESSFULLY!")
    print(" Check the 'visualizations/' folder for the following files:")
    print("   1. t-SNE Feature Clustered plot:  visualizations/tsne_features.png")
    print("   2. Full Confusion Matrix:          visualizations/confusion_matrix_full.png")
    print("   3. Subset (12 Classes) CF Matrix:   visualizations/confusion_matrix_subset.png")
    print("   4. Spatial Joint Attention:        visualizations/spatial_attention.png")
    print("   5. Temporal Frame Attention:       visualizations/temporal_attention.png")
    print("="*60)

if __name__ == '__main__':
    main()
