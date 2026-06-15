import torch
import torch.nn as nn
import numpy as np
import concurrent.futures
from model.utils import import_class
import glob
import os

def find_best_checkpoint(base_dir, prefix):
    pattern = os.path.join(base_dir, "**", f"*{prefix}*.pt")
    files = glob.glob(pattern, recursive=True)
    
    filtered_files = []
    for f in files:
        norm_f = f.replace('\\', '/')
        if prefix == 'Joint_best_acc' and 'Joint_Motion' in norm_f:
            continue
        if prefix == 'Bone_best_acc' and 'Bone_Motion' in norm_f:
            continue
        filtered_files.append(f)
        
    if not filtered_files:
        raise FileNotFoundError(f"Không thể tìm thấy checkpoint '{prefix}' trong '{base_dir}'")
        
    best_file = None
    best_acc = -1.0
    
    for f in filtered_files:
        filename = os.path.basename(f)
        if '_acc_' in filename:
            try:
                name_without_ext = filename[:-3] if filename.endswith('.pt') else filename
                parts = name_without_ext.split('_acc_')
                if len(parts) > 1:
                    suffix = parts[1]
                    sub_parts = suffix.split('_')
                    if len(sub_parts) > 1:
                        acc_str = sub_parts[-1]
                        acc_val = float(f"0.{acc_str}")
                    else:
                        acc_val = float(f"0.{suffix}")
                    
                    if acc_val > best_acc:
                        best_acc = acc_val
                        best_file = f
            except Exception:
                pass
                
    if best_file is None:
        filtered_files.sort(key=os.path.getmtime)
        best_file = filtered_files[-1]
        
    return best_file

class AdaptiveFusionGate(nn.Module):
    def __init__(self, num_classes=200):
        """
        Gating network: Kiến trúc phẳng hóa có Regularization mạnh
        để ngăn chặn Overfitting vào phân phối Softmax cực đoan.
        """
        super(AdaptiveFusionGate, self).__init__()
        self.gate_network = nn.Sequential(
            nn.Linear(4 * num_classes, 128),
            nn.BatchNorm1d(128),    # Ổn định phân phối thang đo xác suất
            nn.ReLU(),
            nn.Dropout(p=0.5),      # Ép mạng không được học vẹt các đỉnh xác suất cực đoan
            nn.Linear(128, 4)
        )
        self.T = 2.0  # Temperature scaling để làm mượt phân phối trọng số

    def forward(self, x):
        # Làm mượt phân phối alpha (Smooth weight distribution)
        alpha = self.gate_network(x) / self.T
        alpha = torch.softmax(alpha, dim=-1)  # Cho ra [a1, a2, a3, a4] mượt mà hơn
        return alpha

AdaptiveFusionModule = AdaptiveFusionGate

class CE_GCN_Pipeline:
    def __init__(self, model_class_path, model_args, weight_paths, fusion_weight_path=None, device='cuda:0'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        Model = import_class(model_class_path)
        self.models = []
        stream_names = ['Joint', 'Bone', 'Joint Motion', 'Bone Motion']
        
        print("Đang nạp 4 mô hình vào bộ nhớ GPU...")
        for i, w_path in enumerate(weight_paths):
            model = Model(**model_args).to(self.device)
            if w_path:
                try:
                    ckpt = torch.load(w_path, map_location=self.device)
                    state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
                    cleaned_state_dict = {}
                    for k, v in state_dict.items():
                        name = k.replace('module.', '').replace('processor.model.', '')
                        cleaned_state_dict[name] = v
                    model.load_state_dict(cleaned_state_dict, strict=False)
                    print(f"[{stream_names[i]}] Loaded xịn từ: {w_path}")
                except Exception as e:
                    print(f"[{stream_names[i]}] Lỗi nạp checkpoint: {e}")
            
            model.eval()
            self.models.append(model)
            
        inward_ori_index = [
            (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (9, 11),
            (12, 13), (12, 14), (12, 16), (12, 18), (12, 20),
            (14, 15), (16, 17), (18, 19), (20, 21),
            (22, 23), (22, 24), (22, 26), (22, 28), (22, 30),
            (24, 25), (26, 27), (28, 29), (30, 31),
            (10, 12), (11, 22)
        ]
        self.bone_conn = torch.zeros(27, dtype=torch.long, device=self.device)
        for (i, j) in inward_ori_index:
            self.bone_conn[j - 5] = i - 5
            
        self.fusion_gate = AdaptiveFusionGate(num_classes=model_args.get('num_class', 200)).to(self.device)
        if fusion_weight_path:
            try:
                ckpt = torch.load(fusion_weight_path, map_location=self.device)
                state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
                self.fusion_gate.load_state_dict(state_dict)
                print(f"[Fusion] Loaded Gate thành công từ: {fusion_weight_path}")
            except Exception as e:
                print(f"[Fusion] Lỗi nạp Gate: {e}")
        self.fusion_gate.eval()
        self.last_weights = None

    def _compute_bone(self, joint_tensor):
        bone = joint_tensor - joint_tensor[:, :, :, self.bone_conn, :]
        norm = torch.norm(bone, p=2, dim=1, keepdim=True)
        bone_normalized = torch.where(norm > 1e-6, bone / norm, torch.zeros_like(bone))
        return bone_normalized

    def _compute_motion(self, x):
        motion = torch.zeros_like(x)
        motion[:, :, :-1, :, :] = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
        motion[:, :, -1, :, :] = motion[:, :, -2, :, :]
        return motion

    def _run_single_model(self, model, data):
        with torch.no_grad():
            output = model(data)
            if isinstance(output, tuple):
                output = output[0]
        return output

    def predict(self, joint_frame, mode='dynamic', static_weights=None, static_temperatures=None):
        joint_frame = joint_frame.to(self.device)
        bone_frame = self._compute_bone(joint_frame)
        jm_frame = self._compute_motion(joint_frame)
        bm_frame = self._compute_motion(bone_frame)

        frames = [joint_frame, bone_frame, jm_frame, bm_frame]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self._run_single_model, self.models[i], frames[i]) for i in range(4)]
            results = [f.result() for f in futures]
            
        with torch.no_grad():
            if mode == 'dynamic':
                softmax_logits = [torch.softmax(r, dim=-1) for r in results]
                gate_input = torch.cat(softmax_logits, dim=-1)
                alpha = self.fusion_gate(gate_input)
                
                softmax_stacked = torch.stack(softmax_logits, dim=0)
                alpha_unsqueezed = alpha.t().unsqueeze(-1)
                fused_predictions = (softmax_stacked * alpha_unsqueezed).sum(dim=0)
                self.last_weights = alpha
            else:
                # Chế độ tĩnh: Sử dụng trọng số alpha cố định và hệ số nhiệt độ (Temperature scaling)
                if static_weights is None:
                    static_weights = [0.347, 0.449, 0.061, 0.143]
                if static_temperatures is None:
                    static_temperatures = [1.0, 1.0, 1.0, 1.0]
                
                w = torch.tensor(static_weights, device=self.device, dtype=torch.float32)
                w = w / (w.sum() + 1e-8)
                
                probs = []
                for r, t in zip(results, static_temperatures):
                    probs.append(torch.softmax(r / t, dim=-1))
                    
                softmax_stacked = torch.stack(probs, dim=0)
                w_unsqueezed = w.view(4, 1, 1)
                fused_predictions = (softmax_stacked * w_unsqueezed).sum(dim=0)
                self.last_weights = w
                
        pred_class = torch.argmax(fused_predictions, dim=1).item()
        confidence = torch.max(fused_predictions).item()
        
        return pred_class, confidence, fused_predictions