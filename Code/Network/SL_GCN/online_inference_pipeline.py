import torch
import torch.nn as nn
import numpy as np
import concurrent.futures
from model.utils import import_class

class AdaptiveFusionModule(nn.Module):
    def __init__(self, num_classes=200):
        super(AdaptiveFusionModule, self).__init__()
        # Gating network: nén 4 logit vectors của 4 luồng
        # Input size: 4 * num_classes
        self.fc = nn.Sequential(
            nn.Linear(4 * num_classes, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
        
    def forward(self, logits_list):
        # logits_list: list 4 tensor shape (N, num_classes)
        # Nối đặc trưng lại dọc theo chiều cuối: (N, 4 * num_classes)
        x = torch.cat(logits_list, dim=-1)
        # MLP sinh ra raw weights
        weights = self.fc(x)
        # Kích hoạt Softmax để tổng trọng số động bằng 1.0: (N, 4)
        alpha = torch.softmax(weights, dim=-1)
        
        # Logits_final = alpha_1 * L_1 + alpha_2 * L_2 + alpha_3 * L_3 + alpha_4 * L_4
        logits_stacked = torch.stack(logits_list, dim=0)  # (4, N, num_classes)
        alpha_unsqueezed = alpha.t().unsqueeze(-1)  # (4, N, 1)
        
        # Nhân trọng số quyết định động theo từng mẫu
        fused_logits = (logits_stacked * alpha_unsqueezed).sum(dim=0)
        return fused_logits, alpha

class CE_GCN_Pipeline:
    def __init__(self, model_class_path, model_args, weight_paths, fusion_weight_path=None, device='cuda:0'):
        """
        Khởi tạo Pipeline suy luận đa luồng trực tuyến cho 4 nhánh CE-GCN.
        - model_class_path: Đường dẫn import mạng, vd: 'model.hand_aware_sl_lgcn.Model'
        - model_args: Dict chứa các tham số khởi tạo mạng
        - weight_paths: List 4 đường dẫn [Joint_pt, Bone_pt, JM_pt, BM_pt]
        - fusion_weight_path: Trọng số của AdaptiveFusionModule
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 1. Khởi tạo 4 mô hình đồng thời
        Model = import_class(model_class_path)
        self.models = []
        stream_names = ['Joint', 'Bone', 'Joint Motion', 'Bone Motion']
        
        print("Đang nạp 4 mô hình vào bộ nhớ...")
        for i, w_path in enumerate(weight_paths):
            model = Model(**model_args).to(self.device)
            if w_path:
                try:
                    ckpt = torch.load(w_path, map_location=self.device)
                    if 'model_state_dict' in ckpt:
                        ckpt = ckpt['model_state_dict']
                    model.load_state_dict(ckpt)
                    print(f"[{stream_names[i]}] Loaded weights từ: {w_path}")
                except Exception as e:
                    print(f"[{stream_names[i]}] Không thể nạp weights: {e}")
            
            model.eval()
            self.models.append(model)
            
        # 2. Khởi tạo đồ thị xương tĩnh cho hàm tính toán Động học (VRAM Tensor)
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
            
        # 3. Tích hợp bộ hợp nhất trọng số tự học AdaptiveFusionModule
        num_classes = model_args.get('num_class', 200)
        self.fusion_module = AdaptiveFusionModule(num_classes=num_classes).to(self.device)
        if fusion_weight_path:
            try:
                ckpt = torch.load(fusion_weight_path, map_location=self.device)
                if 'model_state_dict' in ckpt:
                    ckpt = ckpt['model_state_dict']
                self.fusion_module.load_state_dict(ckpt)
                print(f"[Fusion] Loaded weights từ: {fusion_weight_path}")
            except Exception as e:
                print(f"[Fusion] Không thể nạp weights: {e}")
        self.fusion_module.eval()
        self.last_weights = None

    def _compute_bone(self, joint_tensor):
        """ Tính tensor Bone trực tiếp trên VRAM (Khử nhiễu biên độ, Vector hóa 100%) """
        # joint_tensor shape: (N, C, T, V, M)
        # Vectorized subtraction using advanced indexing (Không sử dụng vòng lặp python)
        bone = joint_tensor - joint_tensor[:, :, :, self.bone_conn, :]
        # Chuẩn hóa L2 Normalization dọc theo trục tọa độ (C=3, tức dim 1)
        bone_normalized = bone / (torch.norm(bone, p=2, dim=1, keepdim=True) + 1e-6)
        return bone_normalized

    def _compute_motion(self, x):
        """ Tính sai phân thời gian bậc nhất (Không sử dụng vòng lặp, Sao chép biên Border Replication) """
        # x shape: (N, C, T, V, M)
        motion = torch.zeros_like(x)
        motion[:, :, :-1, :, :] = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
        # Cơ chế sao chép biên (Border Replication) để bảo toàn dòng chảy thời gian
        motion[:, :, -1, :, :] = motion[:, :, -2, :, :]
        return motion

    def _run_single_model(self, model, data):
        """ Hàm bao bọc để chạy forward 1 mô hình """
        with torch.no_grad():
            output = model(data)
            if isinstance(output, tuple):
                output = output[0]  # Bỏ qua loss l1 nếu có
        return output

    def predict(self, joint_frame):
        """
        Thực hiện dự đoán thời gian thực đa luồng
        joint_frame: Tensor tọa độ (1, C, T, 27, 1)
        """
        joint_frame = joint_frame.to(self.device)
        
        # Tự động tính toán trực tiếp ra 3 luồng còn lại ngay trên VRAM (Vectorized)
        bone_frame = self._compute_bone(joint_frame)
        jm_frame = self._compute_motion(joint_frame)
        bm_frame = self._compute_motion(bone_frame)

        frames = [joint_frame, bone_frame, jm_frame, bm_frame]
        
        # Kỹ thuật Online Multi-threading: Đẩy song song 4 tensor vào 4 luồng GPU
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(self._run_single_model, self.models[i], frames[i])
                for i in range(4)
            ]
            results = [f.result() for f in futures]
            
        # Đưa qua bộ hợp nhất trọng số tự học dynamic gating
        with torch.no_grad():
            fused_logits, weights = self.fusion_module(results)
            fused_scores = torch.softmax(fused_logits, dim=-1)
            
        self.last_weights = weights
        pred_class = torch.argmax(fused_scores, dim=1).item()
        confidence = torch.max(fused_scores).item()
        
        return pred_class, confidence, fused_scores

# Test run (Dummy code)
if __name__ == '__main__':
    # 1. Định nghĩa cấu hình kiến trúc
    model_class_path = 'model.hand_aware_sl_lgcn.Model'
    model_args = {
        'num_class': 200, 
        'num_point': 27, 
        'num_person': 1, 
        'graph': 'graph.sign_27.Graph',  # Sử dụng file đồ thị 27 khớp
        'A_hands': 'graph.sign_27_A_hands.Graph',
        'graph_args': {'labeling_mode': 'spatial'}
    }
    
    # 2. Cập nhật đường dẫn tới 4 checkpoint vừa train xong
    weight_paths = [
        'Code/Network/SL_GCN/work_dir/MultiVSL200/Joint/bs32_f150_lr0.1_warmup0/2026-05-19_20-53-05/checkpoints/Joint_best_acc_116_6543.pt',
        'Code/Network/SL_GCN/work_dir/MultiVSL200/Bone/bs32_f150_lr0.1_warmup0/2026-05-19_22-24-43/checkpoints/Bone_best_acc_44_6419.pt',
        'Code/Network/SL_GCN/work_dir/MultiVSL200/Joint_Motion/bs32_f150_lr0.1_warmup0/2026-05-19_22-45-31/checkpoints/Joint_Motion_best_acc_30_2489.pt',
        'Code/Network/SL_GCN/work_dir/MultiVSL200/Bone_Motion/bs32_f150_lr0.1_warmup0/2026-05-19_23-06-18/checkpoints/Bone_Motion_best_acc_41_5226.pt'
    ]
    
    try:
        pipeline = CE_GCN_Pipeline(model_class_path, model_args, weight_paths)
        
        # Tạo dữ liệu giả lập (Dummy Data) 1 Video: 3 (XYZ), 100 Frames, 27 Khớp, 1 Người
        dummy_joint = torch.randn(1, 3, 100, 27, 1)
        
        # Dự đoán Online
        pred_id, conf, scores = pipeline.predict(dummy_joint)
        print(f"\n[Online Inference] Dự đoán nhãn: {pred_id} | Độ tự tin: {conf*100:.2f}%")
        print(f"[Online Inference] Trọng số Fusion động thu được: {pipeline.last_weights.cpu().numpy()[0]}")
        
    except Exception as e:
        import traceback
        print(f"Chưa thể load mô hình vì cần file trọng số thực: {e}")
        traceback.print_exc()
        # Chạy kiểm thử cấu trúc FusionModule & Vectorized Operations độc lập với file checkpoint thực
        print("\n--- BẮT ĐẦU CHẠY KIỂM THỬ KHÔNG CẦN CHECKPOINT THỰC ---")
        try:
            # Khởi tạo pipeline không load trọng số thực
            pipeline_dummy = CE_GCN_Pipeline(model_class_path, model_args, [None, None, None, None])
            dummy_joint = torch.randn(1, 3, 100, 27, 1)
            pred_id, conf, scores = pipeline_dummy.predict(dummy_joint)
            print(f"[Dummy Test Success] Dự đoán thành công nhãn: {pred_id} | Độ tự tin: {conf*100:.2f}%")
            print(f"[Dummy Test Success] Trọng số Fusion động thu được: {pipeline_dummy.last_weights.cpu().numpy()[0]}")
        except Exception as dummy_err:
            print(f"Lỗi kiểm thử cấu trúc: {dummy_err}")
            traceback.print_exc()
