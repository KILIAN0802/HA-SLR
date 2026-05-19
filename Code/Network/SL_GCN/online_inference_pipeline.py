import torch
import torch.nn as nn
import numpy as np
import concurrent.futures
from model.utils import import_class

class CE_GCN_Pipeline:
    def __init__(self, model_class_path, model_args, weight_paths, device='cuda:0'):
        """
        Khởi tạo Pipeline suy luận đa luồng trực tuyến cho 4 nhánh CE-GCN.
        - model_class_path: Đường dẫn import mạng, vd: 'model.hand_aware_sl_lgcn.Model'
        - model_args: Dict chứa các tham số khởi tạo mạng
        - weight_paths: List 4 đường dẫn [Joint_pt, Bone_pt, JM_pt, BM_pt]
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 1. Khởi tạo 4 mô hình đồng thời
        Model = import_class(model_class_path)
        self.models = []
        stream_names = ['Joint', 'Bone', 'Joint Motion', 'Bone Motion']
        
        print("Đang nạp 4 mô hình vào bộ nhớ...")
        for i, w_path in enumerate(weight_paths):
            model = Model(**model_args).to(self.device)
            # Load trọng số đã Clone & Evolve
            if w_path:
                try:
                    ckpt = torch.load(w_path, map_location=self.device)
                    # Support cả file dict checkpoint
                    if 'model_state_dict' in ckpt:
                        ckpt = ckpt['model_state_dict']
                    model.load_state_dict(ckpt)
                    print(f"[{stream_names[i]}] Loaded weights từ: {w_path}")
                except Exception as e:
                    print(f"[{stream_names[i]}] Không thể nạp weights: {e}")
            
            model.eval()
            self.models.append(model)
            
        # 2. Khởi tạo đồ thị xương tĩnh cho hàm tính toán Động học
        inward_ori_index = [
            (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (9, 11),
            (12, 13), (12, 14), (12, 16), (12, 18), (12, 20),
            (14, 15), (16, 17), (18, 19), (20, 21),
            (22, 23), (22, 24), (22, 26), (22, 28), (22, 30),
            (24, 25), (26, 27), (28, 29), (30, 31),
            (10, 12), (11, 22)
        ]
        self.bone_conn = torch.zeros(27, dtype=torch.long)
        for (i, j) in inward_ori_index:
            self.bone_conn[j - 5] = i - 5
            
        # 3. Trọng số Ensemble Late Fusion
        self.alpha = torch.tensor([0.4, 0.3, 0.15, 0.15], device=self.device).view(4, 1, 1)

    def _compute_bone(self, joint_tensor):
        """ Tính tensor Bone trực tiếp trên VRAM """
        # joint_tensor shape: (N, C, T, V, M)
        bone = torch.zeros_like(joint_tensor)
        for v in range(27):
            u = self.bone_conn[v]
            bone[:, :, :, v, :] = joint_tensor[:, :, :, v, :] - joint_tensor[:, :, :, u, :]
        return bone

    def _compute_motion(self, x):
        """ Tính sai phân thời gian bậc nhất """
        # x shape: (N, C, T, V, M)
        motion = torch.zeros_like(x)
        motion[:, :, :-1, :, :] = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
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
        
        # Tự động tính toán ngoại tuyến ra 3 luồng còn lại ngay trên VRAM
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
            
        # Logits shape: (4, 1, num_classes)
        logits = torch.stack(results) 
        
        # Kỹ thuật Late Fusion mềm (Softmax & Sum)
        softmax_logits = torch.softmax(logits, dim=-1)
        fused_scores = (softmax_logits * self.alpha).sum(dim=0) # (1, num_classes)
        
        pred_class = torch.argmax(fused_scores, dim=1).item()
        confidence = torch.max(fused_scores).item()
        
        return pred_class, confidence, fused_scores

# Test run (Dummy code)
if __name__ == '__main__':
    # 1. Định nghĩa cấu hình kiến trúc
    # Lưu ý: Cần chỉnh model_class_path đúng với import path của mã nguồn hiện hành
    model_class_path = 'model.hand_aware_sl_lgcn.Model'
    model_args = {
        'num_class': 200, 
        'num_point': 27, 
        'num_person': 1, 
        'graph': 'graph.sign_27_cvpr.Graph', 
        'graph_args': {'labeling_mode': 'spatial'}
    }
    
    # 2. Cập nhật đường dẫn tới 4 checkpoint vừa train xong
    weight_paths = [
        None, # Thay bằng: 'work_dir/MultiVSL200/Joint/baseline/checkpoints/xxx_best_acc.pt'
        None, 
        None, 
        None
    ]
    
    try:
        pipeline = CE_GCN_Pipeline(model_class_path, model_args, weight_paths)
        
        # Tạo dữ liệu giả lập (Dummy Data) 1 Video: 3 (XYZ), 100 Frames, 27 Khớp, 1 Người
        dummy_joint = torch.randn(1, 3, 100, 27, 1)
        
        # Dự đoán Online
        pred_id, conf, scores = pipeline.predict(dummy_joint)
        print(f"\n[Online Inference] Dự đoán nhãn: {pred_id} | Độ tự tin: {conf*100:.2f}%")
        
    except Exception as e:
        print(f"Chưa thể load mô hình vì cần file trọng số thực: {e}")
