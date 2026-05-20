import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from feeders.feeder_27 import Feeder
from model.utils import import_class
from online_inference_pipeline import AdaptiveFusionGate

def get_parser():
    parser = argparse.ArgumentParser(description='Huấn luyện Bộ hợp nhất Trọng số Quyết định Động (AdaptiveFusionGate)')
    parser.add_argument('--data_path', type=str, default='data/data/Thi/processed/27_direct/val_data_joint.npy',
                        help='Đường dẫn file npy tọa độ joint (tập val/test để train fusion)')
    parser.add_argument('--label_path', type=str, default='data/data/Thi/processed/27_direct/val_label.pkl',
                        help='Đường dẫn file pkl nhãn tương ứng')
    parser.add_argument('--num_classes', type=int, default=200, help='Số lượng nhãn phân lớp')
    parser.add_argument('--batch_size', type=int, default=32, help='Kích thước batch')
    parser.add_argument('--lr', type=float, default=1e-3, help='Tốc độ học (learning rate)')
    parser.add_argument('--epochs', type=int, default=20, help='Số epoch huấn luyện')
    parser.add_argument('--device', type=str, default='cuda:0', help='Thiết bị huấn luyện (Vd: cuda:0 hoặc cpu)')
    parser.add_argument('--save_path', type=str, default='work_dir/fusion_gate_best.pt', help='Nơi lưu file trọng số tốt nhất')
    
    # 4 Checkpoint weights của các nhánh
    parser.add_argument('--w_joint', type=str, 
                        default='work_dir/MultiVSL200/Joint/bs32_f150_lr0.1_warmup0/2026-05-19_20-53-05/checkpoints/Joint_best_acc_116_6543.pt',
                        help='Weights của nhánh Joint')
    parser.add_argument('--w_bone', type=str, 
                        default='work_dir/MultiVSL200/Bone/bs32_f150_lr0.1_warmup0/2026-05-19_22-24-43/checkpoints/Bone_best_acc_44_6419.pt',
                        help='Weights của nhánh Bone')
    parser.add_argument('--w_jm', type=str, 
                        default='work_dir/MultiVSL200/Joint_Motion/bs32_f150_lr0.1_warmup0/2026-05-19_22-45-31/checkpoints/Joint_Motion_best_acc_30_2489.pt',
                        help='Weights của nhánh Joint Motion')
    parser.add_argument('--w_bm', type=str, 
                        default='work_dir/MultiVSL200/Bone_Motion/bs32_f150_lr0.1_warmup0/2026-05-19_23-06-18/checkpoints/Bone_Motion_best_acc_41_5226.pt',
                        help='Weights của nhánh Bone Motion')
    return parser

def load_backbone(model_class_path, model_args, weight_path, device):
    """Khởi tạo mô hình và nạp trọng số đã huấn luyện, sau đó đóng băng hoàn toàn"""
    Model = import_class(model_class_path)
    model = Model(**model_args).to(device)
    
    if os.path.exists(weight_path):
        try:
            ckpt = torch.load(weight_path, map_location=device)
            if 'model_state_dict' in ckpt:
                ckpt = ckpt['model_state_dict']
            model.load_state_dict(ckpt)
            print(f"-> Nạp thành công trọng số từ: {weight_path}")
        except Exception as e:
            print(f"-> Cảnh báo: Lỗi khi nạp weights từ {weight_path}: {e}. Khởi tạo ngẫu nhiên.")
    else:
        # Thử kiểm tra đường dẫn tương đối khác
        alternative_path = weight_path.replace('Code/Network/SL_GCN/', '')
        if os.path.exists(alternative_path):
            try:
                ckpt = torch.load(alternative_path, map_location=device)
                if 'model_state_dict' in ckpt:
                    ckpt = ckpt['model_state_dict']
                model.load_state_dict(ckpt)
                print(f"-> Nạp thành công trọng số (đường dẫn thay thế) từ: {alternative_path}")
            except Exception as e:
                print(f"-> Cảnh báo: Lỗi khi nạp weights thay thế từ {alternative_path}: {e}")
        else:
            print(f"-> Cảnh báo: Không tìm thấy checkpoint tại {weight_path}. Khởi tạo ngẫu nhiên.")
            
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Đang sử dụng thiết bị: {device}")
    
    # 1. Cấu hình tham số mô hình xương 27 khớp
    model_class_path = 'model.hand_aware_sl_lgcn.Model'
    model_args = {
        'num_class': args.num_classes, 
        'num_point': 27, 
        'num_person': 1, 
        'graph': 'graph.sign_27.Graph',
        'A_hands': 'graph.sign_27_A_hands.Graph',
        'graph_args': {'labeling_mode': 'spatial'}
    }
    
    # 2. Khởi tạo kết nối xương tĩnh (Bone connections) dùng cho tính toán GPU
    inward_ori_index = [
        (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (9, 11),
        (12, 13), (12, 14), (12, 16), (12, 18), (12, 20),
        (14, 15), (16, 17), (18, 19), (20, 21),
        (22, 23), (22, 24), (22, 26), (22, 28), (22, 30),
        (24, 25), (26, 27), (28, 29), (30, 31),
        (10, 12), (11, 22)
    ]
    bone_conn = torch.zeros(27, dtype=torch.long, device=device)
    for (i, j) in inward_ori_index:
        bone_conn[j - 5] = i - 5

    def compute_bone_gpu(joint_tensor):
        bone = joint_tensor - joint_tensor[:, :, :, bone_conn, :]
        norm = torch.norm(bone, p=2, dim=1, keepdim=True)
        return torch.where(norm > 1e-6, bone / norm, torch.zeros_like(bone))

    def compute_motion_gpu(x):
        motion = torch.zeros_like(x)
        motion[:, :, :-1, :, :] = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
        motion[:, :, -1, :, :] = motion[:, :, -2, :, :]
        return motion

    # 3. Nạp 4 backbone độc lập (đóng băng hoàn toàn)
    print("Nạp các nhánh mô hình backbone...")
    model_joint = load_backbone(model_class_path, model_args, args.w_joint, device)
    model_bone  = load_backbone(model_class_path, model_args, args.w_bone, device)
    model_jm    = load_backbone(model_class_path, model_args, args.w_jm, device)
    model_bm    = load_backbone(model_class_path, model_args, args.w_bm, device)
    
    # 4. Khởi tạo bộ nạp dữ liệu validation/test
    print(f"Đang chuẩn bị bộ nạp dữ liệu từ: {args.data_path}")
    if not os.path.exists(args.data_path) or not os.path.exists(args.label_path):
        print(f"Lỗi: Không tìm thấy file dữ liệu hoặc nhãn tại {args.data_path} hoặc {args.label_path}")
        print("Vui lòng đảm bảo các file npy và pkl tồn tại trước khi chạy.")
        return
        
    dataset = Feeder(data_path=args.data_path, label_path=args.label_path, window_size=150)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    
    # 5. Khởi tạo AdaptiveFusionGate phục vụ tối ưu hóa học tập
    print("Khởi tạo mạng AdaptiveFusionGate...")
    fusion_gate = AdaptiveFusionGate(num_classes=args.num_classes).to(device)
    optimizer = optim.Adam(fusion_gate.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # Định nghĩa NLL Loss trên log-probability của giá trị Softmax kết hợp
    criterion = nn.NLLLoss()
    
    best_acc = 0.0
    os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
    
    print("\n--- BẮT ĐẦU HUẤN LUYỆN ADAPTIVE FUSION GATE ---")
    for epoch in range(1, args.epochs + 1):
        fusion_gate.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        
        # Vòng lặp epoch có progress bar chi tiết
        pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d}/{args.epochs:02d}")
        for batch_data, batch_label, _ in pbar:
            batch_data = batch_data.float().to(device)
            batch_label = batch_label.long().to(device)
            
            # Tính toán 3 luồng còn lại động trực tiếp trên GPU (Không có vòng lặp)
            with torch.no_grad():
                bone_data = compute_bone_gpu(batch_data)
                jm_data   = compute_motion_gpu(batch_data)
                bm_data   = compute_motion_gpu(bone_data)
                
                # Lan truyền xuôi qua 4 mô hình để lấy logit
                logits_joint = model_joint(batch_data)
                logits_bone  = model_bone(bone_data)
                logits_jm    = model_jm(jm_data)
                logits_bm    = model_bm(bm_data)
                
                # Trích xuất Softmax logits của từng luồng
                s_joint = torch.softmax(logits_joint, dim=-1)
                s_bone  = torch.softmax(logits_bone, dim=-1)
                s_jm    = torch.softmax(logits_jm, dim=-1)
                s_bm    = torch.softmax(logits_bm, dim=-1)
                
                # Nối các vector để làm đầu vào cho Gating network
                gate_input = torch.cat([s_joint, s_bone, s_jm, s_bm], dim=-1)
                
            # Lan truyền qua fusion gate thu được trọng số alpha tối ưu
            alpha = fusion_gate(gate_input) # shape: (N, 4)
            
            # Nhân trọng số quyết định động với Softmax logits của từng luồng trước khi cộng hợp lại
            softmax_stacked = torch.stack([s_joint, s_bone, s_jm, s_bm], dim=0) # shape: (4, N, C)
            alpha_unsqueezed = alpha.t().unsqueeze(-1) # shape: (4, N, 1)
            
            # Fused probability distribution
            fused_probs = (softmax_stacked * alpha_unsqueezed).sum(dim=0)
            
            # Tính toán hàm lỗi
            log_fused_probs = torch.log(fused_probs + 1e-15)
            loss = criterion(log_fused_probs, batch_label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Thống kê tiến trình
            total_loss += loss.item() * batch_data.size(0)
            preds = torch.argmax(fused_probs, dim=-1)
            correct += (preds == batch_label).sum().item()
            total_samples += batch_data.size(0)
            
            current_loss = loss.item()
            current_acc = (preds == batch_label).float().mean().item() * 100
            pbar.set_postfix({'loss': f"{current_loss:.4f}", 'acc': f"{current_acc:.2f}%"})
            
        epoch_loss = total_loss / total_samples
        epoch_acc = (correct / total_samples) * 100
        print(f"[*] Kết quả Epoch {epoch:02d}: Loss = {epoch_loss:.5f} | Accuracy = {epoch_acc:.2f}%")
        
        # Lưu checkpoint tốt nhất
        if epoch_acc >= best_acc:
            best_acc = epoch_acc
            state = {
                'epoch': epoch,
                'model_state_dict': fusion_gate.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_acc,
                'loss': epoch_loss
            }
            torch.save(state, args.save_path)
            print(f"--> Đã lưu checkpoint Fusion Gate tốt nhất ({best_acc:.2f}%) tại: {args.save_path}")
            
    print(f"\n[X] HUẤN LUYỆN HOÀN TẤT. Accuracy tốt nhất đạt được: {best_acc:.2f}%")
    print(f"File trọng số đã lưu tại: {args.save_path}")

if __name__ == '__main__':
    main()
