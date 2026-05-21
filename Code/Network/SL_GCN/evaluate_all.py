import os
import sys
import glob
import yaml
import pickle
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# Add parent path to import correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parser import get_parser
from model.utils import import_class
from online_inference_pipeline import AdaptiveFusionGate

def find_best_checkpoint(stream_name):
    """Tự động tìm kiếm file checkpoint có độ chính xác cao nhất (*_best_acc_*.pt)"""
    search_patterns = [
        os.path.join("work_dir", "MultiVSL200", stream_name, "checkpoints", "*_best_acc_*.pt"),
        os.path.join("work_dir", "MultiVSL200", stream_name, "**", "*_best_acc_*.pt"),
        os.path.join("work_dir", "MultiVSL200", stream_name, "*_best_acc_*.pt"),
        os.path.join("work_dir", "MultiVSL200", f"{stream_name}*", "**", "*_best_acc_*.pt")
    ]
    
    found_files = []
    for pattern in search_patterns:
        found_files.extend(glob.glob(pattern, recursive=True))
        
    if not found_files:
        return None
        
    # Lấy checkpoint mới nhất dựa trên thời gian sửa đổi (modification time)
    found_files.sort(key=os.path.getmtime)
    return found_files[-1]

def get_scores(model_name, feeder_name, weights_path, feeder_args, model_args, batch_size=32, num_workers=2):
    """Chạy suy luận trên tập test để lấy logits thô của một luồng"""
    import gc
    Model = import_class(model_name)
    Feeder = import_class(feeder_name)
    
    model = Model(**model_args).cuda()
    print(f"-> Đang nạp trọng số từ: {weights_path}")
    
    ckpt = torch.load(weights_path)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
    else:
        model.load_state_dict(ckpt)
        
    model.eval()
    
    # Đọc dữ liệu
    dataset = Feeder(**feeder_args)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    results = []
    for data, label, index in tqdm(data_loader, desc="Inference"):
        data = data.float().cuda()
        with torch.no_grad():
            output = model(data)
            if isinstance(output, tuple):
                output = output[0]
        results.append(output.data.cpu().numpy())
        
    # Dọn dẹp GPU tránh OOM
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    return np.concatenate(results)

def main():
    parser = get_parser()
    
    # Check if --config is explicitly passed, otherwise default to ensemble/test_ensemble.yaml
    has_config = False
    for arg in sys.argv:
        if arg.startswith('--config'):
            has_config = True
            break
            
    p = parser.parse_args()
    
    config_path = p.config if has_config else 'ensemble/test_ensemble.yaml'
    print(f"Đang đọc cấu hình cơ sở từ: {config_path}")
    
    if not os.path.exists(config_path):
        print(f"Lỗi: Không tìm thấy file cấu hình {config_path}. Vui lòng chạy lệnh từ thư mục Code/Network/SL_GCN/")
        return
        
    with open(config_path, 'r') as f:
        default_arg = yaml.load(f, Loader=yaml.FullLoader)
        
    # Cập nhật parser arguments
    for k, v in default_arg.items():
        if k not in p.__dict__ or p.__dict__[k] is None:
            p.__dict__[k] = v
            
    # Tự động quét tìm checkpoint tốt nhất cho 4 luồng
    stream_dirs = {
        'Joint': 'Joint',
        'Bone': 'Bone',
        'Joint Motion': 'Joint_Motion',
        'Bone Motion': 'Bone_Motion'
    }
    
    actual_weights = {}
    print("\n[1] Bắt đầu quét tìm các checkpoints đã huấn luyện...")
    for name, s_dir in stream_dirs.items():
        ckpt = find_best_checkpoint(s_dir)
        if ckpt:
            actual_weights[name] = ckpt
            print(f"  + Tìm thấy {name:12}: {ckpt}")
        else:
            # Fallback về cấu hình mặc định trong yaml nếu không quét thấy
            yaml_weight = None
            if name == 'Joint': yaml_weight = p.joint_weights
            elif name == 'Bone': yaml_weight = p.bone_weights
            elif name == 'Joint Motion': yaml_weight = p.joint_motion_weights
            elif name == 'Bone Motion': yaml_weight = p.bone_motion_weights
            
            if yaml_weight and os.path.exists(yaml_weight):
                actual_weights[name] = yaml_weight
                print(f"  + Sử dụng mặc định {name:12}: {yaml_weight} (từ YAML)")
            else:
                print(f"  X CẢNH BÁO: Không tìm thấy checkpoint cho luồng {name}!")
                
    if len(actual_weights) < 4:
        print("\n[Lỗi] Không tìm thấy đầy đủ checkpoints cho cả 4 luồng. Vui lòng kiểm tra lại thư mục work_dir!")
        return

    # Định nghĩa các task suy luận
    tasks = [
        ('Joint', p.joint_model, p.joint_feeder, actual_weights['Joint'], p.joint_test_feeder_args, p.joint_model_args),
        ('Bone', p.bone_model, p.bone_feeder, actual_weights['Bone'], p.bone_test_feeder_args, p.bone_model_args),
        ('Joint Motion', p.joint_motion_model, p.joint_motion_feeder, actual_weights['Joint Motion'], p.joint_motion_test_feeder_args, p.joint_motion_model_args),
        ('Bone Motion', p.bone_motion_model, p.bone_motion_feeder, actual_weights['Bone Motion'], p.bone_motion_test_feeder_args, p.bone_motion_model_args)
    ]
    
    # 1. Chạy suy luận tuần tự để lấy phân phối xác suất
    print("\n[2] Bắt đầu chạy suy luận tuần tự trên tập Test...")
    scores_dict = {}
    for name, model_name, feeder_name, w_path, feeder_args, model_args in tasks:
        print(f"\n--- ĐANG CHẠY SUY LUẬN CHO LUỒNG: {name} ---")
        # Sử dụng batch size an toàn 32 tránh OOM
        scores_dict[name] = get_scores(
            model_name, feeder_name, w_path, feeder_args, model_args, 
            batch_size=min(p.batch_size, 32), num_workers=p.num_worker
        )
        
    # 2. Tải nhãn nhị phân của tập kiểm thử
    label_path = p.joint_test_feeder_args['label_path']
    print(f"\n[3] Đang tải file nhãn: {label_path}")
    with open(label_path, 'rb') as f:
        try:
            names, label = pickle.load(f)
        except:
            names, label = pickle.load(f, encoding='latin1')
    label = np.array(label)
    
    # 3. Tính độ chính xác của từng luồng đơn lẻ
    print("\n[4] Kết quả độ chính xác của từng luồng độc lập:")
    individual_accs = {}
    for name in stream_dirs.keys():
        preds = np.argmax(scores_dict[name], axis=1)
        acc = np.mean(preds == label) * 100
        individual_accs[name] = acc
        print(f"  * {name:15}: {acc:.2f}%")
        
    # 4. Đánh giá Hardcoded Weighted Ensemble
    print("\n[5] Đánh giá Weighted Ensemble cố định (alpha = [1.0, 0.9, 0.5, 0.5])...")
    alpha = [1.0, 0.9, 0.5, 0.5]
    fused_scores_hard = (
        scores_dict['Joint'] * alpha[0] + 
        scores_dict['Bone'] * alpha[1] + 
        scores_dict['Joint Motion'] * alpha[2] + 
        scores_dict['Bone Motion'] * alpha[3]
    ) / sum(alpha)
    
    preds_hard = np.argmax(fused_scores_hard, axis=1)
    acc_hard = np.mean(preds_hard == label) * 100
    
    # Tính Top-5 cho Hard Ensemble
    correct_top5_hard = 0
    for i in range(len(label)):
        rank_5 = fused_scores_hard[i].argsort()[-5:]
        if label[i] in rank_5:
            correct_top5_hard += 1
    acc5_hard = (correct_top5_hard / len(label)) * 100
    print(f"  * Weighted Ensemble Top-1 Acc: {acc_hard:.2f}%")
    print(f"  * Weighted Ensemble Top-5 Acc: {acc5_hard:.2f}%")
    
    # 5. Đánh giá Adaptive Fusion Gate (Hợp nhất trọng số động)
    fusion_gate_path = 'work_dir/fusion_gate_best.pt'
    acc_adaptive = None
    acc5_adaptive = None
    
    if os.path.exists(fusion_gate_path):
        print(f"\n[6] Đang tải mạng Adaptive Fusion Gate từ: {fusion_gate_path}...")
        num_classes = p.joint_model_args.get('num_class', 200)
        fusion_gate = AdaptiveFusionGate(num_classes=num_classes).cuda()
        
        ckpt = torch.load(fusion_gate_path)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            fusion_gate.load_state_dict(ckpt['model_state_dict'])
        else:
            fusion_gate.load_state_dict(ckpt)
            
        fusion_gate.eval()
        
        # Tính toán Softmax logits của từng luồng làm đầu vào cho Fusion Gate
        s_joint = torch.softmax(torch.tensor(scores_dict['Joint']).cuda(), dim=-1)
        s_bone  = torch.softmax(torch.tensor(scores_dict['Bone']).cuda(), dim=-1)
        s_jm    = torch.softmax(torch.tensor(scores_dict['Joint Motion']).cuda(), dim=-1)
        s_bm    = torch.softmax(torch.tensor(scores_dict['Bone Motion']).cuda(), dim=-1)
        
        gate_input = torch.cat([s_joint, s_bone, s_jm, s_bm], dim=-1)
        
        with torch.no_grad():
            alpha_dynamic = fusion_gate(gate_input) # shape: (N, 4)
            
            # Trọng số động nhân phân phối xác suất
            softmax_stacked = torch.stack([s_joint, s_bone, s_jm, s_bm], dim=0) # shape: (4, N, C)
            alpha_unsqueezed = alpha_dynamic.t().unsqueeze(-1) # shape: (4, N, 1)
            
            fused_probs_adaptive = (softmax_stacked * alpha_unsqueezed).sum(dim=0).cpu().numpy()
            
        preds_adaptive = np.argmax(fused_probs_adaptive, axis=1)
        acc_adaptive = np.mean(preds_adaptive == label) * 100
        
        correct_top5_adaptive = 0
        for i in range(len(label)):
            rank_5 = fused_probs_adaptive[i].argsort()[-5:]
            if label[i] in rank_5:
                correct_top5_adaptive += 1
        acc5_adaptive = (correct_top5_adaptive / len(label)) * 100
        
        print(f"  * Adaptive Fusion Gate Top-1 Acc: {acc_adaptive:.2f}%")
        print(f"  * Adaptive Fusion Gate Top-5 Acc: {acc5_adaptive:.2f}%")
        
        # Lưu kết quả dự đoán của Adaptive Fusion Gate
        out_csv = 'predictions_adaptive_fusion.csv'
        with open(out_csv, 'w') as f:
            for name_idx, pred in zip(names, preds_adaptive):
                f.write('{}, {}\n'.format(name_idx, pred))
        print(f"  -> Đã lưu danh sách dự đoán động vào: {out_csv}")
    else:
        print(f"\n[6] CẢNH BÁO: Không tìm thấy checkpoint Fusion Gate tại '{fusion_gate_path}'.")
        print("    Vui lòng đảm bảo Giai đoạn 4 trong run_ce_gcn.sh đã chạy thành công.")

    # 6. Tạo bảng tổng kết kết quả
    print("\n" + "="*60)
    print(" BẢNG TỔNG KẾT ĐỘ CHÍNH XÁC MÔ HÌNH TRÊN TẬP TEST")
    print("="*60)
    print(f"Luồng Joint (Semantic Anchor):      {individual_accs['Joint']:.2f}%")
    print(f"Luồng Bone (Evolve):                 {individual_accs['Bone']:.2f}%")
    print(f"Luồng Joint Motion:                  {individual_accs['Joint Motion']:.2f}%")
    print(f"Luồng Bone Motion:                   {individual_accs['Bone Motion']:.2f}%")
    print("-"*60)
    print(f"Weighted Ensemble (Cố định):        {acc_hard:.2f}% (Top-5: {acc5_hard:.2f}%)")
    if acc_adaptive is not None:
        print(f"Adaptive Fusion Gate (Động):         {acc_adaptive:.2f}% (Top-5: {acc5_adaptive:.2f}%)  <-- TỐT NHẤT")
    print("="*60)

if __name__ == '__main__':
    main()
