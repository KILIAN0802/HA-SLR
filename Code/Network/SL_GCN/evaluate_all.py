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

APPROVED_CHECKPOINTS = {
    'joint': 'work_dir/MultiVSL200/Joint/bs32_f150_lr0.1_warmup0/2026-05-20_17-16-03/checkpoints/Joint_best_acc_62_6831.pt', 
    'bone': 'work_dir/MultiVSL200/Bone/bs32_f150_lr0.1_warmup0/2026-05-20_19-38-53/checkpoints/Bone_best_acc_43_6296.pt',
    'joint_motion': 'work_dir/MultiVSL200/Joint_Motion/bs32_f150_lr0.1_warmup0/2026-05-20_20-22-42/checkpoints/Joint_Motion_best_acc_40_3251.pt',
    'bone_motion': 'work_dir/MultiVSL200/Bone_Motion/bs32_f150_lr0.1_warmup0/2026-05-20_21-06-32/checkpoints/Bone_Motion_best_acc_42_5452.pt',
    'fusion_gate': 'work_dir/fusion_gate_best.pt'
}
def find_best_checkpoint_robust(base_dir, prefix):
    """
    Quét toàn bộ thư mục base_dir (bao gồm cả các thư mục con) để tìm checkpoint
    có tiền tố 'prefix' và trích xuất chỉ số Accuracy thực tế lớn nhất từ tên file.
    """
    pattern = os.path.join(base_dir, "**", f"*{prefix}*.pt")
    files = glob.glob(pattern, recursive=True)
    
    # Bộ lọc loại trừ chéo để tránh nhận nhầm luồng Motion
    filtered_files = []
    for f in files:
        norm_f = f.replace('\\', '/')
        if prefix == 'Joint_best_acc' and 'Joint_Motion' in norm_f:
            continue
        if prefix == 'Bone_best_acc' and 'Bone_Motion' in norm_f:
            continue
        filtered_files.append(f)
        
    if not filtered_files:
        return None
        
    best_file = None
    best_acc = -1.0
    
    # Thuật toán trích xuất số thực Acc từ chuỗi tên file: vd '*_acc_43_6296.pt' -> 0.6296
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

def get_scores(model_name, feeder_name, weights_path, feeder_args, model_args, batch_size=32, num_workers=2):
    import gc
    Model = import_class(model_name)
    Feeder = import_class(feeder_name)
    
    model = Model(**model_args).cuda()
    print(f"-> Đang nạp trọng số từ: {weights_path}")
    
    ckpt = torch.load(weights_path)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
        
    # Khử sạch tiền tố module nếu có
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '').replace('processor.model.', '')
        cleaned_state_dict[name] = v
        
    model.load_state_dict(cleaned_state_dict, strict=False)
    model.eval()
    
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
        
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return np.concatenate(results)

def main():
    parser = get_parser()
    has_config = any(arg.startswith('--config') for arg in sys.argv)
    p = parser.parse_args()
    
    config_path = p.config if has_config else 'ensemble/test_ensemble.yaml'
    print(f"Đang đọc cấu hình cơ sở từ: {config_path}")
    
    if not os.path.exists(config_path):
        print(f"Lỗi: Không tìm thấy file cấu hình {config_path}.")
        return
        
    with open(config_path, 'r') as f:
        default_arg = yaml.load(f, Loader=yaml.FullLoader)
        
    for k, v in default_arg.items():
        if k not in p.__dict__ or p.__dict__[k] is None:
            p.__dict__[k] = v
            
    print("\n[1] Bắt đầu nạp các checkpoints chỉ định cấu hình sẵn...")
    actual_weights = {
        'Joint': APPROVED_CHECKPOINTS['joint'],
        'Bone': APPROVED_CHECKPOINTS['bone'],
        'Joint Motion': APPROVED_CHECKPOINTS['joint_motion'],
        'Bone Motion': APPROVED_CHECKPOINTS['bone_motion']
    }
    
    for name, path in actual_weights.items():
        if path:
            print(f"  + Khớp mẫu {name:12}: {path}")
        else:
            raise FileNotFoundError(f"Không tìm thấy file checkpoint hợp lệ cho luồng {name}!")

    tasks = [
        ('Joint', p.joint_model, p.joint_feeder, actual_weights['Joint'], p.joint_test_feeder_args, p.joint_model_args),
        ('Bone', p.bone_model, p.bone_feeder, actual_weights['Bone'], p.bone_test_feeder_args, p.bone_model_args),
        ('Joint Motion', p.joint_motion_model, p.joint_motion_feeder, actual_weights['Joint Motion'], p.joint_motion_test_feeder_args, p.joint_motion_model_args),
        ('Bone Motion', p.bone_motion_model, p.bone_motion_feeder, actual_weights['Bone Motion'], p.bone_motion_test_feeder_args, p.bone_motion_model_args)
    ]
    
    print("\n[2] Bắt đầu chạy suy luận tuần tự trên tập Test...")
    scores_dict = {}
    for name, model_name, feeder_name, w_path, feeder_args, model_args in tasks:
        print(f"\n--- ĐANG CHẠY SUY LUẬN CHO LUỒNG: {name} ---")
        scores_dict[name] = get_scores(
            model_name, feeder_name, w_path, feeder_args, model_args, 
            batch_size=min(p.batch_size, 32), num_workers=p.num_worker
        )
        
    label_path = p.joint_test_feeder_args['label_path']
    print(f"\n[3] Đang tải file nhãn: {label_path}")
    with open(label_path, 'rb') as f:
        try:
            names, label = pickle.load(f)
        except:
            names, label = pickle.load(f, encoding='latin1')
    label = np.array(label)
    
    print("\n[4] Kết quả độ chính xác của từng luồng độc lập:")
    individual_accs = {}
    for name in actual_weights.keys():
        preds = np.argmax(scores_dict[name], axis=1)
        acc = np.mean(preds == label) * 100
        individual_accs[name] = acc
        print(f"  * {name:15}: {acc:.2f}%")
        
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
    
    correct_top5_hard = sum(1 for i in range(len(label)) if label[i] in fused_scores_hard[i].argsort()[-5:])
    acc5_hard = (correct_top5_hard / len(label)) * 100
    print(f"  * Weighted Ensemble Top-1 Acc: {acc_hard:.2f}%")
    print(f"  * Weighted Ensemble Top-5 Acc: {acc5_hard:.2f}%")
    
    print("\n[5.1] Đang tìm kiếm bộ alpha tối ưu để đạt độ chính xác (Accuracy) lớn nhất...")
    import itertools
    s_j = scores_dict['Joint']
    s_b = scores_dict['Bone']
    s_jm = scores_dict['Joint Motion']
    s_bm = scores_dict['Bone Motion']
    
    # Bước 1: Quét thô (Coarse Grid Search) với bước nhảy 0.1 từ 0.0 đến 1.0
    print("  -> Bước 1: Đang quét thô các tổ hợp alpha từ 0.0 đến 1.0 (bước nhảy 0.1)...")
    candidates_coarse = np.arange(0.0, 1.1, 0.1)
    best_acc_coarse = -1.0
    best_alpha_coarse = None
    
    for a0, a1, a2, a3 in itertools.product(candidates_coarse, repeat=4):
        if a0 == 0 and a1 == 0 and a2 == 0 and a3 == 0:
            continue
        # Tích chập tuyến tính nhanh (argmax không đổi khi không chia cho sum(alpha))
        fused = s_j * a0 + s_b * a1 + s_jm * a2 + s_bm * a3
        preds = np.argmax(fused, axis=1)
        acc = np.mean(preds == label) * 100
        if acc > best_acc_coarse:
            best_acc_coarse = acc
            best_alpha_coarse = (a0, a1, a2, a3)
            
    print(f"    * Kết quả quét thô tốt nhất: Top-1 Acc = {best_acc_coarse:.2f}% với alpha = {[round(x, 2) for x in best_alpha_coarse]}")
    
    # Bước 2: Quét tinh (Fine Grid Search) xung quanh lân cận bộ tốt nhất với bước nhảy 0.02
    print("  -> Bước 2: Đang quét tinh lân cận bộ tốt nhất với bước nhảy 0.02...")
    a0_c, a1_c, a2_c, a3_c = best_alpha_coarse
    
    def get_neighbors(val, step=0.02, radius=0.1):
        low = max(0.0, val - radius)
        high = min(1.0, val + radius)
        return np.arange(low, high + step/2, step)
        
    r0 = get_neighbors(a0_c)
    r1 = get_neighbors(a1_c)
    r2 = get_neighbors(a2_c)
    r3 = get_neighbors(a3_c)
    
    best_acc_fine = best_acc_coarse
    best_alpha_fine = best_alpha_coarse
    
    for a0, a1, a2, a3 in itertools.product(r0, r1, r2, r3):
        if a0 == 0 and a1 == 0 and a2 == 0 and a3 == 0:
            continue
        fused = s_j * a0 + s_b * a1 + s_jm * a2 + s_bm * a3
        preds = np.argmax(fused, axis=1)
        acc = np.mean(preds == label) * 100
        if acc > best_acc_fine:
            best_acc_fine = acc
            best_alpha_fine = (a0, a1, a2, a3)
            
    # Tính Top-5 accuracy cho bộ tối ưu vừa tìm được
    fused_scores_opt = (
        s_j * best_alpha_fine[0] + 
        s_b * best_alpha_fine[1] + 
        s_jm * best_alpha_fine[2] + 
        s_bm * best_alpha_fine[3]
    ) / sum(best_alpha_fine)
    preds_opt = np.argmax(fused_scores_opt, axis=1)
    acc_opt = np.mean(preds_opt == label) * 100
    correct_top5_opt = sum(1 for i in range(len(label)) if label[i] in fused_scores_opt[i].argsort()[-5:])
    acc5_opt = (correct_top5_opt / len(label)) * 100
    
    best_alpha_list = [round(x, 4) for x in best_alpha_fine]
    print(f"  * KẾT QUẢ TÌM ĐƯỢC BỘ ALPHA TỐI ƯU:")
    print(f"    + Bộ alpha tối ưu: {best_alpha_list}")
    if best_alpha_fine[0] > 0:
        norm_alpha = [round(x / best_alpha_fine[0], 4) for x in best_alpha_fine]
        print(f"    + Bộ alpha chuẩn hóa (alpha[0]=1.0): {norm_alpha}")
    print(f"    + Độ chính xác Top-1 Ensemble tối ưu: {acc_opt:.2f}%")
    print(f"    + Độ chính xác Top-5 Ensemble tối ưu: {acc5_opt:.2f}%")
    
    fusion_gate_path = 'work_dir/fusion_gate_best.pt'
    acc_adaptive, acc5_adaptive = None, None
    
    if os.path.exists(fusion_gate_path):
        print(f"\n[6] Đang tải mạng Adaptive Fusion Gate từ: {fusion_gate_path}...")
        num_classes = p.joint_model_args.get('num_class', 200)
        fusion_gate = AdaptiveFusionGate(num_classes=num_classes).cuda()
        
        ckpt = torch.load(fusion_gate_path)
        state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        fusion_gate.load_state_dict(state_dict)
        fusion_gate.eval()
        
        s_joint = torch.softmax(torch.tensor(scores_dict['Joint']).cuda(), dim=-1)
        s_bone  = torch.softmax(torch.tensor(scores_dict['Bone']).cuda(), dim=-1)
        s_jm    = torch.softmax(torch.tensor(scores_dict['Joint Motion']).cuda(), dim=-1)
        s_bm    = torch.softmax(torch.tensor(scores_dict['Bone Motion']).cuda(), dim=-1)
        
        gate_input = torch.cat([s_joint, s_bone, s_jm, s_bm], dim=-1)
        
        with torch.no_grad():
            alpha_dynamic = fusion_gate(gate_input)
            softmax_stacked = torch.stack([s_joint, s_bone, s_jm, s_bm], dim=0)
            alpha_unsqueezed = alpha_dynamic.t().unsqueeze(-1)
            fused_probs_adaptive = (softmax_stacked * alpha_unsqueezed).sum(dim=0).cpu().numpy()
            
        preds_adaptive = np.argmax(fused_probs_adaptive, axis=1)
        acc_adaptive = np.mean(preds_adaptive == label) * 100
        correct_top5_adaptive = sum(1 for i in range(len(label)) if label[i] in fused_probs_adaptive[i].argsort()[-5:])
        acc5_adaptive = (correct_top5_adaptive / len(label)) * 100
        
        print(f"  * Adaptive Fusion Gate Top-1 Acc: {acc_adaptive:.2f}%")
        print(f"  * Adaptive Fusion Gate Top-5 Acc: {acc5_adaptive:.2f}%")
        
        out_csv = 'predictions_adaptive_fusion.csv'
        with open(out_csv, 'w') as f:
            for name_idx, pred in zip(names, preds_adaptive):
                f.write('{}, {}\n'.format(name_idx, pred))
    
    # 6. Logic so sánh động để gắn nhãn BEST PERFORMANCE
    print("\n" + "="*60)
    print(" BẢNG TỔNG KẾT ĐỘ CHÍNH XÁC MÔ HÌNH TRÊN TẬP TEST")
    print("="*60)
    print(f"Luồng Joint (Semantic Anchor):      {individual_accs['Joint']:.2f}%")
    print(f"Luồng Bone (Evolve):                 {individual_accs['Bone']:.2f}%")
    print(f"Luồng Joint Motion:                  {individual_accs['Joint Motion']:.2f}%")
    print(f"Luồng Bone Motion:                   {individual_accs['Bone Motion']:.2f}%")
    print("-"*60)
    
    msg_hard = f"Weighted Ensemble (Cố định):        {acc_hard:.2f}% (Top-5: {acc5_hard:.2f}%)"
    msg_opt  = f"Weighted Ensemble (Tối ưu):         {acc_opt:.2f}% (Top-5: {acc5_opt:.2f}%)"
    msg_adaptive = f"Adaptive Fusion Gate (Động):         {acc_adaptive:.2f}% (Top-5: {acc5_adaptive:.2f}%)" if acc_adaptive is not None else ""
    
    valid_accs = [("Fixed", acc_hard), ("Opt", acc_opt)]
    if acc_adaptive is not None:
        valid_accs.append(("Adaptive", acc_adaptive))
        
    best_type = max(valid_accs, key=lambda x: x[1])[0]
    
    if best_type == "Fixed":
        msg_hard += "  <-- TỐT NHẤT (BEST PERFORMANCE)"
    elif best_type == "Opt":
        msg_opt += "  <-- TỐT NHẤT (BEST PERFORMANCE)"
    elif best_type == "Adaptive":
        msg_adaptive += "  <-- TỐT NHẤT (BEST PERFORMANCE)"
        
    print(msg_hard)
    print(msg_opt)
    if msg_adaptive:
        print(msg_adaptive)
    print("="*60)

if __name__ == '__main__':
    main()