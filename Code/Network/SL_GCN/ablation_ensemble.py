"""
ablation_ensemble.py
====================
Chạy ablation study ensemble trên tập test MultiVSL.
Thử mọi tổ hợp 2 luồng (C(4,2)=6) và 3 luồng (C(4,3)=4),
rồi in bảng so sánh Top-1 / Top-5 đầy đủ.

Cách dùng:
    cd Code/Network/SL_GCN
    python ablation_ensemble.py --config ensemble/test_ensemble.yaml

Tuỳ chọn thêm:
    --num_worker 4          # số worker DataLoader
    --batch_size 32         # batch size inference
    --alpha_mode equal      # equal | grid | custom
    --custom_alpha 1.0 0.9 0.5 0.5  # chỉ dùng khi alpha_mode=custom
    --cache_path work_dir/ablation_cache.pkl   # lưu/tái sử dụng logit đã tính
    --save_csv results/ablation_results.csv    # xuất bảng kết quả ra CSV
"""

import os, sys, argparse, pickle, itertools, csv
import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parser import get_parser
from model.utils import import_class

# ─────────────────────────────────────────
#  Tiện ích
# ─────────────────────────────────────────
def softmax_np(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def top1(scores, labels):
    return float(np.mean(np.argmax(scores, axis=1) == labels) * 100)

def top5(scores, labels):
    n = len(labels)
    hit = sum(1 for i in range(n) if labels[i] in scores[i].argsort()[-5:])
    return float(hit / n * 100)

def coarse_grid_search(streams_data, labels, step=0.1):
    """Tìm alpha tốt nhất qua Grid Search thô rồi tinh trên Softmax."""
    n = len(streams_data)
    candidates = np.arange(0.0, 1.0 + step / 2, step)
    probs = [softmax_np(d) for d in streams_data]

    best_acc, best_alpha = -1.0, None
    for combo in itertools.product(candidates, repeat=n):
        if sum(combo) == 0:
            continue
        w = np.array(combo)
        fused = sum(p * wi for p, wi in zip(probs, w)) / w.sum()
        acc = top1(fused, labels)
        if acc > best_acc:
            best_acc, best_alpha = acc, combo

    # Quét tinh quanh best_alpha (bước 0.02, bán kính 0.1)
    def neighbors(v, fine_step=0.02, radius=0.1):
        lo, hi = max(0.0, v - radius), min(1.0, v + radius)
        return np.arange(lo, hi + fine_step / 2, fine_step)

    ranges = [neighbors(a) for a in best_alpha]
    for combo in itertools.product(*ranges):
        if sum(combo) == 0:
            continue
        w = np.array(combo)
        fused = sum(p * wi for p, wi in zip(probs, w)) / w.sum()
        acc = top1(fused, labels)
        if acc > best_acc:
            best_acc, best_alpha = acc, combo

    return best_acc, best_alpha


# ─────────────────────────────────────────
#  Inference một luồng
# ─────────────────────────────────────────
def run_inference(model_path, feeder_class_path, feeder_args, model_class_path, model_args, batch_size, num_workers):
    import gc
    Model  = import_class(model_class_path)
    Feeder = import_class(feeder_class_path)

    model = Model(**model_args).cuda()
    ckpt  = torch.load(model_path, map_location='cuda', weights_only=False)

    state_dict = (ckpt['model_state_dict']
                  if isinstance(ckpt, dict) and 'model_state_dict' in ckpt
                  else ckpt)
    cleaned = {k.replace('module.', '').replace('processor.model.', ''): v
               for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=False)
    model.eval()

    loader = torch.utils.data.DataLoader(
        Feeder(**feeder_args),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    all_out = []
    for data, _, _ in tqdm(loader, desc='  inference', leave=False):
        data = data.float().cuda()
        with torch.no_grad():
            out = model(data)
            if isinstance(out, tuple):
                out = out[0]
        all_out.append(out.cpu().numpy())

    del model
    torch.cuda.empty_cache(); gc.collect()
    return np.concatenate(all_out)          # (N, num_class)


# ─────────────────────────────────────────
#  Main
# ─────────────────────────────────────────
def main():
    # ── 1. Đọc tham số ────────────────────────────────────────
    parser = get_parser()
    parser.add_argument('--alpha_mode', default='grid',
                        choices=['equal', 'grid', 'custom'],
                        help='Chiến lược chọn alpha: '
                             'equal=đều nhau, grid=tìm lưới, custom=chỉ định tay')
    parser.add_argument('--custom_alpha', nargs='+', type=float,
                        default=None,
                        help='Bộ alpha tùy chọn theo thứ tự J B JM BM '
                             '(chỉ dùng khi alpha_mode=custom)')
    parser.add_argument('--cache_path', default='',
                        help='Đường dẫn file cache logit (.pkl). '
                             'Nếu tồn tại sẽ bỏ qua inference.')
    parser.add_argument('--save_csv', default='',
                        help='Xuất bảng kết quả ra file CSV.')

    has_cfg = any(a.startswith('--config') for a in sys.argv)
    p = parser.parse_args()
    cfg_path = p.config if has_cfg else 'ensemble/test_ensemble.yaml'

    if not os.path.exists(cfg_path):
        sys.exit(f'[ERROR] Không tìm thấy config: {cfg_path}')
    with open(cfg_path) as f:
        default_arg = yaml.safe_load(f)
    for k, v in default_arg.items():
        if k not in p.__dict__ or p.__dict__[k] is None:
            p.__dict__[k] = v

    # ── 2. Xây dựng danh sách 4 luồng ────────────────────────
    STREAMS = {
        'Joint':       dict(model=p.joint_model,       feeder=p.joint_feeder,
                            weights=p.joint_weights,   feeder_args=p.joint_test_feeder_args,
                            model_args=p.joint_model_args),
        'Bone':        dict(model=p.bone_model,        feeder=p.bone_feeder,
                            weights=p.bone_weights,    feeder_args=p.bone_test_feeder_args,
                            model_args=p.bone_model_args),
        'Joint Motion':dict(model=p.joint_motion_model, feeder=p.joint_motion_feeder,
                            weights=p.joint_motion_weights,
                            feeder_args=p.joint_motion_test_feeder_args,
                            model_args=p.joint_motion_model_args),
        'Bone Motion': dict(model=p.bone_motion_model, feeder=p.bone_motion_feeder,
                            weights=p.bone_motion_weights,
                            feeder_args=p.bone_motion_test_feeder_args,
                            model_args=p.bone_motion_model_args),
    }
    ALL_NAMES = list(STREAMS.keys())

    # ── 3. Tải nhãn ───────────────────────────────────────────
    label_path = p.joint_test_feeder_args['label_path']
    with open(label_path, 'rb') as f:
        try:    _, labels = pickle.load(f)
        except: _, labels = pickle.load(f, encoding='latin1')
    labels = np.array(labels)
    print(f'[INFO] Số mẫu test: {len(labels)}  |  Số lớp: {labels.max()+1}')

    # ── 4. Inference / load cache ─────────────────────────────
    scores = {}
    cache_ok = bool(p.cache_path) and os.path.exists(p.cache_path)
    if cache_ok:
        print(f'[INFO] Tải logit từ cache: {p.cache_path}')
        with open(p.cache_path, 'rb') as f:
            scores = pickle.load(f)
        missing = [n for n in ALL_NAMES if n not in scores]
        if missing:
            print(f'[WARN] Cache thiếu luồng: {missing}. Sẽ chạy inference bổ sung.')
            cache_ok = False      # cần chạy lại đủ

    if not cache_ok:
        print('\n[STEP 1] Chạy inference cho 4 luồng...')
        for name, cfg in STREAMS.items():
            if name in scores:
                print(f'  > {name}: đã có trong cache, bỏ qua.')
                continue
            print(f'\n  > Luồng: {name}')
            print(f'    Checkpoint : {cfg["weights"]}')
            print(f'    Data       : {cfg["feeder_args"]["data_path"]}')
            scores[name] = run_inference(
                model_path        = cfg['weights'],
                feeder_class_path = cfg['feeder'],
                feeder_args       = cfg['feeder_args'],
                model_class_path  = cfg['model'],
                model_args        = cfg['model_args'],
                batch_size        = min(getattr(p, 'batch_size', 32), 32),
                num_workers       = getattr(p, 'num_worker', 2),
            )
        if p.cache_path:
            os.makedirs(os.path.dirname(p.cache_path) or '.', exist_ok=True)
            with open(p.cache_path, 'wb') as f:
                pickle.dump(scores, f)
            print(f'\n[INFO] Đã lưu cache tại: {p.cache_path}')

    # ── 5. Kết quả từng luồng đơn ─────────────────────────────
    print('\n' + '='*68)
    print(' [A] ĐỘ CHÍNH XÁC TỪNG LUỒNG ĐƠN LẺ')
    print('='*68)
    single_results = {}
    for name in ALL_NAMES:
        t1 = top1(scores[name], labels)
        t5 = top5(scores[name], labels)
        single_results[name] = (t1, t5)
        print(f'  {name:<15}: Top-1 = {t1:.2f}%  |  Top-5 = {t5:.2f}%')

    # ── 6. Hàm ensemble chung ─────────────────────────────────
    def ensemble_subset(names_subset, alpha_mode, custom_alpha=None):
        """
        Trả về (top1_acc, top5_acc, alpha_used) cho một tập con luồng.
        """
        data_list = [scores[n] for n in names_subset]
        n = len(names_subset)

        if alpha_mode == 'equal':
            w = np.ones(n) / n
            fused = sum(softmax_np(d) * wi for d, wi in zip(data_list, w))
            return top1(fused, labels), top5(fused, labels), list(w)

        elif alpha_mode == 'custom':
            # Dùng alpha toàn cục, lấy đúng vị trí của luồng trong bộ 4
            full_alpha = custom_alpha or [1.0]*4
            idx_map = {n: i for i, n in enumerate(ALL_NAMES)}
            w = np.array([full_alpha[idx_map[nm]] for nm in names_subset])
            w = w / (w.sum() + 1e-8)
            fused = sum(softmax_np(d) * wi for d, wi in zip(data_list, w))
            return top1(fused, labels), top5(fused, labels), list(w)

        else:  # grid
            best_t1, best_alpha = coarse_grid_search(data_list, labels)
            w = np.array(best_alpha)
            w_norm = w / (w.sum() + 1e-8)
            fused = sum(softmax_np(d) * wi for d, wi in zip(data_list, w_norm))
            return best_t1, top5(fused, labels), [round(float(a), 4) for a in best_alpha]

    # ── 7. Tổ hợp 2 luồng (C(4,2) = 6) ──────────────────────
    print('\n' + '='*68)
    print(' [B] ABLATION: CẶP 2 LUỒNG  (6 tổ hợp)')
    print('='*68)
    pair_results = []
    for combo in itertools.combinations(ALL_NAMES, 2):
        print(f'  Đang tính: {" + ".join(combo)} ...', end='', flush=True)
        t1, t5, alpha = ensemble_subset(list(combo), p.alpha_mode, p.custom_alpha)
        pair_results.append(dict(streams=combo, top1=t1, top5=t5, alpha=alpha))
        print(f'  Top-1 = {t1:.2f}%  |  Top-5 = {t5:.2f}%  |  α = {alpha}')
    pair_results.sort(key=lambda x: -x['top1'])

    print(f'\n  >> TỐT NHẤT trong cặp 2: '
          f'{" + ".join(pair_results[0]["streams"])}  '
          f'→ Top-1 = {pair_results[0]["top1"]:.2f}%')

    # ── 8. Tổ hợp 3 luồng (C(4,3) = 4) ──────────────────────
    print('\n' + '='*68)
    print(' [C] ABLATION: BỘ 3 LUỒNG  (4 tổ hợp)')
    print('='*68)
    triple_results = []
    for combo in itertools.combinations(ALL_NAMES, 3):
        print(f'  Đang tính: {" + ".join(combo)} ...', end='', flush=True)
        t1, t5, alpha = ensemble_subset(list(combo), p.alpha_mode, p.custom_alpha)
        triple_results.append(dict(streams=combo, top1=t1, top5=t5, alpha=alpha))
        print(f'  Top-1 = {t1:.2f}%  |  Top-5 = {t5:.2f}%  |  α = {alpha}')
    triple_results.sort(key=lambda x: -x['top1'])

    print(f'\n  >> TỐT NHẤT trong bộ 3: '
          f'{" + ".join(triple_results[0]["streams"])}  '
          f'→ Top-1 = {triple_results[0]["top1"]:.2f}%')

    # ── 9. Bộ đầy đủ 4 luồng ─────────────────────────────────
    print('\n' + '='*68)
    print(' [D] BỘ ĐẦY ĐỦ 4 LUỒNG')
    print('='*68)
    t1_full, t5_full, alpha_full = ensemble_subset(ALL_NAMES, p.alpha_mode, p.custom_alpha)
    print(f'  Joint + Bone + JointMotion + BoneMotion')
    print(f'  Top-1 = {t1_full:.2f}%  |  Top-5 = {t5_full:.2f}%  |  α = {alpha_full}')

    # ── 10. BẢNG TỔNG KẾT ─────────────────────────────────────
    all_rows = []
    # Luồng đơn
    for name in ALL_NAMES:
        t1, t5 = single_results[name]
        all_rows.append({'combo': name, 'size': 1, 'top1': t1, 'top5': t5, 'alpha': [1.0]})
    # Cặp 2
    for r in pair_results:
        all_rows.append({'combo': ' + '.join(r['streams']), 'size': 2,
                         'top1': r['top1'], 'top5': r['top5'], 'alpha': r['alpha']})
    # Bộ 3
    for r in triple_results:
        all_rows.append({'combo': ' + '.join(r['streams']), 'size': 3,
                         'top1': r['top1'], 'top5': r['top5'], 'alpha': r['alpha']})
    # Bộ 4
    all_rows.append({'combo': 'All 4 streams', 'size': 4,
                     'top1': t1_full, 'top5': t5_full, 'alpha': alpha_full})

    all_rows.sort(key=lambda x: -x['top1'])

    print('\n' + '='*68)
    print(' BẢNG TỔNG KẾT — xếp theo Top-1 giảm dần')
    print('='*68)
    print(f'  {"Tổ hợp":<42} {"#":<3} {"Top-1":>7} {"Top-5":>7}')
    print('  ' + '-'*64)
    best_top1 = all_rows[0]['top1']
    for row in all_rows:
        flag = ' ← BEST' if abs(row['top1'] - best_top1) < 0.001 else ''
        print(f'  {row["combo"]:<42} {row["size"]:<3} {row["top1"]:>6.2f}% {row["top5"]:>6.2f}%{flag}')
    print('='*68)

    # ── 11. Xuất CSV (tuỳ chọn) ───────────────────────────────
    if p.save_csv:
        os.makedirs(os.path.dirname(p.save_csv) or '.', exist_ok=True)
        with open(p.save_csv, 'w', newline='', encoding='utf-8') as csvf:
            writer = csv.DictWriter(csvf,
                fieldnames=['combo', 'num_streams', 'top1', 'top5', 'alpha'])
            writer.writeheader()
            for row in all_rows:
                writer.writerow({
                    'combo': row['combo'],
                    'num_streams': row['size'],
                    'top1': f"{row['top1']:.4f}",
                    'top5': f"{row['top5']:.4f}",
                    'alpha': str(row['alpha'])
                })
        print(f'\n[INFO] Đã lưu kết quả CSV tại: {p.save_csv}')


if __name__ == '__main__':
    main()
