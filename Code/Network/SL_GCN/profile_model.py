import os
import sys
import time
import torch
import numpy as np

# Thêm đường dẫn thư mục hiện tại vào sys.path để import đúng
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.hand_aware_sl_lgcn import Model

def get_model_and_input():
    # Tham số cấu hình lấy từ ensemble/test_ensemble.yaml
    num_class = 200
    num_point = 27
    num_person = 1
    graph_class_name = 'graph.sign_27.Graph'
    a_hands_class_name = 'graph.sign_27_A_hands.Graph'
    graph_args = {'labeling_mode': 'spatial'}
    
    # Khởi tạo mô hình
    model = Model(
        num_class=num_class,
        num_point=num_point,
        num_person=num_person,
        graph=graph_class_name,
        A_hands=a_hands_class_name,
        graph_args=graph_args
    )
    model.eval()
    
    # Kích thước dữ liệu đầu vào chuẩn (N, C, T, V, M)
    # N = 1 (batch size = 1 để đo thời gian/FPS trên từng mẫu riêng lẻ)
    # C = 3 (Tọa độ X, Y, Z)
    # T = 150 (Window size / Số frame của một video)
    # V = 27 (Số lượng keypoints của khung xương)
    # M = 1 (Số người)
    batch_size = 1
    in_channels = 3
    window_size = 150
    num_point = 27
    num_person = 1
    
    x = torch.randn(batch_size, in_channels, window_size, num_point, num_person)
    return model, x, window_size

def calculate_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def estimate_flops_hooks(model, x):
    """
    Ước tính số FLOPS bằng cách sử dụng các Hooks của PyTorch để theo dõi hoạt động của các lớp.
    Độ chính xác cao và không phụ thuộc vào bất kỳ thư viện bên ngoài nào.
    """
    flops_dict = {'conv': 0, 'linear': 0, 'bn': 0, 'gcn_matmul': 0, 'relu': 0, 'others': 0}
    hooks = []
    
    def conv2d_hook(module, input, output):
        batch_size = input[0].size(0)
        out_h, out_w = output.size(2), output.size(3)
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels
        bias_ops = 1 if module.bias is not None else 0
        # Nhân 2 vì mỗi phép MAC (Multiply-Accumulate) gồm 1 phép nhân + 1 phép cộng
        total_ops = (2 * kernel_ops + bias_ops) * module.out_channels * out_h * out_w * batch_size
        flops_dict['conv'] += total_ops

    def conv1d_hook(module, input, output):
        batch_size = input[0].size(0)
        out_len = output.size(2)
        kernel_ops = module.kernel_size[0] * module.in_channels
        bias_ops = 1 if module.bias is not None else 0
        total_ops = (2 * kernel_ops + bias_ops) * module.out_channels * out_len * batch_size
        flops_dict['conv'] += total_ops

    def linear_hook(module, input, output):
        batch_size = input[0].size(0) if input[0].dim() > 1 else 1
        in_features = module.in_features
        out_features = module.out_features
        bias_ops = 1 if module.bias is not None else 0
        total_ops = (2 * in_features + bias_ops) * out_features * batch_size
        flops_dict['linear'] += total_ops

    def bn_hook(module, input, output):
        # BN chuẩn hóa: y = (x - mean) / sqrt(var + eps) * weight + bias
        # Ở chế độ eval, mean/var đã được tính trước, tương đương ~2 FLOPS trên mỗi phần tử
        num_elements = input[0].numel()
        flops_dict['bn'] += 2 * num_elements

    def gcn_hook(module, input, output):
        # Trong unit_gcn: 
        # z = torch.matmul(f.view(N, C * T, V), A[i]).view(N, C, T, V)
        # Thực hiện nhân ma trận (C * T, V) x (V, V) cho mỗi subset (3 subset)
        x_in = input[0]
        N, C, T, V = x_in.size()
        # Phép nhân ma trận cỡ (C*T, V) với (V, V) tốn 2 * C * T * V * V FLOPS cho mỗi subset
        total_ops = module.num_subset * (2 * V * V * C * T) * N
        flops_dict['gcn_matmul'] += total_ops

    def relu_hook(module, input, output):
        num_elements = input[0].numel()
        flops_dict['relu'] += num_elements

    # Đăng ký hook cho các lớp tương ứng
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            hooks.append(module.register_forward_hook(conv2d_hook))
        elif isinstance(module, torch.nn.Conv1d):
            hooks.append(module.register_forward_hook(conv1d_hook))
        elif isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            hooks.append(module.register_forward_hook(bn_hook))
        elif module.__class__.__name__ == 'unit_gcn':
            hooks.append(module.register_forward_hook(gcn_hook))
        elif isinstance(module, torch.nn.ReLU):
            hooks.append(module.register_forward_hook(relu_hook))

    with torch.no_grad():
        _ = model(x)

    for hook in hooks:
        hook.remove()

    total_flops = sum(flops_dict.values())
    return total_flops, flops_dict

def measure_fps(model, x, device_name, window_size, warmup_iters=50, num_iters=200):
    device = torch.device(device_name)
    model = model.to(device)
    x = x.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(x)
            
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    start_time = time.perf_counter()
    
    with torch.no_grad():
        for _ in range(num_iters):
            _ = model(x)
            
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    avg_time_per_sample = total_time / num_iters
    
    # FPS định nghĩa theo 2 cách để tránh gây hiểu nhầm:
    # 1. Số video (mẫu) xử lý được trên giây (Samples per second)
    samples_per_sec = 1.0 / avg_time_per_sample
    # 2. Số khung hình (frames) xử lý được trên giây (Frames per second)
    frames_per_sec = window_size / avg_time_per_sample
    
    return samples_per_sec, frames_per_sec, avg_time_per_sample

def main():
    print("="*70)
    print(" BẮT ĐẦU ĐO PARAMS, FLOPS VÀ FPS CHO MÔ HÌNH HA-SLR-GCN ")
    print("="*70)
    
    model, x, window_size = get_model_and_input()
    
    # 1. ĐO TỔNG SỐ THAM SỐ (PARAMS)
    total_params, trainable_params = calculate_params(model)
    print("\n[1] SỐ LƯỢNG THAM SỐ (PARAMS):")
    print(f"  * Tổng số tham số (Total Params):      {total_params:,} ({total_params/1e6:.3f} M)")
    print(f"  * Tham số cần huấn luyện (Trainable):  {trainable_params:,} ({trainable_params/1e6:.3f} M)")
    
    # 2. ĐO SỐ PHÉP TÍNH SỐ THỰC (FLOPS)
    print("\n[2] PHÉP TÍNH SỐ THỰC (FLOPS / MACs):")
    
    # Cách A: Dùng Hook-based Custom Estimator (an toàn, chi tiết)
    hook_flops, flops_breakdown = estimate_flops_hooks(model, x)
    print("  * Chi tiết FLOPs ước tính từ Hook (Forward Pass):")
    print(f"    - Các lớp Convolution (1D/2D):     {flops_breakdown['conv']:,} ({flops_breakdown['conv']/1e9:.4f} GFLOPs)")
    print(f"    - Các lớp GCN MatMul (Adjacency):  {flops_breakdown['gcn_matmul']:,} ({flops_breakdown['gcn_matmul']/1e9:.4f} GFLOPs)")
    print(f"    - Các lớp Tuyến tính (Linear):     {flops_breakdown['linear']:,} ({flops_breakdown['linear']/1e9:.4f} GFLOPs)")
    print(f"    - Các lớp BatchNorm (1D/2D):       {flops_breakdown['bn']:,} ({flops_breakdown['bn']/1e9:.4f} GFLOPs)")
    print(f"    - Các lớp Kích hoạt (ReLU):        {flops_breakdown['relu']:,} ({flops_breakdown['relu']/1e9:.4f} GFLOPs)")
    print(f"    => TỔNG CỘNG (Estimated FLOPs):    {hook_flops:,} ({hook_flops/1e9:.4f} GFLOPs)")
    
    # Cách B: Dùng PyTorch Native FlopCounter (nếu khả dụng)
    try:
        from torch.utils.flop_counter import FlopCounterMode
        flop_counter = FlopCounterMode(display=False)
        with flop_counter:
            with torch.no_grad():
                _ = model(x)
        
        def recursive_sum(d):
            total = 0
            if isinstance(d, dict) or hasattr(d, 'items'):
                for k, v in d.items():
                    total += recursive_sum(v)
            elif isinstance(d, (int, float)):
                total += d
            return total
            
        native_flops = recursive_sum(flop_counter.flop_counts)
        print(f"  * Tổng FLOPs từ PyTorch Native FlopCounter: {native_flops:,} ({native_flops/1e9:.4f} GFLOPs)")
    except Exception as e:
        print(f"  * PyTorch Native FlopCounter không khả dụng hoặc lỗi: {e}")
        
    # 3. ĐO TỐC ĐỘ SUY LUẬN (FPS)
    print("\n[3] TỐC ĐỘ SUY LUẬN (INFERENCE SPEED & FPS):")
    print("  * Đang chạy benchmark trên CPU...")
    cpu_samples, cpu_fps, cpu_time = measure_fps(model, x, 'cpu', window_size)
    print(f"    - Thời gian trung bình / 1 video:  {cpu_time*1000:.2f} ms")
    print(f"    - Tốc độ xử lý video (Video/s):     {cpu_samples:.2f} samples/s")
    print(f"    - Tốc độ xử lý khung hình (FPS):    {cpu_fps:.2f} frames/s")
    
    if torch.cuda.is_available():
        print("  * Đang chạy benchmark trên GPU (CUDA)...")
        gpu_samples, gpu_fps, gpu_time = measure_fps(model, x, 'cuda', window_size)
        device_name = torch.cuda.get_device_name(0)
        print(f"    - Thiết bị GPU:                     {device_name}")
        print(f"    - Thời gian trung bình / 1 video:  {gpu_time*1000:.2f} ms")
        print(f"    - Tốc độ xử lý video (Video/s):     {gpu_samples:.2f} samples/s")
        print(f"    - Tốc độ xử lý khung hình (FPS):    {gpu_fps:.2f} frames/s")
    else:
        print("  * GPU (CUDA) không khả dụng trên máy này.")
        
    print("="*70)

if __name__ == '__main__':
    main()
