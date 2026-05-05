import argparse
import torch
from thop import profile
from thop import clever_format

# Import lớp Model và Graph của bạn
# Đảm bảo rằng các đường dẫn và tên lớp là chính xác
from model.hand_aware_sl_lgcn import Model 
from graph.sign_27_A_hands import Graph

def calculate_metrics_for_model():
    """
    Hàm này khởi tạo mô hình, tạo đầu vào giả và tính toán
    FLOPs và số lượng tham số.
    """
    # 1. Khởi tạo Graph
    # Lớp Graph này sẽ định nghĩa ma trận kề (adjacency matrix) cho mô hình
    graph = Graph()
    A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)

    # 2. Khởi tạo Model
    # Vui lòng kiểm tra và điều chỉnh các tham số này để khớp với cấu hình của bạn
    model = Model(
        num_class=200,          # Số lớp của bộ dữ liệu (ví dụ: 200 cho MultiVSL200)
        num_point=27,           # Số lượng keypoint
        num_person=1,           # Số người trong mỗi video
        graph='graph.sign_27_A_hands.Graph', # SỬA LỖI: Truyền đường dẫn dưới dạng string
        A_hands='graph.sign_27_A_hands.Graph', # Thêm dòng này
        in_channels=3           # Số kênh đầu vào: 3 cho (x, y, confidence)
    )
    
    # Chuyển mô hình sang chế độ đánh giá
    model.eval()

    # 3. Tạo một đầu vào giả (dummy input)
    # Kích thước: (N, C, T, V, M)
    # N: batch_size, C: channels, T: frames, V: vertices (points), M: members (persons)
    dummy_input = torch.randn(1, 3, 100, 27, 1)

    # 4. Sử dụng 'thop' để tính toán MACs và Params
    # MACs (Multiply-Accumulate Operations) xấp xỉ 1/2 FLOPs
    macs, params = profile(model, inputs=(dummy_input, A), verbose=False)

    # FLOPs gần đúng bằng 2 * MACs
    flops = macs * 2

    # 5. Định dạng và in kết quả
    # Sử dụng clever_format để chuyển đổi sang đơn vị G (Giga) và M (Mega)
    flops_str, params_str = clever_format([flops, params], "%.2f")

    print("="*30)
    print("Kết quả cho mô hình HA-SLR:")
    print(f"  - FLOPs: {flops_str}")
    print(f"  - Params: {params_str}")
    print("="*30)

if __name__ == '__main__':
    calculate_metrics_for_model()
