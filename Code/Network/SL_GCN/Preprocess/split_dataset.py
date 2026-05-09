import os
import argparse

def split_data(input_csv, output_dir):
    # Đọc tất cả các dòng từ file labels.csv tổng
    with open(input_csv, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    train_lines = []
    val_lines = []
    test_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Tên file video: '31_Toan_..._173'
        filename = line.split(',')[0]
        
        try:
            # Lấy Signer ID (2 chữ số đầu tiên)
            signer_id = int(filename[:2])
            
            # Chia tập dựa trên Signer ID (Signer-Independent)
            if 1 <= signer_id <= 24:
                train_lines.append(line)
            elif 25 <= signer_id <= 27:
                val_lines.append(line)
            elif 28 <= signer_id <= 31:
                test_lines.append(line)
            else:
                # Nếu có ID khác ngoài 1-31, cho vào train
                train_lines.append(line)
        except:
            # Nếu không parse được ID, mặc định cho vào train
            train_lines.append(line)

    # Ghi ra các file tương ứng
    with open(os.path.join(output_dir, 'train_labels.csv'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_lines))
    
    with open(os.path.join(output_dir, 'val_labels.csv'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_lines))
        
    with open(os.path.join(output_dir, 'test_labels.csv'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(test_lines))

    print(f"Đã chia xong!")
    print(f" - Train: {len(train_lines)} mẫu (Signer 01-24)")
    print(f" - Val  : {len(val_lines)} mẫu (Signer 25-27)")
    print(f" - Test : {len(test_lines)} mẫu (Signer 28-31)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_file', type=str, default='data/demo/processed_npy/labels.csv')
    parser.add_argument('--output_dir', type=str, default='data/demo/processed_npy')
    args = parser.parse_args()
    
    split_data(args.label_file, args.output_dir)
