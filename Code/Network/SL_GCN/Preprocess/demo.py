import cv2
import mediapipe as mp
import numpy as np
import os
from tqdm import tqdm
import argparse

# Thiết lập giới hạn luồng để tránh lỗi "Can't spawn new thread" trên Server
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENCV_FOR_THREADS_NUM'] = '1'
cv2.setNumThreads(1)

# Nạp MediaPipe Solutions
try:
    import mediapipe.python.solutions.holistic as mp_holistic
    import mediapipe.python.solutions.drawing_utils as mp_drawing
except ModuleNotFoundError:
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

# Cấu trúc 27 điểm khớp
POSE_INDICES = [0, 11, 12, 13, 14, 15, 16]
HAND_INDICES = [0, 4, 5, 8, 9, 12, 13, 16, 17, 20]

def extract_landmarks(video_path, holistic, visualize=False):
    cap = cv2.VideoCapture(video_path)
    sequence = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        
        frame_landmarks = np.zeros((27, 3))
        
        if results.pose_landmarks:
            for i, idx in enumerate(POSE_INDICES):
                lm = results.pose_landmarks.landmark[idx]
                frame_landmarks[i] = [lm.x, lm.y, lm.z]
        
        if results.left_hand_landmarks:
            for i, idx in enumerate(HAND_INDICES):
                lm = results.left_hand_landmarks.landmark[idx]
                frame_landmarks[7 + i] = [lm.x, lm.y, lm.z]
        
        if results.right_hand_landmarks:
            for i, idx in enumerate(HAND_INDICES):
                lm = results.right_hand_landmarks.landmark[idx]
                frame_landmarks[17 + i] = [lm.x, lm.y, lm.z]
        
        if not np.all(frame_landmarks[0] == 0):
            nose_coord = frame_landmarks[0].copy()
            frame_landmarks = frame_landmarks - nose_coord

        sequence.append(frame_landmarks)
        
        if visualize:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            cv2.imshow('MediaPipe Debug', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
                    
    cap.release()
    return np.array(sequence)

def resample_sequence(sequence, target_len=150):
    L = len(sequence)
    if L == 0:
        return np.zeros((target_len, 27, 3))
    
    if L > target_len:
        indices = np.linspace(0, L - 1, target_len, dtype=int)
        new_sequence = sequence[indices]
    elif L < target_len:
        rest = target_len - L
        num = int(np.ceil(rest / L))
        pad = np.concatenate([sequence for _ in range(num)], 0)[:rest]
        new_sequence = np.concatenate([sequence, pad], 0)
    else:
        new_sequence = sequence
        
    return new_sequence

def main():
    parser = argparse.ArgumentParser(description='Preprocess video to skeleton .npy for HA-SLR-GCN')
    parser.add_argument('--input_dir', type=str, default='data/demo/raw_videos')
    parser.add_argument('--output_dir', type=str, default='data/demo/processed_npy')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--max_frame', type=int, default=150)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    video_files = [f for f in os.listdir(args.input_dir) if f.endswith(('.mp4', '.avi'))]
    print(f"Tìm thấy {len(video_files)} video. Bắt đầu xử lý...")

    labels = []
    
    # Khởi tạo Holistic một lần duy nhất
    with mp_holistic.Holistic(
        static_image_mode=False, 
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    ) as holistic:
        
        for video_name in tqdm(video_files):
            video_path = os.path.join(args.input_dir, video_name)
            
            raw_skel = extract_landmarks(video_path, holistic, visualize=args.visualize)
            processed_skel = resample_sequence(raw_skel, target_len=args.max_frame)
            
            output_name = video_name.rsplit('.', 1)[0] + '.npy'
            output_path = os.path.join(args.output_dir, output_name)
            np.save(output_path, processed_skel)
            
            try:
                name_no_ext = video_name.rsplit('.', 1)[0]
                label_str = name_no_ext.split('_')[-1]
                label = int(label_str)
            except:
                label = 0
            labels.append(f"{video_name.rsplit('.', 1)[0]},{label}")

    with open(os.path.join(args.output_dir, 'labels.csv'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(labels))

    print(f"\nHoàn thành! Kết quả lưu tại: {args.output_dir}")

if __name__ == '__main__':
    main()
