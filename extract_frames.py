import cv2
import os

input_dir = "kaggle_dataset/"  # unzipped videos
output_dir = "frames/"
os.makedirs(output_dir, exist_ok=True)

classes = []  # Auto-detect classes
for class_folder in os.listdir(input_dir):
    class_path = os.path.join(input_dir, class_folder)
    if os.path.isdir(class_path):
        classes.append(class_folder)
        os.makedirs(os.path.join(output_dir, class_folder), exist_ok=True)
        for video_file in os.listdir(class_path):
            if video_file.endswith(('.mp4', '.avi', '.mov')):
                cap = cv2.VideoCapture(os.path.join(class_path, video_file))
                frame_count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    if frame_count % 5 == 0:  # Every 5th frame (avoid duplicates)
                        cv2.imwrite(os.path.join(output_dir, class_folder, f"{os.path.splitext(video_file)[0]}_{frame_count}.jpg"), frame)
                    frame_count += 1
                cap.release()
print(f"Classes found: {classes}")  # e.g., ['hello', 'thank you', ...]
print("Frames ready!")