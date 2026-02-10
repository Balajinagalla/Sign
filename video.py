import cv2
import os

video_dir = "path/to/videos/folder"   # e.g. "dataset/ThankYou/video1.mp4"
output_dir = "extracted_frames"

os.makedirs(output_dir, exist_ok=True)

for class_folder in os.listdir(video_dir):
    class_path = os.path.join(video_dir, class_folder)
    if os.path.isdir(class_path):
        for video_file in os.listdir(class_path):
            if video_file.endswith(('.mp4', '.avi')):
                cap = cv2.VideoCapture(os.path.join(class_path, video_file))
                frame_count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # Save every 5th frame (or adjust)
                    if frame_count % 5 == 0:
                        cv2.imwrite(f"{output_dir}/{class_folder}_frame_{frame_count}.jpg", frame)
                    frame_count += 1
                cap.release()