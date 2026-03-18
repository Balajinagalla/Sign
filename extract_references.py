# extract_references.py - Extract reference frames from Kaggle dataset videos
# Extracts a clear middle frame from each sign's video as reference image
import cv2
import os

KAGGLE_DIR = "kaggle_dataset"
OUTPUT_DIR = "sign_references"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get all sign folders
sign_folders = sorted([d for d in os.listdir(KAGGLE_DIR) 
                       if os.path.isdir(os.path.join(KAGGLE_DIR, d))])

print(f"Found {len(sign_folders)} sign categories")
extracted = 0
failed = 0

for sign_name in sign_folders:
    folder = os.path.join(KAGGLE_DIR, sign_name)
    
    # Find the first non-tilted video (original angle is best for reference)
    videos = [f for f in os.listdir(folder) 
              if f.endswith('.mp4') and 'tilt' not in f.lower()]
    
    if not videos:
        # Fallback to any video
        videos = [f for f in os.listdir(folder) if f.endswith('.mp4')]
    
    if not videos:
        print(f"  ❌ No videos found for: {sign_name}")
        failed += 1
        continue
    
    video_path = os.path.join(folder, videos[0])
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"  ❌ Cannot open video: {sign_name}")
        failed += 1
        continue
    
    # Get total frames and go to middle
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mid_frame = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"  ❌ Cannot read frame: {sign_name}")
        failed += 1
        continue
    
    # Resize to 400x400 for reference display
    frame = cv2.resize(frame, (400, 400))
    
    # Add sign name label at bottom
    cv2.rectangle(frame, (0, 350), (400, 400), (0, 0, 0), -1)
    cv2.putText(frame, sign_name.upper(), (10, 385),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # Save as PNG
    clean_name = sign_name.replace(" ", "_").lower()
    output_path = os.path.join(OUTPUT_DIR, f"{clean_name}.png")
    cv2.imwrite(output_path, frame)
    
    print(f"  ✅ {sign_name} → {output_path}")
    extracted += 1

print(f"\n{'='*50}")
print(f"  Extracted: {extracted} reference images")
print(f"  Failed:    {failed}")
print(f"  Output:    {OUTPUT_DIR}/")
print(f"{'='*50}")
