# predict.py
from ultralytics import YOLO

# Load the BEST trained model
model = YOLO("runs/train/sign_lang_yolo11/weights/best.pt")

# Run inference on test images
results = model("Data/test/images/*.jpg", save=True, conf=0.4)