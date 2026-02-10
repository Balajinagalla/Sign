# train.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # ADD THIS LINE

from ultralytics import YOLO

# ---- Paths -------------------------------------------------
WEIGHTS = "weights/yolo11n.pt"
DATA_YAML = "Data/data.yaml"

if __name__ == '__main__':
    if not os.path.exists(WEIGHTS):
        raise FileNotFoundError(f"Download {WEIGHTS} first!")

    model = YOLO(WEIGHTS)

    results = model.train(
        data=DATA_YAML,
        epochs=100,
        imgsz=640,
        batch=8,                # Reduced from 16 → safer for 4GB GPU
        name="sign_lang_yolo11",
        project="runs/train",
        exist_ok=True,
        device=0,               # GPU (CUDA enabled)
        patience=20,
        workers=4,
        cache='ram'             # Speed up if you have 16+ GB RAM
    )

    # Final test
    metrics = model.val(split="test")
    print(f"Test mAP@0.5 = {metrics.box.map50:.3f}")
    print(f"Test mAP@0.5:0.95 = {metrics.box.map:.3f}")