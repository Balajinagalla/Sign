# train.py - High-Accuracy Training for ISL Sign Language Detection
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

# ---- Config ------------------------------------------------
WEIGHTS = "yolo11s.pt"          # Small model (9.4M params vs nano's 2.6M)
DATA_YAML = "Data/data.yaml"

if __name__ == '__main__':
    import torch
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Training on: {'GPU (CUDA)' if device == 0 else 'CPU'}")

    model = YOLO(WEIGHTS)       # Auto-downloads if not present

    results = model.train(
        data=DATA_YAML,
        epochs=50,              # ~220 images → converges fast, 50 is enough
        imgsz=640,
        batch=8 if device == 'cpu' else 4,  # Larger batch OK on CPU (no VRAM limit)
        name="sign_lang_yolo11",
        project="runs/train",
        exist_ok=True,
        device=device,

        # ---- Optimizer & LR Schedule ----
        optimizer="AdamW",      # Better for small datasets
        lr0=0.001,              # Lower initial LR for stability
        lrf=0.01,               # Final LR factor
        cos_lr=True,            # Cosine annealing schedule
        warmup_epochs=3.0,      # Quick warmup for small dataset

        # ---- Regularization ----
        weight_decay=0.001,     # Slightly higher regularization
        dropout=0.1,            # Light dropout to reduce overfitting

        # ---- Heavy Augmentation (compensates for small dataset) ----
        mosaic=1.0,             # Mosaic augmentation
        mixup=0.3,              # Blend images together
        copy_paste=0.2,         # Copy-paste augmentation
        degrees=15.0,           # Rotation ±15°
        translate=0.2,          # Translation ±20%
        scale=0.7,              # Scale variation
        shear=5.0,              # Shear ±5°
        flipud=0.2,             # Vertical flip 20%
        fliplr=0.5,             # Horizontal flip 50%
        hsv_h=0.02,             # Hue variation
        hsv_s=0.8,              # Saturation variation
        hsv_v=0.5,              # Value/brightness variation
        erasing=0.5,            # Random erasing
        close_mosaic=10,        # Disable mosaic for last 10 epochs

        # ---- Transfer Learning ----
        freeze=10,              # Freeze first 10 backbone layers

        # ---- Training Control ----
        patience=15,            # Stop early if no improvement for 15 epochs
        workers=4,
        cache='ram',
    )

    # Final test evaluation
    metrics = model.val(split="test")
    print(f"\n{'='*50}")
    print(f"  Test mAP@0.5     = {metrics.box.map50:.3f}")
    print(f"  Test mAP@0.5:0.95 = {metrics.box.map:.3f}")
    print(f"{'='*50}")