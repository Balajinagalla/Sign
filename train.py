# train.py - High-Accuracy Training for ISL Sign Language Detection
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

# ---- Config ------------------------------------------------
WEIGHTS = "yolo11n.pt"          # Nano model (2.6M params) = 3x Faster Training + Inference
DATA_YAML = "Data/data.yaml"

if __name__ == '__main__':
    import torch
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Training on: {'GPU (CUDA)' if device == 0 else 'CPU'}")

    model = YOLO(WEIGHTS)       # Auto-downloads if not present

    results = model.train(
        data=DATA_YAML,
        epochs=100,             # Increased epochs for higher accuracy
        imgsz=640,
        batch=16 if device == 0 else 8,  # Maximize GPU memory for faster batches
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

        # ---- Heavy Augmentation (tuned for Hand Signs) ----
        mosaic=1.0,             # Mosaic is great for context
        mixup=0.0,              # Disabled: Mixup ruins fine hand shapes
        copy_paste=0.0,         # Disabled: Copy-paste creates artificial edges
        degrees=10.0,           # Lower rotation (signs are directional)
        translate=0.1,          # Lower translation
        scale=0.5,              # Moderate scale variation
        shear=2.0,              # Lower shear (maintains finger shapes)
        flipud=0.0,             # Disabled: Hand signs are never upside down!
        fliplr=0.0,             # Disabled: ISL meaning changes if flipped!
        hsv_h=0.015,            # Hue variation
        hsv_s=0.7,              # Saturation variation
        hsv_v=0.4,              # Dark/light variation
        erasing=0.1,            # Very light erasing (don't cover fingers)
        close_mosaic=10,        # Disable mosaic for last 10 epochs

        # ---- Transfer Learning & Speed ----
        freeze=10,              # Freeze first 10 backbone layers
        
        # ---- Training Control ----
        patience=20,            # Stop early if no improvement
        workers=8,              # Higher parallel workers
        cache='ram',            # Cache to RAM for extreme speed
    )

    # Final test evaluation
    metrics = model.val(split="test")
    print(f"\n{'='*50}")
    print(f"  Test mAP@0.5     = {metrics.box.map50:.3f}")
    print(f"  Test mAP@0.5:0.95 = {metrics.box.map:.3f}")
    print(f"{'='*50}")