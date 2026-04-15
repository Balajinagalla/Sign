# train.py - High-Accuracy Training for ISL Sign Language Detection
# Supports image datasets only
# Usage:
#   python train.py                          # Train with existing image dataset
#   python train.py --epochs 100             # Custom epochs
#   python train.py --batch 32 --epochs 80   # Custom batch + epochs
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO
import shutil
import glob
import sys

# ---- Auto Model Selection --------------------------------
def auto_select_model():
    """Pick the best YOLO size based on training data volume."""
    train_imgs = glob.glob("Data/train/images/*")
    n = len(train_imgs)
    if n < 1000:
        print(f"  Dataset: {n} images → using YOLO11-Nano (best for small datasets)")
        return "yolo11n.pt"
    elif n < 3000:
        print(f"  Dataset: {n} images → using YOLO11-Small")
        return "yolo11s.pt"
    else:
        print(f"  Dataset: {n} images → using YOLO11-Medium (high capacity)")
        return "yolo11m.pt"


# ---- Parse CLI Arguments -----------------------------------
def parse_args():
    """Parse command line arguments."""
    args = {
        'epochs': 50,
        'batch': 16,
        'imgsz': 640,
    }

    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        if argv[i] == '--epochs' and i + 1 < len(argv):
            args['epochs'] = int(argv[i + 1])
            i += 2
        elif argv[i] == '--batch' and i + 1 < len(argv):
            args['batch'] = int(argv[i + 1])
            i += 2
        elif argv[i] == '--imgsz' and i + 1 < len(argv):
            args['imgsz'] = int(argv[i + 1])
            i += 2
        elif argv[i] == '--help':
            print_help()
            sys.exit(0)
        else:
            i += 1

    return args


def print_help():
    print("""
╔═══════════════════════════════════════════════════════════╗
║        ISL Sign Language - Model Training Script          ║
╚═══════════════════════════════════════════════════════════╝

USAGE:
  python train.py [OPTIONS]

OPTIONS:
  --epochs N          Training epochs (default: 50)
  --batch N           Batch size (default: 16)
  --imgsz N           Image size (default: 640)
  --help              Show this help

EXAMPLES:
  # Train with existing image dataset
  python train.py

  # Train for 100 epochs
  python train.py --epochs 100

  # Train with larger batch on GPU
  python train.py --batch 32 --epochs 80

DATASET FORMAT:
  📷 Image Dataset:  Data/train/images/*.jpg + Data/train/labels/*.txt
""")


DATA_YAML = "Data/data.yaml"

if __name__ == '__main__':
    import torch

    # ── Parse arguments ──
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"  🤟 ISL Sign Language — Model Training")
    print(f"{'='*60}")

    # ── Check dataset exists ──
    if not os.path.exists(DATA_YAML):
        print(f"\n  ❌ Error: {DATA_YAML} not found!")
        print(f"  Please provide an image dataset in Data/")
        sys.exit(1)

    # ── Count images ──
    n_train = len(glob.glob("Data/train/images/*"))
    n_val = len(glob.glob("Data/valid/images/*"))
    n_test = len(glob.glob("Data/test/images/*"))

    print(f"\n  📊 Dataset Summary:")
    print(f"     Train: {n_train} images")
    print(f"     Val:   {n_val} images")
    print(f"     Test:  {n_test} images")
    print(f"     Total: {n_train + n_val + n_test} images")

    if n_train == 0:
        print(f"\n  ❌ Error: No training images found in Data/train/images/")
        print(f"  Add images manually or use the Data Collector tool.")
        sys.exit(1)

    # ── Device selection ──
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"\n  🖥️  Training on: {'GPU (CUDA)' if device == 0 else 'CPU'}")

    WEIGHTS = auto_select_model()
    model = YOLO(WEIGHTS)

    is_small = n_train < 1000

    print(f"\n  🚀 Starting training: {args['epochs']} epochs, batch={args['batch']}, imgsz={args['imgsz']}")
    print(f"{'='*60}\n")

    results = model.train(
        data=DATA_YAML,
        epochs=args['epochs'],
        imgsz=args['imgsz'],
        batch=args['batch'],
        name="sign_lang_yolo11",
        project="runs/train",
        exist_ok=True,
        device=device,

        # ---- Optimizer & LR Schedule ----
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3.0,

        # ---- Regularization (stronger for small datasets) ----
        weight_decay=0.002 if is_small else 0.001,
        dropout=0.25 if is_small else 0.1,

        # ---- Augmentation (tuned for Hand Signs) ----
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        flipud=0.0,
        fliplr=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        erasing=0.3 if is_small else 0.1,
        close_mosaic=20 if is_small else 10,

        # ---- Transfer Learning ----
        freeze=5 if is_small else 10,

        # ---- Training Control ----
        patience=15,
        workers=2,
        cache='disk',
        amp=True,
    )

    # Final test evaluation
    metrics = model.val(split="test")
    print(f"\n{'='*50}")
    print(f"  Test mAP@0.5     = {metrics.box.map50:.3f}")
    print(f"  Test mAP@0.5:0.95 = {metrics.box.map:.3f}")
    print(f"{'='*50}")

    # ── AUTO-DEPLOY: Copy best model to project root ──────────
    best_path = "runs/train/sign_lang_yolo11/weights/best.pt"
    if os.path.exists(best_path):
        shutil.copy2(best_path, "best.pt")
        print(f"\n✅ Model auto-deployed to best.pt")
        
        # Export to ONNX for faster inference
        try:
            deploy_model = YOLO("best.pt")
            deploy_model.export(format="onnx", imgsz=640, dynamic=True)
            print(f"✅ ONNX model exported to best.onnx")
        except Exception as e:
            print(f"⚠️ ONNX export failed: {e}")
    
    print(f"\n🏹 Training complete! Your model is ready.")
    print(f"Run 'python isl_gui_app.py' to test it.")
    input("\nPress Enter to close...")