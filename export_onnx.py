# export_onnx.py - Export YOLO model to ONNX for faster CPU inference
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

MODEL_PATH = "runs/train/sign_lang_yolo11/weights/best.pt"

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found! Train the model first.")
        exit(1)

    print("Loading model...")
    model = YOLO(MODEL_PATH)

    print("Exporting to ONNX...")
    model.export(
        format="onnx",
        imgsz=640,
        simplify=True,
        dynamic=False,
    )

    onnx_path = MODEL_PATH.replace(".pt", ".onnx")
    if os.path.exists(onnx_path):
        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"\n✅ ONNX model exported: {onnx_path} ({size_mb:.1f} MB)")
        print("   The GUI app will auto-detect and use it for faster inference.")
    else:
        print("⚠️ Export may have saved to a different path. Check runs/train/sign_lang_yolo11/weights/")
