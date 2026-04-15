# build_exe.py — Package ISL GUI as standalone Windows .exe
# Usage: python build_exe.py
# Output: dist/ISL_Recognition/ folder with executable
import os
import sys
import subprocess
import shutil


def build():
    print("=" * 60)
    print("  🔧 ISL Recognition — Windows .exe Builder")
    print("=" * 60)

    # Check PyInstaller
    try:
        import PyInstaller
        print(f"  ✅ PyInstaller {PyInstaller.__version__} found")
    except ImportError:
        print("  ⚠️  PyInstaller not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        print("  ✅ PyInstaller installed")

    # Files to include
    data_files = []

    # Model files
    for model in ['best.pt', 'best.onnx', 'hand_landmarker.task', 'selfie_segmenter.tflite',
                   'yolo11n.pt', 'yolo11s.pt']:
        if os.path.exists(model):
            data_files.append(f'--add-data={model};.')
            print(f"  📦 Including: {model}")

    # Sign references
    if os.path.isdir('sign_references'):
        data_files.append('--add-data=sign_references;sign_references')
        print(f"  📦 Including: sign_references/ ({len(os.listdir("sign_references"))} files)")

    # Python modules
    for pyfile in ['tts_indic_multi.py', 'sign_constants.py', 'enhancements.py',
                    'pdf_report.py']:
        if os.path.exists(pyfile):
            data_files.append(f'--add-data={pyfile};.')
            print(f"  📦 Including: {pyfile}")

    # Data yaml
    if os.path.exists('Data/data.yaml'):
        data_files.append('--add-data=Data/data.yaml;Data')
        print(f"  📦 Including: Data/data.yaml")

    print(f"\n  🚀 Building executable...")
    print(f"  This may take 3-10 minutes...\n")

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name=ISL_Recognition",
        "--windowed",                 # No console window
        "--onedir",                   # Directory mode (faster startup)
        "--noconfirm",                # Overwrite existing
        "--clean",                    # Clean cache
        "--icon=NONE",                # No custom icon (add .ico if you have one)
        "--hidden-import=ultralytics",
        "--hidden-import=cv2",
        "--hidden-import=mediapipe",
        "--hidden-import=PIL",
        "--hidden-import=torch",
        "--hidden-import=torchvision",
        "--hidden-import=numpy",
        "--hidden-import=yaml",
        "--hidden-import=pygame",
        "--hidden-import=gtts",
        "--hidden-import=pyttsx3",
        "--hidden-import=edge_tts",
        "--hidden-import=tts_indic_multi",
        "--hidden-import=sign_constants",
        "--hidden-import=enhancements",
        "--collect-all=ultralytics",
        "--collect-all=mediapipe",
    ] + data_files + ["isl_gui_app.py"]

    print("  Running: " + " ".join(cmd[:5]) + " ...")

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        exe_path = os.path.join("dist", "ISL_Recognition")
        print(f"\n{'='*60}")
        print(f"  ✅ BUILD SUCCESSFUL!")
        print(f"  📂 Output: {os.path.abspath(exe_path)}")
        print(f"  🚀 Run: dist\\ISL_Recognition\\ISL_Recognition.exe")
        print(f"{'='*60}")

        # Check size
        total_size = 0
        for root, dirs, files in os.walk(exe_path):
            for f in files:
                total_size += os.path.getsize(os.path.join(root, f))
        print(f"  📊 Total size: {total_size / (1024*1024):.0f} MB")
    else:
        print(f"\n  ❌ Build failed with exit code {result.returncode}")
        print(f"  Check the output above for errors.")

    print(f"\n  TIP: To create a single .exe file (slower startup), use:")
    print(f"  python build_exe.py --onefile")


if __name__ == "__main__":
    if "--onefile" in sys.argv:
        print("  Using --onefile mode (single .exe, slower startup)")
    build()
