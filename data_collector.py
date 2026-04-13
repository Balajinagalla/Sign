# data_collector.py - Real-time Image Collection and Labeling Tool for YOLOv11
# Now supports adding NEW signs dynamically with auto-capture and hand detection!
import cv2
import os
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk
import uuid
from datetime import datetime
import yaml
import time
import numpy as np
import threading
import shutil

# TTS (Text-to-Speech) for sign announcements
try:
    from tts_indic_multi import speak_sign
    from sign_constants import LANGUAGES, LANG_CODES, TRANSLATIONS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    LANGUAGES = ["English"]
    LANG_CODES = ["en"]
    TRANSLATIONS = {}

# Try importing mediapipe for hand detection
MEDIAPIPE_AVAILABLE = False
MP_USE_LEGACY = False

try:
    import mediapipe as mp
    # Check for legacy API (mp.solutions)
    if hasattr(mp, 'solutions'):
        MEDIAPIPE_AVAILABLE = True
        MP_USE_LEGACY = True
    # Check for new tasks API (mp.tasks)
    elif hasattr(mp, 'tasks'):
        MEDIAPIPE_AVAILABLE = True
        MP_USE_LEGACY = False
except ImportError:
    print("Warning: mediapipe not installed. Auto hand detection will use skin color detection.")
    print("Install with: pip install mediapipe")

class DataCollectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ISL Data Collector - Dynamic Capture Mode")
        self.root.geometry("1500x1080")
        self.root.state("zoomed")
        self.root.configure(bg="#2c3e50")

        # Paths
        self.data_yaml_path = "Data/data.yaml"
        self.train_images_dir = "Data/train/images"
        self.train_labels_dir = "Data/train/labels"
        self.tts_file_path = "tts_indic_multi.py"

        # TTS state
        self.tts_enabled = True
        self.current_lang_idx = 0  # English default
        self.last_tts_sign = None
        self.last_tts_time = 0
        self.TTS_COOLDOWN = 3.0  # seconds between repeats
        
        # Load existing class names from data.yaml
        self.CLASS_NAMES = self._load_class_names()
        
        # Ensure directories exist
        os.makedirs(self.train_images_dir, exist_ok=True)
        os.makedirs(self.train_labels_dir, exist_ok=True)

        # Variables
        self.cap = None
        self.running = False
        self.current_frame = None
        self.display_frame = None
        self.bbox_start = None
        self.bbox_end = None
        self.drawing = False
        self.collected_count = 0
        
        # Dynamic capture settings
        self.TARGET_IMAGES = 25  # Default target images per class
        self.auto_capture_enabled = False
        self.auto_capture_interval = 5.0  # 5 seconds between captures (user requested)
        self.batch_capture_count = 10  # images per batch
        self.batch_capturing = False
        self.batch_remaining = 0
        self.last_capture_time = 0
        self.use_full_frame = False  # Option to use full frame without bounding box
        
        # Auto-cycle settings (cycle through all signs automatically)
        self.auto_cycle_active = False
        self.auto_cycle_sign_index = 0
        self.auto_cycle_images_per_sign = 10
        self.auto_cycle_captured_for_current = 0
        self.auto_cycle_countdown = 0  # seconds remaining before capture starts
        self.auto_cycle_countdown_start = 0
        self.auto_cycle_sign_delay = 5  # 5 seconds between signs
        self.auto_cycle_phase = 'idle'  # 'idle', 'countdown', 'capturing'
        
        # Hand detection settings
        self.auto_detect_hand = True  # Auto-detect ON by default
        self.detected_hand_bbox = None
        self.hand_detection_padding = 30  # Extra padding around detected hand
        
        # Advanced Features
        self.augment_enabled = False
        self.blur_bg_enabled = False
        self.voice_control_enabled = False
        
        # Live Prediction (shows model's guess during collection)
        self.live_predict_enabled = True
        self.live_model = None
        self.live_predict_label = ""
        self.live_predict_conf = 0.0
        self.live_predict_interval = 0.5  # seconds between predictions
        self.last_predict_time = 0
        self._load_live_model()
        
        # Capture feedback
        self.flash_alpha = 0  # For capture flash effect
        self.capture_countdown = 0  # Countdown seconds before next capture
        self.capture_countdown_start = 0
        
        # Load Image Segmenter for background blur
        self.segmenter = None
        if os.path.exists("selfie_segmenter.tflite"):
            try:
                from mediapipe.tasks import python as mp_python
                from mediapipe.tasks.python import vision as mp_vision
                base_options = mp_python.BaseOptions(model_asset_path="selfie_segmenter.tflite")
                options = mp_vision.ImageSegmenterOptions(base_options=base_options, output_category_mask=True)
                self.segmenter = mp_vision.ImageSegmenter.create_from_options(options)
            except Exception as e:
                print(f"Failed to load segmenter: {e}")
        self.hands = None
        self.mp_hands = None
        self.mp_draw = None
        
        # Initialize hand detection
        self._init_hand_detector()
        
        # Count existing images per class
        self.class_image_counts = self._count_existing_images()

        self._create_ui()
    
    def _load_live_model(self):
        """Load YOLO model for live prediction during data collection."""
        try:
            from ultralytics import YOLO
            pt_path = "best.pt"
            if os.path.exists(pt_path):
                self.live_model = YOLO(pt_path, task='detect')
                print(f"Live prediction model loaded: {pt_path}")
            elif os.path.exists("runs/train/sign_lang_yolo11/weights/best.pt"):
                self.live_model = YOLO("runs/train/sign_lang_yolo11/weights/best.pt", task='detect')
                print("Live prediction model loaded from runs/")
            else:
                print("No model found for live prediction")
        except Exception as e:
            print(f"Could not load live model: {e}")
            self.live_model = None
    
    def _init_hand_detector(self):
        """Initialize hand detector based on available API"""
        global MEDIAPIPE_AVAILABLE, MP_USE_LEGACY
        
        if MEDIAPIPE_AVAILABLE and MP_USE_LEGACY:
            # Use legacy API (mp.solutions)
            try:
                self.mp_hands = mp.solutions.hands
                self.hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.mp_draw = mp.solutions.drawing_utils
                print("Using MediaPipe legacy API for hand detection")
            except Exception as e:
                print(f"Failed to init legacy MediaPipe: {e}")
                MEDIAPIPE_AVAILABLE = False
        elif MEDIAPIPE_AVAILABLE and not MP_USE_LEGACY:
            # New tasks API not fully supported for real-time detection in current version
            # Fall back to skin detection
            print("MediaPipe tasks API detected - using skin color detection fallback")
            MEDIAPIPE_AVAILABLE = False
        
        if not MEDIAPIPE_AVAILABLE:
            print("Using skin color detection for hand detection")

    def _load_class_names(self):
        """Load class names from data.yaml"""
        try:
            with open(self.data_yaml_path, 'r') as f:
                data = yaml.safe_load(f)
                return data.get('names', [])
        except:
            return ['family', 'hello', 'help', 'house', 'ily', 'no', 'please', 'thank you', 'yes']

    def _save_class_names(self):
        """Save updated class names to data.yaml"""
        try:
            with open(self.data_yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            
            data['names'] = self.CLASS_NAMES
            data['nc'] = len(self.CLASS_NAMES)
            
            with open(self.data_yaml_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            return True
        except Exception as e:
            print(f"Error saving data.yaml: {e}")
            return False

    def _count_existing_images(self):
        """Count existing labeled images per class"""
        counts = {name: 0 for name in self.CLASS_NAMES}
        try:
            for label_file in os.listdir(self.train_labels_dir):
                if label_file.endswith('.txt'):
                    label_path = os.path.join(self.train_labels_dir, label_file)
                    with open(label_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                class_id = int(parts[0])
                                if 0 <= class_id < len(self.CLASS_NAMES):
                                    counts[self.CLASS_NAMES[class_id]] += 1
        except Exception as e:
            print(f"Error counting images: {e}")
        return counts

    def _get_progress_color(self, count):
        """Get color based on progress"""
        if count >= self.TARGET_IMAGES:
            return "#27ae60"  # Green - complete
        elif count >= self.TARGET_IMAGES * 0.6:
            return "#f39c12"  # Orange - almost there
        else:
            return "#e74c3c"  # Red - needs more

    def _create_ui(self):
        # Title Bar
        title_bar = tk.Frame(self.root, bg="#1a252f", pady=6)
        title_bar.pack(fill=tk.X)
        tk.Label(title_bar, text="📸 ISL Data Collector - Dynamic Mode", 
                font=("Arial", 18, "bold"), bg="#1a252f", fg="white").pack(side=tk.LEFT, padx=15)
        tk.Label(title_bar, 
            text="Start Camera → Select Sign → Capture → Train",
            font=("Arial", 10), bg="#1a252f", fg="#7f8c8d").pack(side=tk.RIGHT, padx=15)

        # ══════════ MAIN HORIZONTAL LAYOUT ══════════
        main_frame = tk.Frame(self.root, bg="#2c3e50")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # ── LEFT SIDE: Camera + Action Buttons ──
        left_panel = tk.Frame(main_frame, bg="#2c3e50")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Control row (Sign Class, Start, Stop, Train)
        control_frame = tk.Frame(left_panel, bg="#2c3e50", pady=5)
        control_frame.pack(fill=tk.X)

        tk.Label(control_frame, text="Sign:", font=("Arial", 11, "bold"), 
                bg="#2c3e50", fg="white").pack(side=tk.LEFT, padx=(5, 3))
        self.class_combo = ttk.Combobox(control_frame, values=self.CLASS_NAMES, 
                                        state="readonly", width=12, font=("Arial", 11))
        if self.CLASS_NAMES:
            self.class_combo.set(self.CLASS_NAMES[0])
        self.class_combo.pack(side=tk.LEFT, padx=3)
        self.class_combo.bind("<<ComboboxSelected>>", self._on_class_change)

        self.add_sign_btn = tk.Button(control_frame, text="➕ Add", command=self.add_new_sign,
                                      bg="#9b59b6", fg="white", font=("Arial", 10, "bold"), width=6)
        self.add_sign_btn.pack(side=tk.LEFT, padx=3)

        self.start_btn = tk.Button(control_frame, text="▶ Start", command=self.start_camera,
                                   bg="#27ae60", fg="white", font=("Arial", 11, "bold"), width=8)
        self.start_btn.pack(side=tk.LEFT, padx=3)

        self.stop_btn = tk.Button(control_frame, text="⏹ Stop", command=self.stop_camera,
                                  bg="#e74c3c", fg="white", font=("Arial", 11, "bold"), width=8, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=3)

        selected = self.class_combo.get() if self.CLASS_NAMES else "sign"
        self.train_btn = tk.Button(control_frame, text=f"🔥 Train '{selected}'", command=self._train_selected_sign,
                                   bg="#f1c40f", fg="#2c3e50", font=("Arial", 11, "bold"), width=16, borderwidth=3)
        self.train_btn.pack(side=tk.LEFT, padx=5)

        # Video Canvas (TALLER)
        self.canvas = tk.Canvas(left_panel, width=860, height=700, bg="black", cursor="cross")
        self.canvas.pack(pady=5, padx=5)
        
        # Bind mouse events for bounding box drawing
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        # Action buttons row below camera
        action_frame = tk.Frame(left_panel, bg="#2c3e50", pady=3)
        action_frame.pack(fill=tk.X)

        self.capture_btn = tk.Button(action_frame, text="📷 Capture", command=self.capture_frame,
                                     bg="#3498db", fg="white", font=("Arial", 10, "bold"), width=10, state=tk.DISABLED)
        self.capture_btn.pack(side=tk.LEFT, padx=3)

        self.save_btn = tk.Button(action_frame, text="💾 Save & Label", command=self.save_labeled_image,
                                  bg="#27ae60", fg="white", font=("Arial", 10, "bold"), width=12, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=3)

        self.clear_btn = tk.Button(action_frame, text="🔄 Clear Box", command=self.clear_bbox,
                                   bg="#f39c12", fg="white", font=("Arial", 10, "bold"), width=10)
        self.clear_btn.pack(side=tk.LEFT, padx=3)
        
        self.undo_btn = tk.Button(action_frame, text="↩️ Undo", command=self.undo_last_capture,
                                  bg="#c0392b", fg="white", font=("Arial", 10, "bold"), width=8)
        self.undo_btn.pack(side=tk.LEFT, padx=3)

        self.retrain_btn = tk.Button(action_frame, text="🔥 FULL TRAIN", command=self.retrain_model,
                                     bg="#f1c40f", fg="#2c3e50", font=("Arial", 10, "bold"), width=12, borderwidth=2)
        self.retrain_btn.pack(side=tk.LEFT, padx=3)

        self.quick_train_btn = tk.Button(action_frame, text="⚡ QUICK TRAIN", command=self._quick_train,
                                          bg="#e74c3c", fg="white", font=("Arial", 10, "bold"), width=12, borderwidth=2)
        self.quick_train_btn.pack(side=tk.LEFT, padx=3)

        # Status bar
        self.status_label = tk.Label(action_frame, text="Status: Ready", 
                                     font=("Arial", 10), bg="#2c3e50", fg="#ecf0f1")
        self.status_label.pack(side=tk.RIGHT, padx=10)

        # ── RIGHT SIDE: Sidebar with all settings ──
        sidebar = tk.Frame(main_frame, bg="#34495e", width=380)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        sidebar.pack_propagate(False)

        # Scrollable sidebar
        sidebar_canvas = tk.Canvas(sidebar, bg="#34495e", highlightthickness=0)
        sidebar_scroll = tk.Scrollbar(sidebar, orient=tk.VERTICAL, command=sidebar_canvas.yview)
        sidebar_inner = tk.Frame(sidebar_canvas, bg="#34495e")
        
        sidebar_inner.bind("<Configure>", lambda e: sidebar_canvas.configure(scrollregion=sidebar_canvas.bbox("all")))
        sidebar_canvas.create_window((0, 0), window=sidebar_inner, anchor="nw", width=360)
        sidebar_canvas.configure(yscrollcommand=sidebar_scroll.set)
        
        sidebar_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sidebar_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        def _on_sidebar_scroll(event):
            sidebar_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        sidebar_canvas.bind_all("<MouseWheel>", _on_sidebar_scroll)

        # ── SECTION: Progress ──
        prog_section = tk.LabelFrame(sidebar_inner, text="📊 Current Progress", 
                                      font=("Arial", 10, "bold"), bg="#1a252f", fg="white", padx=10, pady=8)
        prog_section.pack(fill=tk.X, padx=8, pady=(8, 4))
        
        self.progress_label = tk.Label(prog_section, 
            text=self._get_current_progress_text(),
            font=("Arial", 12, "bold"), bg="#1a252f", fg="#27ae60", wraplength=330, justify=tk.LEFT)
        self.progress_label.pack(anchor=tk.W, pady=3)
        
        self.progress_bar = ttk.Progressbar(prog_section, length=330, mode='determinate')
        self.progress_bar.pack(anchor=tk.W, pady=3)
        self._update_progress_bar()

        self.bbox_label = tk.Label(prog_section, text="BBox: Not drawn", 
                                   font=("Arial", 9), bg="#1a252f", fg="#f39c12")
        self.bbox_label.pack(anchor=tk.W)

        # ── SECTION: Capture Settings ──
        cap_section = tk.LabelFrame(sidebar_inner, text="⚙️ Capture Settings", 
                                     font=("Arial", 10, "bold"), bg="#2c3e50", fg="white", padx=10, pady=6)
        cap_section.pack(fill=tk.X, padx=8, pady=4)

        t_row = tk.Frame(cap_section, bg="#2c3e50")
        t_row.pack(fill=tk.X, pady=2)
        tk.Label(t_row, text="🎯 Target:", font=("Arial", 9, "bold"), 
                bg="#2c3e50", fg="white").pack(side=tk.LEFT)
        self.target_var = tk.IntVar(value=self.TARGET_IMAGES)
        self.target_slider = tk.Scale(t_row, from_=10, to=100, orient=tk.HORIZONTAL,
                                      variable=self.target_var, length=150,
                                      bg="#2c3e50", fg="white", highlightthickness=0,
                                      command=self._on_target_change)
        self.target_slider.pack(side=tk.LEFT, padx=3)
        self.target_display = tk.Label(t_row, text=f"{self.TARGET_IMAGES}/class", 
                                       font=("Arial", 9), bg="#2c3e50", fg="#3498db")
        self.target_display.pack(side=tk.LEFT)

        self.full_frame_var = tk.BooleanVar(value=False)
        tk.Checkbutton(cap_section, text="📐 Use Full Frame (no bbox)", 
                       variable=self.full_frame_var, bg="#2c3e50", fg="white", 
                       selectcolor="#1a252f", font=("Arial", 9),
                       command=self._on_full_frame_toggle).pack(anchor=tk.W, pady=1)

        ac_row = tk.Frame(cap_section, bg="#2c3e50")
        ac_row.pack(fill=tk.X, pady=2)
        self.auto_capture_var = tk.BooleanVar(value=False)
        tk.Checkbutton(ac_row, text="🔄 Auto", variable=self.auto_capture_var,
                       bg="#2c3e50", fg="white", selectcolor="#1a252f", font=("Arial", 9, "bold"),
                       command=self._on_auto_capture_toggle).pack(side=tk.LEFT)
        self.interval_var = tk.DoubleVar(value=self.auto_capture_interval)
        self.interval_slider = tk.Scale(ac_row, from_=1.0, to=10.0, resolution=0.5,
                                        orient=tk.HORIZONTAL, variable=self.interval_var,
                                        length=100, bg="#2c3e50", fg="white", highlightthickness=0,
                                        command=self._on_interval_change)
        self.interval_slider.pack(side=tk.LEFT, padx=3)
        self.interval_display = tk.Label(ac_row, text=f"{self.auto_capture_interval}s", 
                                         font=("Arial", 9), bg="#2c3e50", fg="#e74c3c")
        self.interval_display.pack(side=tk.LEFT)

        batch_row = tk.Frame(cap_section, bg="#2c3e50")
        batch_row.pack(fill=tk.X, pady=2)
        tk.Label(batch_row, text="Batch:", font=("Arial", 9), 
                bg="#2c3e50", fg="#95a5a6").pack(side=tk.LEFT)
        self.batch_var = tk.IntVar(value=self.batch_capture_count)
        self.batch_spinbox = tk.Spinbox(batch_row, from_=5, to=50, width=4,
                                        textvariable=self.batch_var, font=("Arial", 9))
        self.batch_spinbox.pack(side=tk.LEFT, padx=3)
        self.batch_btn = tk.Button(batch_row, text="🚀 Start Batch", command=self.start_batch_capture,
                                   bg="#1abc9c", fg="white", font=("Arial", 9, "bold"), width=10)
        self.batch_btn.pack(side=tk.LEFT, padx=5)

        # ── SECTION: Detection & Features ──
        feat_section = tk.LabelFrame(sidebar_inner, text="🖐️ Detection & Features", 
                                      font=("Arial", 10, "bold"), bg="#2c3e50", fg="white", padx=10, pady=6)
        feat_section.pack(fill=tk.X, padx=8, pady=4)

        self.auto_detect_var = tk.BooleanVar(value=True)
        detect_state = tk.NORMAL if MEDIAPIPE_AVAILABLE else tk.DISABLED
        self.auto_detect_check = tk.Checkbutton(feat_section, text="🖐️ Auto-Detect Hand", 
                                                 variable=self.auto_detect_var,
                                                 bg="#2c3e50", fg="white", selectcolor="#1a252f",
                                                 font=("Arial", 9, "bold"), state=detect_state,
                                                 command=self._on_auto_detect_toggle)
        self.auto_detect_check.pack(anchor=tk.W, pady=1)

        if not MEDIAPIPE_AVAILABLE:
            tk.Label(feat_section, text="  (pip install mediapipe)", font=("Arial", 8), 
                    bg="#2c3e50", fg="#e74c3c").pack(anchor=tk.W)

        pad_row = tk.Frame(feat_section, bg="#2c3e50")
        pad_row.pack(fill=tk.X, pady=2)
        tk.Label(pad_row, text="Padding:", font=("Arial", 9), 
                bg="#2c3e50", fg="#95a5a6").pack(side=tk.LEFT)
        self.padding_var = tk.IntVar(value=self.hand_detection_padding)
        self.padding_slider = tk.Scale(pad_row, from_=10, to=80, orient=tk.HORIZONTAL,
                                       variable=self.padding_var, length=120,
                                       bg="#2c3e50", fg="white", highlightthickness=0,
                                       command=self._on_padding_change)
        self.padding_slider.pack(side=tk.LEFT, padx=3)
        self.hand_status_label = tk.Label(pad_row, text="Hand: --", 
                                          font=("Arial", 9), bg="#2c3e50", fg="#95a5a6")
        self.hand_status_label.pack(side=tk.RIGHT)

        self.blur_bg_var = tk.BooleanVar(value=False)
        tk.Checkbutton(feat_section, text="🌫️ Blur Background", variable=self.blur_bg_var,
                       bg="#2c3e50", fg="white", selectcolor="#1a252f", font=("Arial", 9, "bold"),
                       command=lambda: setattr(self, 'blur_bg_enabled', self.blur_bg_var.get())
                       ).pack(anchor=tk.W, pady=1)
        
        self.augment_var = tk.BooleanVar(value=False)
        tk.Checkbutton(feat_section, text="✨ Extra Augmentation (5x)", variable=self.augment_var,
                       bg="#2c3e50", fg="white", selectcolor="#1a252f", font=("Arial", 9, "bold"),
                       command=lambda: setattr(self, 'augment_enabled', self.augment_var.get())
                       ).pack(anchor=tk.W, pady=1)

        self.voice_var = tk.BooleanVar(value=False)
        tk.Checkbutton(feat_section, text="🎤 Voice Control", variable=self.voice_var,
                       bg="#2c3e50", fg="white", selectcolor="#1a252f", font=("Arial", 9, "bold"),
                       command=self._on_voice_toggle).pack(anchor=tk.W, pady=1)

        # ── SECTION: Language & TTS ──
        tts_section = tk.LabelFrame(sidebar_inner, text="🔊 Language & TTS", 
                                     font=("Arial", 10, "bold"), bg="#16213e", fg="white", padx=10, pady=6)
        tts_section.pack(fill=tk.X, padx=8, pady=4)

        # Language selector
        lang_row = tk.Frame(tts_section, bg="#16213e")
        lang_row.pack(fill=tk.X, pady=3)
        tk.Label(lang_row, text="🌐 Language:", font=("Arial", 9, "bold"), 
                bg="#16213e", fg="white").pack(side=tk.LEFT)
        self.lang_combo = ttk.Combobox(lang_row, values=LANGUAGES, 
                                       state="readonly", width=12, font=("Arial", 9))
        self.lang_combo.set(LANGUAGES[0])
        self.lang_combo.pack(side=tk.LEFT, padx=5)
        self.lang_combo.bind("<<ComboboxSelected>>", self._on_lang_change)

        # TTS enable/disable
        self.tts_var = tk.BooleanVar(value=True)
        tk.Checkbutton(tts_section, text="🔊 Speak Detected Signs", variable=self.tts_var,
                       bg="#16213e", fg="white", selectcolor="#0f3460", font=("Arial", 9, "bold"),
                       command=lambda: setattr(self, 'tts_enabled', self.tts_var.get())
                       ).pack(anchor=tk.W, pady=1)

        # Manual speak button
        speak_row = tk.Frame(tts_section, bg="#16213e")
        speak_row.pack(fill=tk.X, pady=3)
        self.speak_btn = tk.Button(speak_row, text="🔊 Speak Selected Sign", 
                                   command=self._speak_selected_sign,
                                   bg="#bb86fc", fg="white", font=("Arial", 9, "bold"), width=20)
        self.speak_btn.pack(side=tk.LEFT)
        
        # Current TTS status
        self.tts_status_label = tk.Label(tts_section, text="TTS: Ready", 
                                         font=("Arial", 9), bg="#16213e", fg="#7bed9f")
        self.tts_status_label.pack(anchor=tk.W, pady=2)

        # ── SECTION: Auto-Cycle ──
        cycle_section = tk.LabelFrame(sidebar_inner, text="🔁 Auto-Cycle All Signs", 
                                       font=("Arial", 10, "bold"), bg="#2c3e50", fg="white", padx=10, pady=6)
        cycle_section.pack(fill=tk.X, padx=8, pady=4)

        cyc_row = tk.Frame(cycle_section, bg="#2c3e50")
        cyc_row.pack(fill=tk.X, pady=2)
        self.auto_cycle_btn = tk.Button(cyc_row, text="🔁 Start Cycle", 
                                        command=self.start_auto_cycle,
                                        bg="#e67e22", fg="white", font=("Arial", 9, "bold"), width=12)
        self.auto_cycle_btn.pack(side=tk.LEFT, padx=3)
        tk.Label(cyc_row, text="Imgs/sign:", font=("Arial", 9), 
                bg="#2c3e50", fg="white").pack(side=tk.LEFT, padx=3)
        self.cycle_imgs_var = tk.IntVar(value=10)
        self.cycle_imgs_spinbox = tk.Spinbox(cyc_row, from_=5, to=50, width=4,
                                             textvariable=self.cycle_imgs_var, font=("Arial", 9))
        self.cycle_imgs_spinbox.pack(side=tk.LEFT, padx=3)
        
        self.auto_cycle_status = tk.Label(cycle_section, text="", 
                                          font=("Arial", 9, "bold"), bg="#2c3e50", fg="#f1c40f")
        self.auto_cycle_status.pack(anchor=tk.W, pady=2)

        # ── SECTION: Import Videos ──
        vid_section = tk.LabelFrame(sidebar_inner, text="📹 Import Videos", 
                                     font=("Arial", 10, "bold"), bg="#16213e", fg="#64b5f6", padx=10, pady=6)
        vid_section.pack(fill=tk.X, padx=8, pady=4)

        tk.Label(vid_section, text="Extract frames from video files into training data",
                 font=("Arial", 8), bg="#16213e", fg="#7f8c8d").pack(anchor=tk.W, pady=(0, 4))

        vid_btn_row1 = tk.Frame(vid_section, bg="#16213e")
        vid_btn_row1.pack(fill=tk.X, pady=2)

        tk.Button(vid_btn_row1, text="📹 Import Video Files",
                  command=self.import_video_files,
                  bg="#64b5f6", fg="#000", font=("Arial", 9, "bold"),
                  relief=tk.FLAT, cursor="hand2", width=18).pack(side=tk.LEFT, padx=3)

        tk.Button(vid_btn_row1, text="📁 Import Folder",
                  command=self.import_video_folder,
                  bg="#bb86fc", fg="white", font=("Arial", 9, "bold"),
                  relief=tk.FLAT, cursor="hand2", width=14).pack(side=tk.LEFT, padx=3)

        # Frame interval for video import
        vid_int_row = tk.Frame(vid_section, bg="#16213e")
        vid_int_row.pack(fill=tk.X, pady=2)
        tk.Label(vid_int_row, text="Frame interval:", font=("Arial", 9),
                 bg="#16213e", fg="#95a5a6").pack(side=tk.LEFT)
        self.vid_interval_var = tk.IntVar(value=5)
        tk.Scale(vid_int_row, from_=1, to=30, orient=tk.HORIZONTAL,
                 variable=self.vid_interval_var, length=120,
                 bg="#16213e", fg="white", highlightthickness=0,
                 troughcolor="#0f3460").pack(side=tk.LEFT, padx=3)
        tk.Label(vid_int_row, text="(every Nth frame)", font=("Arial", 8),
                 bg="#16213e", fg="#7f8c8d").pack(side=tk.LEFT)

        self.vid_skip_dup_var = tk.BooleanVar(value=True)
        tk.Checkbutton(vid_section, text="🔄 Skip duplicate frames",
                       variable=self.vid_skip_dup_var, bg="#16213e", fg="white",
                       selectcolor="#0f3460", font=("Arial", 9)).pack(anchor=tk.W, pady=1)

        self.vid_import_status = tk.Label(vid_section, text="",
                                          font=("Arial", 9, "bold"), bg="#16213e", fg="#03dac6")
        self.vid_import_status.pack(anchor=tk.W, pady=2)

        # Open Video Dataset Converter button
        tk.Button(vid_section, text="🚀 Open Full Video Converter Tool",
                  command=self._open_video_converter,
                  bg="#f1c40f", fg="#000", font=("Arial", 9, "bold"),
                  relief=tk.FLAT, cursor="hand2", width=30).pack(pady=(5, 2))

        # ── SECTION: All Signs ──
        signs_section = tk.LabelFrame(sidebar_inner, text="📋 All Signs - Image Counts", 
                                       font=("Arial", 10, "bold"), bg="#1a252f", fg="white", padx=10, pady=6)
        signs_section.pack(fill=tk.X, padx=8, pady=4)
        
        self.class_list_label = tk.Label(signs_section, 
            text=self._format_class_list_with_counts(),
            font=("Arial", 9), bg="#1a252f", fg="#bdc3c7", justify=tk.LEFT, wraplength=340)
        self.class_list_label.pack(anchor=tk.W, pady=3)

    def _format_class_list_with_counts(self):
        """Format class list with image counts"""
        items = []
        for i, name in enumerate(self.CLASS_NAMES):
            count = self.class_image_counts.get(name, 0)
            if count >= self.TARGET_IMAGES:
                status = "✅"
            elif count >= self.TARGET_IMAGES * 0.6:
                status = "🟡"
            else:
                status = "❌"
            items.append(f"{status} {name}: {count}/{self.TARGET_IMAGES}")
        return "  |  ".join(items)

    def _get_current_progress_text(self):
        """Get progress text for current class"""
        if not self.CLASS_NAMES:
            return "No classes available"
        current_class = self.class_combo.get() if hasattr(self, 'class_combo') else self.CLASS_NAMES[0]
        count = self.class_image_counts.get(current_class, 0)
        remaining = max(0, self.TARGET_IMAGES - count)
        if count >= self.TARGET_IMAGES:
            return f"✅ '{current_class}': {count} images (Target reached!)"
        else:
            return f"📷 '{current_class}': {count}/{self.TARGET_IMAGES} images ({remaining} more needed)"

    def _update_progress_bar(self):
        """Update progress bar for current class"""
        if not hasattr(self, 'progress_bar'):
            return
        current_class = self.class_combo.get() if hasattr(self, 'class_combo') else self.CLASS_NAMES[0]
        count = self.class_image_counts.get(current_class, 0)
        progress = min(100, (count / self.TARGET_IMAGES) * 100)
        self.progress_bar['value'] = progress
        
        # Update color of progress label
        color = self._get_progress_color(count)
        if hasattr(self, 'progress_label'):
            self.progress_label.config(fg=color)

    def _on_class_change(self, event=None):
        """Handle class selection change"""
        self._update_progress_display()
        # Update train button label to show selected sign
        if hasattr(self, 'train_btn'):
            selected = self.class_combo.get()
            self.train_btn.config(text=f"🔥 Train '{selected}'")
    
    def _on_target_change(self, value):
        """Handle target images slider change"""
        self.TARGET_IMAGES = int(float(value))
        self.target_display.config(text=f"{self.TARGET_IMAGES} images/class")
        self._update_progress_display()
    
    def _on_lang_change(self, event=None):
        """Handle language selection change"""
        lang = self.lang_combo.get()
        if lang in LANGUAGES:
            self.current_lang_idx = LANGUAGES.index(lang)
            if hasattr(self, 'tts_status_label'):
                self.tts_status_label.config(text=f"🌐 Language: {lang}", fg="#00d2ff")
    
    def _speak_selected_sign(self):
        """Manually speak the currently selected sign in the chosen language"""
        if not TTS_AVAILABLE:
            messagebox.showwarning("TTS Not Available", "Text-to-Speech is not installed!")
            return
        selected = self.class_combo.get().lower()
        if selected:
            self.last_tts_sign = None  # Reset so it speaks even if same
            speak_sign(selected, self.current_lang_idx)
            lang_name = LANGUAGES[self.current_lang_idx]
            self.tts_status_label.config(text=f"🔊 Spoke: '{selected}' in {lang_name}", fg="#7bed9f")
    
    def _on_full_frame_toggle(self):
        """Handle full frame checkbox toggle"""
        self.use_full_frame = self.full_frame_var.get()
        if self.use_full_frame:
            self.status_label.config(text="Mode: Full Frame - No bounding box needed")
        else:
            self.status_label.config(text="Mode: Manual bbox - Draw around hand sign")
    
    def _on_auto_capture_toggle(self):
        """Handle auto-capture checkbox toggle"""
        self.auto_capture_enabled = self.auto_capture_var.get()
        if self.auto_capture_enabled:
            self.status_label.config(text=f"🔄 Auto-Capture ON - Every {self.auto_capture_interval}s")
            self.last_capture_time = time.time()
        else:
            self.status_label.config(text="Auto-Capture OFF")
    
    def _on_interval_change(self, value):
        """Handle interval slider change"""
        self.auto_capture_interval = float(value)
        self.interval_display.config(text=f"{self.auto_capture_interval}s")
    
    def undo_last_capture(self):
        """Removes tracking from last capture block"""
        if not getattr(self, 'last_saved_files', []):
            messagebox.showinfo("Undo", "Nothing to undo!")
            return
        
        class_name = None
        count_deleted = 0
        for f in self.last_saved_files:
            if os.path.exists(f):
                os.remove(f)
                count_deleted += 1
                if f.endswith('.jpg') and not class_name:
                    base = os.path.basename(f)
                    class_name = base.split('_2026')[0].replace('_', ' ')
        
        if class_name and count_deleted > 0:
            imgs_deleted = count_deleted // 2
            self.class_image_counts[class_name] = max(0, self.class_image_counts.get(class_name, 0) - imgs_deleted)
            self.collected_count = max(0, self.collected_count - imgs_deleted)
            self._update_progress_display()
            self.status_label.config(text=f"↩️ Undo successful: Removed {imgs_deleted} images")
        
        self.last_saved_files = []

    def _on_voice_toggle(self):
        """Handle voice control checkbox toggle"""
        self.voice_control_enabled = self.voice_var.get()
        if self.voice_control_enabled:
            import speech_recognition as sr
            import threading
            self.recognizer = sr.Recognizer()
            self.mic = sr.Microphone()
            
            def listen_loop():
                while getattr(self, 'voice_control_enabled', False) and getattr(self, 'running', False):
                    try:
                        with self.mic as source:
                            audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=3)
                        text = self.recognizer.recognize_google(audio).lower()
                        print("Voice heard:", text)
                        if "capture" in text or "take" in text or "picture" in text:
                            self.root.after(0, self._capture_and_save_auto)
                            self.root.after(0, lambda: self.status_label.config(text="🎤 Voice: Captured!"))
                        elif "next" in text or "change" in text:
                            idx = self.CLASS_NAMES.index(self.class_combo.get())
                            idx = (idx + 1) % len(self.CLASS_NAMES)
                            self.root.after(0, lambda: self.class_combo.set(self.CLASS_NAMES[idx]))
                            self.root.after(0, self._update_progress_display)
                            self.root.after(0, lambda: self.status_label.config(text=f"🎤 Voice: Changed to {self.CLASS_NAMES[idx]}"))
                        elif "train" in text:
                            self.root.after(0, self.retrain_model)
                    except:
                        pass
            
            self.status_label.config(text="🎤 Voice Control ON: Say 'Capture', 'Next', or 'Train'")
            threading.Thread(target=listen_loop, daemon=True).start()
        else:
            self.status_label.config(text="Voice Control OFF")

    def _on_auto_detect_toggle(self):
        """Handle auto-detect hand checkbox toggle"""
        self.auto_detect_hand = self.auto_detect_var.get()
        if self.auto_detect_hand:
            self.status_label.config(text="🖐️ Auto Hand Detection ON - Hand bbox auto-drawn")
            # Disable full frame mode when auto-detect is on
            self.full_frame_var.set(False)
            self.use_full_frame = False
        else:
            self.status_label.config(text="Hand Detection OFF")
            self.detected_hand_bbox = None
    
    def _on_padding_change(self, value):
        """Handle padding slider change"""
        self.hand_detection_padding = int(float(value))
    
    def _detect_hand(self, frame):
        """Detect hands (up to 2) in frame using MediaPipe or skin color detection"""
        # Try MediaPipe first if available
        if MEDIAPIPE_AVAILABLE and self.hands is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                h, w, _ = frame.shape
                # Collect landmarks from ALL detected hands (up to 2)
                all_x = []
                all_y = []
                for hand_landmarks in results.multi_hand_landmarks:
                    all_x.extend([lm.x * w for lm in hand_landmarks.landmark])
                    all_y.extend([lm.y * h for lm in hand_landmarks.landmark])
                
                # Single bounding box encompassing both hands
                x_min = int(max(0, min(all_x) - self.hand_detection_padding))
                y_min = int(max(0, min(all_y) - self.hand_detection_padding))
                x_max = int(min(w, max(all_x) + self.hand_detection_padding))
                y_max = int(min(h, max(all_y) + self.hand_detection_padding))
                
                return (x_min, y_min, x_max, y_max)
            return None
        
        # Fallback: Skin color detection using HSV
        return self._detect_hand_by_skin_color(frame)
    
    def _detect_hand_by_skin_color(self, frame):
        """Detect hand using skin color segmentation in HSV space"""
        h, w, _ = frame.shape
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Skin color range in HSV (works for various skin tones)
        lower_skin = np.array([0, 30, 60], dtype=np.uint8)
        upper_skin = np.array([20, 150, 255], dtype=np.uint8)
        
        # Create mask
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get top 2 largest contours (both hands)
            min_area = (h * w) * 0.02  # At least 2% of frame
            valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
            valid_contours.sort(key=cv2.contourArea, reverse=True)
            valid_contours = valid_contours[:2]  # Keep up to 2 hands
            
            if valid_contours:
                # Merge bounding boxes of all valid contours
                x_min_all, y_min_all = w, h
                x_max_all, y_max_all = 0, 0
                for contour in valid_contours:
                    cx, cy, cbw, cbh = cv2.boundingRect(contour)
                    x_min_all = min(x_min_all, cx)
                    y_min_all = min(y_min_all, cy)
                    x_max_all = max(x_max_all, cx + cbw)
                    y_max_all = max(y_max_all, cy + cbh)
                
                # Add padding
                x_min = max(0, x_min_all - self.hand_detection_padding)
                y_min = max(0, y_min_all - self.hand_detection_padding)
                x_max = min(w, x_max_all + self.hand_detection_padding)
                y_max = min(h, y_max_all + self.hand_detection_padding)
                
                return (x_min, y_min, x_max, y_max)
        
        return None
    
    def start_batch_capture(self):
        """Start batch capture mode"""
        if not self.running:
            messagebox.showwarning("Warning", "Please start the camera first!")
            return
        
        if not self.use_full_frame and (self.bbox_start is None or self.bbox_end is None):
            messagebox.showwarning("Warning", 
                "Please either:\n1. Draw a bounding box first, OR\n2. Enable 'Use Full Frame' mode")
            return
        
        try:
            self.batch_capture_count = int(self.batch_var.get())
        except:
            self.batch_capture_count = 10
        
        self.batch_remaining = self.batch_capture_count
        self.batch_capturing = True
        self.batch_btn.config(text="⏹ Stop Batch", command=self.stop_batch_capture, bg="#e74c3c")
        self.status_label.config(text=f"🚀 Batch Capture: {self.batch_remaining} images remaining...")
        self.last_capture_time = time.time()
    
    def stop_batch_capture(self):
        """Stop batch capture mode"""
        self.batch_capturing = False
        self.batch_remaining = 0
        self.batch_btn.config(text="🚀 Start Batch", command=self.start_batch_capture, bg="#1abc9c")
        captured = self.batch_capture_count - self.batch_remaining
        self.status_label.config(text=f"Batch stopped. Captured {captured} images.")
    
    def start_auto_cycle(self):
        """Start auto-cycle mode: cycles through all signs, 5s countdown each"""
        if not self.running:
            messagebox.showwarning("Warning", "Please start the camera first!")
            return
        
        try:
            self.auto_cycle_images_per_sign = int(self.cycle_imgs_var.get())
        except:
            self.auto_cycle_images_per_sign = 10
        
        self.auto_cycle_active = True
        self.auto_cycle_sign_index = 0
        self.auto_cycle_captured_for_current = 0
        self.auto_cycle_phase = 'countdown'
        self.auto_cycle_countdown_start = time.time()
        self.last_capture_time = time.time()
        
        # Force full-frame mode
        self.full_frame_var.set(True)
        self.use_full_frame = True
        
        # Select first sign
        self.class_combo.set(self.CLASS_NAMES[0])
        self._update_progress_display()
        
        self.auto_cycle_btn.config(text="⏹ Stop Auto-Cycle", command=self.stop_auto_cycle, bg="#e74c3c")
        self.status_label.config(text=f"🔁 Auto-Cycle: Prepare sign '{self.CLASS_NAMES[0]}' — 5s countdown...")
    
    def stop_auto_cycle(self):
        """Stop auto-cycle mode"""
        self.auto_cycle_active = False
        self.auto_cycle_phase = 'idle'
        self.auto_cycle_btn.config(text="🔁 Auto-Cycle All Signs", command=self.start_auto_cycle, bg="#e67e22")
        self.auto_cycle_status.config(text="")
        total = self.auto_cycle_sign_index * self.auto_cycle_images_per_sign + self.auto_cycle_captured_for_current
        self.status_label.config(text=f"Auto-Cycle stopped. Total captured: {total} images.")
    
    def _auto_cycle_tick(self):
        """Handle auto-cycle state machine"""
        if not self.auto_cycle_active:
            return
        
        current_time = time.time()
        current_sign = self.CLASS_NAMES[self.auto_cycle_sign_index]
        
        if self.auto_cycle_phase == 'countdown':
            elapsed = current_time - self.auto_cycle_countdown_start
            remaining = max(0, self.auto_cycle_sign_delay - elapsed)
            self.auto_cycle_countdown = int(remaining) + 1
            
            sign_num = self.auto_cycle_sign_index + 1
            total_signs = len(self.CLASS_NAMES)
            self.auto_cycle_status.config(
                text=f"Sign {sign_num}/{total_signs}: '{current_sign}' — {self.auto_cycle_countdown}s")
            self.status_label.config(
                text=f"🔁 Get ready! Show sign '{current_sign.upper()}' in {self.auto_cycle_countdown}s...")
            
            if remaining <= 0:
                # Countdown done, start capturing
                self.auto_cycle_phase = 'capturing'
                self.auto_cycle_captured_for_current = 0
                self.last_capture_time = current_time
        
        elif self.auto_cycle_phase == 'capturing':
            # Capture at the auto_capture_interval rate
            if current_time - self.last_capture_time >= self.auto_capture_interval:
                self._capture_and_save_auto()
                self.auto_cycle_captured_for_current += 1
                self.last_capture_time = current_time
                
                remaining = self.auto_cycle_images_per_sign - self.auto_cycle_captured_for_current
                self.auto_cycle_status.config(
                    text=f"📷 '{current_sign}': {self.auto_cycle_captured_for_current}/{self.auto_cycle_images_per_sign}")
                self.status_label.config(
                    text=f"🔁 Capturing '{current_sign.upper()}' — {remaining} images left...")
                
                if self.auto_cycle_captured_for_current >= self.auto_cycle_images_per_sign:
                    # Move to next sign
                    self.auto_cycle_sign_index += 1
                    
                    if self.auto_cycle_sign_index >= len(self.CLASS_NAMES):
                        # All signs done!
                        total = len(self.CLASS_NAMES) * self.auto_cycle_images_per_sign
                        self.auto_cycle_active = False
                        self.auto_cycle_phase = 'idle'
                        self.auto_cycle_btn.config(text="🔁 Auto-Cycle All Signs", 
                                                   command=self.start_auto_cycle, bg="#e67e22")
                        self.auto_cycle_status.config(text="✅ All signs done!")
                        self.status_label.config(text=f"✅ Auto-Cycle complete! Captured {total} images across all signs.")
                        messagebox.showinfo("Auto-Cycle Complete", 
                            f"✅ Captured {self.auto_cycle_images_per_sign} images for each of {len(self.CLASS_NAMES)} signs!\n"
                            f"Total: {total} new images.")
                        return
                    
                    # Start countdown for next sign
                    next_sign = self.CLASS_NAMES[self.auto_cycle_sign_index]
                    self.class_combo.set(next_sign)
                    self._update_progress_display()
                    self.auto_cycle_phase = 'countdown'
                    self.auto_cycle_countdown_start = time.time()
                    self.status_label.config(text=f"🔁 Next sign: '{next_sign.upper()}' — prepare in 5s...")

    def _auto_capture_tick(self):
        """Check if it's time to auto-capture"""
        current_time = time.time()
        
        # Auto-cycle mode takes priority
        if self.auto_cycle_active:
            self._auto_cycle_tick()
            return
        
        # Batch capture mode
        if self.batch_capturing and self.batch_remaining > 0:
            if current_time - self.last_capture_time >= self.auto_capture_interval:
                self._capture_and_save_auto()
                self.batch_remaining -= 1
                self.last_capture_time = current_time
                
                if self.batch_remaining > 0:
                    self.status_label.config(text=f"🚀 Batch: {self.batch_remaining} images remaining...")
                else:
                    self.stop_batch_capture()
                    messagebox.showinfo("Batch Complete", 
                        f"✅ Captured {self.batch_capture_count} images for '{self.class_combo.get()}'!")
        
        # Regular auto-capture mode
        elif self.auto_capture_enabled and self.use_full_frame:
            if current_time - self.last_capture_time >= self.auto_capture_interval:
                self._capture_and_save_auto()
                self.last_capture_time = current_time
    
    def _save_augmented_image_set(self, frame, class_name, class_id, center_x, center_y, width, height, timestamp, unique_id):
        """Saves originally captured frame, plus heavily augmented versions if enabled"""
        images_to_save = [("orig", frame)]
        
        if getattr(self, 'augment_enabled', False):
            # Rotations
            h, w = frame.shape[:2]
            M_left = cv2.getRotationMatrix2D((w/2, h/2), 5, 1.0)
            M_right = cv2.getRotationMatrix2D((w/2, h/2), -5, 1.0)
            images_to_save.append(("rot_L", cv2.warpAffine(frame, M_left, (w, h))))
            images_to_save.append(("rot_R", cv2.warpAffine(frame, M_right, (w, h))))
            
            # Brightness variations
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hh, ss, vv = cv2.split(hsv)
            v_bright = cv2.add(vv, 30)
            v_dark = cv2.subtract(vv, 30)
            images_to_save.append(("bright", cv2.cvtColor(cv2.merge((hh, ss, v_bright)), cv2.COLOR_HSV2BGR)))
            images_to_save.append(("dark", cv2.cvtColor(cv2.merge((hh, ss, v_dark)), cv2.COLOR_HSV2BGR)))

        self.last_saved_files = []
        saved_count = 0
        for aug_suffix, aug_frame in images_to_save:
            filename = f"{class_name.replace(' ', '_')}_{timestamp}_{unique_id}_{aug_suffix}"
            img_path = os.path.join(self.train_images_dir, f"{filename}.jpg")
            lbl_path = os.path.join(self.train_labels_dir, f"{filename}.txt")
            
            cv2.imwrite(img_path, aug_frame)
            with open(lbl_path, 'w') as f:
                f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
            
            self.last_saved_files.extend([img_path, lbl_path])
            saved_count += 1
            
        return saved_count

    def _capture_and_save_auto(self):
        """Automatically capture and save current frame"""
        if self.display_frame is None:
            return
        
        frame_to_save = self.display_frame.copy()
        
        # Get class
        class_name = self.class_combo.get()
        class_id = self.CLASS_NAMES.index(class_name)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        filename = f"{class_name.replace(' ', '_')}_{timestamp}_{unique_id}"
        
        # Calculate bounding box in YOLO format
        h_img, w_img = frame_to_save.shape[:2]
        if self.use_full_frame:
            center_x, center_y, width, height = 0.5, 0.5, 1.0, 1.0
        elif self.bbox_start and self.bbox_end:
            x1, y1 = self.bbox_start
            x2, y2 = self.bbox_end
            center_x = ((x1 + x2) / 2.0) / w_img
            center_y = ((y1 + y2) / 2.0) / h_img
            width = abs(x2 - x1) / w_img
            height = abs(y2 - y1) / h_img
        else:
            # Fallback: use full frame
            center_x, center_y, width, height = 0.5, 0.5, 1.0, 1.0
        
        # Save augmented batch
        saved_count = self._save_augmented_image_set(frame_to_save, class_name, class_id, center_x, center_y, width, height, timestamp, unique_id)
        
        # Trigger capture flash effect
        self.flash_alpha = 6
        
        self.collected_count += saved_count
        self.class_image_counts[class_name] = self.class_image_counts.get(class_name, 0) + saved_count
        self._update_progress_display()

    def _update_progress_display(self):
        """Update all progress displays"""
        if hasattr(self, 'progress_label'):
            self.progress_label.config(text=self._get_current_progress_text())
        self._update_progress_bar()
        if hasattr(self, 'class_list_label'):
            self.class_list_label.config(text=self._format_class_list_with_counts())

    def add_new_sign(self):
        """Add a new sign class"""
        new_sign = simpledialog.askstring("Add New Sign", 
            "Enter the name of the new sign:\n(e.g., 'good morning', 'sorry', 'water')",
            parent=self.root)
        
        if new_sign:
            new_sign = new_sign.strip().lower()
            
            if new_sign in self.CLASS_NAMES:
                messagebox.showwarning("Warning", f"Sign '{new_sign}' already exists!")
                return
            
            if not new_sign:
                messagebox.showwarning("Warning", "Sign name cannot be empty!")
                return
            
            # Add to class list
            self.CLASS_NAMES.append(new_sign)
            
            # Save to data.yaml
            if self._save_class_names():
                # Update UI
                self.class_combo['values'] = self.CLASS_NAMES
                self.class_combo.set(new_sign)
                self.class_list_label.config(text=self._format_class_list_with_counts())
                
                messagebox.showinfo("Success", 
                    f"✅ New sign '{new_sign}' added!\n\n"
                    f"Class ID: {len(self.CLASS_NAMES) - 1}\n"
                    f"Total classes: {len(self.CLASS_NAMES)}\n\n"
                    "Now capture images for this sign and retrain the model.")
                
                # Remind about TTS
                add_tts = messagebox.askyesno("Add TTS Translations?",
                    f"Would you like to add text-to-speech translations for '{new_sign}'?\n\n"
                    "This will open a dialog to enter translations in all languages.")
                
                if add_tts:
                    self._add_tts_translations(new_sign)
            else:
                messagebox.showerror("Error", "Failed to save to data.yaml!")
                self.CLASS_NAMES.remove(new_sign)

    def _add_tts_translations(self, sign_name):
        """Add TTS translations for a new sign — auto-saves to sign_constants.py"""
        languages = LANGUAGES
        lang_codes = LANG_CODES
        
        translations = {}
        
        for lang, code in zip(languages, lang_codes):
            translation = simpledialog.askstring(f"Translation - {lang}",
                f"Enter '{sign_name}' in {lang}:\n\n(Leave empty to use English)",
                parent=self.root)
            
            if translation and translation.strip():
                translations[code] = translation.strip()
            else:
                translations[code] = sign_name.title()
        
        # Auto-append to sign_constants.py
        try:
            with open("sign_constants.py", 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Build the translation line using string concat (no f-string brace issues)
            parts = []
            for code in lang_codes:
                parts.append('"' + code + '": "' + translations[code] + '"')
            dict_str = '{' + ', '.join(parts) + '}'
            new_line = '    "' + sign_name.lower() + '": ' + dict_str + ','
            
            # Insert before the closing brace of TRANSLATIONS dict
            insert_marker = '\n}\n'
            if insert_marker in content:
                content = content.replace(insert_marker, '\n' + new_line + '\n}\n')
                with open("sign_constants.py", 'w', encoding='utf-8') as f:
                    f.write(content)
                print("Auto-saved translations for '" + sign_name + "' to sign_constants.py")
        except Exception as e:
            print("Could not auto-save translations: " + str(e))
        
        # Show summary
        summary = "\n".join([f"{lang}: {translations[code]}" for lang, code in zip(languages, lang_codes)])
        messagebox.showinfo("Translations Added", 
            f"Translations for '{sign_name}':\n\n{summary}\n\n"
            f"✅ Auto-saved to sign_constants.py")

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open webcam!")
            return
        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.capture_btn.config(state=tk.NORMAL)
        self.status_label.config(text=f"Status: Camera running | Images collected: {self.collected_count}")
        self.update_frame()

    def stop_camera(self):
        self.running = False
        self.batch_capturing = False
        self.auto_capture_enabled = False
        self.auto_capture_var.set(False)
        if self.cap:
            self.cap.release()
        self.canvas.delete("all")
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.capture_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.DISABLED)
        self.batch_btn.config(text="🚀 Start Batch", command=self.start_batch_capture, bg="#1abc9c")
        self.status_label.config(text=f"Status: Stopped | Images collected: {self.collected_count}")

    def update_frame(self):
        if not self.running:
            return
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (860, 700))
            self.display_frame = frame.copy()
            
            # Auto-detect hand if enabled
            if self.auto_detect_hand:
                hand_bbox = self._detect_hand(frame)
                if hand_bbox:
                    self.detected_hand_bbox = hand_bbox
                    self.bbox_start = (hand_bbox[0], hand_bbox[1])
                    self.bbox_end = (hand_bbox[2], hand_bbox[3])
                    # Draw detected hand bbox on frame (cyan color)
                    cv2.rectangle(frame, (hand_bbox[0], hand_bbox[1]), 
                                 (hand_bbox[2], hand_bbox[3]), (255, 255, 0), 3)
                    cv2.putText(frame, "HANDS DETECTED", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    self.hand_status_label.config(text="Hands: ✅ Detected", fg="#27ae60")
                else:
                    self.detected_hand_bbox = None
                    self.hand_status_label.config(text="Hand: ❌ Not detected", fg="#e74c3c")
            
            # Add visual indicator for auto-capture/batch/cycle mode
            if self.auto_cycle_active:
                current_sign = self.CLASS_NAMES[self.auto_cycle_sign_index]
                if self.auto_cycle_phase == 'countdown':
                    # Big countdown overlay
                    cv2.rectangle(frame, (0, 0), (860, 700), (0, 0, 0), -1)  # dim background
                    frame[:] = (frame * 0.3).astype(np.uint8)  # darken
                    cv2.putText(frame, f"PREPARE SIGN:", (250, 250),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                    cv2.putText(frame, f"{current_sign.upper()}", (200, 370),
                               cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)
                    cv2.putText(frame, f"{self.auto_cycle_countdown}", (390, 530),
                               cv2.FONT_HERSHEY_SIMPLEX, 4.0, (255, 255, 255), 6)
                    sign_num = self.auto_cycle_sign_index + 1
                    cv2.putText(frame, f"Sign {sign_num}/{len(self.CLASS_NAMES)}", (340, 650),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                elif self.auto_cycle_phase == 'capturing':
                    cv2.putText(frame, f"CAPTURING: {current_sign.upper()}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    captured = self.auto_cycle_captured_for_current
                    total = self.auto_cycle_images_per_sign
                    cv2.putText(frame, f"{captured}/{total}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            elif self.batch_capturing:
                cv2.putText(frame, f"BATCH: {self.batch_remaining} left", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            elif self.auto_capture_enabled:
                cv2.putText(frame, "AUTO-CAPTURE ON", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if self.use_full_frame and not self.auto_cycle_active:
                cv2.putText(frame, "FULL FRAME MODE", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
                           
            # Apply Background Blur if enabled
            if getattr(self, 'blur_bg_enabled', False) and getattr(self, 'segmenter', None):
                import mediapipe as mp
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                seg_result = self.segmenter.segment(mp_image)
                mask = seg_result.category_mask.numpy_view()
                condition = np.stack((mask,) * 3, axis=-1) > 0.1
                bg_blur = cv2.GaussianBlur(frame, (55, 55), 0)
                frame = np.where(condition, frame, bg_blur)

            # Quality Indicator (Blur & Lighting)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            brightness = hsv[:, :, 2].mean()

            quality_color = (0, 255, 0)
            quality_text = "Good"
            if blur_score < 40:
                quality_color = (0, 0, 255)
                quality_text = "Too Blurry!"
            elif brightness < 30:
                quality_color = (0, 255, 255)
                quality_text = "Too Dark!"
            
            cv2.putText(frame, f"Quality: {quality_text}", (10, 690), cv2.FONT_HERSHEY_SIMPLEX, 0.6, quality_color, 2)
            
            # ── LIVE SIGN PREDICTION HUD ──────────────────────────
            if self.live_predict_enabled and self.live_model is not None:
                now = time.time()
                if now - self.last_predict_time >= self.live_predict_interval:
                    self.last_predict_time = now
                    try:
                        results = self.live_model(self.display_frame, conf=0.3, verbose=False)[0]
                        if results.boxes and len(results.boxes) > 0:
                            best_idx = results.boxes.conf.argmax()
                            self.live_predict_conf = results.boxes.conf[best_idx].item()
                            cls_id = int(results.boxes.cls[best_idx].item())
                            self.live_predict_label = self.live_model.names.get(cls_id, "?")
                        else:
                            self.live_predict_label = ""
                            self.live_predict_conf = 0.0
                    except:
                        pass
                
                if self.live_predict_label and self.live_predict_conf > 0.3:
                    # Draw prediction HUD panel (bottom-left)
                    cv2.rectangle(frame, (5, 400), (320, 460), (0, 0, 0), -1)
                    cv2.rectangle(frame, (5, 400), (320, 460), (187, 134, 252), 2)
                    cv2.putText(frame, f"AI: {self.live_predict_label.upper()}", (15, 425),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (187, 134, 252), 2)
                    # Confidence bar
                    bar_w = int(180 * self.live_predict_conf)
                    bar_color = (0, 230, 118) if self.live_predict_conf > 0.6 else (0, 200, 255)
                    cv2.rectangle(frame, (15, 435), (15 + bar_w, 452), bar_color, -1)
                    cv2.rectangle(frame, (15, 435), (195, 452), (100, 100, 100), 1)
                    cv2.putText(frame, f"{self.live_predict_conf:.0%}", (205, 450),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Match indicator
                    selected = self.class_combo.get().lower()
                    if self.live_predict_label.lower() == selected:
                        cv2.putText(frame, "MATCH!", (255, 425),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 118), 2)
                    
                    # ── TTS: Speak detected sign ──
                    if self.tts_enabled and TTS_AVAILABLE and self.live_predict_conf > 0.5:
                        now_tts = time.time()
                        detected = self.live_predict_label.lower()
                        if detected != self.last_tts_sign or (now_tts - self.last_tts_time) > self.TTS_COOLDOWN:
                            self.last_tts_sign = detected
                            self.last_tts_time = now_tts
                            speak_sign(detected, self.current_lang_idx)
                            lang_name = LANGUAGES[self.current_lang_idx]
                            self.root.after(0, lambda l=detected, ln=lang_name: 
                                self.tts_status_label.config(
                                    text=f"🔊 Spoke: '{l}' in {ln}", fg="#7bed9f"))
            
            # ── CAPTURE COUNTDOWN TIMER ───────────────────────────
            if (self.auto_capture_enabled or self.batch_capturing) and not self.auto_cycle_active:
                elapsed = time.time() - self.last_capture_time
                remaining = max(0, self.auto_capture_interval - elapsed)
                if remaining > 0:
                    # Draw countdown arc
                    progress = 1.0 - (remaining / self.auto_capture_interval)
                    angle = int(360 * progress)
                    center = (810, 660)
                    cv2.ellipse(frame, center, (25, 25), -90, 0, angle, (0, 230, 118), 3)
                    cv2.ellipse(frame, center, (25, 25), -90, angle, 360, (80, 80, 80), 1)
                    cv2.putText(frame, f"{remaining:.0f}", (center[0]-8, center[1]+7),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # ── CAPTURE FLASH EFFECT ──────────────────────────────
            if self.flash_alpha > 0:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (860, 700), (0, 230, 118), -1)
                alpha = min(0.4, self.flash_alpha / 10.0)
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                self.flash_alpha -= 1
            
            # Add capture count indicator (top-right corner)
            current_class = self.class_combo.get()
            class_count = self.class_image_counts.get(current_class, 0)
            
            # Background rectangle for better visibility
            cv2.rectangle(frame, (650, 5), (855, 75), (0, 0, 0), -1)
            cv2.rectangle(frame, (650, 5), (855, 75), (0, 255, 0), 2)
            
            # Session total
            cv2.putText(frame, f"Session: {self.collected_count}", (660, 28),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Current class count
            cv2.putText(frame, f"{current_class}: {class_count}/{self.TARGET_IMAGES}", (660, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.imgtk = imgtk
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            
            # Redraw bounding box if exists (manual mode - not auto-detect)
            if not self.auto_detect_hand and self.bbox_start and self.bbox_end:
                self.canvas.create_rectangle(
                    self.bbox_start[0], self.bbox_start[1],
                    self.bbox_end[0], self.bbox_end[1],
                    outline="#00ff00", width=3, tags="bbox"
                )
            
            # Check for auto-capture
            self._auto_capture_tick()
            
        self.root.after(30, self.update_frame)

    def capture_frame(self):
        if self.display_frame is not None:
            self.current_frame = self.display_frame.copy()
            self.running = False  # Pause camera to draw bbox
            
            # Display captured frame
            cv2image = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.imgtk = imgtk
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            
            self.status_label.config(text="Status: Frame captured - Draw bounding box around the hand sign")
            self.capture_btn.config(text="📷 Resume Camera", command=self.resume_camera)
            self.save_btn.config(state=tk.NORMAL)

    def resume_camera(self):
        self.running = True
        self.current_frame = None
        self.bbox_start = None
        self.bbox_end = None
        self.capture_btn.config(text="📷 Capture Frame", command=self.capture_frame)
        self.save_btn.config(state=tk.DISABLED)
        self.bbox_label.config(text="Bounding Box: Not drawn")
        self.update_frame()

    def on_mouse_down(self, event):
        if self.current_frame is not None:
            self.bbox_start = (event.x, event.y)
            self.drawing = True

    def on_mouse_drag(self, event):
        if self.drawing and self.current_frame is not None:
            self.bbox_end = (event.x, event.y)
            self.canvas.delete("bbox")
            self.canvas.create_rectangle(
                self.bbox_start[0], self.bbox_start[1],
                self.bbox_end[0], self.bbox_end[1],
                outline="#00ff00", width=3, tags="bbox"
            )

    def on_mouse_up(self, event):
        if self.drawing and self.current_frame is not None:
            self.bbox_end = (event.x, event.y)
            self.drawing = False
            
            # Ensure valid bbox
            x1 = min(self.bbox_start[0], self.bbox_end[0])
            y1 = min(self.bbox_start[1], self.bbox_end[1])
            x2 = max(self.bbox_start[0], self.bbox_end[0])
            y2 = max(self.bbox_start[1], self.bbox_end[1])
            
            self.bbox_start = (x1, y1)
            self.bbox_end = (x2, y2)
            
            self.bbox_label.config(text=f"Bounding Box: ({x1}, {y1}) to ({x2}, {y2})")

    def clear_bbox(self):
        self.bbox_start = None
        self.bbox_end = None
        self.canvas.delete("bbox")
        self.bbox_label.config(text="Bounding Box: Not drawn")

    def save_labeled_image(self):
        if self.current_frame is None:
            messagebox.showwarning("Warning", "No frame captured!")
            return
        if self.bbox_start is None or self.bbox_end is None:
            messagebox.showwarning("Warning", "Please draw a bounding box around the hand sign!")
            return

        # Get class
        class_name = self.class_combo.get()
        class_id = self.CLASS_NAMES.index(class_name)

        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        filename = f"{class_name.replace(' ', '_')}_{timestamp}_{unique_id}"

        # Calculate YOLO bounding box
        h_img, w_img = self.current_frame.shape[:2]
        x1, y1 = self.bbox_start
        x2, y2 = self.bbox_end
        center_x = ((x1 + x2) / 2.0) / w_img
        center_y = ((y1 + y2) / 2.0) / h_img
        width = abs(x2 - x1) / w_img
        height = abs(y2 - y1) / h_img

        # Save augmented batch
        saved_count = self._save_augmented_image_set(self.current_frame, class_name, class_id, center_x, center_y, width, height, timestamp, unique_id)

        self.collected_count += saved_count
        self.class_image_counts[class_name] = self.class_image_counts.get(class_name, 0) + saved_count
        current_count = self.class_image_counts[class_name]
        remaining = max(0, self.TARGET_IMAGES - current_count)
        
        self.status_label.config(text=f"✅ Saved: {class_name} | Total collected: {self.collected_count}")
        
        # Update progress display
        self._update_progress_display()
        
        # Show different message based on progress
        if current_count >= self.TARGET_IMAGES:
            msg = f"🎉 Image saved!\n\n" \
                  f"Sign: {class_name}\n" \
                  f"Images for this sign: {current_count}\n\n" \
                  f"✅ Target reached! You can add more or move to another sign."
        else:
            msg = f"📷 Image saved!\n\n" \
                  f"Sign: {class_name}\n" \
                  f"Progress: {current_count}/{self.TARGET_IMAGES}\n" \
                  f"Remaining: {remaining} more images needed"
        
        messagebox.showinfo("Saved", msg)
        
        # Resume camera
        self.resume_camera()

    def retrain_model(self):
        # Refresh image counts
        self.class_image_counts = self._count_existing_images()
        self._update_progress_display()
        
        # Auto-split data into train/val/test (80/15/5)
        self._auto_split_data()
        
        # Build detailed dataset summary
        total_images = sum(self.class_image_counts.values())
        lines = []
        warnings = 0
        for name in self.CLASS_NAMES:
            count = self.class_image_counts.get(name, 0)
            if count < 10:
                lines.append(f"  \u274c {name}: {count} images (too few!)")
                warnings += 1
            elif count < self.TARGET_IMAGES:
                lines.append(f"  \U0001f7e1 {name}: {count} images")
            else:
                lines.append(f"  \u2705 {name}: {count} images")
        
        summary = "\n".join(lines)
        
        warn_msg = ""
        if warnings > 0:
            warn_msg = f"\n\u26a0\ufe0f {warnings} class(es) have fewer than 10 images!\nConsider collecting more data first.\n"
        
        result = messagebox.askyesno("\U0001f680 Train Model", 
            f"Start training with collected images?\n\n"
            f"\U0001f4ca Dataset Summary:\n"
            f"  Classes: {len(self.CLASS_NAMES)}\n"
            f"  Total images: {total_images}\n\n"
            f"{summary}\n"
            f"{warn_msg}\n"
            f"Training will open in a new window.\n"
            f"This may take 1-3 hours depending on your hardware.")
        
        if result:
            # Delete old cache files
            cache_files = [
                "Data/train/labels.cache",
                "Data/valid/labels.cache", 
                "Data/test/labels.cache"
            ]
            for cache in cache_files:
                if os.path.exists(cache):
                    os.remove(cache)
                    print(f"Deleted cache: {cache}")
            
            import subprocess
            subprocess.Popen(["python", "train.py"], 
                           creationflags=subprocess.CREATE_NEW_CONSOLE)
            self.status_label.config(text="\U0001f680 Training started in new window! You can continue collecting data.")
            messagebox.showinfo("Training Started", 
                "\u2705 Training started in a new window!\n\n"
                "You can continue collecting more data.\n"
                "Check the training window for progress.\n\n"
                "After training completes, the new model\n"
                "will be auto-deployed to best.pt")

    def _train_selected_sign(self):
        """Train model using only the currently selected sign's captured images."""
        selected_sign = self.class_combo.get()
        if not selected_sign:
            messagebox.showwarning("No Sign Selected", "Please select a sign class first!")
            return
        
        # Refresh counts
        self.class_image_counts = self._count_existing_images()
        selected_count = self.class_image_counts.get(selected_sign, 0)
        
        if selected_count < 3:
            messagebox.showwarning("Not Enough Data", 
                f"Sign '{selected_sign}' only has {selected_count} images!\n"
                f"Capture at least 3 images for this sign first.")
            return
        
        selected_class_id = self.CLASS_NAMES.index(selected_sign)
        
        result = messagebox.askyesno(f"🔥 Train '{selected_sign}'",
            f"Train model for sign: '{selected_sign.upper()}'\n\n"
            f"📸 Images for '{selected_sign}': {selected_count}\n"
            f"🔧 Fine-tune from: best.pt (existing model)\n"
            f"⚡ Epochs: 15 (focused training)\n\n"
            f"This will create a temporary dataset with only\n"
            f"'{selected_sign}' images and fine-tune the model.\n\n"
            f"Continue?")
        
        if not result:
            return
        
        # ── Build temp dataset with only selected sign's images ──
        temp_dir = os.path.join("Data", "_temp_selected_train")
        temp_train_imgs = os.path.join(temp_dir, "train", "images")
        temp_train_lbls = os.path.join(temp_dir, "train", "labels")
        temp_val_imgs = os.path.join(temp_dir, "valid", "images")
        temp_val_lbls = os.path.join(temp_dir, "valid", "labels")
        
        # Clean up any previous temp data
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        for d in [temp_train_imgs, temp_train_lbls, temp_val_imgs, temp_val_lbls]:
            os.makedirs(d, exist_ok=True)
        
        # Copy only the selected sign's images + labels
        import random
        copied_files = []
        for label_file in os.listdir(self.train_labels_dir):
            if not label_file.endswith('.txt'):
                continue
            label_path = os.path.join(self.train_labels_dir, label_file)
            with open(label_path, 'r') as f:
                lines = f.readlines()
            # Check if this label file contains the selected class
            has_selected = any(line.strip().startswith(f"{selected_class_id} ") for line in lines)
            if has_selected:
                img_name = os.path.splitext(label_file)[0]
                # Find the matching image
                for ext in ['.jpg', '.png', '.jpeg']:
                    img_path = os.path.join(self.train_images_dir, img_name + ext)
                    if os.path.exists(img_path):
                        copied_files.append((img_path, label_path, img_name + ext, label_file))
                        break
        
        if not copied_files:
            messagebox.showwarning("No Images Found", 
                f"Could not find labeled images for '{selected_sign}'!")
            return
        
        # Shuffle and split 80/20 for train/val
        random.shuffle(copied_files)
        val_count = max(1, len(copied_files) // 5)
        val_set = copied_files[:val_count]
        train_set = copied_files[val_count:]
        if not train_set:
            train_set = copied_files  # If too few, use all for training too
        
        for img_path, lbl_path, img_name, lbl_name in train_set:
            shutil.copy2(img_path, os.path.join(temp_train_imgs, img_name))
            shutil.copy2(lbl_path, os.path.join(temp_train_lbls, lbl_name))
        for img_path, lbl_path, img_name, lbl_name in val_set:
            shutil.copy2(img_path, os.path.join(temp_val_imgs, img_name))
            shutil.copy2(lbl_path, os.path.join(temp_val_lbls, lbl_name))
        
        # Write temp data.yaml pointing to this filtered dataset
        temp_yaml_path = os.path.join(temp_dir, "data.yaml")
        abs_temp_dir = os.path.abspath(temp_dir).replace("\\", "/")
        with open(temp_yaml_path, 'w') as f:
            yaml.dump({
                'path': abs_temp_dir,
                'train': 'train/images',
                'val': 'valid/images',
                'nc': len(self.CLASS_NAMES),
                'names': self.CLASS_NAMES
            }, f, default_flow_style=False, allow_unicode=True)
        
        train_count = len(train_set)
        val_c = len(val_set)
        
        # ── Auto-split the main data too (for FULL TRAIN) ──
        self._auto_split_data()
        
        # Clear caches
        for cache_dir in [temp_dir, "Data"]:
            for sub in ["train/labels.cache", "valid/labels.cache", "test/labels.cache"]:
                cache_path = os.path.join(cache_dir, sub)
                if os.path.exists(cache_path):
                    os.remove(cache_path)
        
        # Disable buttons during training
        self.quick_train_btn.config(state=tk.DISABLED, text="\u23f3 Training...")
        self.retrain_btn.config(state=tk.DISABLED)
        if hasattr(self, 'train_btn'):
            self.train_btn.config(state=tk.DISABLED, text="\u23f3 Training...")
        self.status_label.config(text=f"\u26a1 Training '{selected_sign}' in progress...")
        
        # ── Open the Training Dashboard Window ──
        self._open_train_dashboard(train_count + val_c, sign_name=selected_sign)
        
        # ── Training state shared between threads ──
        self.train_epochs = 15
        self.train_start_time = time.time()
        self.train_loss_history = []
        
        def train_thread():
            try:
                import torch
                from ultralytics import YOLO
                
                device = 0 if torch.cuda.is_available() else 'cpu'
                dev_name = 'GPU \u26a1' if device == 0 else 'CPU'
                self.root.after(0, lambda: self._dashboard_update_device(dev_name))
                
                # Fine-tune from existing best.pt if available, else yolo11s.pt
                base_model = "best.pt" if os.path.exists("best.pt") else "yolo11s.pt"
                model = YOLO(base_model)
                
                # ── Register callback for live metrics ──
                def on_train_epoch_end(trainer):
                    epoch = trainer.epoch + 1
                    loss_items = trainer.label_loss_items(trainer.tloss)
                    box_loss = loss_items.get('train/box_loss', 0)
                    cls_loss = loss_items.get('train/cls_loss', 0)
                    dfl_loss = loss_items.get('train/dfl_loss', 0)
                    self.train_loss_history.append((epoch, box_loss, cls_loss, dfl_loss))
                    self.root.after(0, lambda e=epoch, b=box_loss, c=cls_loss, d=dfl_loss:
                                    self._dashboard_update_epoch(e, b, c, d))
                
                model.add_callback("on_train_epoch_end", on_train_epoch_end)
                
                model.train(
                    data=os.path.abspath(temp_yaml_path),
                    epochs=15,
                    imgsz=640,
                    batch=8,
                    name="sign_lang_yolo11",
                    project="runs/train",
                    exist_ok=True,
                    device=device,
                    optimizer="AdamW",
                    lr0=0.0005,
                    lrf=0.01,
                    cos_lr=True,
                    warmup_epochs=1.0,
                    weight_decay=0.001,
                    dropout=0.1,
                    mosaic=1.0,
                    mixup=0.0,
                    flipud=0.0,
                    fliplr=0.0,
                    degrees=10.0,
                    hsv_h=0.015,
                    hsv_s=0.7,
                    hsv_v=0.4,
                    freeze=10,
                    patience=8,
                    workers=4,
                    cache='ram',
                    verbose=True,
                )
                
                # Auto-deploy best model
                best_path = "runs/train/sign_lang_yolo11/weights/best.pt"
                if os.path.exists(best_path):
                    shutil.copy2(best_path, "best.pt")
                    try:
                        self.live_model = YOLO("best.pt", task='detect')
                    except:
                        pass
                
                # Cleanup temp directory
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass
                
                self.root.after(0, lambda: self._on_quick_train_done(True, sign_name=selected_sign))
            except Exception as e:
                # Cleanup temp directory on failure too
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass
                self.root.after(0, lambda: self._on_quick_train_done(False, str(e)))
        
        threading.Thread(target=train_thread, daemon=True).start()
        self._dashboard_tick_timer()

    def _quick_train(self):
        """Full training with ALL signs (20 epochs) with live dashboard."""
        # Refresh counts
        self.class_image_counts = self._count_existing_images()
        total_images = sum(self.class_image_counts.values())
        
        if total_images < 5:
            messagebox.showwarning("Not Enough Data", 
                "Need at least 5 images to train!\n"
                "Capture more images first.")
            return
        
        result = messagebox.askyesno("\u26a1 Full Train (All Signs)",
            f"Train on ALL sign images?\n\n"
            f"Images: {total_images}\n"
            f"Classes: {len(self.CLASS_NAMES)}\n"
            f"Epochs: 20\n\n"
            f"This trains on ALL captured data with a LIVE dashboard.\n"
            f"The model will be ready to use immediately!")
        
        if not result:
            return
        
        # Auto-split data
        self._auto_split_data()
        
        # Clear caches
        for cache in ["Data/train/labels.cache", "Data/valid/labels.cache", "Data/test/labels.cache"]:
            if os.path.exists(cache):
                os.remove(cache)
        
        # Disable buttons during training
        self.quick_train_btn.config(state=tk.DISABLED, text="\u23f3 Training...")
        self.retrain_btn.config(state=tk.DISABLED)
        if hasattr(self, 'train_btn'):
            self.train_btn.config(state=tk.DISABLED, text="\u23f3 Training...")
        self.status_label.config(text="\u26a1 Full Training in progress...")
        
        # ── Open the Training Dashboard Window ──
        self._open_train_dashboard(total_images)
        
        # ── Training state shared between threads ──
        self.train_epochs = 20
        self.train_start_time = time.time()
        self.train_loss_history = []
        
        def train_thread():
            try:
                import torch
                from ultralytics import YOLO
                
                device = 0 if torch.cuda.is_available() else 'cpu'
                dev_name = 'GPU \u26a1' if device == 0 else 'CPU'
                self.root.after(0, lambda: self._dashboard_update_device(dev_name))
                
                model = YOLO("yolo11s.pt")
                
                def on_train_epoch_end(trainer):
                    epoch = trainer.epoch + 1
                    loss_items = trainer.label_loss_items(trainer.tloss)
                    box_loss = loss_items.get('train/box_loss', 0)
                    cls_loss = loss_items.get('train/cls_loss', 0)
                    dfl_loss = loss_items.get('train/dfl_loss', 0)
                    self.train_loss_history.append((epoch, box_loss, cls_loss, dfl_loss))
                    self.root.after(0, lambda e=epoch, b=box_loss, c=cls_loss, d=dfl_loss:
                                    self._dashboard_update_epoch(e, b, c, d))
                
                model.add_callback("on_train_epoch_end", on_train_epoch_end)
                
                model.train(
                    data="Data/data.yaml",
                    epochs=20,
                    imgsz=640,
                    batch=8,
                    name="sign_lang_yolo11",
                    project="runs/train",
                    exist_ok=True,
                    device=device,
                    optimizer="AdamW",
                    lr0=0.001,
                    lrf=0.01,
                    cos_lr=True,
                    warmup_epochs=2.0,
                    weight_decay=0.001,
                    dropout=0.1,
                    mosaic=1.0,
                    mixup=0.0,
                    flipud=0.0,
                    fliplr=0.0,
                    degrees=10.0,
                    hsv_h=0.015,
                    hsv_s=0.7,
                    hsv_v=0.4,
                    freeze=10,
                    patience=10,
                    workers=4,
                    cache='ram',
                    verbose=True,
                )
                
                # Auto-deploy best model
                best_path = "runs/train/sign_lang_yolo11/weights/best.pt"
                if os.path.exists(best_path):
                    shutil.copy2(best_path, "best.pt")
                    try:
                        self.live_model = YOLO("best.pt", task='detect')
                    except:
                        pass
                
                self.root.after(0, lambda: self._on_quick_train_done(True))
            except Exception as e:
                self.root.after(0, lambda: self._on_quick_train_done(False, str(e)))
        
        threading.Thread(target=train_thread, daemon=True).start()
        self._dashboard_tick_timer()

    # ═══════════════════════════════════════════════════════════════
    #  TRAINING DASHBOARD WINDOW
    # ═══════════════════════════════════════════════════════════════
    def _open_train_dashboard(self, total_images, sign_name=None):
        """Create a live training progress window."""
        self.dash = tk.Toplevel(self.root)
        if sign_name:
            self.dash.title(f"🔥 Training '{sign_name}' — Live Progress")
        else:
            self.dash.title("🔥 Training Dashboard — Live Progress")
        self.dash.geometry("620x580")
        self.dash.configure(bg="#1a1a2e")
        self.dash.resizable(False, False)
        self.dash.attributes("-topmost", True)
        
        # ── Header ──
        hdr = tk.Frame(self.dash, bg="#16213e", pady=10)
        hdr.pack(fill=tk.X)
        if sign_name:
            tk.Label(hdr, text=f"🔥 Training Sign: '{sign_name.upper()}'", font=("Segoe UI", 16, "bold"),
                     bg="#16213e", fg="#e94560").pack()
            epochs = self.train_epochs if hasattr(self, 'train_epochs') else 15
            tk.Label(hdr, text=f"📸 {total_images} images  |  Fine-tuning from best.pt  |  {epochs} epochs",
                     font=("Segoe UI", 10), bg="#16213e", fg="#a0a0a0").pack()
        else:
            tk.Label(hdr, text="🔥 YOLO Training Dashboard", font=("Segoe UI", 16, "bold"),
                     bg="#16213e", fg="#e94560").pack()
            tk.Label(hdr, text=f"📊 {total_images} images  |  {len(self.CLASS_NAMES)} classes  |  20 epochs",
                     font=("Segoe UI", 10), bg="#16213e", fg="#a0a0a0").pack()
        
        # ── Device & Timer Row ──
        info_row = tk.Frame(self.dash, bg="#1a1a2e", pady=5)
        info_row.pack(fill=tk.X, padx=20)
        self.dash_device_label = tk.Label(info_row, text="⏳ Detecting device...",
                                          font=("Consolas", 11), bg="#1a1a2e", fg="#0f3460")
        self.dash_device_label.pack(side=tk.LEFT)
        self.dash_timer_label = tk.Label(info_row, text="⏱ 0:00",
                                         font=("Consolas", 11, "bold"), bg="#1a1a2e", fg="#e94560")
        self.dash_timer_label.pack(side=tk.RIGHT)
        
        # ── Epoch Progress ──
        prog_frame = tk.Frame(self.dash, bg="#1a1a2e", pady=8)
        prog_frame.pack(fill=tk.X, padx=20)
        self.dash_epoch_label = tk.Label(prog_frame, text="Epoch: 0 / 20",
                                         font=("Segoe UI", 13, "bold"), bg="#1a1a2e", fg="white")
        self.dash_epoch_label.pack(anchor=tk.W)
        
        # Custom progress bar using canvas
        self.dash_prog_canvas = tk.Canvas(prog_frame, width=560, height=28, bg="#0f3460",
                                           highlightthickness=0, bd=0)
        self.dash_prog_canvas.pack(pady=5)
        self.dash_prog_canvas.create_rectangle(0, 0, 0, 28, fill="#e94560", tags="bar")
        self.dash_prog_canvas.create_text(280, 14, text="0%", fill="white",
                                           font=("Segoe UI", 11, "bold"), tags="pct")
        
        # ── Loss Values ──
        loss_frame = tk.LabelFrame(self.dash, text="📉 Training Losses", font=("Segoe UI", 11, "bold"),
                                    bg="#16213e", fg="white", padx=15, pady=10)
        loss_frame.pack(fill=tk.X, padx=20, pady=8)
        
        loss_grid = tk.Frame(loss_frame, bg="#16213e")
        loss_grid.pack(fill=tk.X)
        
        # Box Loss
        tk.Label(loss_grid, text="📦 Box Loss:", font=("Segoe UI", 11), bg="#16213e", fg="#a0a0a0"
                 ).grid(row=0, column=0, sticky=tk.W, padx=5, pady=3)
        self.dash_box_loss = tk.Label(loss_grid, text="—", font=("Consolas", 14, "bold"),
                                      bg="#16213e", fg="#00d2ff")
        self.dash_box_loss.grid(row=0, column=1, sticky=tk.W, padx=10)
        
        # Class Loss
        tk.Label(loss_grid, text="🏷️ Class Loss:", font=("Segoe UI", 11), bg="#16213e", fg="#a0a0a0"
                 ).grid(row=0, column=2, sticky=tk.W, padx=5, pady=3)
        self.dash_cls_loss = tk.Label(loss_grid, text="—", font=("Consolas", 14, "bold"),
                                      bg="#16213e", fg="#7bed9f")
        self.dash_cls_loss.grid(row=0, column=3, sticky=tk.W, padx=10)
        
        # DFL Loss
        tk.Label(loss_grid, text="🎯 DFL Loss:", font=("Segoe UI", 11), bg="#16213e", fg="#a0a0a0"
                 ).grid(row=1, column=0, sticky=tk.W, padx=5, pady=3)
        self.dash_dfl_loss = tk.Label(loss_grid, text="—", font=("Consolas", 14, "bold"),
                                      bg="#16213e", fg="#ffa502")
        self.dash_dfl_loss.grid(row=1, column=1, sticky=tk.W, padx=10)
        
        # Total Loss
        tk.Label(loss_grid, text="Σ Total:", font=("Segoe UI", 11, "bold"), bg="#16213e", fg="#a0a0a0"
                 ).grid(row=1, column=2, sticky=tk.W, padx=5, pady=3)
        self.dash_total_loss = tk.Label(loss_grid, text="—", font=("Consolas", 14, "bold"),
                                        bg="#16213e", fg="#ff6b81")
        self.dash_total_loss.grid(row=1, column=3, sticky=tk.W, padx=10)
        
        # ── Loss Chart ──
        chart_frame = tk.LabelFrame(self.dash, text="📈 Loss Over Epochs", font=("Segoe UI", 11, "bold"),
                                     bg="#16213e", fg="white", padx=10, pady=5)
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)
        
        self.dash_chart = tk.Canvas(chart_frame, bg="#0f3460", highlightthickness=0, height=160)
        self.dash_chart.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # ── Legend ──
        legend = tk.Frame(chart_frame, bg="#16213e")
        legend.pack(fill=tk.X, padx=5)
        for color, name in [("#00d2ff", "Box"), ("#7bed9f", "Class"), ("#ffa502", "DFL")]:
            tk.Canvas(legend, width=12, height=12, bg=color, highlightthickness=0).pack(side=tk.LEFT, padx=(8, 2))
            tk.Label(legend, text=name, font=("Segoe UI", 9), bg="#16213e", fg="#a0a0a0").pack(side=tk.LEFT)
        
        # ── Status ──
        self.dash_status = tk.Label(self.dash, text="⏳ Initializing model and dataset...",
                                     font=("Segoe UI", 11, "bold"), bg="#1a1a2e", fg="#ffa502")
        self.dash_status.pack(pady=8)
        
        # Animation dots counter
        self.dash_dots = 0
        self.dash_training_active = True
    
    def _dashboard_update_device(self, dev_name):
        """Update the device label on the dashboard."""
        if hasattr(self, 'dash') and self.dash.winfo_exists():
            self.dash_device_label.config(text=f"🖥️ Device: {dev_name}", fg="#7bed9f")
            self.dash_status.config(text="🚀 Training started! Watching for metrics...")
    
    def _dashboard_tick_timer(self):
        """Update elapsed time every second."""
        if not hasattr(self, 'dash_training_active') or not self.dash_training_active:
            return
        if not hasattr(self, 'dash') or not self.dash.winfo_exists():
            return
        
        elapsed = int(time.time() - self.train_start_time)
        mins, secs = divmod(elapsed, 60)
        self.dash_timer_label.config(text=f"⏱ {mins}:{secs:02d}")
        
        # Animate status dots
        self.dash_dots = (self.dash_dots + 1) % 4
        dots = "." * self.dash_dots
        current = self.dash_status.cget("text").rstrip(".")
        # Only animate if it's a "working" status
        if "Training" in current or "Initializing" in current or "Watching" in current:
            base = current.split("...")[0].split("..")[0].split(".")[0]
            self.dash_status.config(text=f"{base}{'.' * (self.dash_dots + 1)}")
        
        self.root.after(1000, self._dashboard_tick_timer)
    
    def _dashboard_update_epoch(self, epoch, box_loss, cls_loss, dfl_loss):
        """Update the dashboard with new epoch data."""
        if not hasattr(self, 'dash') or not self.dash.winfo_exists():
            return
        
        total_loss = box_loss + cls_loss + dfl_loss
        total_epochs = self.train_epochs
        pct = int((epoch / total_epochs) * 100)
        
        # Update epoch label
        self.dash_epoch_label.config(text=f"Epoch: {epoch} / {total_epochs}")
        
        # Update progress bar
        bar_w = int(560 * (epoch / total_epochs))
        self.dash_prog_canvas.coords("bar", 0, 0, bar_w, 28)
        # Gradient color: red → orange → green
        if pct < 40:
            bar_color = "#e94560"
        elif pct < 75:
            bar_color = "#ffa502"
        else:
            bar_color = "#7bed9f"
        self.dash_prog_canvas.itemconfig("bar", fill=bar_color)
        self.dash_prog_canvas.itemconfig("pct", text=f"{pct}%")
        
        # Update loss values
        self.dash_box_loss.config(text=f"{box_loss:.4f}")
        self.dash_cls_loss.config(text=f"{cls_loss:.4f}")
        self.dash_dfl_loss.config(text=f"{dfl_loss:.4f}")
        self.dash_total_loss.config(text=f"{total_loss:.4f}")
        
        # Update status
        elapsed = int(time.time() - self.train_start_time)
        if epoch > 0:
            eta = int((elapsed / epoch) * (total_epochs - epoch))
            eta_m, eta_s = divmod(eta, 60)
            self.dash_status.config(
                text=f"🚀 Training epoch {epoch}/{total_epochs}  |  ETA: {eta_m}:{eta_s:02d}",
                fg="#7bed9f" if pct >= 75 else "#ffa502")
        
        # ── Redraw Chart ──
        self._dashboard_draw_chart()
    
    def _dashboard_draw_chart(self):
        """Draw the loss history chart on the canvas."""
        if not hasattr(self, 'dash_chart') or not self.dash_chart.winfo_exists():
            return
        
        chart = self.dash_chart
        chart.delete("all")
        
        w = chart.winfo_width() or 540
        h = chart.winfo_height() or 160
        pad_l, pad_r, pad_t, pad_b = 45, 15, 10, 25
        plot_w = w - pad_l - pad_r
        plot_h = h - pad_t - pad_b
        
        if not self.train_loss_history:
            return
        
        # Gather all loss values for scaling
        all_vals = []
        for _, b, c, d in self.train_loss_history:
            all_vals.extend([b, c, d])
        max_val = max(all_vals) if all_vals else 1
        min_val = 0
        val_range = max(max_val - min_val, 0.001)
        
        n = len(self.train_loss_history)
        
        # Draw grid lines
        for i in range(5):
            y = pad_t + int(plot_h * i / 4)
            chart.create_line(pad_l, y, w - pad_r, y, fill="#1a3a5c", dash=(2, 4))
            val = max_val - (val_range * i / 4)
            chart.create_text(pad_l - 5, y, text=f"{val:.2f}", fill="#607080",
                             font=("Consolas", 8), anchor=tk.E)
        
        # Draw epoch labels on x-axis  
        for i, (ep, _, _, _) in enumerate(self.train_loss_history):
            x = pad_l + int(plot_w * i / max(n - 1, 1))
            if n <= 10 or i % max(1, n // 10) == 0 or i == n - 1:
                chart.create_text(x, h - 5, text=str(ep), fill="#607080",
                                 font=("Consolas", 8))
        
        # Plot lines for each loss type
        colors = {"box": "#00d2ff", "cls": "#7bed9f", "dfl": "#ffa502"}
        for loss_idx, (key, color) in enumerate(colors.items()):
            points = []
            for i, (ep, b, c, d) in enumerate(self.train_loss_history):
                vals = [b, c, d]
                x = pad_l + int(plot_w * i / max(n - 1, 1))
                y = pad_t + int(plot_h * (1 - (vals[loss_idx] - min_val) / val_range))
                points.append((x, y))
            
            # Draw line segments
            if len(points) >= 2:
                for j in range(1, len(points)):
                    chart.create_line(points[j-1][0], points[j-1][1],
                                     points[j][0], points[j][1],
                                     fill=color, width=2, smooth=True)
            
            # Draw dots
            for x, y in points:
                chart.create_oval(x-3, y-3, x+3, y+3, fill=color, outline="")
    
    def _on_quick_train_done(self, success, error="", sign_name=None):
        """Callback when quick training finishes."""
        self.quick_train_btn.config(state=tk.NORMAL, text="\u26a1 QUICK TRAIN")
        self.retrain_btn.config(state=tk.NORMAL)
        if hasattr(self, 'train_btn'):
            selected = self.class_combo.get() if hasattr(self, 'class_combo') else "sign"
            self.train_btn.config(state=tk.NORMAL, text=f"🔥 Train '{selected}'")
        
        # Stop the timer
        self.dash_training_active = False
        
        if hasattr(self, 'dash') and self.dash.winfo_exists():
            if success:
                elapsed = int(time.time() - self.train_start_time)
                mins, secs = divmod(elapsed, 60)
                # Update dashboard to show completion
                self.dash_epoch_label.config(text=f"Epoch: {self.train_epochs} / {self.train_epochs}")
                self.dash_prog_canvas.coords("bar", 0, 0, 560, 28)
                self.dash_prog_canvas.itemconfig("bar", fill="#7bed9f")
                self.dash_prog_canvas.itemconfig("pct", text="100%")
                done_msg = f"✅ Training Complete!  Total time: {mins}:{secs:02d}  |  Model deployed to best.pt"
                if sign_name:
                    done_msg = f"✅ '{sign_name}' trained!  Time: {mins}:{secs:02d}  |  Model deployed"
                self.dash_status.config(text=done_msg, fg="#7bed9f")
                self.dash_timer_label.config(text=f"✅ {mins}:{secs:02d}")
                self.dash.title("✅ Training Complete!")
            else:
                self.dash_status.config(text=f"❌ Training Failed: {error}", fg="#e94560")
                self.dash.title("❌ Training Failed")
        
        if success:
            sign_msg = f" for '{sign_name}'" if sign_name else ""
            self.status_label.config(text=f"\u2705 Training{sign_msg} complete! Model deployed to best.pt")
            messagebox.showinfo("Training Complete",
                f"\u2705 Training Done{sign_msg}!\n\n"
                "Model has been auto-deployed to best.pt\n"
                "Live prediction is now using the new model.\n\n"
                "Run isl_gui_app.py to test the updated model!")
        else:
            self.status_label.config(text=f"\u274c Training failed: {error}")
            messagebox.showerror("Training Failed", f"Error: {error}")

    def _auto_split_data(self):
        """Automatically split training data into train/val/test (80/15/5)."""
        import random
        
        train_imgs = os.path.join("Data", "train", "images")
        train_lbls = os.path.join("Data", "train", "labels")
        val_imgs = os.path.join("Data", "valid", "images")
        val_lbls = os.path.join("Data", "valid", "labels")
        test_imgs = os.path.join("Data", "test", "images")
        test_lbls = os.path.join("Data", "test", "labels")
        
        for d in [val_imgs, val_lbls, test_imgs, test_lbls]:
            os.makedirs(d, exist_ok=True)
        
        # Get all training images
        if not os.path.exists(train_imgs):
            return
        
        all_images = [f for f in os.listdir(train_imgs) if f.endswith(('.jpg', '.png', '.jpeg'))]
        random.shuffle(all_images)
        
        total = len(all_images)
        val_count = max(1, int(total * 0.15))
        test_count = max(1, int(total * 0.05))
        
        val_set = all_images[:val_count]
        test_set = all_images[val_count:val_count + test_count]
        
        moved = 0
        for img_name in val_set:
            lbl_name = os.path.splitext(img_name)[0] + '.txt'
            src_img = os.path.join(train_imgs, img_name)
            src_lbl = os.path.join(train_lbls, lbl_name)
            if os.path.exists(src_img):
                shutil.copy2(src_img, os.path.join(val_imgs, img_name))
                moved += 1
            if os.path.exists(src_lbl):
                shutil.copy2(src_lbl, os.path.join(val_lbls, lbl_name))
        
        for img_name in test_set:
            lbl_name = os.path.splitext(img_name)[0] + '.txt'
            src_img = os.path.join(train_imgs, img_name)
            src_lbl = os.path.join(train_lbls, lbl_name)
            if os.path.exists(src_img):
                shutil.copy2(src_img, os.path.join(test_imgs, img_name))
                moved += 1
            if os.path.exists(src_lbl):
                shutil.copy2(src_lbl, os.path.join(test_lbls, lbl_name))
        
        print(f"\u2705 Auto-split: {len(val_set)} val + {len(test_set)} test images (from {total} total)")

    # ═══════════════════════════════════════════════════════════
    #  VIDEO IMPORT METHODS
    # ═══════════════════════════════════════════════════════════
    def import_video_files(self):
        """Import individual video files — extract frames for the selected sign class."""
        filetypes = [
            ("Video files", "*.mp4 *.avi *.mkv *.mov *.webm *.flv *.wmv *.m4v"),
            ("All files", "*.*")
        ]
        files = filedialog.askopenfilenames(
            title="Select Video Files to Import",
            filetypes=filetypes,
            parent=self.root
        )
        if not files:
            return

        class_name = self.class_combo.get()
        if not class_name:
            messagebox.showwarning("Warning", "Please select a sign class first!")
            return

        class_id = self.CLASS_NAMES.index(class_name)
        interval = max(1, self.vid_interval_var.get())
        skip_dup = self.vid_skip_dup_var.get()

        self.vid_import_status.config(text=f"📹 Importing {len(files)} video(s)...")
        self.root.update_idletasks()

        # Run in background thread
        threading.Thread(
            target=self._process_video_import,
            args=(list(files), class_name, class_id, interval, skip_dup),
            daemon=True
        ).start()

    def import_video_folder(self):
        """Import a folder of videos — each subfolder = a sign class.
        Structure: folder/class_name/video.mp4
        Or: folder/video.mp4 (uses currently selected class)
        """
        folder = filedialog.askdirectory(
            title="Select Video Folder (class_name/video.mp4 structure)",
            parent=self.root
        )
        if not folder:
            return

        video_ext = ('.mp4', '.avi', '.mkv', '.mov', '.webm', '.flv', '.wmv', '.m4v')
        interval = max(1, self.vid_interval_var.get())
        skip_dup = self.vid_skip_dup_var.get()
        total_imported = 0

        # Check for class subfolders
        has_subfolders = False
        for item in os.listdir(folder):
            if os.path.isdir(os.path.join(folder, item)):
                sub_videos = [f for f in os.listdir(os.path.join(folder, item))
                              if f.lower().endswith(video_ext)]
                if sub_videos:
                    has_subfolders = True
                    break

        if has_subfolders:
            # Process each subfolder as a class
            for item in sorted(os.listdir(folder)):
                item_path = os.path.join(folder, item)
                if not os.path.isdir(item_path):
                    continue

                class_name = item.lower().strip()
                videos = [os.path.join(item_path, f) for f in os.listdir(item_path)
                          if f.lower().endswith(video_ext)]
                if not videos:
                    continue

                # Add class if new
                if class_name not in self.CLASS_NAMES:
                    self.CLASS_NAMES.append(class_name)
                    self._save_class_names()
                    self.class_combo['values'] = self.CLASS_NAMES

                class_id = self.CLASS_NAMES.index(class_name)
                self.vid_import_status.config(
                    text=f"📹 Processing: {class_name} ({len(videos)} videos)...")
                self.root.update_idletasks()

                threading.Thread(
                    target=self._process_video_import,
                    args=(videos, class_name, class_id, interval, skip_dup),
                    daemon=True
                ).start()
        else:
            # All videos in root — use currently selected class
            videos = [os.path.join(folder, f) for f in os.listdir(folder)
                      if f.lower().endswith(video_ext)]
            if not videos:
                messagebox.showinfo("No Videos", "No video files found in the selected folder!")
                return

            class_name = self.class_combo.get()
            class_id = self.CLASS_NAMES.index(class_name)
            self.vid_import_status.config(
                text=f"📹 Importing {len(videos)} video(s) as '{class_name}'...")
            self.root.update_idletasks()

            threading.Thread(
                target=self._process_video_import,
                args=(videos, class_name, class_id, interval, skip_dup),
                daemon=True
            ).start()

    def _process_video_import(self, video_paths, class_name, class_id, interval, skip_dup):
        """Background worker: extract frames from videos and save as YOLO training data."""
        import uuid as _uuid
        total_extracted = 0
        total_skipped = 0
        prev_frame = None

        for vid_path in video_paths:
            cap = cv2.VideoCapture(vid_path)
            if not cap.isOpened():
                continue

            vid_name = os.path.splitext(os.path.basename(vid_path))[0]
            frame_idx = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % interval != 0:
                    frame_idx += 1
                    continue

                # Skip near-duplicate frames via histogram comparison
                if skip_dup and prev_frame is not None:
                    h1 = cv2.calcHist([frame], [0,1,2], None, [16,16,16],
                                      [0,256,0,256,0,256])
                    h2 = cv2.calcHist([prev_frame], [0,1,2], None, [16,16,16],
                                      [0,256,0,256,0,256])
                    cv2.normalize(h1, h1)
                    cv2.normalize(h2, h2)
                    if cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL) >= 0.95:
                        total_skipped += 1
                        frame_idx += 1
                        continue

                # Detect hand bounding box
                h_img, w_img = frame.shape[:2]
                cx, cy, bw, bh = 0.5, 0.5, 1.0, 1.0  # default full frame

                if self.auto_detect_hand and self.hands is not None:
                    bbox = self._detect_hand(frame)
                    if bbox:
                        x1, y1, x2, y2 = bbox
                        cx = ((x1 + x2) / 2.0) / w_img
                        cy = ((y1 + y2) / 2.0) / h_img
                        bw = (x2 - x1) / w_img
                        bh = (y2 - y1) / h_img

                # Save image + label
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                uid = _uuid.uuid4().hex[:8]
                fname = f"{class_name.replace(' ', '_')}_{vid_name}_f{frame_idx}_{uid}"
                img_path = os.path.join(self.train_images_dir, f"{fname}.jpg")
                lbl_path = os.path.join(self.train_labels_dir, f"{fname}.txt")

                cv2.imwrite(img_path, frame)
                with open(lbl_path, 'w') as f:
                    f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

                total_extracted += 1
                prev_frame = frame.copy()
                frame_idx += 1

                # Update UI periodically
                if total_extracted % 10 == 0:
                    pct = (frame_idx / max(1, total_frames)) * 100
                    self.root.after(0, lambda e=total_extracted, s=total_skipped:
                        self.vid_import_status.config(
                            text=f"📹 Extracted: {e} | Skipped: {s}"))

            cap.release()

        # Update counts
        self.class_image_counts[class_name] = self.class_image_counts.get(class_name, 0) + total_extracted
        self.collected_count += total_extracted

        # Update UI on main thread
        def _done():
            self._update_progress_display()
            self.vid_import_status.config(
                text=f"✅ Done! {total_extracted} frames extracted, {total_skipped} skipped")
            self.status_label.config(
                text=f"Video import done: {total_extracted} frames for '{class_name}'")
            # Announce via TTS if enabled
            if TTS_AVAILABLE and self.tts_enabled:
                speak_sign(class_name, self.current_lang_idx)

        self.root.after(0, _done)

    def _open_video_converter(self):
        """Launch the full Video Dataset Converter GUI tool."""
        try:
            import subprocess
            import sys
            subprocess.Popen([sys.executable, "video_dataset.py"])
            self.status_label.config(text="Launched Video Dataset Converter tool")
        except Exception as e:
            messagebox.showerror("Error", f"Could not launch video_dataset.py:\n{e}")

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()

    def on_close(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = DataCollectorApp(root)
    app.run()
