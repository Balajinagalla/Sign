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
        self.root.geometry("1050x900")
        self.root.configure(bg="#2c3e50")

        # Paths
        self.data_yaml_path = "Data/data.yaml"
        self.train_images_dir = "Data/train/images"
        self.train_labels_dir = "Data/train/labels"
        self.tts_file_path = "tts_indic_multi.py"
        
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
        self.auto_capture_interval = 2.0  # seconds between captures
        self.batch_capture_count = 10  # images per batch
        self.batch_capturing = False
        self.batch_remaining = 0
        self.last_capture_time = 0
        self.use_full_frame = False  # Option to use full frame without bounding box
        
        # Hand detection settings
        self.auto_detect_hand = False
        self.detected_hand_bbox = None
        self.hand_detection_padding = 30  # Extra padding around detected hand
        self.hands = None
        self.mp_hands = None
        self.mp_draw = None
        
        # Initialize hand detection
        self._init_hand_detector()
        
        # Count existing images per class
        self.class_image_counts = self._count_existing_images()

        self._create_ui()
    
    def _init_hand_detector(self):
        """Initialize hand detector based on available API"""
        global MEDIAPIPE_AVAILABLE, MP_USE_LEGACY
        
        if MEDIAPIPE_AVAILABLE and MP_USE_LEGACY:
            # Use legacy API (mp.solutions)
            try:
                self.mp_hands = mp.solutions.hands
                self.hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
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
        # Title
        title = tk.Label(self.root, text="📸 ISL Data Collector - Dynamic Mode", 
                        font=("Arial", 22, "bold"), bg="#2c3e50", fg="white")
        title.pack(pady=10)

        # Instructions
        instructions = tk.Label(self.root, 
            text="1. Start Camera → 2. Select/Add Sign → 3. Set Region or Use Full Frame → 4. Capture (Manual/Auto/Batch)",
            font=("Arial", 11), bg="#2c3e50", fg="#bdc3c7")
        instructions.pack(pady=5)

        # ========== DYNAMIC SETTINGS PANEL ==========
        settings_frame = tk.LabelFrame(self.root, text="⚙️ Dynamic Capture Settings", 
                                       font=("Arial", 11, "bold"), bg="#34495e", fg="white",
                                       padx=15, pady=10)
        settings_frame.pack(pady=10, fill=tk.X, padx=20)
        
        # Row 1: Target images per class
        row1 = tk.Frame(settings_frame, bg="#34495e")
        row1.pack(fill=tk.X, pady=5)
        
        tk.Label(row1, text="🎯 Target Images:", font=("Arial", 10, "bold"), 
                bg="#34495e", fg="white").pack(side=tk.LEFT, padx=5)
        
        self.target_var = tk.IntVar(value=self.TARGET_IMAGES)
        self.target_slider = tk.Scale(row1, from_=10, to=100, orient=tk.HORIZONTAL,
                                      variable=self.target_var, length=200,
                                      bg="#34495e", fg="white", highlightthickness=0,
                                      command=self._on_target_change)
        self.target_slider.pack(side=tk.LEFT, padx=5)
        
        self.target_display = tk.Label(row1, text=f"{self.TARGET_IMAGES} images/class", 
                                       font=("Arial", 10), bg="#34495e", fg="#3498db")
        self.target_display.pack(side=tk.LEFT, padx=10)
        
        # Full frame checkbox
        self.full_frame_var = tk.BooleanVar(value=False)
        self.full_frame_check = tk.Checkbutton(row1, text="📐 Use Full Frame (no bbox)", 
                                                variable=self.full_frame_var,
                                                bg="#34495e", fg="white", selectcolor="#2c3e50",
                                                font=("Arial", 10),
                                                command=self._on_full_frame_toggle)
        self.full_frame_check.pack(side=tk.RIGHT, padx=10)
        
        # Row 2: Auto-capture settings
        row2 = tk.Frame(settings_frame, bg="#34495e")
        row2.pack(fill=tk.X, pady=5)
        
        self.auto_capture_var = tk.BooleanVar(value=False)
        self.auto_check = tk.Checkbutton(row2, text="🔄 Auto-Capture", 
                                         variable=self.auto_capture_var,
                                         bg="#34495e", fg="white", selectcolor="#2c3e50",
                                         font=("Arial", 10, "bold"),
                                         command=self._on_auto_capture_toggle)
        self.auto_check.pack(side=tk.LEFT, padx=5)
        
        tk.Label(row2, text="Interval:", font=("Arial", 10), 
                bg="#34495e", fg="white").pack(side=tk.LEFT, padx=5)
        
        self.interval_var = tk.DoubleVar(value=self.auto_capture_interval)
        self.interval_slider = tk.Scale(row2, from_=0.5, to=5.0, resolution=0.5,
                                        orient=tk.HORIZONTAL, variable=self.interval_var,
                                        length=150, bg="#34495e", fg="white", highlightthickness=0,
                                        command=self._on_interval_change)
        self.interval_slider.pack(side=tk.LEFT, padx=5)
        
        self.interval_display = tk.Label(row2, text=f"{self.auto_capture_interval}s", 
                                         font=("Arial", 10), bg="#34495e", fg="#e74c3c")
        self.interval_display.pack(side=tk.LEFT, padx=5)
        
        # Batch capture settings
        tk.Label(row2, text="│  Batch:", font=("Arial", 10), 
                bg="#34495e", fg="#95a5a6").pack(side=tk.LEFT, padx=10)
        
        self.batch_var = tk.IntVar(value=self.batch_capture_count)
        self.batch_spinbox = tk.Spinbox(row2, from_=5, to=50, width=5,
                                        textvariable=self.batch_var,
                                        font=("Arial", 10))
        self.batch_spinbox.pack(side=tk.LEFT, padx=5)
        
        self.batch_btn = tk.Button(row2, text="🚀 Start Batch", command=self.start_batch_capture,
                                   bg="#1abc9c", fg="white", font=("Arial", 10, "bold"), width=12)
        self.batch_btn.pack(side=tk.LEFT, padx=10)
        
        # Row 3: Hand Auto-Detection
        row3 = tk.Frame(settings_frame, bg="#34495e")
        row3.pack(fill=tk.X, pady=5)
        
        self.auto_detect_var = tk.BooleanVar(value=False)
        detect_state = tk.NORMAL if MEDIAPIPE_AVAILABLE else tk.DISABLED
        self.auto_detect_check = tk.Checkbutton(row3, text="🖐️ Auto-Detect Hand (MediaPipe)", 
                                                 variable=self.auto_detect_var,
                                                 bg="#34495e", fg="white", selectcolor="#2c3e50",
                                                 font=("Arial", 10, "bold"),
                                                 state=detect_state,
                                                 command=self._on_auto_detect_toggle)
        self.auto_detect_check.pack(side=tk.LEFT, padx=5)
        
        if not MEDIAPIPE_AVAILABLE:
            tk.Label(row3, text="(pip install mediapipe)", font=("Arial", 9), 
                    bg="#34495e", fg="#e74c3c").pack(side=tk.LEFT, padx=5)
        
        tk.Label(row3, text="│  Padding:", font=("Arial", 10), 
                bg="#34495e", fg="#95a5a6").pack(side=tk.LEFT, padx=10)
        
        self.padding_var = tk.IntVar(value=self.hand_detection_padding)
        self.padding_slider = tk.Scale(row3, from_=10, to=80, orient=tk.HORIZONTAL,
                                       variable=self.padding_var, length=100,
                                       bg="#34495e", fg="white", highlightthickness=0,
                                       command=self._on_padding_change)
        self.padding_slider.pack(side=tk.LEFT, padx=5)
        
        self.hand_status_label = tk.Label(row3, text="Hand: Not detected", 
                                          font=("Arial", 10), bg="#34495e", fg="#95a5a6")
        self.hand_status_label.pack(side=tk.RIGHT, padx=10)
        control_frame = tk.Frame(self.root, bg="#2c3e50")
        control_frame.pack(pady=10)

        # Class Selection
        tk.Label(control_frame, text="Sign Class:", font=("Arial", 12, "bold"), 
                bg="#2c3e50", fg="white").pack(side=tk.LEFT, padx=5)
        self.class_combo = ttk.Combobox(control_frame, values=self.CLASS_NAMES, 
                                        state="readonly", width=15, font=("Arial", 11))
        if self.CLASS_NAMES:
            self.class_combo.set(self.CLASS_NAMES[0])
        self.class_combo.pack(side=tk.LEFT, padx=5)
        self.class_combo.bind("<<ComboboxSelected>>", self._on_class_change)

        # Add New Sign Button
        self.add_sign_btn = tk.Button(control_frame, text="➕ Add New Sign", command=self.add_new_sign,
                                      bg="#9b59b6", fg="white", font=("Arial", 10, "bold"), width=14)
        self.add_sign_btn.pack(side=tk.LEFT, padx=5)

        # Camera buttons
        self.start_btn = tk.Button(control_frame, text="▶ Start", command=self.start_camera,
                                   bg="#27ae60", fg="white", font=("Arial", 11, "bold"), width=10)
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = tk.Button(control_frame, text="⏹ Stop", command=self.stop_camera,
                                  bg="#e74c3c", fg="white", font=("Arial", 11, "bold"), width=10, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        # Video Frame with Canvas for drawing
        self.canvas = tk.Canvas(self.root, width=640, height=480, bg="black", cursor="cross")
        self.canvas.pack(pady=15)
        
        # Bind mouse events for bounding box drawing
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        # Second row controls
        action_frame = tk.Frame(self.root, bg="#2c3e50")
        action_frame.pack(pady=10)

        self.capture_btn = tk.Button(action_frame, text="📷 Capture Frame", command=self.capture_frame,
                                     bg="#3498db", fg="white", font=("Arial", 11, "bold"), width=14, state=tk.DISABLED)
        self.capture_btn.pack(side=tk.LEFT, padx=5)

        self.save_btn = tk.Button(action_frame, text="💾 Save & Label", command=self.save_labeled_image,
                                  bg="#27ae60", fg="white", font=("Arial", 12, "bold"), width=14, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=5)

        self.clear_btn = tk.Button(action_frame, text="🔄 Clear Box", command=self.clear_bbox,
                                   bg="#f39c12", fg="white", font=("Arial", 11, "bold"), width=12)
        self.clear_btn.pack(side=tk.LEFT, padx=5)

        self.retrain_btn = tk.Button(action_frame, text="🚀 Retrain Model", command=self.retrain_model,
                                     bg="#1abc9c", fg="white", font=("Arial", 11, "bold"), width=14)
        self.retrain_btn.pack(side=tk.LEFT, padx=5)

        # Status Frame
        status_frame = tk.Frame(self.root, bg="#2c3e50")
        status_frame.pack(pady=10, fill=tk.X, padx=20)

        self.status_label = tk.Label(status_frame, text="Status: Ready | Images collected: 0", 
                                     font=("Arial", 12), bg="#2c3e50", fg="#ecf0f1")
        self.status_label.pack()

        self.bbox_label = tk.Label(status_frame, text="Bounding Box: Not drawn", 
                                   font=("Arial", 11), bg="#2c3e50", fg="#f39c12")
        self.bbox_label.pack()

        # Progress indicator for current class
        progress_frame = tk.Frame(self.root, bg="#1a252f", padx=15, pady=10)
        progress_frame.pack(pady=5, fill=tk.X, padx=20)
        
        tk.Label(progress_frame, text="📊 Current Sign Progress:", font=("Arial", 11, "bold"), 
                bg="#1a252f", fg="white").pack(anchor=tk.W)
        
        self.progress_label = tk.Label(progress_frame, 
            text=self._get_current_progress_text(),
            font=("Arial", 14, "bold"), bg="#1a252f", fg="#27ae60")
        self.progress_label.pack(anchor=tk.W, pady=5)
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(progress_frame, length=400, mode='determinate')
        self.progress_bar.pack(anchor=tk.W, pady=5)
        self._update_progress_bar()
        
        tk.Label(progress_frame, text=f"🎯 Target: {self.TARGET_IMAGES} images per sign (minimum recommended)",
                font=("Arial", 10), bg="#1a252f", fg="#95a5a6").pack(anchor=tk.W)

        # Class list display with counts
        class_frame = tk.Frame(self.root, bg="#34495e", padx=10, pady=10)
        class_frame.pack(pady=10, fill=tk.X, padx=20)
        
        tk.Label(class_frame, text="📋 All Signs - Image Counts:", font=("Arial", 11, "bold"), 
                bg="#34495e", fg="white").pack(anchor=tk.W)
        
        self.class_list_label = tk.Label(class_frame, 
            text=self._format_class_list_with_counts(),
            font=("Arial", 10), bg="#34495e", fg="#bdc3c7", justify=tk.LEFT, wraplength=900)
        self.class_list_label.pack(anchor=tk.W, pady=5)

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
    
    def _on_target_change(self, value):
        """Handle target images slider change"""
        self.TARGET_IMAGES = int(float(value))
        self.target_display.config(text=f"{self.TARGET_IMAGES} images/class")
        self._update_progress_display()
    
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
        """Detect hand in frame using MediaPipe or skin color detection"""
        # Try MediaPipe first if available
        if MEDIAPIPE_AVAILABLE and self.hands is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                h, w, _ = frame.shape
                hand_landmarks = results.multi_hand_landmarks[0]
                
                x_coords = [lm.x * w for lm in hand_landmarks.landmark]
                y_coords = [lm.y * h for lm in hand_landmarks.landmark]
                
                x_min = int(max(0, min(x_coords) - self.hand_detection_padding))
                y_min = int(max(0, min(y_coords) - self.hand_detection_padding))
                x_max = int(min(w, max(x_coords) + self.hand_detection_padding))
                y_max = int(min(h, max(y_coords) + self.hand_detection_padding))
                
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
            # Get largest contour (assumed to be hand)
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            # Filter by minimum area (avoid noise)
            min_area = (h * w) * 0.02  # At least 2% of frame
            if area > min_area:
                x, y, bw, bh = cv2.boundingRect(largest_contour)
                
                # Add padding
                x_min = max(0, x - self.hand_detection_padding)
                y_min = max(0, y - self.hand_detection_padding)
                x_max = min(w, x + bw + self.hand_detection_padding)
                y_max = min(h, y + bh + self.hand_detection_padding)
                
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
    
    def _auto_capture_tick(self):
        """Check if it's time to auto-capture"""
        current_time = time.time()
        
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
        
        # Save image
        image_path = os.path.join(self.train_images_dir, f"{filename}.jpg")
        cv2.imwrite(image_path, frame_to_save)
        
        # Determine bounding box
        img_h, img_w = frame_to_save.shape[:2]
        
        if self.use_full_frame:
            # Use full frame as bounding box (with small margin)
            margin = 0.02
            center_x, center_y = 0.5, 0.5
            width, height = 1.0 - margin*2, 1.0 - margin*2
        else:
            # Use drawn bounding box
            x1, y1 = self.bbox_start
            x2, y2 = self.bbox_end
            center_x = ((x1 + x2) / 2) / img_w
            center_y = ((y1 + y2) / 2) / img_h
            width = abs(x2 - x1) / img_w
            height = abs(y2 - y1) / img_h
        
        # Save label
        label_path = os.path.join(self.train_labels_dir, f"{filename}.txt")
        with open(label_path, 'w') as f:
            f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
        
        self.collected_count += 1
        
        # Update class image count
        self.class_image_counts[class_name] = self.class_image_counts.get(class_name, 0) + 1
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
                self.class_list_label.config(text=self._format_class_list())
                
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
        """Add TTS translations for a new sign"""
        languages = ["English", "Hindi", "Tamil", "Telugu", "Bengali", "Malayalam", "Kannada", "Gujarati", "Marathi"]
        lang_codes = ["en", "hi", "ta", "te", "bn", "ml", "kn", "gu", "mr"]
        
        translations = {}
        
        for lang, code in zip(languages, lang_codes):
            translation = simpledialog.askstring(f"Translation - {lang}",
                f"Enter '{sign_name}' in {lang}:\n\n(Leave empty to use English)",
                parent=self.root)
            
            if translation and translation.strip():
                translations[code] = translation.strip()
            else:
                translations[code] = sign_name.title()
        
        # Show summary
        summary = "\n".join([f"{lang}: {translations[code]}" for lang, code in zip(languages, lang_codes)])
        messagebox.showinfo("Translations Added", 
            f"Translations for '{sign_name}':\n\n{summary}\n\n"
            f"Note: You'll need to manually add these to tts_indic_multi.py\n"
            f"in the TRANSLATIONS dictionary.")
        
        # Print the code to add
        print(f"\n# Add this to TRANSLATIONS in tts_indic_multi.py:")
        print(f'    "{sign_name}": {translations},')

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
            frame = cv2.resize(frame, (640, 480))
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
                    cv2.putText(frame, "HAND DETECTED", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    self.hand_status_label.config(text="Hand: ✅ Detected", fg="#27ae60")
                else:
                    self.detected_hand_bbox = None
                    self.hand_status_label.config(text="Hand: ❌ Not detected", fg="#e74c3c")
            
            # Add visual indicator for auto-capture/batch mode
            if self.batch_capturing:
                cv2.putText(frame, f"BATCH: {self.batch_remaining} left", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            elif self.auto_capture_enabled:
                cv2.putText(frame, "AUTO-CAPTURE ON", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if self.use_full_frame:
                cv2.putText(frame, "FULL FRAME MODE", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
            
            # Add capture count indicator (top-right corner)
            current_class = self.class_combo.get()
            class_count = self.class_image_counts.get(current_class, 0)
            
            # Background rectangle for better visibility
            cv2.rectangle(frame, (430, 5), (635, 75), (0, 0, 0), -1)
            cv2.rectangle(frame, (430, 5), (635, 75), (0, 255, 0), 2)
            
            # Session total
            cv2.putText(frame, f"Session: {self.collected_count}", (440, 28),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Current class count
            cv2.putText(frame, f"{current_class}: {class_count}/{self.TARGET_IMAGES}", (440, 55),
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

        # Save image
        image_path = os.path.join(self.train_images_dir, f"{filename}.jpg")
        cv2.imwrite(image_path, self.current_frame)

        # Convert bbox to YOLO format (normalized center x, center y, width, height)
        img_h, img_w = self.current_frame.shape[:2]
        x1, y1 = self.bbox_start
        x2, y2 = self.bbox_end
        
        center_x = ((x1 + x2) / 2) / img_w
        center_y = ((y1 + y2) / 2) / img_h
        width = abs(x2 - x1) / img_w
        height = abs(y2 - y1) / img_h

        # Save label
        label_path = os.path.join(self.train_labels_dir, f"{filename}.txt")
        with open(label_path, 'w') as f:
            f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")

        self.collected_count += 1
        
        # Update class image count
        self.class_image_counts[class_name] = self.class_image_counts.get(class_name, 0) + 1
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
        result = messagebox.askyesno("Retrain Model", 
            f"Start retraining with the updated dataset?\n\n"
            f"Classes: {len(self.CLASS_NAMES)}\n"
            f"New images: {self.collected_count}\n\n"
            "This will run train.py in a new window.")
        
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
            messagebox.showinfo("Training Started", 
                "Training started in a new window!\n\n"
                "You can continue collecting more data.\n"
                "Check the training window for progress.")

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
