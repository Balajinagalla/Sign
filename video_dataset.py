# video_dataset.py - Video Dataset → YOLO Image Dataset Converter
# Supports: .mp4, .avi, .mkv, .mov, .webm → YOLO format images + labels
# Features: Auto hand detection, configurable frame extraction, train/val/test split
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import os
import numpy as np
import threading
import time
import random
import shutil
import yaml
from datetime import datetime
from PIL import Image, ImageTk

# ── Try importing MediaPipe for hand detection ──
MEDIAPIPE_AVAILABLE = False
try:
    import mediapipe as mp
    if hasattr(mp, 'solutions'):
        MEDIAPIPE_AVAILABLE = True
except ImportError:
    pass

# ── Supported video formats ──
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mkv', '.mov', '.webm', '.flv', '.wmv', '.m4v')

# ── Color Palette (Dark Theme — matches project style) ──
BG_DARK      = "#121212"
BG_CARD      = "#1e1e2f"
BG_ACCENT    = "#16213e"
FG_PRIMARY   = "#e0e0e0"
FG_SECONDARY = "#b0b0b0"
COLOR_GREEN  = "#03dac6"
COLOR_RED    = "#cf6679"
COLOR_CYAN   = "#03dac6"
COLOR_ORANGE = "#ffb74d"
COLOR_PURPLE = "#bb86fc"
COLOR_YELLOW = "#fbc02d"
COLOR_BLUE   = "#64b5f6"


class VideoDatasetConverter:
    """GUI tool to convert video folders into YOLO image datasets."""

    def __init__(self, root):
        self.root = root
        self.root.title("📹 Video Dataset Converter — Video → YOLO Training Data")
        self.root.geometry("1100x800")
        self.root.configure(bg=BG_DARK)
        self.root.resizable(True, True)

        # ── Paths ──
        self.video_dir = ""
        self.output_dir = "Data"
        self.data_yaml_path = os.path.join(self.output_dir, "data.yaml")

        # ── Settings ──
        self.frame_mode = "interval"   # "interval" or "fps"
        self.frame_interval = 5        # Extract every Nth frame
        self.target_fps = 2.0          # Extract N frames per second
        self.skip_similar = True       # Skip near-duplicate frames
        self.similarity_threshold = 0.95
        self.auto_hand_detect = True   # Auto-detect hand bounding box
        self.hand_padding = 40         # Padding around detected hand
        self.use_full_frame = False    # Use entire frame as bounding box
        self.split_ratio = (0.70, 0.20, 0.10)  # train/val/test

        # ── State ──
        self.video_classes = {}        # {class_name: [video_paths]}
        self.processing = False
        self.cancel_requested = False
        self.total_extracted = 0
        self.total_skipped = 0

        # ── MediaPipe Hands ──
        self.mp_hands = None
        self.hands = None
        if MEDIAPIPE_AVAILABLE:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=2,
                min_detection_confidence=0.5
            )

        self._build_ui()

    # ═══════════════════════════════════════════════════════════
    #  UI CONSTRUCTION
    # ═══════════════════════════════════════════════════════════
    def _build_ui(self):
        # ── Title Bar ──
        title_frame = tk.Frame(self.root, bg=BG_ACCENT, pady=8)
        title_frame.pack(fill=tk.X)
        tk.Label(title_frame, text="📹 Video Dataset → YOLO Converter",
                 font=("Segoe UI", 16, "bold"), bg=BG_ACCENT, fg=FG_PRIMARY
                 ).pack(side=tk.LEFT, padx=15)
        tk.Label(title_frame,
                 text="Convert video folders into YOLO training images with auto hand detection",
                 font=("Segoe UI", 9), bg=BG_ACCENT, fg=FG_SECONDARY
                 ).pack(side=tk.RIGHT, padx=15)

        # ── Main Content ──
        main = tk.Frame(self.root, bg=BG_DARK)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # ── Left Panel: Settings ──
        left = tk.Frame(main, bg=BG_DARK, width=450)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        left.pack_propagate(False)

        # -- Source Selection --
        src_card = tk.LabelFrame(left, text="📂 VIDEO SOURCE", font=("Segoe UI", 10, "bold"),
                                  bg=BG_CARD, fg=COLOR_CYAN, padx=10, pady=8)
        src_card.pack(fill=tk.X, pady=(0, 5))

        tk.Label(src_card, text="Expected structure:\n  videos/\n    hello/\n      video1.mp4\n    help/\n      clip1.avi",
                 font=("Consolas", 8), bg=BG_CARD, fg=FG_SECONDARY, justify=tk.LEFT).pack(anchor=tk.W)

        src_row = tk.Frame(src_card, bg=BG_CARD)
        src_row.pack(fill=tk.X, pady=5)

        self.src_entry = tk.Entry(src_row, font=("Segoe UI", 10), bg=BG_DARK, fg=FG_PRIMARY,
                                   insertbackground=FG_PRIMARY, relief=tk.FLAT, width=30)
        self.src_entry.pack(side=tk.LEFT, padx=(0, 5), ipady=3, fill=tk.X, expand=True)

        tk.Button(src_row, text="📁 Browse", command=self._browse_source,
                  bg=COLOR_PURPLE, fg="white", font=("Segoe UI", 9, "bold"),
                  relief=tk.FLAT, cursor="hand2").pack(side=tk.LEFT)

        tk.Button(src_card, text="🔍 Scan Videos", command=self._scan_videos,
                  bg=COLOR_BLUE, fg="#000", font=("Segoe UI", 10, "bold"),
                  relief=tk.FLAT, cursor="hand2", width=20).pack(pady=5)

        self.scan_result = tk.Label(src_card, text="No folder selected",
                                     font=("Segoe UI", 9), bg=BG_CARD, fg=FG_SECONDARY,
                                     wraplength=400, justify=tk.LEFT)
        self.scan_result.pack(anchor=tk.W)

        # -- Extraction Settings --
        ext_card = tk.LabelFrame(left, text="⚙️ EXTRACTION SETTINGS", font=("Segoe UI", 10, "bold"),
                                  bg=BG_CARD, fg=COLOR_ORANGE, padx=10, pady=8)
        ext_card.pack(fill=tk.X, pady=5)

        # Frame extraction mode
        mode_row = tk.Frame(ext_card, bg=BG_CARD)
        mode_row.pack(fill=tk.X, pady=3)
        tk.Label(mode_row, text="Mode:", font=("Segoe UI", 9, "bold"),
                 bg=BG_CARD, fg=FG_PRIMARY).pack(side=tk.LEFT)
        self.mode_var = tk.StringVar(value="interval")
        tk.Radiobutton(mode_row, text="Every Nth frame", variable=self.mode_var, value="interval",
                       bg=BG_CARD, fg=FG_PRIMARY, selectcolor=BG_DARK,
                       command=self._on_mode_change, font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(mode_row, text="Target FPS", variable=self.mode_var, value="fps",
                       bg=BG_CARD, fg=FG_PRIMARY, selectcolor=BG_DARK,
                       command=self._on_mode_change, font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=5)

        # Interval slider
        self.interval_frame = tk.Frame(ext_card, bg=BG_CARD)
        self.interval_frame.pack(fill=tk.X, pady=2)
        tk.Label(self.interval_frame, text="Extract every:", font=("Segoe UI", 9),
                 bg=BG_CARD, fg=FG_SECONDARY).pack(side=tk.LEFT)
        self.interval_var = tk.IntVar(value=5)
        self.interval_slider = tk.Scale(self.interval_frame, from_=1, to=30, orient=tk.HORIZONTAL,
                                         variable=self.interval_var, length=150,
                                         bg=BG_CARD, fg=FG_PRIMARY, highlightthickness=0,
                                         troughcolor=BG_DARK)
        self.interval_slider.pack(side=tk.LEFT, padx=5)
        self.interval_label = tk.Label(self.interval_frame, text="5th frame",
                                        font=("Segoe UI", 9, "bold"), bg=BG_CARD, fg=COLOR_CYAN)
        self.interval_label.pack(side=tk.LEFT)
        self.interval_slider.config(command=lambda v: self.interval_label.config(
            text=f"{v} frame{'s' if int(v)>1 else ''}"))

        # FPS slider (hidden by default)
        self.fps_frame = tk.Frame(ext_card, bg=BG_CARD)
        tk.Label(self.fps_frame, text="Target FPS:", font=("Segoe UI", 9),
                 bg=BG_CARD, fg=FG_SECONDARY).pack(side=tk.LEFT)
        self.fps_var = tk.DoubleVar(value=2.0)
        self.fps_slider = tk.Scale(self.fps_frame, from_=0.5, to=10.0, resolution=0.5,
                                    orient=tk.HORIZONTAL, variable=self.fps_var, length=150,
                                    bg=BG_CARD, fg=FG_PRIMARY, highlightthickness=0,
                                    troughcolor=BG_DARK)
        self.fps_slider.pack(side=tk.LEFT, padx=5)
        self.fps_label = tk.Label(self.fps_frame, text="2.0 fps",
                                   font=("Segoe UI", 9, "bold"), bg=BG_CARD, fg=COLOR_CYAN)
        self.fps_label.pack(side=tk.LEFT)
        self.fps_slider.config(command=lambda v: self.fps_label.config(text=f"{v} fps"))

        # Skip similar frames
        self.skip_var = tk.BooleanVar(value=True)
        tk.Checkbutton(ext_card, text="🔄 Skip near-duplicate frames (histogram similarity)",
                       variable=self.skip_var, bg=BG_CARD, fg=FG_PRIMARY, selectcolor=BG_DARK,
                       font=("Segoe UI", 9)).pack(anchor=tk.W, pady=2)

        # Similarity threshold
        sim_row = tk.Frame(ext_card, bg=BG_CARD)
        sim_row.pack(fill=tk.X, pady=2)
        tk.Label(sim_row, text="Similarity threshold:", font=("Segoe UI", 9),
                 bg=BG_CARD, fg=FG_SECONDARY).pack(side=tk.LEFT)
        self.sim_var = tk.DoubleVar(value=0.95)
        tk.Scale(sim_row, from_=0.80, to=0.99, resolution=0.01, orient=tk.HORIZONTAL,
                 variable=self.sim_var, length=120, bg=BG_CARD, fg=FG_PRIMARY,
                 highlightthickness=0, troughcolor=BG_DARK).pack(side=tk.LEFT, padx=5)

        # -- Hand Detection / Labeling Settings --
        det_card = tk.LabelFrame(left, text="🖐️ BOUNDING BOX / LABELING", font=("Segoe UI", 10, "bold"),
                                  bg=BG_CARD, fg=COLOR_GREEN, padx=10, pady=8)
        det_card.pack(fill=tk.X, pady=5)

        self.hand_detect_var = tk.BooleanVar(value=True)
        mp_state = tk.NORMAL if MEDIAPIPE_AVAILABLE else tk.DISABLED
        tk.Checkbutton(det_card, text="🖐️ Auto-detect hand bounding box (MediaPipe)",
                       variable=self.hand_detect_var, bg=BG_CARD, fg=COLOR_GREEN,
                       selectcolor=BG_DARK, font=("Segoe UI", 9, "bold"),
                       state=mp_state).pack(anchor=tk.W, pady=2)

        if not MEDIAPIPE_AVAILABLE:
            tk.Label(det_card, text="⚠️ mediapipe not installed — will use full frame as bbox",
                     font=("Segoe UI", 8), bg=BG_CARD, fg=COLOR_RED).pack(anchor=tk.W)

        pad_row = tk.Frame(det_card, bg=BG_CARD)
        pad_row.pack(fill=tk.X, pady=2)
        tk.Label(pad_row, text="Hand padding (px):", font=("Segoe UI", 9),
                 bg=BG_CARD, fg=FG_SECONDARY).pack(side=tk.LEFT)
        self.pad_var = tk.IntVar(value=40)
        tk.Scale(pad_row, from_=10, to=100, orient=tk.HORIZONTAL, variable=self.pad_var,
                 length=120, bg=BG_CARD, fg=FG_PRIMARY, highlightthickness=0,
                 troughcolor=BG_DARK).pack(side=tk.LEFT, padx=5)

        self.fullframe_var = tk.BooleanVar(value=False)
        tk.Checkbutton(det_card, text="📐 Fallback: use full frame when no hand detected",
                       variable=self.fullframe_var, bg=BG_CARD, fg=FG_PRIMARY,
                       selectcolor=BG_DARK, font=("Segoe UI", 9)).pack(anchor=tk.W, pady=2)

        # -- Split Ratio --
        split_card = tk.LabelFrame(left, text="📊 TRAIN / VAL / TEST SPLIT", font=("Segoe UI", 10, "bold"),
                                    bg=BG_CARD, fg=COLOR_YELLOW, padx=10, pady=8)
        split_card.pack(fill=tk.X, pady=5)

        split_row = tk.Frame(split_card, bg=BG_CARD)
        split_row.pack(fill=tk.X)

        tk.Label(split_row, text="Train %:", font=("Segoe UI", 9), bg=BG_CARD, fg=FG_SECONDARY).pack(side=tk.LEFT)
        self.train_pct = tk.IntVar(value=70)
        tk.Spinbox(split_row, from_=50, to=90, width=4, textvariable=self.train_pct,
                   font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=3)

        tk.Label(split_row, text="Val %:", font=("Segoe UI", 9), bg=BG_CARD, fg=FG_SECONDARY).pack(side=tk.LEFT, padx=(10,0))
        self.val_pct = tk.IntVar(value=20)
        tk.Spinbox(split_row, from_=5, to=30, width=4, textvariable=self.val_pct,
                   font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=3)

        tk.Label(split_row, text="Test %:", font=("Segoe UI", 9), bg=BG_CARD, fg=FG_SECONDARY).pack(side=tk.LEFT, padx=(10,0))
        self.test_pct = tk.IntVar(value=10)
        tk.Spinbox(split_row, from_=0, to=30, width=4, textvariable=self.test_pct,
                   font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=3)

        # -- Action Buttons --
        action_card = tk.Frame(left, bg=BG_DARK)
        action_card.pack(fill=tk.X, pady=8)

        self.convert_btn = tk.Button(action_card, text="🚀 CONVERT VIDEOS → YOLO DATASET",
                                      command=self._start_conversion,
                                      bg=COLOR_GREEN, fg="#000", font=("Segoe UI", 12, "bold"),
                                      relief=tk.FLAT, cursor="hand2", width=35, pady=6)
        self.convert_btn.pack(pady=3)

        self.cancel_btn = tk.Button(action_card, text="⏹ Cancel",
                                     command=self._cancel_conversion,
                                     bg=COLOR_RED, fg="white", font=("Segoe UI", 10, "bold"),
                                     relief=tk.FLAT, cursor="hand2", width=20, state=tk.DISABLED)
        self.cancel_btn.pack(pady=3)

        # ── Right Panel: Progress & Preview ──
        right = tk.Frame(main, bg=BG_DARK)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # -- Progress --
        prog_card = tk.LabelFrame(right, text="📊 PROGRESS", font=("Segoe UI", 10, "bold"),
                                   bg=BG_CARD, fg=COLOR_CYAN, padx=10, pady=8)
        prog_card.pack(fill=tk.X, pady=(0, 5))

        self.progress_bar = ttk.Progressbar(prog_card, length=500, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=5)

        self.progress_label = tk.Label(prog_card, text="Ready — Select a video folder to begin",
                                        font=("Segoe UI", 11, "bold"), bg=BG_CARD, fg=COLOR_GREEN,
                                        wraplength=500, justify=tk.LEFT)
        self.progress_label.pack(anchor=tk.W)

        self.stats_label = tk.Label(prog_card, text="",
                                     font=("Consolas", 9), bg=BG_CARD, fg=FG_SECONDARY,
                                     wraplength=500, justify=tk.LEFT)
        self.stats_label.pack(anchor=tk.W, pady=3)

        # -- Class Summary --
        cls_card = tk.LabelFrame(right, text="📋 CLASS SUMMARY", font=("Segoe UI", 10, "bold"),
                                  bg=BG_CARD, fg=COLOR_ORANGE, padx=10, pady=8)
        cls_card.pack(fill=tk.BOTH, expand=True, pady=5)

        # Scrollable text for class info
        self.class_text = tk.Text(cls_card, bg=BG_DARK, fg=FG_PRIMARY,
                                   font=("Consolas", 9), height=10,
                                   wrap=tk.WORD, relief=tk.FLAT, state=tk.DISABLED)
        cls_scroll = tk.Scrollbar(cls_card, command=self.class_text.yview)
        self.class_text.configure(yscrollcommand=cls_scroll.set)
        self.class_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        cls_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # -- Preview Panel --
        prev_card = tk.LabelFrame(right, text="🖼️ FRAME PREVIEW", font=("Segoe UI", 10, "bold"),
                                   bg=BG_CARD, fg=COLOR_PURPLE, padx=10, pady=8)
        prev_card.pack(fill=tk.X, pady=5)

        self.preview_label = tk.Label(prev_card, bg=BG_CARD, text="Preview will appear during conversion",
                                       font=("Segoe UI", 9), fg=FG_SECONDARY)
        self.preview_label.pack(pady=5)

        # ── Status Bar ──
        status_bar = tk.Frame(self.root, bg=BG_ACCENT, pady=3)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        mp_status = "✅ MediaPipe" if MEDIAPIPE_AVAILABLE else "❌ MediaPipe (using full-frame fallback)"
        self.status_label = tk.Label(status_bar, text=f"Status: Ready │ {mp_status}",
                                      font=("Segoe UI", 9), bg=BG_ACCENT, fg=FG_SECONDARY)
        self.status_label.pack(side=tk.LEFT, padx=10)

    # ═══════════════════════════════════════════════════════════
    #  EVENT HANDLERS
    # ═══════════════════════════════════════════════════════════
    def _on_mode_change(self):
        if self.mode_var.get() == "interval":
            self.fps_frame.pack_forget()
            self.interval_frame.pack(fill=tk.X, pady=2)
        else:
            self.interval_frame.pack_forget()
            self.fps_frame.pack(fill=tk.X, pady=2)

    def _browse_source(self):
        folder = filedialog.askdirectory(title="Select Video Folder (containing class subfolders)")
        if folder:
            self.src_entry.delete(0, tk.END)
            self.src_entry.insert(0, folder)
            self.video_dir = folder
            self._scan_videos()

    def _scan_videos(self):
        """Scan the selected folder for video files organized by class."""
        path = self.src_entry.get().strip()
        if not path or not os.path.isdir(path):
            messagebox.showwarning("Warning", "Please select a valid video folder!")
            return

        self.video_dir = path
        self.video_classes = {}
        total_videos = 0
        total_size_mb = 0

        for item in sorted(os.listdir(path)):
            item_path = os.path.join(path, item)

            if os.path.isdir(item_path):
                # Class subfolder — look for videos inside
                videos = []
                for f in os.listdir(item_path):
                    if f.lower().endswith(VIDEO_EXTENSIONS):
                        fpath = os.path.join(item_path, f)
                        videos.append(fpath)
                        total_size_mb += os.path.getsize(fpath) / (1024 * 1024)
                if videos:
                    class_name = item.lower().strip()
                    self.video_classes[class_name] = videos
                    total_videos += len(videos)
            elif item.lower().endswith(VIDEO_EXTENSIONS):
                # Video file directly in root — use filename as class name
                class_name = os.path.splitext(item)[0].lower().strip()
                if class_name not in self.video_classes:
                    self.video_classes[class_name] = []
                self.video_classes[class_name].append(os.path.join(path, item))
                total_videos += 1
                total_size_mb += os.path.getsize(os.path.join(path, item)) / (1024 * 1024)

        if not self.video_classes:
            self.scan_result.config(
                text="❌ No videos found! Expected structure:\n  folder/class_name/video.mp4",
                fg=COLOR_RED)
            return

        # Update UI
        summary = (f"✅ Found {len(self.video_classes)} classes, "
                   f"{total_videos} videos ({total_size_mb:.1f} MB)")
        self.scan_result.config(text=summary, fg=COLOR_GREEN)

        # Update class text
        self.class_text.config(state=tk.NORMAL)
        self.class_text.delete("1.0", tk.END)
        for cls_name, vids in sorted(self.video_classes.items()):
            vid_names = [os.path.basename(v) for v in vids]
            self.class_text.insert(tk.END, f"📁 {cls_name.upper()} ({len(vids)} videos)\n")
            for vn in vid_names:
                self.class_text.insert(tk.END, f"    📹 {vn}\n")
            self.class_text.insert(tk.END, "\n")
        self.class_text.config(state=tk.DISABLED)

        self.status_label.config(text=f"Status: Scanned │ {len(self.video_classes)} classes, {total_videos} videos")

    # ═══════════════════════════════════════════════════════════
    #  HAND DETECTION
    # ═══════════════════════════════════════════════════════════
    def _detect_hand_bbox(self, frame):
        """Detect hands in frame, return YOLO-format bbox (cx, cy, w, h) normalized."""
        h, w = frame.shape[:2]

        if MEDIAPIPE_AVAILABLE and self.hands and self.hand_detect_var.get():
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            if results.multi_hand_landmarks:
                all_x, all_y = [], []
                for hand_lm in results.multi_hand_landmarks:
                    all_x.extend([lm.x * w for lm in hand_lm.landmark])
                    all_y.extend([lm.y * h for lm in hand_lm.landmark])

                pad = self.pad_var.get()
                x_min = max(0, min(all_x) - pad)
                y_min = max(0, min(all_y) - pad)
                x_max = min(w, max(all_x) + pad)
                y_max = min(h, max(all_y) + pad)

                cx = ((x_min + x_max) / 2.0) / w
                cy = ((y_min + y_max) / 2.0) / h
                bw = (x_max - x_min) / w
                bh = (y_max - y_min) / h

                return (cx, cy, bw, bh), True

        # Fallback: skin color detection
        bbox = self._detect_hand_skin(frame)
        if bbox:
            return bbox, True

        # No hand found — use full frame if allowed
        if self.fullframe_var.get():
            return (0.5, 0.5, 1.0, 1.0), False

        return None, False

    def _detect_hand_skin(self, frame):
        """Fallback hand detection using skin color in HSV."""
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_skin = np.array([0, 30, 60], dtype=np.uint8)
        upper_skin = np.array([20, 150, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            min_area = (h * w) * 0.02
            valid = [c for c in contours if cv2.contourArea(c) > min_area]
            valid.sort(key=cv2.contourArea, reverse=True)
            valid = valid[:2]

            if valid:
                x_min_all, y_min_all = w, h
                x_max_all, y_max_all = 0, 0
                for c in valid:
                    cx, cy, cw, ch = cv2.boundingRect(c)
                    x_min_all = min(x_min_all, cx)
                    y_min_all = min(y_min_all, cy)
                    x_max_all = max(x_max_all, cx + cw)
                    y_max_all = max(y_max_all, cy + ch)

                pad = self.pad_var.get()
                x_min = max(0, x_min_all - pad)
                y_min = max(0, y_min_all - pad)
                x_max = min(w, x_max_all + pad)
                y_max = min(h, y_max_all + pad)

                ncx = ((x_min + x_max) / 2.0) / w
                ncy = ((y_min + y_max) / 2.0) / h
                nbw = (x_max - x_min) / w
                nbh = (y_max - y_min) / h

                return (ncx, ncy, nbw, nbh)

        return None

    # ═══════════════════════════════════════════════════════════
    #  FRAME SIMILARITY CHECK
    # ═══════════════════════════════════════════════════════════
    def _frames_similar(self, frame1, frame2):
        """Check if two frames are near-duplicates via histogram correlation."""
        if frame1 is None or frame2 is None:
            return False

        h1 = cv2.calcHist([frame1], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
        h2 = cv2.calcHist([frame2], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])

        cv2.normalize(h1, h1)
        cv2.normalize(h2, h2)

        score = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
        return score >= self.sim_var.get()

    # ═══════════════════════════════════════════════════════════
    #  CONVERSION ENGINE
    # ═══════════════════════════════════════════════════════════
    def _start_conversion(self):
        """Start the video → YOLO dataset conversion in a background thread."""
        if not self.video_classes:
            messagebox.showwarning("Warning", "Please scan a video folder first!")
            return

        # Validate split ratios
        total_pct = self.train_pct.get() + self.val_pct.get() + self.test_pct.get()
        if total_pct != 100:
            messagebox.showwarning("Warning",
                f"Split percentages must add up to 100%!\nCurrent: {total_pct}%")
            return

        self.processing = True
        self.cancel_requested = False
        self.total_extracted = 0
        self.total_skipped = 0

        # Disable UI
        self.convert_btn.config(state=tk.DISABLED)
        self.cancel_btn.config(state=tk.NORMAL)

        # Run conversion in background thread
        threading.Thread(target=self._convert_worker, daemon=True).start()

    def _cancel_conversion(self):
        self.cancel_requested = True
        self.cancel_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Cancelling...")

    def _convert_worker(self):
        """Background worker: extract frames from all videos."""
        try:
            self._do_conversion()
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Conversion failed:\n{e}"))
        finally:
            self.processing = False
            self.root.after(0, self._conversion_done)

    def _do_conversion(self):
        """Core conversion logic."""
        # ── Step 1: Create temp staging area ──
        staging_dir = os.path.join(self.output_dir, "_video_staging")
        os.makedirs(staging_dir, exist_ok=True)

        # Load existing class names from data.yaml if it exists
        existing_classes = []
        if os.path.exists(self.data_yaml_path):
            try:
                with open(self.data_yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                    existing_classes = data.get('names', [])
            except Exception:
                pass

        # Merge classes: existing + new from videos
        all_classes = list(existing_classes)
        for cls_name in sorted(self.video_classes.keys()):
            if cls_name not in all_classes:
                all_classes.append(cls_name)

        # ── Step 2: Count total videos for progress ──
        total_videos = sum(len(vids) for vids in self.video_classes.values())
        processed_videos = 0

        # Per-class extracted counts
        class_frame_counts = {cls: 0 for cls in self.video_classes}
        # Collected frames: {class_name: [(frame_path, label_path), ...]}
        all_frames = {cls: [] for cls in self.video_classes}

        # ── Step 3: Extract frames from each video ──
        for cls_name, video_paths in sorted(self.video_classes.items()):
            class_id = all_classes.index(cls_name)
            cls_staging = os.path.join(staging_dir, cls_name)
            os.makedirs(cls_staging, exist_ok=True)

            for vid_path in video_paths:
                if self.cancel_requested:
                    return

                vid_name = os.path.splitext(os.path.basename(vid_path))[0]
                self.root.after(0, lambda c=cls_name, v=vid_name: self.progress_label.config(
                    text=f"📹 Processing: {c.upper()} / {v}"))

                cap = cv2.VideoCapture(vid_path)
                if not cap.isOpened():
                    processed_videos += 1
                    continue

                video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_idx = 0
                prev_frame = None

                # Calculate extraction interval
                if self.mode_var.get() == "interval":
                    step = max(1, self.interval_var.get())
                else:
                    target_fps = max(0.1, self.fps_var.get())
                    step = max(1, int(round(video_fps / target_fps)))

                while cap.isOpened():
                    if self.cancel_requested:
                        cap.release()
                        return

                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Only process every Nth frame
                    if frame_idx % step != 0:
                        frame_idx += 1
                        continue

                    # Skip similar frames
                    if self.skip_var.get() and prev_frame is not None:
                        if self._frames_similar(frame, prev_frame):
                            self.total_skipped += 1
                            frame_idx += 1
                            continue

                    # Detect hand bounding box
                    bbox, hand_found = self._detect_hand_bbox(frame)

                    if bbox is None:
                        # No hand detected and full-frame fallback is off — skip
                        self.total_skipped += 1
                        frame_idx += 1
                        continue

                    cx, cy, bw, bh = bbox

                    # Save frame + label
                    timestamp = datetime.now().strftime("%H%M%S%f")[:10]
                    fname = f"{cls_name}_{vid_name}_f{frame_idx}_{timestamp}"
                    img_path = os.path.join(cls_staging, f"{fname}.jpg")
                    lbl_path = os.path.join(cls_staging, f"{fname}.txt")

                    cv2.imwrite(img_path, frame)
                    with open(lbl_path, 'w') as f:
                        f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

                    all_frames[cls_name].append((img_path, lbl_path))
                    class_frame_counts[cls_name] += 1
                    self.total_extracted += 1

                    prev_frame = frame.copy()
                    frame_idx += 1

                    # Update preview periodically
                    if self.total_extracted % 5 == 0:
                        self._update_preview(frame, cls_name, bbox, hand_found)

                    # Update progress
                    if total_frames > 0:
                        vid_pct = (frame_idx / total_frames) * 100
                        overall_pct = ((processed_videos + vid_pct/100) / total_videos) * 100
                        self.root.after(0, lambda p=overall_pct: self.progress_bar.configure(value=p))

                    self.root.after(0, lambda: self.stats_label.config(
                        text=f"Extracted: {self.total_extracted} │ Skipped: {self.total_skipped}"))

                cap.release()
                processed_videos += 1

        # ── Step 4: Split into train/val/test ──
        if self.cancel_requested:
            return

        self.root.after(0, lambda: self.progress_label.config(text="📊 Splitting into train/val/test..."))

        train_pct = self.train_pct.get() / 100.0
        val_pct = self.val_pct.get() / 100.0
        # test_pct = remainder

        # Ensure output directories exist
        for split in ['train', 'valid', 'test']:
            os.makedirs(os.path.join(self.output_dir, split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, split, 'labels'), exist_ok=True)

        split_counts = {'train': 0, 'valid': 0, 'test': 0}

        for cls_name, frames in all_frames.items():
            if not frames:
                continue

            random.shuffle(frames)
            n = len(frames)
            n_train = int(n * train_pct)
            n_val = int(n * val_pct)

            splits = {
                'train': frames[:n_train],
                'valid': frames[n_train:n_train + n_val],
                'test': frames[n_train + n_val:]
            }

            for split_name, split_frames in splits.items():
                img_dir = os.path.join(self.output_dir, split_name, 'images')
                lbl_dir = os.path.join(self.output_dir, split_name, 'labels')

                for img_path, lbl_path in split_frames:
                    img_name = os.path.basename(img_path)
                    lbl_name = os.path.basename(lbl_path)

                    shutil.move(img_path, os.path.join(img_dir, img_name))
                    shutil.move(lbl_path, os.path.join(lbl_dir, lbl_name))
                    split_counts[split_name] += 1

        # ── Step 5: Update data.yaml ──
        self.root.after(0, lambda: self.progress_label.config(text="📝 Updating data.yaml..."))
        self._update_data_yaml(all_classes)

        # ── Step 6: Cleanup staging ──
        try:
            shutil.rmtree(staging_dir, ignore_errors=True)
        except Exception:
            pass

        # ── Done ──
        summary = (
            f"✅ CONVERSION COMPLETE!\n\n"
            f"Total frames extracted: {self.total_extracted}\n"
            f"Duplicate frames skipped: {self.total_skipped}\n\n"
            f"Split: Train={split_counts['train']} │ "
            f"Val={split_counts['valid']} │ Test={split_counts['test']}\n\n"
            f"Classes: {len(all_classes)} total\n"
        )
        for cls, count in sorted(class_frame_counts.items()):
            summary += f"  • {cls}: {count} frames\n"

        self.root.after(0, lambda: self.progress_label.config(text="✅ Conversion complete!"))
        self.root.after(0, lambda: self.stats_label.config(text=summary))
        self.root.after(0, lambda s=summary: messagebox.showinfo("Conversion Complete", s))

    def _update_preview(self, frame, cls_name, bbox, hand_found):
        """Update the preview panel with a processed frame."""
        def _do_update():
            try:
                display = frame.copy()
                h, w = display.shape[:2]

                if bbox and hand_found:
                    cx, cy, bw, bh = bbox
                    x1 = int((cx - bw/2) * w)
                    y1 = int((cy - bh/2) * h)
                    x2 = int((cx + bw/2) * w)
                    y2 = int((cy + bh/2) * h)
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 230, 118), 2)
                    cv2.putText(display, f"{cls_name.upper()}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 118), 2)

                # Resize for preview
                preview_w = 400
                scale = preview_w / w
                preview_h = int(h * scale)
                display = cv2.resize(display, (preview_w, preview_h))

                # Convert to ImageTk
                rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                tk_img = ImageTk.PhotoImage(pil_img)
                self.preview_label.config(image=tk_img, text="")
                self.preview_label.image = tk_img
            except Exception:
                pass

        self.root.after(0, _do_update)

    def _update_data_yaml(self, all_classes):
        """Update Data/data.yaml with merged class list."""
        data = {
            'names': all_classes,
            'nc': len(all_classes),
            'train': '../train/images',
            'val': '../valid/images',
            'test': '../test/images'
        }

        os.makedirs(os.path.dirname(self.data_yaml_path), exist_ok=True)
        with open(self.data_yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    def _conversion_done(self):
        """Re-enable UI after conversion."""
        self.convert_btn.config(state=tk.NORMAL)
        self.cancel_btn.config(state=tk.DISABLED)
        self.progress_bar.configure(value=100)

        if self.cancel_requested:
            self.status_label.config(text="Status: Cancelled")
            self.progress_label.config(text="⏹ Conversion cancelled by user")
        else:
            self.status_label.config(
                text=f"Status: Done │ {self.total_extracted} frames extracted │ Ready to train!")

        # Update class text with final counts
        self.class_text.config(state=tk.NORMAL)
        self.class_text.delete("1.0", tk.END)
        self.class_text.insert(tk.END, "── FINAL DATASET ──\n\n")

        for split in ['train', 'valid', 'test']:
            img_dir = os.path.join(self.output_dir, split, 'images')
            if os.path.exists(img_dir):
                count = len([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
                self.class_text.insert(tk.END, f"  {split:6s}: {count} images\n")

        self.class_text.insert(tk.END, "\n── PER-CLASS (from videos) ──\n\n")
        for cls_name in sorted(self.video_classes.keys()):
            # Count frames for this class across all splits
            total = 0
            for split in ['train', 'valid', 'test']:
                img_dir = os.path.join(self.output_dir, split, 'images')
                if os.path.exists(img_dir):
                    total += len([f for f in os.listdir(img_dir)
                                  if f.startswith(cls_name)])
            self.class_text.insert(tk.END, f"  {cls_name.upper():20s}: {total} frames\n")

        self.class_text.config(state=tk.DISABLED)


# ═══════════════════════════════════════════════════════════════
#  STANDALONE FUNCTION (for use by train.py)
# ═══════════════════════════════════════════════════════════════
def convert_video_dir(video_dir, output_dir="Data", frame_interval=5,
                      split_ratio=(0.70, 0.20, 0.10), hand_detect=True,
                      hand_padding=40, skip_similar=True, sim_threshold=0.95,
                      full_frame_fallback=True, verbose=True):
    """
    Convert a folder of videos into YOLO image dataset (CLI/script usage).

    Args:
        video_dir: Path to video folder (class_name/video.ext structure)
        output_dir: Path to output Data/ directory
        frame_interval: Extract every Nth frame
        split_ratio: (train, val, test) percentages as decimals
        hand_detect: Use MediaPipe for auto hand bounding box
        hand_padding: Pixels of padding around detected hand
        skip_similar: Skip near-duplicate frames
        sim_threshold: Histogram similarity threshold (0-1)
        full_frame_fallback: Use full frame when no hand detected
        verbose: Print progress

    Returns:
        dict with extraction statistics
    """
    if not os.path.isdir(video_dir):
        raise ValueError(f"Video directory not found: {video_dir}")

    # ── Discover videos ──
    video_classes = {}
    for item in sorted(os.listdir(video_dir)):
        item_path = os.path.join(video_dir, item)
        if os.path.isdir(item_path):
            videos = [os.path.join(item_path, f) for f in os.listdir(item_path)
                      if f.lower().endswith(VIDEO_EXTENSIONS)]
            if videos:
                video_classes[item.lower().strip()] = videos
        elif item.lower().endswith(VIDEO_EXTENSIONS):
            cls = os.path.splitext(item)[0].lower().strip()
            video_classes.setdefault(cls, []).append(os.path.join(video_dir, item))

    if not video_classes:
        raise ValueError(f"No videos found in {video_dir}")

    total_videos = sum(len(v) for v in video_classes.values())
    if verbose:
        print(f"\n{'='*60}")
        print(f"  📹 Video Dataset Converter")
        print(f"  Found: {len(video_classes)} classes, {total_videos} videos")
        print(f"  Frame interval: every {frame_interval} frames")
        print(f"  Hand detection: {'ON' if hand_detect else 'OFF'}")
        print(f"{'='*60}\n")

    # ── Init hand detector ──
    hands = None
    if hand_detect and MEDIAPIPE_AVAILABLE:
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2,
                               min_detection_confidence=0.5)
        if verbose:
            print("  ✅ MediaPipe hand detection active")
    elif verbose:
        print("  ⚠️ MediaPipe not available — using skin/full-frame detection")

    # ── Load existing classes ──
    data_yaml = os.path.join(output_dir, "data.yaml")
    existing_classes = []
    if os.path.exists(data_yaml):
        try:
            with open(data_yaml, 'r') as f:
                existing_classes = yaml.safe_load(f).get('names', [])
        except Exception:
            pass

    all_classes = list(existing_classes)
    for cls in sorted(video_classes.keys()):
        if cls not in all_classes:
            all_classes.append(cls)

    # ── Staging ──
    staging = os.path.join(output_dir, "_video_staging")
    os.makedirs(staging, exist_ok=True)

    stats = {'extracted': 0, 'skipped': 0, 'per_class': {}, 'split': {}}
    all_frames = {cls: [] for cls in video_classes}

    # ── Extract frames ──
    for cls_name, vids in sorted(video_classes.items()):
        class_id = all_classes.index(cls_name)
        cls_dir = os.path.join(staging, cls_name)
        os.makedirs(cls_dir, exist_ok=True)
        cls_count = 0

        for vid_path in vids:
            cap = cv2.VideoCapture(vid_path)
            if not cap.isOpened():
                continue

            frame_idx = 0
            prev_frame = None
            vid_name = os.path.splitext(os.path.basename(vid_path))[0]

            if verbose:
                total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                print(f"  📹 {cls_name}/{os.path.basename(vid_path)} ({total_f} frames)...", end=" ")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_interval != 0:
                    frame_idx += 1
                    continue

                # Skip similar
                if skip_similar and prev_frame is not None:
                    h1 = cv2.calcHist([frame], [0,1,2], None, [16,16,16], [0,256,0,256,0,256])
                    h2 = cv2.calcHist([prev_frame], [0,1,2], None, [16,16,16], [0,256,0,256,0,256])
                    cv2.normalize(h1, h1); cv2.normalize(h2, h2)
                    if cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL) >= sim_threshold:
                        stats['skipped'] += 1
                        frame_idx += 1
                        continue

                # Detect hand
                h_img, w_img = frame.shape[:2]
                bbox = None

                if hands:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = hands.process(rgb)
                    if res.multi_hand_landmarks:
                        ax = [lm.x * w_img for hand in res.multi_hand_landmarks for lm in hand.landmark]
                        ay = [lm.y * h_img for hand in res.multi_hand_landmarks for lm in hand.landmark]
                        x1 = max(0, min(ax) - hand_padding)
                        y1 = max(0, min(ay) - hand_padding)
                        x2 = min(w_img, max(ax) + hand_padding)
                        y2 = min(h_img, max(ay) + hand_padding)
                        bbox = (((x1+x2)/2)/w_img, ((y1+y2)/2)/h_img,
                                (x2-x1)/w_img, (y2-y1)/h_img)

                if bbox is None and full_frame_fallback:
                    bbox = (0.5, 0.5, 1.0, 1.0)

                if bbox is None:
                    stats['skipped'] += 1
                    frame_idx += 1
                    continue

                cx, cy, bw, bh = bbox
                fname = f"{cls_name}_{vid_name}_f{frame_idx}"
                img_p = os.path.join(cls_dir, f"{fname}.jpg")
                lbl_p = os.path.join(cls_dir, f"{fname}.txt")

                cv2.imwrite(img_p, frame)
                with open(lbl_p, 'w') as f:
                    f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

                all_frames[cls_name].append((img_p, lbl_p))
                cls_count += 1
                stats['extracted'] += 1
                prev_frame = frame.copy()
                frame_idx += 1

            cap.release()

        stats['per_class'][cls_name] = cls_count
        if verbose:
            print(f"{cls_count} frames")

    # ── Split ──
    train_r, val_r, _ = split_ratio
    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)

    stats['split'] = {'train': 0, 'valid': 0, 'test': 0}
    for cls_name, frames in all_frames.items():
        random.shuffle(frames)
        n = len(frames)
        n_train = int(n * train_r)
        n_val = int(n * val_r)
        splits = {'train': frames[:n_train], 'valid': frames[n_train:n_train+n_val],
                   'test': frames[n_train+n_val:]}
        for sname, sframes in splits.items():
            for ip, lp in sframes:
                shutil.move(ip, os.path.join(output_dir, sname, 'images', os.path.basename(ip)))
                shutil.move(lp, os.path.join(output_dir, sname, 'labels', os.path.basename(lp)))
                stats['split'][sname] += 1

    # ── Update data.yaml ──
    yaml_data = {'names': all_classes, 'nc': len(all_classes),
                 'train': '../train/images', 'val': '../valid/images', 'test': '../test/images'}
    with open(data_yaml, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)

    # ── Cleanup ──
    shutil.rmtree(staging, ignore_errors=True)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  ✅ Conversion complete!")
        print(f"  Extracted: {stats['extracted']} │ Skipped: {stats['skipped']}")
        print(f"  Train: {stats['split']['train']} │ Val: {stats['split']['valid']} │ Test: {stats['split']['test']}")
        print(f"  Classes: {len(all_classes)} │ data.yaml updated")
        print(f"{'='*60}\n")

    if hands:
        hands.close()

    return stats


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys

    # CLI mode: python video_dataset.py --dir videos/
    if len(sys.argv) > 1 and "--dir" in sys.argv:
        idx = sys.argv.index("--dir")
        if idx + 1 < len(sys.argv):
            video_path = sys.argv[idx + 1]
            interval = 5
            if "--interval" in sys.argv:
                i = sys.argv.index("--interval")
                interval = int(sys.argv[i+1])
            convert_video_dir(video_path, frame_interval=interval)
        else:
            print("Usage: python video_dataset.py --dir path/to/videos [--interval 5]")
    else:
        # GUI mode
        root = tk.Tk()
        app = VideoDatasetConverter(root)
        root.mainloop()
