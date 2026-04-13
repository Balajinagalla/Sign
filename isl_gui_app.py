# isl_gui_app.py - Enhanced Indian Sign Language Recognition GUI
# Features: 7 Unique Advanced Features + Temporal smoothing, Sentence builder,
#           Sign history, Quiz mode, Text-to-Sign, Hand Intelligence
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import cv2
from ultralytics import YOLO
import threading
import numpy as np
from PIL import Image, ImageTk
from collections import deque
import time
import random
import os
import json
import mediapipe as mp
from tts_indic_multi import speak_sign, LANGUAGES, LANG_CODES, TRANSLATIONS

# ── Enhancement Modules ─────────────────────────────────────────
try:
    from enhancements import (
        FaceBlur, GradCAMVisualizer, SignDictionary, SessionRecorder,
        ProgressTracker, ConfidenceCalibrator, RealtimeCharts,
        GestureShortcuts, THEMES, ConversationMode
    )
    ENHANCEMENTS_AVAILABLE = True
except ImportError:
    ENHANCEMENTS_AVAILABLE = False

# ── Color Palette (Premium Theme) ───────────────────────────────
BG_DARK      = "#121212"
BG_CARD      = "#1e1e2f"
BG_ACCENT    = "#16213e"
FG_PRIMARY   = "#e0e0e0"
FG_SECONDARY = "#b0b0b0"
COLOR_GREEN  = "#03dac6"  # Aqua/Mint
COLOR_RED    = "#cf6679"  # Soft Red
COLOR_CYAN   = "#03dac6"  # Aqua
COLOR_ORANGE = "#ffb74d"
COLOR_PURPLE = "#bb86fc"  # Electric Indigo
COLOR_YELLOW = "#fbc02d"

SIGN_REF_DIR = "sign_references"


class ISLRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🤟 ISL Recognition — Indian Sign Language → Speech")
        self.root.geometry("1150x850")
        self.root.configure(bg=BG_DARK)
        self.root.resizable(True, True)
        
        # ── Window Protocol (Fast Exit) ──────────────────────────
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # ── Model (High-Accuracy Original) ──────────────────────
        pt_path = "best.pt"
        onnx_path = "best.onnx"
        if os.path.exists(pt_path):
            self.model = YOLO(pt_path, task='detect')
            self.model_type = "YOLOv11-Small (High Accuracy)"
        elif os.path.exists(onnx_path):
            self.model = YOLO(onnx_path, task='detect')
            self.model_type = "YOLOv11 (ONNX)"
        else:
            self.model = YOLO("runs/train/sign_lang_yolo11/weights/best.pt", task='detect')
            self.model_type = "YOLOv11 (Fallback)"

        # ── MediaPipe Hands (Hand Intelligence) ──────────────────
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.show_skeleton = True

        self.cap = None
        self.running = False
        self.current_lang_idx = 1  # Hindi

        # ── Temporal Smoothing ───────────────────────────────────
        self.prediction_buffer = deque(maxlen=7)
        self.STABLE_THRESHOLD = 4

        # ── Sentence Builder ─────────────────────────────────────
        self.sentence = []
        self.last_added_sign = None
        self.last_added_time = 0
        self.SIGN_COOLDOWN = 3.0

        # ── Sign History ─────────────────────────────────────────
        self.history = deque(maxlen=20)

        # ── TTS Debounce ─────────────────────────────────────────
        self.last_spoken_sign = None
        self.last_spoken_time = 0
        self.TTS_COOLDOWN = 2.5

        # ── FPS ──────────────────────────────────────────────────
        self.frame_times = deque(maxlen=30)

        # ── Confidence Threshold ─────────────────────────────────
        self.conf_threshold = 0.4

        # ── Quiz Mode ────────────────────────────────────────────
        self.quiz_active = False
        self.quiz_target = None
        self.quiz_score = 0
        self.quiz_total = 0
        self.quiz_streak = 0
        self.quiz_best_streak = 0
        self.quiz_start_time = 0

        # ── Mode ─────────────────────────────────────────────────
        self.current_mode = "recognition"  # recognition, quiz, text2sign

        # ═══════════ 7 UNIQUE FEATURES ═══════════════════════════

        # ── [1] Voice-to-Sign Reverse Translator ─────────────────
        self.voice_listening = False

        # ── [2] Few-Shot Sign Learner ────────────────────────────
        self.fewshot_images = []
        self.fewshot_sign_name = ""

        # ── [3] Emotion-Aware TTS ────────────────────────────────
        self.emotion_enabled = True
        self.current_emotion = "neutral"
        self.face_mesh = None
        try:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1, min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        except Exception:
            pass

        # ── [4] Multi-Person Sign Detection ──────────────────────
        self.person_colors = [
            (0, 230, 118),   # Green
            (100, 149, 237), # Cornflower Blue
            (255, 165, 0),   # Orange
            (220, 20, 60),   # Crimson
        ]

        # ── [5] Emergency Sign Priority Alert ────────────────────
        self.emergency_signs = {"help", "no"}
        self.emergency_flash = 0
        self.last_emergency_time = 0

        # ── [6] Sign Speed Coach ─────────────────────────────────
        self.sign_timestamps = deque(maxlen=10)
        self.current_speed_text = ""
        self.current_speed_color = (200, 200, 200)

        # ── [7] Confusion Matrix Tracker ─────────────────────────
        self.confusion_data = {}  # {predicted: {actual: count}}
        self.confusion_log = deque(maxlen=100)

        # ═════════════════════════════════════════════════════════

        # ── Sign Reference Images ────────────────────────────────
        self.sign_images = {}
        self._load_sign_references()

        # ═══════════ ENHANCEMENT MODULES ═════════════════════════
        if ENHANCEMENTS_AVAILABLE:
            self.face_blur = FaceBlur()
            self.gradcam = GradCAMVisualizer()
            self.recorder = SessionRecorder()
            self.progress = ProgressTracker()
            self.progress.start_session()
            self.calibrator = ConfidenceCalibrator()
            self.charts = RealtimeCharts()
            self.conversation = ConversationMode()
            self.dictionary = SignDictionary()
            self.current_theme = 'dark'
            self.is_fullscreen = False
        else:
            self.face_blur = None
            self.gradcam = None
            self.recorder = None
            self.progress = None
            self.calibrator = None
            self.charts = None
            self.conversation = None
            self.dictionary = None

        self._build_ui()

    def _load_sign_references(self):
        """Load ALL reference images from the sign_references directory."""
        self.all_sign_names = []  # All available sign names (for Text→Sign)
        if not os.path.exists(SIGN_REF_DIR):
            return
        for fname in sorted(os.listdir(SIGN_REF_DIR)):
            if not fname.endswith('.png'):
                continue
            name = fname.replace('.png', '').replace('_', ' ').title()
            path = os.path.join(SIGN_REF_DIR, fname)
            try:
                img = Image.open(path).resize((200, 200), Image.LANCZOS)
                self.sign_images[name] = ImageTk.PhotoImage(img)
                self.all_sign_names.append(name)
            except Exception:
                pass

    # ═════════════════════════════════════════════════════════════
    #  UI CONSTRUCTION
    # ═════════════════════════════════════════════════════════════
    def _build_ui(self):
        # ── Title Bar ────────────────────────────────────────────
        title_frame = tk.Frame(self.root, bg=BG_ACCENT, pady=8)
        title_frame.pack(fill=tk.X)

        tk.Label(title_frame, text="🤟 Indian Sign Language → Speech",
                 font=("Segoe UI", 16, "bold"), bg=BG_ACCENT, fg=FG_PRIMARY
                 ).pack(side=tk.LEFT, padx=15)

        self.fps_label = tk.Label(title_frame, text="FPS: --",
                                  font=("Consolas", 10), bg=BG_ACCENT, fg=COLOR_CYAN)
        self.fps_label.pack(side=tk.RIGHT, padx=15)

        self.model_info = tk.Label(title_frame,
                                   text=f"Model: {self.model_type} | {len(self.model.names)} classes",
                                   font=("Consolas", 10), bg=BG_ACCENT, fg=FG_SECONDARY)
        self.model_info.pack(side=tk.RIGHT, padx=15)

        # ── Controls Row ─────────────────────────────────────────
        ctrl = tk.Frame(self.root, bg=BG_DARK, pady=6)
        ctrl.pack(fill=tk.X, padx=10)

        # Mode selector
        tk.Label(ctrl, text="Mode:", font=("Segoe UI", 10, "bold"),
                 bg=BG_DARK, fg=FG_SECONDARY).pack(side=tk.LEFT, padx=(5, 3))

        self.mode_var = tk.StringVar(value="recognition")
        modes = [("🎤 Recognition", "recognition"), ("📝 Quiz", "quiz"), ("🔄 Text→Sign", "text2sign")]
        for text, val in modes:
            tk.Radiobutton(ctrl, text=text, variable=self.mode_var, value=val,
                          bg=BG_DARK, fg=FG_PRIMARY, selectcolor=BG_CARD,
                          font=("Segoe UI", 9, "bold"), activebackground=BG_DARK,
                          activeforeground=COLOR_CYAN,
                          command=self._on_mode_change).pack(side=tk.LEFT, padx=3)

        # Separator
        tk.Label(ctrl, text="│", fg=FG_SECONDARY, bg=BG_DARK).pack(side=tk.LEFT, padx=5)

        # Language
        tk.Label(ctrl, text="Lang:", font=("Segoe UI", 10),
                 bg=BG_DARK, fg=FG_SECONDARY).pack(side=tk.LEFT, padx=2)
        self.lang_combo = ttk.Combobox(ctrl, values=LANGUAGES, state="readonly", width=10)
        self.lang_combo.set(LANGUAGES[1])
        self.lang_combo.pack(side=tk.LEFT, padx=2)
        self.lang_combo.bind("<<ComboboxSelected>>", self._on_lang_change)

        # Confidence slider
        tk.Label(ctrl, text="│  Conf:", font=("Segoe UI", 10),
                 bg=BG_DARK, fg=FG_SECONDARY).pack(side=tk.LEFT, padx=3)
        self.conf_slider = tk.Scale(ctrl, from_=0.1, to=0.9, resolution=0.05,
                                    orient=tk.HORIZONTAL, length=100,
                                    bg=BG_DARK, fg=FG_PRIMARY, highlightthickness=0,
                                    troughcolor=BG_CARD, command=self._on_conf_change)
        self.conf_slider.set(0.4)
        self.conf_slider.pack(side=tk.LEFT, padx=2)

        # Camera buttons
        self.start_btn = tk.Button(ctrl, text="▶ Start",
                                   command=self.start_camera,
                                   bg=COLOR_GREEN, fg="#000", font=("Segoe UI", 10, "bold"),
                                   width=10, relief=tk.FLAT, cursor="hand2")
        self.start_btn.pack(side=tk.RIGHT, padx=3)

        self.stop_btn = tk.Button(ctrl, text="⏹ Stop",
                                  command=self.stop_camera,
                                  bg=COLOR_RED, fg="white", font=("Segoe UI", 10, "bold"),
                                  width=8, relief=tk.FLAT, cursor="hand2", state=tk.DISABLED)
        self.stop_btn.pack(side=tk.RIGHT, padx=3)

        # ── Hand Intelligence Toggle ─────────────────────────────
        self.skel_var = tk.BooleanVar(value=True)
        tk.Checkbutton(ctrl, text="✨ Hand Intelligence", variable=self.skel_var,
                      command=self._on_skel_toggle, bg=BG_DARK, fg=COLOR_PURPLE,
                      selectcolor=BG_CARD, activebackground=BG_DARK,
                      activeforeground=COLOR_CYAN, font=("Segoe UI", 9, "bold")).pack(side=tk.RIGHT, padx=5)

        # ── Enhancement Toolbar Buttons ───────────────────────────
        if ENHANCEMENTS_AVAILABLE:
            # Face Blur
            self.blur_var = tk.BooleanVar(value=False)
            tk.Checkbutton(ctrl, text="😶 Blur Face", variable=self.blur_var,
                          command=lambda: setattr(self.face_blur, 'enabled', self.blur_var.get()),
                          bg=BG_DARK, fg=COLOR_ORANGE, selectcolor=BG_CARD,
                          font=("Segoe UI", 8, "bold")).pack(side=tk.RIGHT, padx=2)

            # Record button
            self.rec_btn = tk.Button(ctrl, text="⏺ Rec", command=self._toggle_recording,
                                     bg=COLOR_RED, fg="white", font=("Segoe UI", 8, "bold"),
                                     relief=tk.FLAT, cursor="hand2", width=5)
            self.rec_btn.pack(side=tk.RIGHT, padx=2)

            # Fullscreen
            tk.Button(ctrl, text="⛶", command=self._toggle_fullscreen,
                      bg=BG_CARD, fg=FG_PRIMARY, font=("Segoe UI", 10, "bold"),
                      relief=tk.FLAT, cursor="hand2", width=2).pack(side=tk.RIGHT, padx=2)

        # ── Main Content ─────────────────────────────────────────
        content = tk.Frame(self.root, bg=BG_DARK)
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Video panel
        video_panel = tk.Frame(content, bg=BG_CARD, bd=2, relief=tk.GROOVE)
        video_panel.pack(side=tk.LEFT, padx=(0, 8))
        self.video_label = tk.Label(video_panel, bg="black", width=640, height=480)
        self.video_label.pack(padx=2, pady=2)

        # ── Sidebar (notebook with tabs) ─────────────────────────
        sidebar = tk.Frame(content, bg=BG_DARK, width=350)
        sidebar.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        sidebar.pack_propagate(False)

        # Use a notebook for switching between panels
        style = ttk.Style()
        style.configure("Dark.TNotebook", background=BG_DARK)
        style.configure("Dark.TNotebook.Tab", background=BG_CARD, foreground=FG_PRIMARY,
                        padding=[10, 4])
        style.map("Dark.TNotebook.Tab", background=[("selected", BG_ACCENT)])

        self.notebook = ttk.Notebook(sidebar, style="Dark.TNotebook")
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # ── TAB 1: Recognition ───────────────────────────────────
        rec_tab = tk.Frame(self.notebook, bg=BG_DARK)
        self.notebook.add(rec_tab, text="🎤 Recognition")

        # Detected sign
        det_card = tk.Frame(rec_tab, bg=BG_CARD, bd=1, relief=tk.RIDGE, pady=8, padx=8)
        det_card.pack(fill=tk.X, pady=(5, 5), padx=5)
        tk.Label(det_card, text="🔍 DETECTED SIGN",
                 font=("Segoe UI", 9, "bold"), bg=BG_CARD, fg=FG_SECONDARY).pack(anchor=tk.W)
        self.sign_label = tk.Label(det_card, text="—",
                                   font=("Segoe UI", 24, "bold"), bg=BG_CARD, fg=COLOR_RED)
        self.sign_label.pack(pady=3)
        self.conf_label = tk.Label(det_card, text="Confidence: --",
                                   font=("Consolas", 9), bg=BG_CARD, fg=FG_SECONDARY)
        self.conf_label.pack()
        self.stability_label = tk.Label(det_card, text="Stability: ○○○○○○○",
                                        font=("Consolas", 9), bg=BG_CARD, fg=FG_SECONDARY)
        self.stability_label.pack()

        # Sentence builder
        sent_card = tk.Frame(rec_tab, bg=BG_CARD, bd=1, relief=tk.RIDGE, pady=8, padx=8)
        sent_card.pack(fill=tk.X, pady=(0, 5), padx=5)
        sent_hdr = tk.Frame(sent_card, bg=BG_CARD)
        sent_hdr.pack(fill=tk.X)
        tk.Label(sent_hdr, text="💬 SENTENCE",
                 font=("Segoe UI", 9, "bold"), bg=BG_CARD, fg=FG_SECONDARY).pack(side=tk.LEFT)
        tk.Button(sent_hdr, text="Clear", command=self._clear_sentence,
                  bg=COLOR_RED, fg="white", font=("Segoe UI", 7, "bold"),
                  relief=tk.FLAT, cursor="hand2", padx=6).pack(side=tk.RIGHT)
        tk.Button(sent_hdr, text="🔊 Speak", command=self._speak_sentence,
                  bg=COLOR_PURPLE, fg="white", font=("Segoe UI", 7, "bold"),
                  relief=tk.FLAT, cursor="hand2", padx=6).pack(side=tk.RIGHT, padx=3)
        tk.Button(sent_hdr, text="💾 Export", command=self._export_session,
                  bg=COLOR_GREEN, fg="#000", font=("Segoe UI", 7, "bold"),
                  relief=tk.FLAT, cursor="hand2", padx=6).pack(side=tk.RIGHT)
        self.sentence_label = tk.Label(sent_card, text="(Show signs to build a sentence)",
                                       font=("Segoe UI", 12, "bold"), bg=BG_CARD,
                                       fg=COLOR_CYAN, wraplength=280, justify=tk.LEFT)
        self.sentence_label.pack(pady=5, anchor=tk.W)

        # History
        hist_card = tk.Frame(rec_tab, bg=BG_CARD, bd=1, relief=tk.RIDGE, pady=8, padx=8)
        hist_card.pack(fill=tk.BOTH, expand=True, pady=(0, 5), padx=5)
        tk.Label(hist_card, text="📋 SIGN HISTORY",
                 font=("Segoe UI", 9, "bold"), bg=BG_CARD, fg=FG_SECONDARY).pack(anchor=tk.W)
        self.history_text = tk.Text(hist_card, bg=BG_DARK, fg=FG_PRIMARY,
                                    font=("Consolas", 9), height=8,
                                    wrap=tk.WORD, relief=tk.FLAT, state=tk.DISABLED)
        self.history_text.pack(fill=tk.BOTH, expand=True, pady=3)

        # ── TAB 2: Quiz Mode ────────────────────────────────────
        quiz_tab = tk.Frame(self.notebook, bg=BG_DARK)
        self.notebook.add(quiz_tab, text="📝 Quiz")

        quiz_card = tk.Frame(quiz_tab, bg=BG_CARD, bd=1, relief=tk.RIDGE, pady=12, padx=12)
        quiz_card.pack(fill=tk.X, pady=5, padx=5)

        tk.Label(quiz_card, text="📝 QUIZ MODE",
                 font=("Segoe UI", 12, "bold"), bg=BG_CARD, fg=COLOR_YELLOW).pack()
        tk.Label(quiz_card, text="Show the correct sign to score points!",
                 font=("Segoe UI", 9), bg=BG_CARD, fg=FG_SECONDARY).pack()

        self.quiz_target_label = tk.Label(quiz_card, text="Press Start Quiz!",
                                          font=("Segoe UI", 22, "bold"), bg=BG_CARD, fg=COLOR_CYAN)
        self.quiz_target_label.pack(pady=8)

        # Reference image for quiz
        self.quiz_ref_label = tk.Label(quiz_card, bg=BG_CARD)
        self.quiz_ref_label.pack(pady=5)

        self.quiz_feedback = tk.Label(quiz_card, text="",
                                      font=("Segoe UI", 14, "bold"), bg=BG_CARD, fg=FG_SECONDARY)
        self.quiz_feedback.pack(pady=3)

        quiz_btns = tk.Frame(quiz_card, bg=BG_CARD)
        quiz_btns.pack(pady=5)

        self.quiz_start_btn = tk.Button(quiz_btns, text="▶ Start Quiz",
                                        command=self._start_quiz,
                                        bg=COLOR_GREEN, fg="#000", font=("Segoe UI", 10, "bold"),
                                        relief=tk.FLAT, cursor="hand2", width=12)
        self.quiz_start_btn.pack(side=tk.LEFT, padx=5)

        self.quiz_skip_btn = tk.Button(quiz_btns, text="⏭ Skip",
                                       command=self._next_quiz_sign,
                                       bg=COLOR_ORANGE, fg="#000", font=("Segoe UI", 10, "bold"),
                                       relief=tk.FLAT, cursor="hand2", width=8, state=tk.DISABLED)
        self.quiz_skip_btn.pack(side=tk.LEFT, padx=5)

        self.quiz_stop_btn = tk.Button(quiz_btns, text="⏹ Stop",
                                       command=self._stop_quiz,
                                       bg=COLOR_RED, fg="white", font=("Segoe UI", 10, "bold"),
                                       relief=tk.FLAT, cursor="hand2", width=8, state=tk.DISABLED)
        self.quiz_stop_btn.pack(side=tk.LEFT, padx=5)

        # Score card
        score_card = tk.Frame(quiz_tab, bg=BG_CARD, bd=1, relief=tk.RIDGE, pady=10, padx=10)
        score_card.pack(fill=tk.X, pady=5, padx=5)

        tk.Label(score_card, text="📊 SCORE",
                 font=("Segoe UI", 9, "bold"), bg=BG_CARD, fg=FG_SECONDARY).pack(anchor=tk.W)

        score_row = tk.Frame(score_card, bg=BG_CARD)
        score_row.pack(fill=tk.X, pady=5)

        self.score_label = tk.Label(score_row, text="0 / 0",
                                    font=("Segoe UI", 20, "bold"), bg=BG_CARD, fg=COLOR_GREEN)
        self.score_label.pack(side=tk.LEFT, padx=15)

        score_details = tk.Frame(score_row, bg=BG_CARD)
        score_details.pack(side=tk.LEFT, padx=10)

        self.accuracy_label = tk.Label(score_details, text="Accuracy: --%",
                                       font=("Segoe UI", 10), bg=BG_CARD, fg=FG_PRIMARY)
        self.accuracy_label.pack(anchor=tk.W)

        self.streak_label = tk.Label(score_details, text="🔥 Streak: 0  |  Best: 0",
                                     font=("Segoe UI", 10), bg=BG_CARD, fg=COLOR_ORANGE)
        self.streak_label.pack(anchor=tk.W)

        # ── TAB 3: Text → Sign ──────────────────────────────────
        t2s_tab = tk.Frame(self.notebook, bg=BG_DARK)
        self.notebook.add(t2s_tab, text="🔄 Text→Sign")

        t2s_card = tk.Frame(t2s_tab, bg=BG_CARD, bd=1, relief=tk.RIDGE, pady=12, padx=12)
        t2s_card.pack(fill=tk.X, pady=5, padx=5)

        tk.Label(t2s_card, text="🔄 TEXT → SIGN LANGUAGE",
                 font=("Segoe UI", 12, "bold"), bg=BG_CARD, fg=COLOR_PURPLE).pack()
        tk.Label(t2s_card, text="Type a word to see how to sign it",
                 font=("Segoe UI", 9), bg=BG_CARD, fg=FG_SECONDARY).pack(pady=3)

        # Input field
        input_frame = tk.Frame(t2s_card, bg=BG_CARD)
        input_frame.pack(fill=tk.X, pady=8)

        self.t2s_entry = tk.Entry(input_frame, font=("Segoe UI", 14),
                                  bg=BG_DARK, fg=FG_PRIMARY, insertbackground=FG_PRIMARY,
                                  relief=tk.FLAT, width=18)
        self.t2s_entry.pack(side=tk.LEFT, padx=5, ipady=4)
        self.t2s_entry.bind("<Return>", self._on_t2s_search)

        tk.Button(input_frame, text="🔍 Show Sign", command=self._on_t2s_search,
                  bg=COLOR_PURPLE, fg="white", font=("Segoe UI", 10, "bold"),
                  relief=tk.FLAT, cursor="hand2").pack(side=tk.LEFT, padx=5)

        tk.Button(input_frame, text="🔊", command=self._speak_t2s,
                  bg=COLOR_CYAN, fg="#000", font=("Segoe UI", 10, "bold"),
                  relief=tk.FLAT, cursor="hand2", width=3).pack(side=tk.LEFT, padx=3)

        # Result display
        self.t2s_sign_name = tk.Label(t2s_card, text="",
                                      font=("Segoe UI", 18, "bold"), bg=BG_CARD, fg=COLOR_GREEN)
        self.t2s_sign_name.pack(pady=5)

        self.t2s_image_label = tk.Label(t2s_card, bg=BG_CARD)
        self.t2s_image_label.pack(pady=5)

        self.t2s_translation = tk.Label(t2s_card, text="",
                                         font=("Segoe UI", 11), bg=BG_CARD, fg=FG_PRIMARY,
                                         wraplength=300, justify=tk.CENTER)
        self.t2s_translation.pack(pady=3)

        # Scrollable sign library
        quick_frame = tk.Frame(t2s_tab, bg=BG_CARD, bd=1, relief=tk.RIDGE, pady=8, padx=8)
        quick_frame.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)

        total_signs = len(self.all_sign_names) if hasattr(self, 'all_sign_names') else 0
        tk.Label(quick_frame, text=f"📖 SIGN LIBRARY ({total_signs} signs)",
                 font=("Segoe UI", 9, "bold"), bg=BG_CARD, fg=FG_SECONDARY).pack(anchor=tk.W, pady=(0, 5))

        # Scrollable canvas for many buttons
        canvas = tk.Canvas(quick_frame, bg=BG_CARD, highlightthickness=0)
        scrollbar = tk.Scrollbar(quick_frame, orient=tk.VERTICAL, command=canvas.yview)
        btn_container = tk.Frame(canvas, bg=BG_CARD)

        btn_container.bind("<Configure>",
                           lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=btn_container, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        all_signs = self.all_sign_names if hasattr(self, 'all_sign_names') else list(self.model.names.values())
        for i, sign in enumerate(all_signs):
            row, col = divmod(i, 3)
            # Highlight model-detectable signs in green
            is_model_sign = any(sign.lower() == n.lower() for n in self.model.names.values())
            btn_bg = COLOR_GREEN if is_model_sign else BG_ACCENT
            btn_fg = "#000" if is_model_sign else FG_PRIMARY
            tk.Button(btn_container, text=sign.upper(), width=14,
                      command=lambda s=sign: self._show_sign_reference(s),
                      bg=btn_bg, fg=btn_fg, font=("Segoe UI", 8, "bold"),
                      relief=tk.FLAT, cursor="hand2"
                      ).grid(row=row, column=col, padx=2, pady=2)

        # ── Status Bar ───────────────────────────────────────────
        status_bar = tk.Frame(self.root, bg=BG_ACCENT, pady=3)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        self.status_label = tk.Label(status_bar, text="Status: Ready │ Language: Hindi",
                                     font=("Segoe UI", 9), bg=BG_ACCENT, fg=FG_SECONDARY)
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        # Speed Coach display in status bar
        self.speed_label = tk.Label(status_bar, text="",
                                     font=("Segoe UI", 9, "bold"), bg=BG_ACCENT, fg=COLOR_CYAN)
        self.speed_label.pack(side=tk.RIGHT, padx=10)
        
        # Emotion display in status bar
        self.emotion_label = tk.Label(status_bar, text="",
                                      font=("Segoe UI", 9), bg=BG_ACCENT, fg=COLOR_PURPLE)
        self.emotion_label.pack(side=tk.RIGHT, padx=10)

        # ── TAB 4: Advanced Features ─────────────────────────────
        adv_tab = tk.Frame(self.notebook, bg=BG_DARK)
        self.notebook.add(adv_tab, text="🧪 Advanced")

        # ── TAB 5: Tools / Enhancements ──────────────────────────
        if ENHANCEMENTS_AVAILABLE:
            tools_tab = tk.Frame(self.notebook, bg=BG_DARK)
            self.notebook.add(tools_tab, text="🔧 Tools")
            self._build_tools_tab(tools_tab)

        # [1] Voice-to-Sign
        v2s_card = tk.Frame(adv_tab, bg=BG_CARD, bd=1, relief=tk.RIDGE, pady=6, padx=8)
        v2s_card.pack(fill=tk.X, pady=(5, 3), padx=5)
        tk.Label(v2s_card, text="🗣️ VOICE → SIGN TRANSLATOR",
                 font=("Segoe UI", 9, "bold"), bg=BG_CARD, fg=COLOR_ORANGE).pack(anchor=tk.W)
        tk.Label(v2s_card, text="Speak a word to see how to sign it",
                 font=("Segoe UI", 8), bg=BG_CARD, fg=FG_SECONDARY).pack(anchor=tk.W)
        v2s_row = tk.Frame(v2s_card, bg=BG_CARD)
        v2s_row.pack(fill=tk.X, pady=3)
        self.mic_btn = tk.Button(v2s_row, text="🎤 Listen", command=self._voice_to_sign,
                                  bg=COLOR_RED, fg="white", font=("Segoe UI", 9, "bold"),
                                  relief=tk.FLAT, cursor="hand2", width=10)
        self.mic_btn.pack(side=tk.LEFT, padx=3)
        self.voice_result = tk.Label(v2s_row, text="",
                                      font=("Segoe UI", 10, "bold"), bg=BG_CARD, fg=COLOR_GREEN)
        self.voice_result.pack(side=tk.LEFT, padx=8)

        # [2] Few-Shot Learner
        fs_card = tk.Frame(adv_tab, bg=BG_CARD, bd=1, relief=tk.RIDGE, pady=6, padx=8)
        fs_card.pack(fill=tk.X, pady=3, padx=5)
        tk.Label(fs_card, text="🧠 FEW-SHOT SIGN LEARNER",
                 font=("Segoe UI", 9, "bold"), bg=BG_CARD, fg=COLOR_CYAN).pack(anchor=tk.W)
        tk.Label(fs_card, text="Teach a NEW sign with just 5 images!",
                 font=("Segoe UI", 8), bg=BG_CARD, fg=FG_SECONDARY).pack(anchor=tk.W)
        fs_row = tk.Frame(fs_card, bg=BG_CARD)
        fs_row.pack(fill=tk.X, pady=3)
        self.fewshot_btn = tk.Button(fs_row, text="🧠 Quick Learn", command=self._start_fewshot,
                                      bg=COLOR_PURPLE, fg="white", font=("Segoe UI", 9, "bold"),
                                      relief=tk.FLAT, cursor="hand2", width=12)
        self.fewshot_btn.pack(side=tk.LEFT, padx=3)
        self.fewshot_status = tk.Label(fs_row, text="",
                                       font=("Segoe UI", 9), bg=BG_CARD, fg=FG_SECONDARY)
        self.fewshot_status.pack(side=tk.LEFT, padx=8)

        # [3] Emotion-Aware TTS Toggle
        emo_card = tk.Frame(adv_tab, bg=BG_CARD, bd=1, relief=tk.RIDGE, pady=6, padx=8)
        emo_card.pack(fill=tk.X, pady=3, padx=5)
        emo_row = tk.Frame(emo_card, bg=BG_CARD)
        emo_row.pack(fill=tk.X)
        self.emotion_var = tk.BooleanVar(value=True)
        tk.Checkbutton(emo_row, text="😊 Emotion-Aware TTS", variable=self.emotion_var,
                       command=lambda: setattr(self, 'emotion_enabled', self.emotion_var.get()),
                       bg=BG_CARD, fg=COLOR_YELLOW, selectcolor=BG_DARK,
                       font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)
        self.emo_display = tk.Label(emo_row, text="Emotion: —",
                                     font=("Segoe UI", 9), bg=BG_CARD, fg=FG_SECONDARY)
        self.emo_display.pack(side=tk.RIGHT, padx=5)

        # [5] Emergency Alert Toggle
        emg_card = tk.Frame(adv_tab, bg=BG_CARD, bd=1, relief=tk.RIDGE, pady=6, padx=8)
        emg_card.pack(fill=tk.X, pady=3, padx=5)
        self.emg_var = tk.BooleanVar(value=True)
        tk.Checkbutton(emg_card, text="🚨 Emergency Sign Alerts (Help, No)",
                       variable=self.emg_var, bg=BG_CARD, fg=COLOR_RED,
                       selectcolor=BG_DARK, font=("Segoe UI", 9, "bold")).pack(anchor=tk.W)

        # [6] Speed Coach
        spd_card = tk.Frame(adv_tab, bg=BG_CARD, bd=1, relief=tk.RIDGE, pady=6, padx=8)
        spd_card.pack(fill=tk.X, pady=3, padx=5)
        tk.Label(spd_card, text="🏋️ SIGN SPEED COACH",
                 font=("Segoe UI", 9, "bold"), bg=BG_CARD, fg=COLOR_GREEN).pack(anchor=tk.W)
        self.speed_display = tk.Label(spd_card, text="Pace: — │ Avg: —",
                                      font=("Segoe UI", 10), bg=BG_CARD, fg=FG_PRIMARY)
        self.speed_display.pack(anchor=tk.W, pady=2)

        # [7] Confusion Matrix Viewer
        cm_card = tk.Frame(adv_tab, bg=BG_CARD, bd=1, relief=tk.RIDGE, pady=6, padx=8)
        cm_card.pack(fill=tk.BOTH, expand=True, pady=3, padx=5)
        tk.Label(cm_card, text="📊 LIVE CONFUSION MATRIX",
                 font=("Segoe UI", 9, "bold"), bg=BG_CARD, fg=COLOR_ORANGE).pack(anchor=tk.W)
        tk.Label(cm_card, text="Shows which signs are most commonly confused",
                 font=("Segoe UI", 8), bg=BG_CARD, fg=FG_SECONDARY).pack(anchor=tk.W)
        self.confusion_text = tk.Text(cm_card, bg=BG_DARK, fg=FG_PRIMARY,
                                      font=("Consolas", 8), height=6,
                                      wrap=tk.WORD, relief=tk.FLAT, state=tk.DISABLED)
        self.confusion_text.pack(fill=tk.BOTH, expand=True, pady=3)

    # ═════════════════════════════════════════════════════════════
    #  TOOLS TAB (Enhancement Controls)
    # ═════════════════════════════════════════════════════════════
    def _build_tools_tab(self, parent):
        """Build the Tools tab with all enhancement controls."""
        # Scrollable
        canvas = tk.Canvas(parent, bg=BG_DARK, highlightthickness=0)
        scrollbar = tk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        tools_inner = tk.Frame(canvas, bg=BG_DARK)
        tools_inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=tools_inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # ── Theme Selector ──
        theme_card = tk.LabelFrame(tools_inner, text="🎨 THEME", font=("Segoe UI", 9, "bold"),
                                    bg=BG_CARD, fg=COLOR_PURPLE, padx=8, pady=6)
        theme_card.pack(fill=tk.X, pady=3, padx=5)
        theme_row = tk.Frame(theme_card, bg=BG_CARD)
        theme_row.pack(fill=tk.X)
        for name, label in [('dark', '🌙 Dark'), ('light', '☀️ Light'), ('high_contrast', '🔲 High Contrast')]:
            tk.Button(theme_row, text=label, command=lambda n=name: self._switch_theme(n),
                      bg=BG_ACCENT, fg=FG_PRIMARY, font=("Segoe UI", 8, "bold"),
                      relief=tk.FLAT, cursor="hand2", width=14).pack(side=tk.LEFT, padx=2)

        # ── Privacy ──
        priv_card = tk.LabelFrame(tools_inner, text="🔒 PRIVACY", font=("Segoe UI", 9, "bold"),
                                   bg=BG_CARD, fg=COLOR_GREEN, padx=8, pady=6)
        priv_card.pack(fill=tk.X, pady=3, padx=5)
        self.blur_face_var = tk.BooleanVar(value=False)
        tk.Checkbutton(priv_card, text="😶 Auto-blur face in camera feed",
                       variable=self.blur_face_var,
                       command=lambda: setattr(self.face_blur, 'enabled', self.blur_face_var.get()),
                       bg=BG_CARD, fg=FG_PRIMARY, selectcolor=BG_DARK,
                       font=("Segoe UI", 9)).pack(anchor=tk.W)

        # ── Heatmap ──
        heat_card = tk.LabelFrame(tools_inner, text="🔥 GRAD-CAM HEATMAP", font=("Segoe UI", 9, "bold"),
                                   bg=BG_CARD, fg=COLOR_ORANGE, padx=8, pady=6)
        heat_card.pack(fill=tk.X, pady=3, padx=5)
        self.heatmap_var = tk.BooleanVar(value=False)
        tk.Checkbutton(heat_card, text="Show attention heatmap overlay on camera",
                       variable=self.heatmap_var,
                       command=lambda: setattr(self.gradcam, 'enabled', self.heatmap_var.get()),
                       bg=BG_CARD, fg=FG_PRIMARY, selectcolor=BG_DARK,
                       font=("Segoe UI", 9)).pack(anchor=tk.W)

        # ── Charts ──
        chart_card = tk.LabelFrame(tools_inner, text="📈 LIVE CHARTS", font=("Segoe UI", 9, "bold"),
                                    bg=BG_CARD, fg=COLOR_CYAN, padx=8, pady=6)
        chart_card.pack(fill=tk.X, pady=3, padx=5)
        self.chart_conf_label = tk.Label(chart_card, bg=BG_CARD)
        self.chart_conf_label.pack(pady=2)
        self.chart_dist_label = tk.Label(chart_card, bg=BG_CARD)
        self.chart_dist_label.pack(pady=2)

        # ── Progress Stats ──
        prog_card = tk.LabelFrame(tools_inner, text="📊 LEARNING PROGRESS", font=("Segoe UI", 9, "bold"),
                                   bg=BG_CARD, fg=COLOR_YELLOW, padx=8, pady=6)
        prog_card.pack(fill=tk.X, pady=3, padx=5)
        self.progress_stats_label = tk.Label(prog_card, text="Loading...",
                                              font=("Consolas", 9), bg=BG_CARD, fg=FG_PRIMARY,
                                              justify=tk.LEFT, wraplength=300)
        self.progress_stats_label.pack(anchor=tk.W, pady=3)
        tk.Button(prog_card, text="🏆 Show Achievements", command=self._show_achievements,
                  bg=COLOR_YELLOW, fg="#000", font=("Segoe UI", 9, "bold"),
                  relief=tk.FLAT, cursor="hand2").pack(anchor=tk.W, pady=2)

        # ── Report & Export ──
        export_card = tk.LabelFrame(tools_inner, text="📄 REPORTS & EXPORT", font=("Segoe UI", 9, "bold"),
                                     bg=BG_CARD, fg=COLOR_PURPLE, padx=8, pady=6)
        export_card.pack(fill=tk.X, pady=3, padx=5)
        exp_row = tk.Frame(export_card, bg=BG_CARD)
        exp_row.pack(fill=tk.X)
        tk.Button(exp_row, text="📄 Generate PDF Report", command=self._generate_report,
                  bg=COLOR_PURPLE, fg="white", font=("Segoe UI", 9, "bold"),
                  relief=tk.FLAT, cursor="hand2", width=20).pack(side=tk.LEFT, padx=3, pady=2)
        tk.Button(exp_row, text="💾 Export Conversation", command=self._export_conversation,
                  bg=COLOR_GREEN, fg="#000", font=("Segoe UI", 9, "bold"),
                  relief=tk.FLAT, cursor="hand2", width=18).pack(side=tk.LEFT, padx=3, pady=2)

        # ── External Tools ──
        tools_card = tk.LabelFrame(tools_inner, text="🚀 LAUNCH TOOLS", font=("Segoe UI", 9, "bold"),
                                    bg=BG_CARD, fg=COLOR_RED, padx=8, pady=6)
        tools_card.pack(fill=tk.X, pady=3, padx=5)
        tool_row1 = tk.Frame(tools_card, bg=BG_CARD)
        tool_row1.pack(fill=tk.X, pady=2)
        tk.Button(tool_row1, text="🌐 Start API Server", command=self._launch_api,
                  bg="#64b5f6", fg="#000", font=("Segoe UI", 9, "bold"),
                  relief=tk.FLAT, cursor="hand2", width=18).pack(side=tk.LEFT, padx=3)
        tk.Button(tool_row1, text="📹 Video Converter", command=self._launch_video_tool,
                  bg="#bb86fc", fg="white", font=("Segoe UI", 9, "bold"),
                  relief=tk.FLAT, cursor="hand2", width=16).pack(side=tk.LEFT, padx=3)
        tool_row2 = tk.Frame(tools_card, bg=BG_CARD)
        tool_row2.pack(fill=tk.X, pady=2)
        tk.Button(tool_row2, text="📷 Data Collector", command=self._launch_collector,
                  bg=COLOR_ORANGE, fg="#000", font=("Segoe UI", 9, "bold"),
                  relief=tk.FLAT, cursor="hand2", width=18).pack(side=tk.LEFT, padx=3)
        tk.Button(tool_row2, text="📦 Build .exe", command=self._launch_build,
                  bg=COLOR_RED, fg="white", font=("Segoe UI", 9, "bold"),
                  relief=tk.FLAT, cursor="hand2", width=16).pack(side=tk.LEFT, padx=3)

        # Update progress stats on load
        self._update_progress_stats()

    # ═════════════════════════════════════════════════════════════
    #  ENHANCEMENT METHODS
    # ═════════════════════════════════════════════════════════════
    def _toggle_recording(self):
        if not ENHANCEMENTS_AVAILABLE:
            return
        if self.recorder.recording:
            path, dur, frames = self.recorder.stop()
            self.rec_btn.config(text="⏺ Rec", bg=COLOR_RED)
            self.status_label.config(text=f"Recording saved: {path} ({dur:.0f}s, {frames} frames)")
            messagebox.showinfo("Recording Saved", f"Video saved as:\n{path}\nDuration: {dur:.1f}s")
        else:
            path = self.recorder.start(640, 480, 15.0)
            self.rec_btn.config(text="⏹ Stop", bg="#ff1744")
            self.status_label.config(text=f"🔴 Recording to {path}...")

    def _toggle_fullscreen(self):
        if ENHANCEMENTS_AVAILABLE:
            self.is_fullscreen = not self.is_fullscreen
            self.root.attributes("-fullscreen", self.is_fullscreen)

    def _switch_theme(self, theme_name):
        if not ENHANCEMENTS_AVAILABLE:
            return
        self.current_theme = theme_name
        t = THEMES.get(theme_name, THEMES['dark'])
        self.root.configure(bg=t['bg'])
        self.status_label.config(text=f"Theme: {theme_name.replace('_', ' ').title()}")

    def _show_achievements(self):
        if not ENHANCEMENTS_AVAILABLE or not self.progress:
            return
        stats = self.progress.get_stats_summary()
        achievements = stats.get('achievements', [])
        mastered = self.progress.get_mastered_signs()
        msg = f"🏆 ACHIEVEMENTS\n{'='*30}\n"
        if achievements:
            for a in achievements:
                msg += f"  🏅 {a}\n"
        else:
            msg += "  No achievements yet!\n"
        msg += f"\n📊 STATS\n{'='*30}\n"
        msg += f"  Sessions: {stats['sessions']}\n"
        msg += f"  Total detected: {stats['total_detected']}\n"
        msg += f"  Practice time: {stats['practice_minutes']} min\n"
        msg += f"  Streak: {stats['streak']} days 🔥\n"
        msg += f"  Signs mastered: {stats['signs_mastered']}\n"
        if mastered:
            msg += f"\n✅ MASTERED: {', '.join(mastered)}\n"
        struggling = self.progress.get_struggling_signs()
        if struggling:
            msg += f"\n⚠️ PRACTICE MORE: {', '.join(struggling)}\n"
        messagebox.showinfo("Learning Progress", msg)

    def _update_progress_stats(self):
        if not ENHANCEMENTS_AVAILABLE or not self.progress:
            return
        stats = self.progress.get_stats_summary()
        text = (f"Sessions: {stats['sessions']}  │  Detected: {stats['total_detected']}\n"
                f"Practice: {stats['practice_minutes']} min  │  Streak: {stats['streak']} 🔥\n"
                f"Signs learned: {stats['signs_learned']}  │  Mastered: {stats['signs_mastered']}\n"
                f"Achievements: {', '.join(stats['achievements']) if stats['achievements'] else 'None yet'}")
        if hasattr(self, 'progress_stats_label'):
            self.progress_stats_label.config(text=text)

    def _update_charts(self):
        if not ENHANCEMENTS_AVAILABLE or not self.charts:
            return
        try:
            conf_img = self.charts.render_confidence_chart(330, 130)
            dist_img = self.charts.render_sign_distribution(330, 130)
            conf_rgb = cv2.cvtColor(conf_img, cv2.COLOR_BGR2RGB)
            dist_rgb = cv2.cvtColor(dist_img, cv2.COLOR_BGR2RGB)
            conf_tk = ImageTk.PhotoImage(image=Image.fromarray(conf_rgb))
            dist_tk = ImageTk.PhotoImage(image=Image.fromarray(dist_rgb))
            self.chart_conf_label.config(image=conf_tk)
            self.chart_conf_label.image = conf_tk
            self.chart_dist_label.config(image=dist_tk)
            self.chart_dist_label.image = dist_tk
        except Exception:
            pass

    def _generate_report(self):
        try:
            from pdf_report import generate_report
            path = generate_report()
            os.startfile(path)
            self.status_label.config(text=f"Report generated: {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Report generation failed:\n{e}")

    def _export_conversation(self):
        if ENHANCEMENTS_AVAILABLE and self.conversation:
            for entry in self.sentence:
                self.conversation.add_sign_message(entry)
            path = self.conversation.export_conversation()
            messagebox.showinfo("Exported", f"Conversation saved as:\n{path}")

    def _launch_api(self):
        import subprocess, sys
        subprocess.Popen([sys.executable, "api_server.py"])
        self.status_label.config(text="🌐 API Server started at http://localhost:8000")

    def _launch_video_tool(self):
        import subprocess, sys
        subprocess.Popen([sys.executable, "video_dataset.py"])

    def _launch_collector(self):
        import subprocess, sys
        subprocess.Popen([sys.executable, "data_collector.py"])

    def _launch_build(self):
        import subprocess, sys
        subprocess.Popen([sys.executable, "build_exe.py"])
        messagebox.showinfo("Build", "Build process started in new window.\nThis may take 3-10 minutes.")

    # ═════════════════════════════════════════════════════════════
    #  EVENT HANDLERS
    # ═════════════════════════════════════════════════════════════
    def _on_mode_change(self):
        mode = self.mode_var.get()
        self.current_mode = mode
        tab_map = {"recognition": 0, "quiz": 1, "text2sign": 2, "advanced": 3}
        self.notebook.select(tab_map.get(mode, 0))

    def _on_lang_change(self, event=None):
        self.current_lang_idx = LANGUAGES.index(self.lang_combo.get())
        self.status_label.config(text=f"Status: Running │ Language: {self.lang_combo.get()}")

    def _on_conf_change(self, val):
        self.conf_threshold = float(val)

    def _on_skel_toggle(self):
        self.show_skeleton = self.skel_var.get()

    # ═════════════════════════════════════════════════════════════
    #  CAMERA
    # ═════════════════════════════════════════════════════════════
    def start_camera(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open webcam!")
            return
        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_label.config(text=f"Status: Running │ Language: {self.lang_combo.get()}")
        self._update_frame()

    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.video_label.config(image='')
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Stopped")

    def on_closing(self):
        """Standard window closing handler for instant exit."""
        self.running = False
        if self.cap:
            self.cap.release()
        # Save progress
        if ENHANCEMENTS_AVAILABLE and self.progress:
            self.progress.end_session()
        # Stop recording
        if ENHANCEMENTS_AVAILABLE and self.recorder and self.recorder.recording:
            self.recorder.stop()
        self.root.destroy()
        # Force exit to stop any hung threads
        os._exit(0)

    # ═════════════════════════════════════════════════════════════
    #  TEMPORAL SMOOTHING
    # ═════════════════════════════════════════════════════════════
    def _get_smoothed_sign(self, current_sign, current_conf):
        self.prediction_buffer.append((current_sign, current_conf))
        sign_counts = {}
        sign_confs = {}
        for sign, conf in self.prediction_buffer:
            if sign is not None:
                sign_counts[sign] = sign_counts.get(sign, 0) + 1
                sign_confs.setdefault(sign, []).append(conf)
        if not sign_counts:
            return None, 0
        best_sign = max(sign_counts, key=sign_counts.get)
        count = sign_counts[best_sign]
        avg_conf = sum(sign_confs[best_sign]) / len(sign_confs[best_sign])
        if count >= self.STABLE_THRESHOLD:
            return best_sign, avg_conf
        return None, 0

    def _get_stability_indicator(self):
        if not self.prediction_buffer:
            return "○" * 7
        signs = [s for s, c in self.prediction_buffer if s is not None]
        if not signs:
            return "○" * 7
        most_common = max(set(signs), key=signs.count)
        return "".join("●" if s == most_common else ("◐" if s else "○")
                       for s, c in self.prediction_buffer)

    # ═════════════════════════════════════════════════════════════
    #  SENTENCE BUILDER
    # ═════════════════════════════════════════════════════════════
    def _add_to_sentence(self, sign):
        now = time.time()
        if sign == self.last_added_sign and (now - self.last_added_time) < self.SIGN_COOLDOWN:
            return
        self.sentence.append(sign.upper())
        self.last_added_sign = sign
        self.last_added_time = now
        self.sentence_label.config(text=" ".join(self.sentence))

    def _clear_sentence(self):
        self.sentence.clear()
        self.last_added_sign = None
        self.sentence_label.config(text="(Show signs to build a sentence)")

    def _speak_sentence(self):
        if not self.sentence:
            return
        threading.Thread(target=speak_sign,
                        args=(" ".join(self.sentence), self.current_lang_idx),
                        daemon=True).start()

    def _export_session(self):
        """Export session history to a timestamped file."""
        if not self.sentence and not self.history:
            messagebox.showinfo("Export", "No session data to export yet.")
            return
            
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"ISL_Session_{timestamp}.txt"
        
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write("🏹 INDIAN SIGN LANGUAGE RECOGNITION SESSION\n")
                f.write(f"📅 Date/Time: {time.ctime()}\n")
                f.write("-" * 50 + "\n")
                f.write(f"📝 FINAL SENTENCE: {' '.join(self.sentence)}\n")
                f.write("-" * 50 + "\n")
                f.write("📋 DETAILED LOG:\n")
                for entry in reversed(self.history):
                    f.write(f"{entry}\n")
            
            messagebox.showinfo("Export Success", f"Session saved as:\n{filename}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to save file: {e}")

    # ═════════════════════════════════════════════════════════════
    #  SIGN HISTORY
    # ═════════════════════════════════════════════════════════════
    def _add_to_history(self, sign, conf):
        entry = f"[{time.strftime('%H:%M:%S')}]  {sign.upper():12s}  ({conf:.0%})"
        self.history.appendleft(entry)
        self.history_text.config(state=tk.NORMAL)
        self.history_text.delete("1.0", tk.END)
        self.history_text.insert(tk.END, "\n".join(self.history))
        self.history_text.config(state=tk.DISABLED)

    # ═════════════════════════════════════════════════════════════
    #  TTS DEBOUNCE
    # ═════════════════════════════════════════════════════════════
    def _speak_debounced(self, sign):
        now = time.time()
        if sign == self.last_spoken_sign and (now - self.last_spoken_time) < self.TTS_COOLDOWN:
            return
        self.last_spoken_sign = sign
        self.last_spoken_time = now
        threading.Thread(target=speak_sign,
                        args=(sign, self.current_lang_idx), daemon=True).start()

    # ═════════════════════════════════════════════════════════════
    #  QUIZ MODE
    # ═════════════════════════════════════════════════════════════
    def _start_quiz(self):
        if not self.running:
            messagebox.showwarning("Warning", "Start the camera first!")
            return
        self.quiz_active = True
        self.quiz_score = 0
        self.quiz_total = 0
        self.quiz_streak = 0
        self.quiz_best_streak = 0
        self.current_mode = "quiz"
        self.mode_var.set("quiz")
        self.notebook.select(1)
        self.quiz_start_btn.config(state=tk.DISABLED)
        self.quiz_skip_btn.config(state=tk.NORMAL)
        self.quiz_stop_btn.config(state=tk.NORMAL)
        self._next_quiz_sign()

    def _stop_quiz(self):
        self.quiz_active = False
        self.quiz_start_btn.config(state=tk.NORMAL)
        self.quiz_skip_btn.config(state=tk.DISABLED)
        self.quiz_stop_btn.config(state=tk.DISABLED)
        self.quiz_target_label.config(text="Quiz ended!")
        self.quiz_ref_label.config(image='')
        acc = (self.quiz_score / self.quiz_total * 100) if self.quiz_total > 0 else 0
        self.quiz_feedback.config(
            text=f"Final: {self.quiz_score}/{self.quiz_total} ({acc:.0f}%)",
            fg=COLOR_GREEN if acc >= 60 else COLOR_RED)

    def _next_quiz_sign(self):
        signs = list(self.model.names.values())
        self.quiz_target = random.choice(signs)
        self.quiz_start_time = time.time()
        self.quiz_target_label.config(text=f"Show: {self.quiz_target.upper()}")
        self.quiz_feedback.config(text="Waiting for your sign...", fg=COLOR_YELLOW)

        # Show reference image
        if self.quiz_target in self.sign_images:
            self.quiz_ref_label.config(image=self.sign_images[self.quiz_target])
            self.quiz_ref_label.image = self.sign_images[self.quiz_target]
        else:
            self.quiz_ref_label.config(image='')

        # Reset smoothing buffer for fresh detection
        self.prediction_buffer.clear()

    def _check_quiz_answer(self, detected_sign):
        if not self.quiz_active or not self.quiz_target:
            return
        if detected_sign.lower() == self.quiz_target.lower():
            elapsed = time.time() - self.quiz_start_time
            self.quiz_score += 1
            self.quiz_total += 1
            self.quiz_streak += 1
            self.quiz_best_streak = max(self.quiz_best_streak, self.quiz_streak)
            self.quiz_feedback.config(text=f"✅ Correct! ({elapsed:.1f}s)", fg=COLOR_GREEN)
            self._update_quiz_score()
            # Auto-advance after short delay
            self.root.after(1500, self._next_quiz_sign)

    def _update_quiz_score(self):
        self.score_label.config(text=f"{self.quiz_score} / {self.quiz_total}")
        acc = (self.quiz_score / self.quiz_total * 100) if self.quiz_total > 0 else 0
        self.accuracy_label.config(text=f"Accuracy: {acc:.0f}%")
        self.streak_label.config(text=f"🔥 Streak: {self.quiz_streak}  |  Best: {self.quiz_best_streak}")

    # ═════════════════════════════════════════════════════════════
    #  TEXT → SIGN (Two-Way Communication)
    # ═════════════════════════════════════════════════════════════
    def _on_t2s_search(self, event=None):
        word = self.t2s_entry.get().strip().lower()
        if word:
            self._show_sign_reference(word)

    def _show_sign_reference(self, sign_name):
        sign_name = sign_name.lower().strip()

        # Search in ALL loaded references (not just model classes)
        matched = None
        for name in self.sign_images.keys():
            if name.lower() == sign_name or sign_name in name.lower():
                matched = name
                break

        if not matched:
            self.t2s_sign_name.config(text=f"'{sign_name}' not found", fg=COLOR_RED)
            self.t2s_image_label.config(image='')
            available = list(self.sign_images.keys())[:15]
            self.t2s_translation.config(text=f"Try: {', '.join(available)}...")
            return

        # Show sign name
        self.t2s_sign_name.config(text=f"✋ {matched.upper()}", fg=COLOR_GREEN)
        self.t2s_entry.delete(0, tk.END)
        self.t2s_entry.insert(0, matched)

        # Show reference image
        if matched in self.sign_images:
            self.t2s_image_label.config(image=self.sign_images[matched])
            self.t2s_image_label.image = self.sign_images[matched]
        else:
            self.t2s_image_label.config(image='')

        # Show translations
        trans = TRANSLATIONS.get(matched.lower(), {})
        if trans:
            lang_code = LANG_CODES[self.current_lang_idx]
            lang_name = LANGUAGES[self.current_lang_idx]
            translated = trans.get(lang_code, matched)
            self.t2s_translation.config(
                text=f"English: {trans.get('en', matched)}\n"
                     f"{lang_name}: {translated}")
        else:
            self.t2s_translation.config(text=f"English: {matched.title()}")

        # Switch to text2sign tab
        self.notebook.select(2)

    def _speak_t2s(self):
        word = self.t2s_entry.get().strip()
        if word:
            threading.Thread(target=speak_sign,
                            args=(word, self.current_lang_idx), daemon=True).start()

    # ═════════════════════════════════════════════════════════════
    #  [1] VOICE-TO-SIGN REVERSE TRANSLATOR
    # ═════════════════════════════════════════════════════════════
    def _voice_to_sign(self):
        """Listen for a spoken word and show the corresponding sign."""
        if self.voice_listening:
            return
        self.voice_listening = True
        self.mic_btn.config(text="🔴 Listening...", bg="#ff5252")
        self.voice_result.config(text="Speak now...", fg=COLOR_YELLOW)
        
        def listen_thread():
            try:
                import speech_recognition as sr
                recognizer = sr.Recognizer()
                with sr.Microphone() as source:
                    recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=4)
                text = recognizer.recognize_google(audio).lower().strip()
                self.root.after(0, lambda: self._process_voice_result(text))
            except Exception as e:
                self.root.after(0, lambda: self.voice_result.config(
                    text=f"Could not hear. Try again.", fg=COLOR_RED))
            finally:
                self.voice_listening = False
                self.root.after(0, lambda: self.mic_btn.config(
                    text="🎤 Listen", bg=COLOR_RED))
        
        threading.Thread(target=listen_thread, daemon=True).start()
    
    def _process_voice_result(self, text):
        """Process the recognized speech and show the sign."""
        self.voice_result.config(text=f'Heard: "{text}"', fg=COLOR_GREEN)
        # Search for matching sign
        self._show_sign_reference(text)
        self.notebook.select(2)  # Switch to Text→Sign tab to show result

    # ═════════════════════════════════════════════════════════════
    #  [2] FEW-SHOT SIGN LEARNER
    # ═════════════════════════════════════════════════════════════
    def _start_fewshot(self):
        """Start capturing 5 images for a new sign using few-shot learning."""
        if not self.running:
            messagebox.showwarning("Warning", "Start the camera first!")
            return
        
        sign_name = simpledialog.askstring("New Sign",
            "Enter the name of the new sign to learn:\n(e.g., 'water', 'good morning')",
            parent=self.root)
        
        if not sign_name or not sign_name.strip():
            return
        
        self.fewshot_sign_name = sign_name.strip().lower()
        self.fewshot_images = []
        self.fewshot_btn.config(state=tk.DISABLED)
        self.fewshot_status.config(text=f"Capturing 0/5 for '{self.fewshot_sign_name}'...",
                                    fg=COLOR_YELLOW)
        
        # Start capturing in background
        self._fewshot_capture_next()
    
    def _fewshot_capture_next(self):
        """Capture next few-shot image."""
        if len(self.fewshot_images) >= 5:
            self._fewshot_train()
            return
        
        if not self.running or self.cap is None:
            self.fewshot_btn.config(state=tk.NORMAL)
            return
        
        # Capture current frame
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (640, 480))
            self.fewshot_images.append(frame.copy())
            count = len(self.fewshot_images)
            self.fewshot_status.config(
                text=f"Captured {count}/5 — move hand slightly...",
                fg=COLOR_CYAN)
        
        # Schedule next capture after 1 second
        if len(self.fewshot_images) < 5:
            self.root.after(1000, self._fewshot_capture_next)
    
    def _fewshot_train(self):
        """Save captured images and prepare for retraining."""
        import yaml, uuid
        
        # Save images to training directory
        train_imgs = "Data/train/images"
        train_lbls = "Data/train/labels"
        os.makedirs(train_imgs, exist_ok=True)
        os.makedirs(train_lbls, exist_ok=True)
        
        # Load/update data.yaml
        data_yaml = "Data/data.yaml"
        with open(data_yaml, 'r') as f:
            data = yaml.safe_load(f)
        
        class_names = data.get('names', [])
        if self.fewshot_sign_name not in class_names:
            class_names.append(self.fewshot_sign_name)
            data['names'] = class_names
            data['nc'] = len(class_names)
            with open(data_yaml, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        
        class_id = class_names.index(self.fewshot_sign_name)
        
        # Save images with full-frame bounding box
        for i, frame in enumerate(self.fewshot_images):
            uid = uuid.uuid4().hex[:8]
            name = f"fewshot_{self.fewshot_sign_name}_{uid}"
            cv2.imwrite(os.path.join(train_imgs, f"{name}.jpg"), frame)
            with open(os.path.join(train_lbls, f"{name}.txt"), 'w') as f:
                f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
        
        self.fewshot_images = []
        self.fewshot_btn.config(state=tk.NORMAL)
        self.fewshot_status.config(
            text=f"✅ '{self.fewshot_sign_name}' saved! Click TRAIN in data_collector.",
            fg=COLOR_GREEN)
        messagebox.showinfo("Few-Shot Complete",
            f"✅ 5 images saved for '{self.fewshot_sign_name}'!\n\n"
            f"Class ID: {class_id}\n"
            f"To train: run python data_collector.py → click 🔥 TRAIN MODEL NOW")

    # ═════════════════════════════════════════════════════════════
    #  [3] EMOTION-AWARE TTS
    # ═════════════════════════════════════════════════════════════
    def _detect_emotion(self, frame):
        """Detect facial emotion using MediaPipe FaceMesh."""
        if not self.emotion_enabled or self.face_mesh is None:
            return "neutral"
        
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            
            if results.multi_face_landmarks:
                face = results.multi_face_landmarks[0]
                h, w = frame.shape[:2]
                
                # Mouth openness (landmarks 13=top lip, 14=bottom lip)
                top_lip = face.landmark[13].y * h
                bottom_lip = face.landmark[14].y * h
                mouth_open = abs(bottom_lip - top_lip)
                
                # Eyebrow raise (landmarks 70=left brow, 107=forehead)
                left_brow = face.landmark[70].y * h
                forehead = face.landmark[10].y * h
                brow_raise = abs(left_brow - forehead)
                
                # Mouth corners (48=left, 54=right)
                left_corner = face.landmark[61].y * h
                right_corner = face.landmark[291].y * h
                mouth_center = (top_lip + bottom_lip) / 2
                
                # Classify emotion
                if mouth_open > 15:
                    return "surprised"  # 😮
                elif left_corner < mouth_center - 3 and right_corner < mouth_center - 3:
                    return "happy"  # 😊
                elif left_corner > mouth_center + 2:
                    return "sad"  # 😢
                else:
                    return "neutral"  # 😐
        except:
            pass
        return "neutral"
    
    def _speak_with_emotion(self, sign):
        """Speak with emotion-adjusted parameters."""
        # For now, emotion affects the spoken text prefix
        if self.emotion_enabled and self.current_emotion != "neutral":
            emotion_prefix = {
                "happy": "(happily) ",
                "sad": "(sadly) ",
                "surprised": "(with surprise) "
            }
            prefix = emotion_prefix.get(self.current_emotion, "")
            threading.Thread(target=speak_sign,
                            args=(prefix + sign, self.current_lang_idx), daemon=True).start()
        else:
            threading.Thread(target=speak_sign,
                            args=(sign, self.current_lang_idx), daemon=True).start()

    # ═════════════════════════════════════════════════════════════
    #  [5] EMERGENCY SIGN PRIORITY ALERT
    # ═════════════════════════════════════════════════════════════
    def _check_emergency(self, sign, frame):
        """Check if detected sign is an emergency sign."""
        if not self.emg_var.get():
            return
        if sign.lower() in self.emergency_signs:
            now = time.time()
            if now - self.last_emergency_time > 3.0:  # 3s cooldown
                self.last_emergency_time = now
                self.emergency_flash = 8  # Flash frames
                # Urgent TTS
                threading.Thread(target=speak_sign,
                                args=(f"ALERT! {sign}!", self.current_lang_idx),
                                daemon=True).start()

    # ═════════════════════════════════════════════════════════════
    #  [6] SIGN SPEED COACH
    # ═════════════════════════════════════════════════════════════
    def _update_speed_coach(self, sign):
        """Track signing speed and provide coaching feedback."""
        now = time.time()
        self.sign_timestamps.append(now)
        
        if len(self.sign_timestamps) >= 2:
            intervals = [self.sign_timestamps[i] - self.sign_timestamps[i-1]
                        for i in range(1, len(self.sign_timestamps))]
            avg_interval = sum(intervals) / len(intervals)
            latest = intervals[-1]
            
            if latest < 1.5:
                self.current_speed_text = "⚡ Too Fast!"
                self.current_speed_color = (0, 100, 255)  # Orange
                pace = "FAST"
            elif latest > 5.0:
                self.current_speed_text = "🐌 Too Slow"
                self.current_speed_color = (0, 200, 255)  # Yellow
                pace = "SLOW"
            else:
                self.current_speed_text = "✅ Perfect Pace"
                self.current_speed_color = (0, 230, 118)  # Green
                pace = "GOOD"
            
            self.speed_display.config(
                text=f"Pace: {pace} │ Avg: {avg_interval:.1f}s/sign")
            self.speed_label.config(text=f"⏱ {avg_interval:.1f}s/sign")

    # ═════════════════════════════════════════════════════════════
    #  [7] CONFUSION MATRIX TRACKER
    # ═════════════════════════════════════════════════════════════
    def _update_confusion_matrix(self, raw_detections):
        """Track which signs appear together (potential confusions)."""
        if len(raw_detections) >= 2:
            # If multiple signs detected in same frame, they might be confused
            sorted_dets = sorted(raw_detections, key=lambda x: x[1], reverse=True)
            primary = sorted_dets[0][0]
            for sign, conf in sorted_dets[1:]:
                if conf > 0.2:  # Only track meaningful alternatives
                    key = f"{primary} vs {sign}"
                    self.confusion_data[key] = self.confusion_data.get(key, 0) + 1
        
        # Update display periodically
        if self.confusion_data:
            sorted_conf = sorted(self.confusion_data.items(), key=lambda x: x[1], reverse=True)
            lines = [f"{pair}: {count}x" for pair, count in sorted_conf[:8]]
            self.confusion_text.config(state=tk.NORMAL)
            self.confusion_text.delete("1.0", tk.END)
            self.confusion_text.insert(tk.END, "\n".join(lines) if lines else "No confusions yet")
            self.confusion_text.config(state=tk.DISABLED)
    # ═════════════════════════════════════════════════════════════
    #  MAIN FRAME LOOP (Enhanced with 7 Unique Features)
    # ═════════════════════════════════════════════════════════════
    def _update_frame(self):
        if not self.running:
            return

        frame_start = time.time()
        ret, frame = self.cap.read()

        if ret:
            frame = cv2.resize(frame, (640, 480))
            results = self.model(frame, conf=self.conf_threshold * 0.7, verbose=False)[0]

            best_sign, best_conf = None, 0
            raw_detections = []  # For confusion matrix

            # ── [4] Multi-Person Color-Coded Detection ────────────
            person_idx = 0
            for box in results.boxes:
                conf = box.conf[0].item()
                cls_id = int(box.cls[0].item())
                label = self.model.names[cls_id]
                raw_detections.append((label, conf))

                if conf > best_conf:
                    best_conf = conf
                    best_sign = label

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Each detection gets a unique person color
                color = self.person_colors[person_idx % len(self.person_colors)]
                person_idx += 1
                
                # If it's an emergency sign, use RED
                is_emergency = label.lower() in self.emergency_signs
                if is_emergency:
                    color = (0, 0, 255)  # Bright Red

                thick = 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
                # Corner accents
                length = 25
                cv2.line(frame, (x1, y1), (x1+length, y1), color, thick+2)
                cv2.line(frame, (x1, y1), (x1, y1+length), color, thick+2)
                cv2.line(frame, (x2, y2), (x2-length, y2), color, thick+2)
                cv2.line(frame, (x2, y2), (x2, y2-length), color, thick+2)

                # Person-labeled tag
                tag = f"P{person_idx}: {label} {conf:.0%}"
                if is_emergency:
                    tag = f"🚨 {label.upper()} {conf:.0%}"
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
                cv2.putText(frame, tag, (x1 + 3, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # ── [7] Confusion Matrix Tracking ─────────────────────
            self._update_confusion_matrix(raw_detections)

            # ── [3] Emotion Detection ─────────────────────────────
            self.current_emotion = self._detect_emotion(frame)
            emoji_map = {"happy": "😊", "sad": "😢", "surprised": "😮", "neutral": "😐"}
            self.emo_display.config(
                text=f"Emotion: {emoji_map.get(self.current_emotion, '😐')} {self.current_emotion}")
            self.emotion_label.config(
                text=f"{emoji_map.get(self.current_emotion, '')} {self.current_emotion.title()}")

            # --- MediaPipe Hand Intelligence ---
            if self.show_skeleton:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_results = self.hands.process(rgb_frame)
                if mp_results.multi_hand_landmarks:
                    for lmarks in mp_results.multi_hand_landmarks:
                        self.mp_draw.draw_landmarks(
                            frame, lmarks, None,
                            self.mp_draw.DrawingSpec(color=(187, 134, 252), thickness=2, circle_radius=2)
                        )

            # Temporal smoothing
            raw_sign = best_sign if best_conf >= self.conf_threshold else None
            smoothed_sign, avg_conf = self._get_smoothed_sign(raw_sign, best_conf)
            self.stability_label.config(text=f"Stability: {self._get_stability_indicator()}")

            if smoothed_sign:
                self.sign_label.config(text=smoothed_sign.upper(), fg=COLOR_GREEN)
                self.conf_label.config(text=f"Confidence: {avg_conf:.0%}")

                # ── Track in enhancements ──
                if ENHANCEMENTS_AVAILABLE:
                    if self.progress:
                        self.progress.log_detection(smoothed_sign, avg_conf)
                    if self.charts:
                        self.charts.add_detection(smoothed_sign, avg_conf)

                # Mode-specific behavior
                if self.current_mode == "recognition":
                    self._add_to_sentence(smoothed_sign)
                    self._add_to_history(smoothed_sign, avg_conf)
                    
                    # ── [5] Emergency Alert ────────────────────────
                    self._check_emergency(smoothed_sign, frame)
                    
                    # ── [6] Speed Coach ────────────────────────────
                    self._update_speed_coach(smoothed_sign)
                    
                    # ── [3] Emotion-Aware TTS (replaces normal TTS) ─
                    now = time.time()
                    if smoothed_sign != self.last_spoken_sign or (now - self.last_spoken_time) >= self.TTS_COOLDOWN:
                        self.last_spoken_sign = smoothed_sign
                        self.last_spoken_time = now
                        if self.emotion_enabled:
                            self._speak_with_emotion(smoothed_sign)
                        else:
                            self._speak_debounced(smoothed_sign)
                    
                elif self.current_mode == "quiz":
                    self._check_quiz_answer(smoothed_sign)

                cv2.putText(frame, f"STABLE: {smoothed_sign.upper()}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 118), 2)
            else:
                self.sign_label.config(text="—", fg=COLOR_RED)
                self.conf_label.config(text="Confidence: --")

            # ── [6] Speed Coach overlay ───────────────────────────
            if self.current_speed_text:
                cv2.putText(frame, self.current_speed_text, (440, 470),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.current_speed_color, 2)

            # ── [5] Emergency Flash Effect ────────────────────────
            if self.emergency_flash > 0:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (640, 480), (0, 0, 255), -1)
                alpha = min(0.4, self.emergency_flash / 12.0)
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                cv2.putText(frame, "🚨 EMERGENCY SIGN!", (120, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
                self.emergency_flash -= 1

            # Quiz mode overlay
            if self.quiz_active and self.quiz_target:
                cv2.putText(frame, f"QUIZ: Show '{self.quiz_target.upper()}'", (10, 470),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 214, 255), 2)

            # Multi-person count overlay
            if person_idx > 1:
                cv2.putText(frame, f"👥 {person_idx} persons detected", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 149, 237), 2)

            # FPS
            ft = time.time() - frame_start
            self.frame_times.append(ft)
            if len(self.frame_times) > 5:
                avg_t = sum(self.frame_times) / len(self.frame_times)
                self.fps_label.config(text=f"FPS: {1.0/avg_t:.1f}" if avg_t > 0 else "FPS: --")

            # ── Enhancement processing ────────────────────────────
            if ENHANCEMENTS_AVAILABLE:
                # Face blur
                if self.face_blur and self.face_blur.enabled:
                    frame = self.face_blur.process(frame)

                # Grad-CAM heatmap
                if self.gradcam and self.gradcam.enabled and results.boxes and len(results.boxes) > 0:
                    boxes_xy = [box.xyxy[0].tolist() for box in results.boxes]
                    confs_list = [float(box.conf[0]) for box in results.boxes]
                    frame = self.gradcam.generate_simple_heatmap(frame, boxes_xy, confs_list)

                # Session recording
                if self.recorder and self.recorder.recording:
                    self.recorder.write_frame(frame)
                    dur = self.recorder.get_duration()
                    cv2.circle(frame, (620, 20), 8, (0, 0, 255), -1)
                    cv2.putText(frame, f"REC {dur:.0f}s", (555, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Update charts every 2 seconds
                if hasattr(self, '_last_chart_update'):
                    if time.time() - self._last_chart_update > 2.0:
                        self._update_charts()
                        self._update_progress_stats()
                        self._last_chart_update = time.time()
                else:
                    self._last_chart_update = time.time()

            # Display
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imgtk = ImageTk.PhotoImage(image=Image.fromarray(cv2image))
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.root.after(15, self._update_frame)

    def run(self):
        self.root.mainloop()


# ═════════════════════ RUN ═══════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    app = ISLRecognitionApp(root)
    app.run()