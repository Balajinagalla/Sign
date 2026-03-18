# isl_gui_app.py - Enhanced Indian Sign Language Recognition GUI
# Features: Temporal smoothing, Sentence builder, Confidence slider,
#           Sign history, Dark theme, FPS counter, Quiz mode, Text-to-Sign
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from ultralytics import YOLO
import threading
import numpy as np
from PIL import Image, ImageTk
from collections import deque
import time
import random
import os
from tts_indic_multi import speak_sign, LANGUAGES, LANG_CODES, TRANSLATIONS

# ── Color Palette (Dark Theme) ──────────────────────────────────
BG_DARK      = "#1a1a2e"
BG_CARD      = "#16213e"
BG_ACCENT    = "#0f3460"
FG_PRIMARY   = "#e8e8e8"
FG_SECONDARY = "#a0a0b0"
COLOR_GREEN  = "#00e676"
COLOR_RED    = "#ff5252"
COLOR_CYAN   = "#00e5ff"
COLOR_ORANGE = "#ffab40"
COLOR_PURPLE = "#bb86fc"
COLOR_YELLOW = "#ffd600"

SIGN_REF_DIR = "sign_references"


class ISLRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🤟 ISL Recognition — Indian Sign Language → Speech")
        self.root.geometry("1150x850")
        self.root.configure(bg=BG_DARK)
        self.root.resizable(True, True)

        # ── Model (auto-detect ONNX) ─────────────────────────────
        onnx_path = "runs/train/sign_lang_yolo11/weights/best.onnx"
        pt_path = "runs/train/sign_lang_yolo11/weights/best.pt"
        if os.path.exists(onnx_path):
            self.model = YOLO(onnx_path)
            self.model_type = "ONNX"
        else:
            self.model = YOLO(pt_path)
            self.model_type = "PyTorch"

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

        # ── Sign Reference Images ────────────────────────────────
        self.sign_images = {}
        self._load_sign_references()

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

    # ═════════════════════════════════════════════════════════════
    #  EVENT HANDLERS
    # ═════════════════════════════════════════════════════════════
    def _on_mode_change(self):
        mode = self.mode_var.get()
        self.current_mode = mode
        tab_map = {"recognition": 0, "quiz": 1, "text2sign": 2}
        self.notebook.select(tab_map.get(mode, 0))

    def _on_lang_change(self, event=None):
        self.current_lang_idx = LANGUAGES.index(self.lang_combo.get())
        self.status_label.config(text=f"Status: Running │ Language: {self.lang_combo.get()}")

    def _on_conf_change(self, val):
        self.conf_threshold = float(val)

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
        self.video_label.config(image='')
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Stopped")

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
    #  MAIN FRAME LOOP
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

            for box in results.boxes:
                conf = box.conf[0].item()
                cls_id = int(box.cls[0].item())
                label = self.model.names[cls_id]

                if conf > best_conf:
                    best_conf = conf
                    best_sign = label

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = (0, 230, 118) if conf >= self.conf_threshold else (80, 80, 200)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label_text = f"{label} {conf:.0%}"
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
                cv2.putText(frame, label_text, (x1 + 3, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # Temporal smoothing
            raw_sign = best_sign if best_conf >= self.conf_threshold else None
            smoothed_sign, avg_conf = self._get_smoothed_sign(raw_sign, best_conf)
            self.stability_label.config(text=f"Stability: {self._get_stability_indicator()}")

            if smoothed_sign:
                self.sign_label.config(text=smoothed_sign.upper(), fg=COLOR_GREEN)
                self.conf_label.config(text=f"Confidence: {avg_conf:.0%}")

                # Mode-specific behavior
                if self.current_mode == "recognition":
                    self._add_to_sentence(smoothed_sign)
                    self._add_to_history(smoothed_sign, avg_conf)
                    self._speak_debounced(smoothed_sign)
                elif self.current_mode == "quiz":
                    self._check_quiz_answer(smoothed_sign)

                cv2.putText(frame, f"STABLE: {smoothed_sign.upper()}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 118), 2)
            else:
                self.sign_label.config(text="—", fg=COLOR_RED)
                self.conf_label.config(text="Confidence: --")

            # Quiz mode overlay
            if self.quiz_active and self.quiz_target:
                cv2.putText(frame, f"QUIZ: Show '{self.quiz_target.upper()}'", (10, 470),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 214, 255), 2)

            # FPS
            ft = time.time() - frame_start
            self.frame_times.append(ft)
            if len(self.frame_times) > 5:
                avg_t = sum(self.frame_times) / len(self.frame_times)
                self.fps_label.config(text=f"FPS: {1.0/avg_t:.1f}" if avg_t > 0 else "FPS: --")

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