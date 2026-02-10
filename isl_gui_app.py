# isl_gui_app.py - Professional Indian Sign Language Recognition GUI
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from ultralytics import YOLO
import threading
import numpy as np
from PIL import Image, ImageTk
from tts_indic_multi import speak_sign, LANGUAGES, LANG_CODES, TRANSLATIONS

class ISLRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Indian Sign Language Recognition System")
        self.root.geometry("1000x700")
        self.root.configure(bg="#f0f2f5")

        # Variables
        self.model = YOLO("runs/train/sign_lang_yolo11/weights/best.pt")
        self.cap = None
        self.running = False
        self.current_lang_idx = 1  # Default: Hindi

        # Title
        title = tk.Label(root, text="Indian Sign Language → Speech", font=("Arial", 20, "bold"), bg="#f0f2f5", fg="#2c3e50")
        title.pack(pady=10)

        # Control Frame
        control_frame = tk.Frame(root, bg="#f0f2f5")
        control_frame.pack(pady=10)

        tk.Label(control_frame, text="Select Language:", font=("Arial", 12), bg="#f0f2f5").pack(side=tk.LEFT, padx=10)
        self.lang_combo = ttk.Combobox(control_frame, values=LANGUAGES, state="readonly", width=18)
        self.lang_combo.set(LANGUAGES[1])  # Hindi
        self.lang_combo.pack(side=tk.LEFT, padx=10)
        self.lang_combo.bind("<<ComboboxSelected>>", self.update_language)

        self.start_btn = tk.Button(control_frame, text="Start Camera", command=self.start_camera, bg="#27ae60", fg="white", font=("Arial", 12, "bold"), width=15)
        self.start_btn.pack(side=tk.LEFT, padx=10)

        self.stop_btn = tk.Button(control_frame, text="Stop Camera", command=self.stop_camera, bg="#c0392b", fg="white", font=("Arial", 12, "bold"), width=15, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=10)

        # Video Frame
        self.video_frame = tk.Label(root, bg="black", width=640, height=480)
        self.video_frame.pack(pady=20)

        # Status Label
        self.status_label = tk.Label(root, text="Status: Ready | Language: Hindi", font=("Arial", 12), bg="#f0f2f5", fg="#2c3e50")
        self.status_label.pack(pady=5)

        # Detected Sign Display
        self.sign_label = tk.Label(root, text="Detected: -", font=("Arial", 16, "bold"), bg="#f0f2f5", fg="#e74c3c")
        self.sign_label.pack(pady=10)

    def update_language(self, event=None):
        selected = self.lang_combo.get()
        self.current_lang_idx = LANGUAGES.index(selected)
        self.status_label.config(text=f"Status: Running | Language: {selected}")

    def start_camera(self):
        if not self.running:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Cannot open webcam!")
                return
            self.running = True
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.status_label.config(text=f"Status: Running | Language: {self.lang_combo.get()}")
            self.update_frame()

    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.video_frame.config(image='')
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Stopped")
        self.sign_label.config(text="Detected: -")

    def update_frame(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (640, 480))
            results = self.model(frame, conf=0.3, verbose=False)[0]  # Lower threshold for better detection

            best_sign = None
            best_conf = 0

            for box in results.boxes:
                conf = box.conf[0].item()
                cls_id = int(box.cls[0].item())
                label = self.model.names[cls_id]

                if conf > best_conf:
                    best_conf = conf
                    best_sign = label

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Update detected sign
            if best_sign and best_conf > 0.4:  # Lower threshold for TTS
                self.sign_label.config(text=f"Detected: {best_sign.upper()}", fg="#27ae60")
                # Speak in selected language
                threading.Thread(
                    target=speak_sign,
                    args=(best_sign, self.current_lang_idx),
                    daemon=True
                ).start()
            else:
                self.sign_label.config(text="Detected: -", fg="#e74c3c")

            # Convert frame to Tkinter format
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_frame.imgtk = imgtk
            self.video_frame.configure(image=imgtk)

        self.root.after(10, self.update_frame)

    def run(self):
        self.root.mainloop()

# ================ RUN THE APP ================
if __name__ == "__main__":
    root = tk.Tk()
    app = ISLRecognitionApp(root)
    app.run()