# enhancements.py — Core Enhancement Utilities for ISL Project
# Contains: Face blur, Grad-CAM, Charts, Dictionary, Recording, Progress, Calibration
import cv2
import numpy as np
import os
import json
import time
from datetime import datetime, timedelta
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════
#  1. FACE BLUR (Privacy)
# ═══════════════════════════════════════════════════════════════
class FaceBlur:
    """Auto-blur faces in camera feed for privacy."""

    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.enabled = False

    def process(self, frame):
        if not self.enabled:
            return frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(40, 40))
        result = frame.copy()
        for (x, y, w, h) in faces:
            roi = result[y:y+h, x:x+w]
            blur = cv2.GaussianBlur(roi, (99, 99), 30)
            result[y:y+h, x:x+w] = blur
        return result


# ═══════════════════════════════════════════════════════════════
#  2. GRAD-CAM HEATMAP VISUALIZATION
# ═══════════════════════════════════════════════════════════════
class GradCAMVisualizer:
    """Generate attention heatmaps showing where the model focuses."""

    def __init__(self):
        self.enabled = False
        self.last_heatmap = None

    def generate_simple_heatmap(self, frame, boxes, confs):
        """Generate a simple attention heatmap from detection bounding boxes.
        For a lightweight approximation — areas with high confidence get hot colors."""
        if not self.enabled or not boxes:
            return frame

        h, w = frame.shape[:2]
        heatmap = np.zeros((h, w), dtype=np.float32)

        for box, conf in zip(boxes, confs):
            x1, y1, x2, y2 = [int(v) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Create gaussian blob at detection center
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            bw, bh = x2 - x1, y2 - y1
            sigma_x, sigma_y = max(bw // 2, 1), max(bh // 2, 1)

            Y, X = np.ogrid[0:h, 0:w]
            gaussian = np.exp(-((X - cx)**2 / (2 * sigma_x**2) + (Y - cy)**2 / (2 * sigma_y**2)))
            heatmap += gaussian.astype(np.float32) * float(conf)

        # Normalize and colorize
        if heatmap.max() > 0:
            heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
        else:
            heatmap = heatmap.astype(np.uint8)

        colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        self.last_heatmap = colored

        # Blend with original
        result = cv2.addWeighted(frame, 0.6, colored, 0.4, 0)
        return result


# ═══════════════════════════════════════════════════════════════
#  3. SIGN DICTIONARY WITH FUZZY SEARCH
# ═══════════════════════════════════════════════════════════════
class SignDictionary:
    """Searchable sign dictionary with fuzzy matching."""

    def __init__(self):
        self.signs = {}  # {name: {translations, image_path, category}}
        self._load_signs()

    def _load_signs(self):
        """Load signs from sign_constants and sign_references."""
        try:
            from sign_constants import TRANSLATIONS
            for sign_name, trans in TRANSLATIONS.items():
                self.signs[sign_name] = {
                    'translations': trans,
                    'image': f"sign_references/{sign_name.replace(' ', '_')}.png",
                    'category': self._categorize(sign_name)
                }
        except ImportError:
            pass

    def _categorize(self, sign_name):
        greetings = {'hello', 'thank you', 'please', 'congratulations'}
        emotions = {'angry', 'cry', 'ily', 'ilove you'}
        actions = {'drink', 'give', 'close', 'sleep', 'wear', 'doing', 'chat'}
        people = {'family', 'man', 'friend'}
        objects = {'house', 'knife', 'tea', 'phone', 'shirt', 'deer', 'elephant'}
        states = {'busy', 'free', 'fever', 'injury', 'thirsty', 'hungry'}
        responses = {'yes', 'no', 'agree', 'sure', 'help', 'promise'}

        for cat, words in [('Greetings', greetings), ('Emotions', emotions),
                            ('Actions', actions), ('People', people),
                            ('Objects', objects), ('States', states),
                            ('Responses', responses)]:
            if sign_name.lower() in words:
                return cat
        return 'Other'

    def search(self, query):
        """Fuzzy search for signs. Returns list of (name, score) tuples."""
        query = query.lower().strip()
        if not query:
            return list(self.signs.keys())

        results = []
        for name in self.signs:
            score = self._similarity(query, name.lower())
            if score > 0.3 or query in name.lower():
                results.append((name, max(score, 1.0 if query in name.lower() else score)))

        results.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in results]

    def _similarity(self, s1, s2):
        """Simple character-based similarity (0-1)."""
        if s1 == s2:
            return 1.0
        common = sum(1 for c in s1 if c in s2)
        return (2.0 * common) / (len(s1) + len(s2)) if (len(s1) + len(s2)) > 0 else 0

    def get_by_category(self):
        """Group signs by category."""
        categories = defaultdict(list)
        for name, info in self.signs.items():
            categories[info['category']].append(name)
        return dict(categories)


# ═══════════════════════════════════════════════════════════════
#  4. SESSION RECORDING
# ═══════════════════════════════════════════════════════════════
class SessionRecorder:
    """Record sign detection sessions as video with overlay."""

    def __init__(self):
        self.recording = False
        self.writer = None
        self.filepath = ""
        self.start_time = 0
        self.frame_count = 0

    def start(self, width=640, height=480, fps=15.0):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = f"ISL_Recording_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(self.filepath, fourcc, fps, (width, height))
        self.recording = True
        self.start_time = time.time()
        self.frame_count = 0
        return self.filepath

    def write_frame(self, frame):
        if self.recording and self.writer:
            h, w = frame.shape[:2]
            if (w, h) != (self.writer.get(3), self.writer.get(4)):
                frame = cv2.resize(frame, (int(self.writer.get(3)), int(self.writer.get(4))))
            self.writer.write(frame)
            self.frame_count += 1

    def stop(self):
        if self.writer:
            self.writer.release()
            self.writer = None
        self.recording = False
        duration = time.time() - self.start_time
        return self.filepath, duration, self.frame_count

    def get_duration(self):
        if self.recording:
            return time.time() - self.start_time
        return 0


# ═══════════════════════════════════════════════════════════════
#  5. LEARNING PROGRESS TRACKER
# ═══════════════════════════════════════════════════════════════
class ProgressTracker:
    """Track user learning progress across sessions."""

    SAVE_FILE = "isl_progress.json"

    def __init__(self):
        self.data = self._load()

    def _load(self):
        if os.path.exists(self.SAVE_FILE):
            try:
                with open(self.SAVE_FILE, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            'total_signs_detected': 0,
            'total_sessions': 0,
            'total_time_minutes': 0,
            'daily_streak': 0,
            'last_practice_date': '',
            'per_sign': {},
            'quiz_history': [],
            'daily_log': {},
            'achievements': []
        }

    def save(self):
        with open(self.SAVE_FILE, 'w') as f:
            json.dump(self.data, f, indent=2)

    def log_detection(self, sign_name, confidence):
        self.data['total_signs_detected'] += 1
        if sign_name not in self.data['per_sign']:
            self.data['per_sign'][sign_name] = {
                'count': 0, 'total_conf': 0, 'first_seen': datetime.now().isoformat(),
                'last_seen': '', 'mastered': False
            }
        ps = self.data['per_sign'][sign_name]
        ps['count'] += 1
        ps['total_conf'] += confidence
        ps['last_seen'] = datetime.now().isoformat()
        if ps['count'] >= 50 and (ps['total_conf'] / ps['count']) >= 0.7:
            ps['mastered'] = True
        self._update_daily()

    def log_quiz(self, score, total, accuracy):
        self.data['quiz_history'].append({
            'date': datetime.now().isoformat(),
            'score': score, 'total': total, 'accuracy': accuracy
        })
        self._check_achievements(score, total)

    def start_session(self):
        self.data['total_sessions'] += 1
        self._session_start = time.time()

    def end_session(self):
        if hasattr(self, '_session_start'):
            minutes = (time.time() - self._session_start) / 60
            self.data['total_time_minutes'] += minutes
        self.save()

    def _update_daily(self):
        today = datetime.now().strftime("%Y-%m-%d")
        if today not in self.data['daily_log']:
            self.data['daily_log'][today] = 0
        self.data['daily_log'][today] += 1

        # Update streak
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        if self.data['last_practice_date'] == yesterday or self.data['last_practice_date'] == today:
            if self.data['last_practice_date'] != today:
                self.data['daily_streak'] += 1
        elif self.data['last_practice_date'] != today:
            self.data['daily_streak'] = 1
        self.data['last_practice_date'] = today

    def _check_achievements(self, score, total):
        achv = self.data['achievements']
        if score == total and total >= 5 and 'Perfect Quiz' not in achv:
            achv.append('Perfect Quiz')
        if self.data['total_signs_detected'] >= 100 and 'Century' not in achv:
            achv.append('Century')
        if self.data['total_signs_detected'] >= 1000 and 'Sign Master' not in achv:
            achv.append('Sign Master')
        if self.data['daily_streak'] >= 7 and 'Week Streak' not in achv:
            achv.append('Week Streak')
        if len(self.data['per_sign']) >= 20 and 'Polyglot' not in achv:
            achv.append('Polyglot')

    def get_mastered_signs(self):
        return [s for s, d in self.data['per_sign'].items() if d.get('mastered')]

    def get_struggling_signs(self):
        struggling = []
        for s, d in self.data['per_sign'].items():
            if d['count'] >= 10 and (d['total_conf'] / d['count']) < 0.5:
                struggling.append(s)
        return struggling

    def get_stats_summary(self):
        mastered = len(self.get_mastered_signs())
        total_signs = len(self.data['per_sign'])
        return {
            'sessions': self.data['total_sessions'],
            'total_detected': self.data['total_signs_detected'],
            'practice_minutes': round(self.data['total_time_minutes'], 1),
            'streak': self.data['daily_streak'],
            'signs_learned': total_signs,
            'signs_mastered': mastered,
            'achievements': self.data['achievements'],
        }


# ═══════════════════════════════════════════════════════════════
#  6. CONFIDENCE CALIBRATION
# ═══════════════════════════════════════════════════════════════
class ConfidenceCalibrator:
    """Temperature scaling for better-calibrated confidence scores."""

    def __init__(self, temperature=1.5):
        self.temperature = temperature
        self.enabled = True

    def calibrate(self, confidence):
        if not self.enabled:
            return confidence
        # Temperature scaling (softmax-style)
        scaled = confidence ** (1.0 / self.temperature)
        return min(scaled, 1.0)


# ═══════════════════════════════════════════════════════════════
#  7. REAL-TIME CHARTS
# ═══════════════════════════════════════════════════════════════
class RealtimeCharts:
    """Generate real-time chart images for the GUI."""

    def __init__(self, max_points=50):
        self.confidence_history = []
        self.sign_counts = defaultdict(int)
        self.max_points = max_points

    def add_detection(self, sign_name, confidence):
        self.confidence_history.append({
            'time': time.time(),
            'sign': sign_name,
            'conf': confidence
        })
        self.sign_counts[sign_name] += 1
        # Keep only recent
        if len(self.confidence_history) > self.max_points:
            self.confidence_history.pop(0)

    def render_confidence_chart(self, width=350, height=150):
        """Render a confidence-over-time chart as a numpy image."""
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:] = (30, 30, 47)  # BG_CARD color

        if len(self.confidence_history) < 2:
            cv2.putText(img, "Collecting data...", (20, height//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            return img

        # Draw grid
        for i in range(5):
            y = int(15 + i * (height - 30) / 4)
            cv2.line(img, (40, y), (width - 10, y), (50, 50, 70), 1)
            pct = 100 - i * 25
            cv2.putText(img, f"{pct}%", (2, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 120), 1)

        # Plot confidence points
        n = len(self.confidence_history)
        points = []
        for i, entry in enumerate(self.confidence_history):
            x = int(40 + i * (width - 50) / max(n - 1, 1))
            y = int(15 + (1 - entry['conf']) * (height - 30))
            points.append((x, y))

        # Draw line
        for i in range(1, len(points)):
            conf = self.confidence_history[i]['conf']
            color = (0, 230, 118) if conf >= 0.7 else (255, 183, 77) if conf >= 0.4 else (207, 102, 121)
            cv2.line(img, points[i-1], points[i], color, 2)

        # Draw dots
        for i, (x, y) in enumerate(points):
            conf = self.confidence_history[i]['conf']
            color = (0, 230, 118) if conf >= 0.7 else (255, 183, 77) if conf >= 0.4 else (207, 102, 121)
            cv2.circle(img, (x, y), 3, color, -1)

        # Title
        cv2.putText(img, "Confidence Over Time", (50, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (187, 134, 252), 1)
        return img

    def render_sign_distribution(self, width=350, height=150):
        """Render sign frequency bar chart as numpy image."""
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:] = (30, 30, 47)

        if not self.sign_counts:
            cv2.putText(img, "No signs detected yet", (20, height//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            return img

        # Top 8 signs
        sorted_signs = sorted(self.sign_counts.items(), key=lambda x: x[1], reverse=True)[:8]
        max_count = max(c for _, c in sorted_signs)

        bar_height = max(8, (height - 30) // len(sorted_signs) - 4)
        colors = [(3, 218, 198), (187, 134, 252), (255, 183, 77), (100, 181, 246),
                  (207, 102, 121), (251, 192, 45), (0, 230, 118), (156, 39, 176)]

        for i, (sign, count) in enumerate(sorted_signs):
            y = 20 + i * (bar_height + 4)
            bar_w = int((count / max_count) * (width - 120))
            color = colors[i % len(colors)]

            cv2.rectangle(img, (80, y), (80 + bar_w, y + bar_height), color, -1)
            cv2.putText(img, sign[:8].upper(), (2, y + bar_height - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
            cv2.putText(img, str(count), (85 + bar_w, y + bar_height - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1)

        cv2.putText(img, "Sign Frequency", (100, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (187, 134, 252), 1)
        return img


# ═══════════════════════════════════════════════════════════════
#  8. GESTURE SHORTCUTS
# ═══════════════════════════════════════════════════════════════
class GestureShortcuts:
    """Map detected signs to system actions."""

    def __init__(self):
        self.enabled = False
        self.mappings = {
            # sign_name: action_function
        }
        self.cooldown = 3.0
        self.last_action_time = 0

    def register(self, sign_name, action_func):
        self.mappings[sign_name.lower()] = action_func

    def check(self, detected_sign):
        if not self.enabled:
            return False
        now = time.time()
        if now - self.last_action_time < self.cooldown:
            return False
        sign = detected_sign.lower()
        if sign in self.mappings:
            self.last_action_time = now
            self.mappings[sign]()
            return True
        return False


# ═══════════════════════════════════════════════════════════════
#  9. HIGH CONTRAST MODE
# ═══════════════════════════════════════════════════════════════
THEMES = {
    'dark': {
        'bg': "#121212", 'card': "#1e1e2f", 'accent': "#16213e",
        'fg': "#e0e0e0", 'fg2': "#b0b0b0",
        'green': "#03dac6", 'red': "#cf6679", 'purple': "#bb86fc",
        'orange': "#ffb74d", 'yellow': "#fbc02d", 'cyan': "#03dac6",
    },
    'light': {
        'bg': "#f5f5f5", 'card': "#ffffff", 'accent': "#e3f2fd",
        'fg': "#212121", 'fg2': "#616161",
        'green': "#00897b", 'red': "#e53935", 'purple': "#7b1fa2",
        'orange': "#f57c00", 'yellow': "#f9a825", 'cyan': "#00838f",
    },
    'high_contrast': {
        'bg': "#000000", 'card': "#1a1a1a", 'accent': "#000000",
        'fg': "#ffffff", 'fg2': "#ffff00",
        'green': "#00ff00", 'red': "#ff0000", 'purple': "#ff00ff",
        'orange': "#ff8800", 'yellow': "#ffff00", 'cyan': "#00ffff",
    }
}


# ═══════════════════════════════════════════════════════════════
#  10. CONVERSATION MODE
# ═══════════════════════════════════════════════════════════════
class ConversationMode:
    """Two-person conversation with sign detection."""

    def __init__(self):
        self.active = False
        self.messages = []  # [{sender: 'signer'/'speaker', text: '...', time: '...'}]

    def add_sign_message(self, sign_text):
        self.messages.append({
            'sender': 'signer',
            'text': sign_text,
            'time': datetime.now().strftime("%H:%M:%S")
        })

    def add_voice_message(self, text):
        self.messages.append({
            'sender': 'speaker',
            'text': text,
            'time': datetime.now().strftime("%H:%M:%S")
        })

    def get_chat_display(self, max_messages=20):
        recent = self.messages[-max_messages:]
        lines = []
        for msg in recent:
            prefix = "🤟" if msg['sender'] == 'signer' else "🗣️"
            lines.append(f"[{msg['time']}] {prefix} {msg['text']}")
        return "\n".join(lines)

    def export_conversation(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ISL_Conversation_{timestamp}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("🤟 ISL Conversation Log\n")
            f.write(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write("=" * 50 + "\n\n")
            for msg in self.messages:
                prefix = "🤟 [SIGN]" if msg['sender'] == 'signer' else "🗣️ [VOICE]"
                f.write(f"[{msg['time']}] {prefix}  {msg['text']}\n")
        return filename
