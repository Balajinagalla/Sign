# api_server.py — REST API for ISL Sign Language Detection
# Endpoints: /detect (image), /signs (list), /translate, /health
# Run: python api_server.py (starts on http://localhost:8000)
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import base64
import numpy as np
import cv2
import threading
import time

# ── Load model ──
MODEL = None
MODEL_NAMES = {}

def load_model():
    global MODEL, MODEL_NAMES
    from ultralytics import YOLO
    pt_path = "best.pt"
    onnx_path = "best.onnx"
    if os.path.exists(pt_path):
        MODEL = YOLO(pt_path, task='detect')
    elif os.path.exists(onnx_path):
        MODEL = YOLO(onnx_path, task='detect')
    else:
        print("⚠️  No model found! Place best.pt or best.onnx in project root.")
        return False
    MODEL_NAMES = MODEL.names
    print(f"✅ Model loaded: {len(MODEL_NAMES)} classes")
    return True

# ── Load translations ──
TRANSLATIONS = {}
LANGUAGES = []
LANG_CODES = []
try:
    from sign_constants import TRANSLATIONS as T, LANGUAGES as L, LANG_CODES as LC
    TRANSLATIONS = T
    LANGUAGES = L
    LANG_CODES = LC
except ImportError:
    pass


class ISLAPIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for ISL API."""

    def do_GET(self):
        if self.path == '/health':
            self._json_response({'status': 'ok', 'model_loaded': MODEL is not None,
                                  'classes': len(MODEL_NAMES), 'languages': len(LANGUAGES)})

        elif self.path == '/signs':
            signs = []
            for idx, name in MODEL_NAMES.items():
                ref_path = f"sign_references/{name.replace(' ', '_')}.png"
                signs.append({
                    'id': idx, 'name': name,
                    'has_reference': os.path.exists(ref_path),
                    'translations': TRANSLATIONS.get(name.lower(), {})
                })
            self._json_response({'signs': signs, 'total': len(signs)})

        elif self.path == '/languages':
            langs = [{'name': n, 'code': c} for n, c in zip(LANGUAGES, LANG_CODES)]
            self._json_response({'languages': langs})

        elif self.path.startswith('/sign/'):
            sign_name = self.path[6:].lower().replace('%20', ' ')
            trans = TRANSLATIONS.get(sign_name, {})
            ref_path = f"sign_references/{sign_name.replace(' ', '_')}.png"
            has_img = os.path.exists(ref_path)

            result = {
                'name': sign_name,
                'translations': trans,
                'has_reference_image': has_img,
            }

            if has_img:
                with open(ref_path, 'rb') as f:
                    result['reference_image_base64'] = base64.b64encode(f.read()).decode('utf-8')

            self._json_response(result)

        elif self.path == '/':
            self._html_response(API_DOCS_HTML)

        else:
            self._json_response({'error': 'Not found'}, 404)

    def do_POST(self):
        if self.path == '/detect':
            self._handle_detect()
        elif self.path == '/translate':
            self._handle_translate()
        else:
            self._json_response({'error': 'Not found'}, 404)

    def _handle_detect(self):
        """POST /detect — Detect signs in an image."""
        if MODEL is None:
            self._json_response({'error': 'Model not loaded'}, 503)
            return

        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._json_response({'error': 'Invalid JSON'}, 400)
            return

        if 'image' not in data:
            self._json_response({'error': 'Missing "image" field (base64 encoded)'}, 400)
            return

        conf_threshold = data.get('confidence', 0.4)

        try:
            img_bytes = base64.b64decode(data['image'])
            np_arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is None:
                self._json_response({'error': 'Could not decode image'}, 400)
                return

            start = time.time()
            results = MODEL(img, conf=conf_threshold, verbose=False)[0]
            inference_ms = (time.time() - start) * 1000

            detections = []
            if results.boxes and len(results.boxes) > 0:
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    det = {
                        'sign': MODEL_NAMES.get(cls_id, 'unknown'),
                        'confidence': round(conf, 4),
                        'bbox': {'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2)},
                        'class_id': cls_id,
                        'translations': TRANSLATIONS.get(MODEL_NAMES.get(cls_id, '').lower(), {})
                    }
                    detections.append(det)

            self._json_response({
                'detections': detections,
                'count': len(detections),
                'inference_ms': round(inference_ms, 1),
                'image_size': {'width': img.shape[1], 'height': img.shape[0]}
            })

        except Exception as e:
            self._json_response({'error': str(e)}, 500)

    def _handle_translate(self):
        """POST /translate — Translate sign name to different languages."""
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._json_response({'error': 'Invalid JSON'}, 400)
            return

        sign = data.get('sign', '').lower()
        lang = data.get('language', 'all')

        trans = TRANSLATIONS.get(sign, {})
        if not trans:
            self._json_response({'error': f'No translations for "{sign}"'}, 404)
            return

        if lang == 'all':
            self._json_response({'sign': sign, 'translations': trans})
        else:
            text = trans.get(lang, trans.get('en', sign))
            self._json_response({'sign': sign, 'language': lang, 'text': text})

    def _json_response(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False, indent=2).encode('utf-8'))

    def _html_response(self, html, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def log_message(self, format, *args):
        print(f"  [{time.strftime('%H:%M:%S')}] {args[0]}")


# ── API Documentation Page ──
API_DOCS_HTML = """<!DOCTYPE html>
<html><head><title>ISL API</title>
<style>
body{font-family:system-ui;background:#121212;color:#e0e0e0;max-width:800px;margin:0 auto;padding:20px}
h1{color:#bb86fc}h2{color:#03dac6;margin-top:20px}
.endpoint{background:#1e1e2f;border:1px solid #333;border-radius:10px;padding:15px;margin:10px 0}
.method{font-weight:bold;padding:3px 8px;border-radius:5px;font-size:0.8rem}
.get{background:#03dac6;color:#000}.post{background:#bb86fc;color:#000}
code{background:#2a2a3a;padding:2px 6px;border-radius:4px;font-size:0.9rem}
pre{background:#1a1a2e;padding:15px;border-radius:8px;overflow-x:auto;font-size:0.85rem}
</style></head><body>
<h1>🤟 ISL Recognition API</h1>
<p>Indian Sign Language Detection REST API</p>

<div class="endpoint">
<span class="method get">GET</span> <code>/health</code>
<p>Health check — returns model status</p>
</div>

<div class="endpoint">
<span class="method get">GET</span> <code>/signs</code>
<p>List all supported signs with translations</p>
</div>

<div class="endpoint">
<span class="method get">GET</span> <code>/languages</code>
<p>List supported languages</p>
</div>

<div class="endpoint">
<span class="method get">GET</span> <code>/sign/{name}</code>
<p>Get sign details + reference image (base64)</p>
</div>

<div class="endpoint">
<span class="method post">POST</span> <code>/detect</code>
<p>Detect signs in image</p>
<pre>{
  "image": "base64_encoded_image",
  "confidence": 0.4
}</pre>
</div>

<div class="endpoint">
<span class="method post">POST</span> <code>/translate</code>
<p>Translate sign name</p>
<pre>{
  "sign": "hello",
  "language": "hi"
}</pre>
</div>

<h2>Example (Python)</h2>
<pre>
import requests, base64

with open("test.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

resp = requests.post("http://localhost:8000/detect",
    json={"image": img_b64, "confidence": 0.4})
print(resp.json())
</pre>
</body></html>"""


def run_server(host="0.0.0.0", port=8000):
    """Start the API server."""
    print(f"\n{'='*55}")
    print(f"  🤟 ISL Recognition REST API Server")
    print(f"{'='*55}")

    if not load_model():
        print("  ❌ Failed to load model. Exiting.")
        return

    server = HTTPServer((host, port), ISLAPIHandler)
    print(f"\n  🌐 API running at: http://localhost:{port}")
    print(f"  📖 API docs at:    http://localhost:{port}/")
    print(f"  🏥 Health check:   http://localhost:{port}/health")
    print(f"\n  Press Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  ⏹ Server stopped.")
        server.server_close()


if __name__ == "__main__":
    import sys
    port = 8000
    if "--port" in sys.argv:
        idx = sys.argv.index("--port")
        port = int(sys.argv[idx + 1])
    run_server(port=port)
