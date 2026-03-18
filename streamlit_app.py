# streamlit_app.py - ISL Sign Language Recognition Web App
# Run with: streamlit run streamlit_app.py
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
import time
from collections import deque

# ── Page Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="ISL Sign Language Recognition",
    page_icon="🤟",
    layout="wide"
)

# ── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #1a1a2e; }
    .stApp { background-color: #1a1a2e; }
    h1, h2, h3 { color: #e8e8e8; }
    .sign-detected { font-size: 2.5rem; font-weight: bold; color: #00e676; }
    .sign-waiting { font-size: 2.5rem; font-weight: bold; color: #ff5252; }
    .sentence-display { font-size: 1.3rem; color: #00e5ff; padding: 10px;
                        background: #16213e; border-radius: 8px; }
    .score-card { background: #16213e; padding: 15px; border-radius: 10px;
                  text-align: center; }
    .metric-value { font-size: 2rem; font-weight: bold; color: #00e676; }
</style>
""", unsafe_allow_html=True)

# ── Load Model ───────────────────────────────────────────────────
@st.cache_resource
def load_model():
    onnx_path = "best.onnx"
    pt_path = "best.pt"
    if os.path.exists(onnx_path):
        return YOLO(onnx_path), "ONNX"
    return YOLO(pt_path), "PyTorch"

model, model_type = load_model()

# ── Session State ────────────────────────────────────────────────
if 'sentence' not in st.session_state:
    st.session_state.sentence = []
if 'history' not in st.session_state:
    st.session_state.history = []
if 'quiz_score' not in st.session_state:
    st.session_state.quiz_score = 0
if 'quiz_total' not in st.session_state:
    st.session_state.quiz_total = 0

# ── Header ───────────────────────────────────────────────────────
st.title("🤟 Indian Sign Language Recognition")
st.caption(f"Model: {model_type} | {len(model.names)} sign classes")

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    conf_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.4, 0.05)
    mode = st.radio("Mode", ["🎤 Recognition", "📝 Quiz", "🔄 Text → Sign"])

    st.markdown("---")
    st.header("📊 Session Stats")
    st.metric("Signs Detected", len(st.session_state.history))
    st.metric("Sentence Words", len(st.session_state.sentence))

    if st.button("🗑️ Clear Sentence"):
        st.session_state.sentence = []

    if st.button("🗑️ Clear History"):
        st.session_state.history = []

# ═════════════════════════════════════════════════════════════════
#  TEXT → SIGN MODE
# ═════════════════════════════════════════════════════════════════
if mode == "🔄 Text → Sign":
    st.header("🔄 Text → Sign Language")
    st.write("Type a word or click a button to see how to sign it")

    col1, col2 = st.columns([1, 1])

    with col1:
        search = st.text_input("Enter a word:", placeholder="e.g., hello, family, help...")

        st.subheader("⚡ Quick Select")
        sign_names = list(model.names.values())
        cols = st.columns(3)
        for i, sign in enumerate(sign_names):
            with cols[i % 3]:
                if st.button(sign.upper(), key=f"sign_{sign}", use_container_width=True):
                    search = sign

    with col2:
        if search:
            sign_lower = search.lower().strip()
            matched = None
            for name in model.names.values():
                if name.lower() == sign_lower or sign_lower in name.lower():
                    matched = name
                    break

            if matched:
                st.markdown(f"### ✋ {matched.upper()}")
                ref_path = f"sign_references/{matched.replace(' ', '_').lower()}.png"
                if os.path.exists(ref_path):
                    st.image(ref_path, width=250)
                else:
                    st.info("Reference image not available")

                from sign_constants import TRANSLATIONS
                trans = TRANSLATIONS.get(matched.lower(), {})
                if trans:
                    st.write("**Translations:**")
                    for lang, code in [("English", "en"), ("Hindi", "hi"), ("Tamil", "ta"),
                                       ("Telugu", "te"), ("Bengali", "bn")]:
                        if code in trans:
                            st.write(f"- **{lang}:** {trans[code]}")
            else:
                st.error(f"'{search}' not found. Available signs: {', '.join(sign_names)}")

# ═════════════════════════════════════════════════════════════════
#  QUIZ MODE
# ═════════════════════════════════════════════════════════════════
elif mode == "📝 Quiz":
    st.header("📝 Quiz Mode")
    st.write("Upload an image of a sign and test if the model recognizes it!")

    col1, col2 = st.columns([1, 1])

    with col1:
        import random
        if 'quiz_target' not in st.session_state:
            st.session_state.quiz_target = random.choice(list(model.names.values()))

        st.markdown(f"### Show this sign: **{st.session_state.quiz_target.upper()}**")

        ref_path = f"sign_references/{st.session_state.quiz_target.replace(' ', '_').lower()}.png"
        if os.path.exists(ref_path):
            st.image(ref_path, width=200)

        if st.button("⏭ Next Sign"):
            st.session_state.quiz_target = random.choice(list(model.names.values()))
            st.rerun()

        uploaded = st.file_uploader("Upload your sign image:", type=['jpg', 'png', 'jpeg'])

        if uploaded:
            img = Image.open(uploaded)
            img_array = np.array(img)
            results = model(img_array, conf=conf_threshold, verbose=False)[0]

            if results.boxes and len(results.boxes) > 0:
                best_idx = results.boxes.conf.argmax()
                detected = model.names[int(results.boxes.cls[best_idx])]
                conf = results.boxes.conf[best_idx].item()

                st.session_state.quiz_total += 1
                if detected.lower() == st.session_state.quiz_target.lower():
                    st.session_state.quiz_score += 1
                    st.success(f"✅ Correct! Detected: {detected.upper()} ({conf:.0%})")
                else:
                    st.error(f"❌ Wrong. Detected: {detected.upper()} ({conf:.0%})")
            else:
                st.warning("No sign detected in the image. Try again!")

    with col2:
        st.subheader("📊 Score")
        sc, tot = st.session_state.quiz_score, st.session_state.quiz_total
        st.metric("Score", f"{sc} / {tot}")
        if tot > 0:
            st.metric("Accuracy", f"{sc/tot*100:.0f}%")
        if st.button("🔄 Reset Score"):
            st.session_state.quiz_score = 0
            st.session_state.quiz_total = 0
            st.rerun()

        if uploaded:
            st.image(uploaded, caption="Your uploaded image", width=300)

# ═════════════════════════════════════════════════════════════════
#  RECOGNITION MODE (Image Upload)
# ═════════════════════════════════════════════════════════════════
else:
    st.header("🎤 Sign Recognition")

    col1, col2 = st.columns([2, 1])

    with col1:
        source = st.radio("Input source:", ["📷 Upload Image", "📁 Test from Dataset"],
                          horizontal=True)

        if source == "📷 Upload Image":
            uploaded = st.file_uploader("Upload a sign language image:", type=['jpg', 'png', 'jpeg'])
            if uploaded:
                img = Image.open(uploaded)
                img_array = np.array(img)
        else:
            test_dir = "Data/test/images"
            if os.path.exists(test_dir):
                test_files = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png'))]
                if test_files:
                    selected = st.selectbox("Select test image:", test_files)
                    img = Image.open(os.path.join(test_dir, selected))
                    img_array = np.array(img)
                    uploaded = True
                else:
                    st.warning("No test images found")
                    uploaded = None
            else:
                st.warning("Test directory not found")
                uploaded = None

        if uploaded:
            results = model(img_array, conf=conf_threshold, verbose=False)[0]
            annotated = results.plot()
            st.image(annotated, channels="BGR", caption="Detection Results", use_container_width=True)

            if results.boxes and len(results.boxes) > 0:
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    conf = box.conf[0].item()
                    sign = model.names[cls_id]

                    st.session_state.history.append(
                        f"[{time.strftime('%H:%M:%S')}] {sign.upper()} ({conf:.0%})")
                    st.session_state.sentence.append(sign.upper())

    with col2:
        st.subheader("💬 Sentence")
        if st.session_state.sentence:
            st.markdown(f'<div class="sentence-display">{" ".join(st.session_state.sentence)}</div>',
                       unsafe_allow_html=True)
        else:
            st.caption("Upload images to build a sentence")

        st.subheader("📋 History")
        if st.session_state.history:
            for entry in reversed(st.session_state.history[-10:]):
                st.text(entry)
        else:
            st.caption("No detections yet")

        # Sign reference gallery
        st.subheader("📖 Sign Guide")
        sign_names = list(model.names.values())
        for sign in sign_names[:5]:
            ref_path = f"sign_references/{sign.replace(' ', '_').lower()}.png"
            if os.path.exists(ref_path):
                with st.expander(f"✋ {sign.upper()}"):
                    st.image(ref_path, width=150)
