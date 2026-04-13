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

# ── Custom CSS Design System ─────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap" rel="stylesheet">

<style>
    /* Main Background & Glassmorphism */
    .stApp {
        background: radial-gradient(circle at top right, #1e1e2f, #121212);
        font-family: 'Outfit', sans-serif;
        color: #e0e0e0;
    }
    
    .main {
        background: transparent;
    }

    /* Glass Cards */
    [data-testid="stVerticalBlock"] > div:has(div.stMarkdown) {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    [data-testid="stVerticalBlock"] > div:has(div.stMarkdown):hover {
        border: 1px solid rgba(187, 134, 252, 0.3);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
    }

    /* Typography & Headers */
    h1 {
        background: linear-gradient(90deg, #bb86fc, #03dac6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 600;
        letter-spacing: -1px;
    }
    
    h2, h3 { color: #bb86fc; font-weight: 400; }

    /* Buttons & Inputs */
    .stButton > button {
        background: rgba(187, 134, 252, 0.1);
        border: 1px solid #bb86fc !important;
        color: #bb86fc !important;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: #bb86fc !important;
        color: #121212 !important;
        box-shadow: 0 0 15px rgba(187, 134, 252, 0.4);
    }

    /* Status Indicators */
    .sign-detected { 
        font-size: 2.8rem; 
        font-weight: 600; 
        color: #03dac6; 
        text-shadow: 0 0 10px rgba(3, 218, 198, 0.4);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.05); opacity: 0.8; }
        100% { transform: scale(1); opacity: 1; }
    }

    .sentence-display { 
        font-size: 1.4rem; 
        color: #e0e0e0; 
        padding: 20px;
        background: rgba(187, 134, 252, 0.05); 
        border: 1px solid rgba(187, 134, 252, 0.2);
        border-radius: 12px;
        line-height: 1.6;
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #333; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #444; }

    /* Metric Styling */
    [data-testid="stMetricValue"] { color: #03dac6 !important; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Load Model ───────────────────────────────────────────────────
@st.cache_resource
def load_model():
    pt_path = "best.pt"  # Use your high-accuracy model
    if os.path.exists(pt_path):
        return YOLO(pt_path), "YOLOv11-Premium"
    return YOLO("yolo11n.pt"), "Baseline"

model, model_name = load_model()

# ── Session State ────────────────────────────────────────────────
if 'sentence' not in st.session_state:
    st.session_state.sentence = []
if 'history' not in st.session_state:
    st.session_state.history = []
if 'quiz_score' not in st.session_state:
    st.session_state.quiz_score = 0
if 'quiz_total' not in st.session_state:
    st.session_state.quiz_total = 0

# ── Header Section ───────────────────────────────────────────────
st.title("🏹 Indian Sign Language Recognition")
st.markdown(f"**Artificial Intelligence Engine Active** | 🔌 Model: `{model_name}` | 🏷️ `{len(model.names)}` sign classes supported")
st.markdown("---")

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
    st.header("🔍 Visual Dictionary")
    st.write("Browse the ISL library or search for a specific sign to see its visual reference.")

    col1, col2 = st.columns([1.5, 1])

    with col1:
        search = st.text_input("✨ Search Sign:", placeholder="Type a word like 'hello', 'family', 'help'...")

        # Load ALL available references
        all_signs = []
        if os.path.exists("sign_references"):
            for fname in sorted(os.listdir("sign_references")):
                if fname.endswith(".png"):
                    all_signs.append(fname.replace('.png', '').replace('_', ' ').title())

        st.subheader(f"📖 Sign Library ({len(all_signs)} classes)")
        
        # Creating a high-density, stylized scrollable grid
        with st.container(height=450):
            cols = st.columns(4)
            for i, sign in enumerate(all_signs):
                is_model_sign = any(sign.lower() == n.lower() for n in model.names.values())
                btn_type = "primary" if is_model_sign else "secondary"
                with cols[i % 4]:
                    # Using a custom emoji prefix for better visual scanning
                    if st.button(f"✋ {sign.upper()}", key=f"v_sign_{i}", type=btn_type, use_container_width=True):
                        st.session_state.search_query = sign
                        st.rerun()

    with col2:
        current_search = st.session_state.get('search_query', search)
        if current_search:
            sign_lower = current_search.lower().strip()
            matched = None
            for name in all_signs:
                if name.lower() == sign_lower or sign_lower in name.lower():
                    matched = name
                    break

            if matched:
                st.markdown(f"### 👐 {matched.upper()}")
                ref_path = f"sign_references/{matched.replace(' ', '_').lower()}.png"
                if os.path.exists(ref_path):
                    st.image(ref_path, use_container_width=True)
                
                # Metadata / Translation Card
                with st.expander("🌐 Multilingual Translations", expanded=True):
                    from sign_constants import TRANSLATIONS
                    trans = TRANSLATIONS.get(matched.lower(), {})
                    if trans:
                        for lang, code in [("Hindi", "hi"), ("Tamil", "ta"), ("Telugu", "te")]:
                            if code in trans:
                                st.write(f"**{lang}:** {trans[code]}")
                    else:
                        st.info("Translation not available for this sign.")
            else:
                st.warning(f"No exact match for '{current_search}'. Try another word!")

# ═════════════════════════════════════════════════════════════════
#  QUIZ MODE
# ═════════════════════════════════════════════════════════════════
elif mode == "📝 Quiz":
    st.header("🎮 Practice & Quiz")
    st.write("Challenge yourself! Can you perform the signs correctly?")

    col1, col2 = st.columns([1.5, 1])

    with col1:
        import random
        if 'quiz_target' not in st.session_state:
            st.session_state.quiz_target = random.choice(list(model.names.values()))

        st.markdown(f"### Target: <span class='sign-detected'>{st.session_state.quiz_target.upper()}</span>", unsafe_allow_html=True)

        ref_path = f"sign_references/{st.session_state.quiz_target.replace(' ', '_').lower()}.png"
        if os.path.exists(ref_path):
            st.image(ref_path, width=400)

        if st.button("⏭️ Generate New Challenge", use_container_width=True):
            st.session_state.quiz_target = random.choice(list(model.names.values()))
            st.rerun()

    with col2:
        st.subheader("🏁 Live Test")
        uploaded = st.file_uploader("Upload your performance:", type=['jpg', 'png', 'jpeg'])

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
                    st.success(f"🎯 **AMAZING!** Detected {detected.upper()} at {conf:.0%}")
                    st.balloons()
                else:
                    st.error(f"⚠️ **CLOSE!** You showed {detected.upper()} ({conf:.0%})")
            else:
                st.warning("No signs detected. Please ensure your hand is clearly visible.")

        st.markdown("---")
        st.subheader("📊 Performance Track")
        sc, tot = st.session_state.quiz_score, st.session_state.quiz_total
        acc = (sc/tot*100) if tot > 0 else 0
        
        c_sc, c_acc = st.columns(2)
        c_sc.metric("Success Rate", f"{sc}/{tot}")
        c_acc.metric("Accuracy", f"{acc:.0f}%")
        
        if st.button("🔄 Reset Statistics", type="secondary"):
            st.session_state.quiz_score = 0
            st.session_state.quiz_total = 0
            st.rerun()

# ═════════════════════════════════════════════════════════════════
#  RECOGNITION MODE (Image Upload)
# ═════════════════════════════════════════════════════════════════
else:
    st.header("🎤 Sign Recognition")

    col1, col2 = st.columns([2, 1])

    with col1:
        source = st.radio("Input source:", ["📷 Upload Image", "📁 Validation Images", "🖼️ Try Reference Library"],
                          horizontal=True)

        if source == "📷 Upload Image":
            uploaded = st.file_uploader("Upload a sign language image:", type=['jpg', 'png', 'jpeg'])
            if uploaded:
                img = Image.open(uploaded)
                img_array = np.array(img)
        elif source == "📁 Validation Images":
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
        else: # Try Reference Library
            ref_dir = "sign_references"
            if os.path.exists(ref_dir):
                ref_files = sorted([f for f in os.listdir(ref_dir) if f.endswith('.png')])
                if ref_files:
                    selected = st.selectbox("Select a reference sign to test:", ref_files)
                    img = Image.open(os.path.join(ref_dir, selected))
                    img_array = np.array(img)
                    uploaded = True
                else:
                    st.warning("No reference images found")
                    uploaded = None
            else:
                st.warning("Reference directory not found")
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
        
        all_signs = []
        if os.path.exists("sign_references"):
            for fname in sorted(os.listdir("sign_references")):
                if fname.endswith(".png"):
                    all_signs.append(fname.replace('.png', '').replace('_', ' ').title())
                    
        with st.container(height=400):
            for sign in all_signs:
                ref_path = f"sign_references/{sign.replace(' ', '_').lower()}.png"
                if os.path.exists(ref_path):
                    with st.expander(f"✋ {sign.upper()}"):
                        st.image(ref_path, width=200)
