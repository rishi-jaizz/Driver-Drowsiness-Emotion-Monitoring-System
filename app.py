import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time

# ── Must be the very first Streamlit command ───────────────────────────────────
st.set_page_config(page_title="EMOTE-ION", layout="centered")


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    def safe_load(path):
        try:
            return load_model(path)
        except Exception:
            return None
    return safe_load('models/fl3d_model_whts_3.h5'), safe_load('models/affectnet_model_whts_2.h5')


drowsiness_model, emotion_model = load_models()

# ── Labels & scoring ──────────────────────────────────────────────────────────
drowsiness_labels = ['alert', 'microsleep', 'yawning']
emotion_labels    = ['angry', 'happy', 'neutral', 'sad']

drowsiness_scores = {'alert': 1, 'yawning': 0, 'microsleep': -1}
emotion_scores    = {'neutral': 0, 'happy': 1, 'sad': -1, 'angry': -1}

safety_matrix = {
    'alert':      {'angry': 'Neutral', 'happy': 'Safe',    'neutral': 'Safe',    'sad': 'Neutral'},
    'yawning':    {'angry': 'Unsafe',  'happy': 'Neutral', 'neutral': 'Neutral', 'sad': 'Unsafe'},
    'microsleep': {'angry': 'Unsafe',  'happy': 'Unsafe',  'neutral': 'Unsafe',  'sad': 'Unsafe'},
}

safety_colors = {'Safe': '🟢', 'Neutral': '🟡', 'Unsafe': '🔴', 'N/A': '⚪'}

# ── Face detector ─────────────────────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ── Page styles ───────────────────────────────────────────────────────────────
st.markdown("""
    <style>
        body { background-color: #03040B; color: #909398; }
        .title { font-size: 55px; color: #ffffff; font-weight: bold; }
        .desc  { font-size: 18px; color: #909398; margin-bottom: 20px; }
        div[data-testid="metric-container"] {
            background: #111827;
            border: 1px solid #1f2937;
            border-radius: 10px;
            padding: 12px 16px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">EMOTE-ION</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="desc">Real-time driver drowsiness & emotion monitoring. '
    'Allow camera access when prompted, then click <b>▶ Start</b>.</div>',
    unsafe_allow_html=True
)

# ── Session state ─────────────────────────────────────────────────────────────
if "running" not in st.session_state:
    st.session_state.running = False

col1, col2 = st.columns(2)
with col1:
    if st.button("▶ Start Demo", use_container_width=True, type="primary"):
        st.session_state.running = True
with col2:
    if st.button("⏹ Stop Demo", use_container_width=True):
        st.session_state.running = False
        st.rerun()

# ── Main loop ─────────────────────────────────────────────────────────────────
if st.session_state.running:
    st.info("📷 Camera active — allow browser camera access if prompted.")

    # st.camera_input captures a single frame from the browser webcam
    img_file = st.camera_input(
        label="Live feed",
        label_visibility="collapsed",
        key=f"cam_{int(time.time())}"   # rotating key forces a new snapshot each rerun
    )

    metrics_placeholder = st.empty()

    if img_file is not None:
        # Decode image
        file_bytes = np.frombuffer(img_file.read(), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30)
        )

        drowsiness_label = emotion_label = safety_state = "N/A"
        S = 0.0

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            face_in  = np.expand_dims(
                np.expand_dims(cv2.resize(face_roi, (48, 48)), axis=-1),
                axis=0
            ) / 255.0

            if drowsiness_model and emotion_model:
                drowsiness_label = drowsiness_labels[
                    np.argmax(drowsiness_model.predict(face_in, verbose=0))
                ]
                emotion_label = emotion_labels[
                    np.argmax(emotion_model.predict(face_in, verbose=0))
                ]
                D = drowsiness_scores.get(drowsiness_label, 0)
                E = emotion_scores.get(emotion_label, 0)
                S = D + 0.7 * E
                safety_state = safety_matrix[drowsiness_label][emotion_label]

            # Annotate frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Drowsiness: {drowsiness_label}",
                        (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Emotion: {emotion_label}",
                        (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 120, 0), 2)
            cv2.putText(frame, f"Safety: {safety_state} (S={S:.2f})",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)

        # Show annotated frame
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

        # Show metrics
        icon = safety_colors.get(safety_state, '⚪')
        with metrics_placeholder.container():
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("👁 Drowsiness",  drowsiness_label)
            m2.metric("😐 Emotion",     emotion_label)
            m3.metric("🛡 Safety",      f"{icon} {safety_state}")
            m4.metric("📊 Score (S)",   f"{S:.2f}")

        if len(faces) == 0:
            st.warning("⚠️ No face detected — ensure your face is well-lit and centred.")

    # Auto-refresh every second for continuous monitoring
    time.sleep(1)
    st.rerun()

elif not st.session_state.running:
    st.markdown(
        "<h4 style='color:#909398; text-align:center; margin-top:40px;'>"
        "Press ▶ Start Demo to begin monitoring.</h4>",
        unsafe_allow_html=True
    )