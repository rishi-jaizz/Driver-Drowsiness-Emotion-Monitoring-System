import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import threading
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# ── Must be the very first Streamlit command ──────────────────────────────────
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

# ── Face detector ─────────────────────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ── Thread-safe inference lock ────────────────────────────────────────────────
model_lock = threading.Lock()

# ── Page styles ───────────────────────────────────────────────────────────────
st.markdown("""
    <style>
        body, .stApp { background-color: #03040B !important; }
        .title { font-size: 52px; color: #ffffff; font-weight: 900;
                 letter-spacing: -1px; margin-bottom: 8px; }
        .desc  { font-size: 16px; color: #909398; margin-bottom: 24px; line-height: 1.6; }
        div[data-testid="stButton"] button {
            background: transparent;
            border: 1px solid #4b5563;
            color: #e5e7eb;
            border-radius: 6px;
        }
        div[data-testid="stButton"] button:hover { border-color: #9ca3af; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">EMOTE-ION</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="desc">Experience real-time drowsiness and emotion detection using '
    'advanced AI models. Click below to start the demo.</div>',
    unsafe_allow_html=True
)

# ── Session state ─────────────────────────────────────────────────────────────
if "running" not in st.session_state:
    st.session_state.running = False

col1, col2 = st.columns(2)
with col1:
    if st.button("Let's Start the Demo", use_container_width=True):
        st.session_state.running = True
        st.rerun()
with col2:
    if st.button("Stop Demo", use_container_width=True):
        st.session_state.running = False
        st.rerun()


# ── Video processor class ─────────────────────────────────────────────────────
class VideoProcessor:
    def __init__(self):
        self._frame_count = 0
        self._last_drowsiness = "N/A"
        self._last_emotion    = "N/A"
        self._last_safety     = "N/A"
        self._last_S          = 0.0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img  = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30)
        )

        self._frame_count += 1
        run_inference = (self._frame_count % 5 == 0)  # infer every 5th frame

        for (x, y, w, h) in faces:
            if run_inference and drowsiness_model and emotion_model:
                face_roi = gray[y:y + h, x:x + w]
                face_in  = np.expand_dims(
                    np.expand_dims(cv2.resize(face_roi, (48, 48)), axis=-1),
                    axis=0
                ) / 255.0

                with model_lock:
                    d_pred = drowsiness_model.predict(face_in, verbose=0)
                    e_pred = emotion_model.predict(face_in, verbose=0)

                self._last_drowsiness = drowsiness_labels[np.argmax(d_pred)]
                self._last_emotion    = emotion_labels[np.argmax(e_pred)]
                D = drowsiness_scores.get(self._last_drowsiness, 0)
                E = emotion_scores.get(self._last_emotion, 0)
                self._last_S      = D + 0.7 * E
                self._last_safety = safety_matrix[self._last_drowsiness][self._last_emotion]

            # Draw annotations using cached last result
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, f"Drowsiness: {self._last_drowsiness}",
                        (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(img, f"Emotion: {self._last_emotion}",
                        (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 120, 0), 2)
            cv2.putText(img, f"Safety: {self._last_safety} (S={self._last_S:.2f})",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ── WebRTC streamer ───────────────────────────────────────────────────────────
if st.session_state.running:
    webrtc_streamer(
        key="drowsiness-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }),
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
else:
    st.markdown(
        "<h4 style='color:#909398; text-align:center; margin-top:60px;'>"
        "Click <b>Let's Start the Demo</b> to begin real-time monitoring.</h4>",
        unsafe_allow_html=True
    )