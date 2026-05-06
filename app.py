import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import threading
import concurrent.futures
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

# ── Face detector & thread pool ───────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
# Single-worker pool so inference never runs concurrently (TF isn't thread-safe)
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
_model_lock = threading.Lock()


def _run_inference(face_in: np.ndarray):
    """Runs model inference in a background thread."""
    with _model_lock:
        d_pred = drowsiness_model.predict(face_in, verbose=0)
        e_pred = emotion_model.predict(face_in, verbose=0)
    d_label = drowsiness_labels[np.argmax(d_pred)]
    e_label = emotion_labels[np.argmax(e_pred)]
    D = drowsiness_scores.get(d_label, 0)
    E = emotion_scores.get(e_label, 0)
    S = D + 0.7 * E
    safety = safety_matrix[d_label][e_label]
    return d_label, e_label, safety, S


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


# ── Video processor ───────────────────────────────────────────────────────────
class VideoProcessor:
    """
    Non-blocking video processor:
    - Face detection runs on a downscaled frame every frame (fast, CPU-cheap)
    - Model inference runs in a background thread; latest result is cached
    - Video frames are NEVER blocked waiting for inference
    """

    def __init__(self):
        self._future: concurrent.futures.Future | None = None
        self._last = ("N/A", "N/A", "N/A", 0.0)   # (drowsiness, emotion, safety, S)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h_orig, w_orig = img.shape[:2]

        # ── Scale down for faster face detection ──────────────────────────────
        scale = 0.5
        small = cv2.resize(img, (int(w_orig * scale), int(h_orig * scale)))
        gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray_small, scaleFactor=1.3, minNeighbors=5, minSize=(20, 20)
        )

        for (sx, sy, sw, sh) in faces:
            # Map coordinates back to original resolution
            x, y, w, h = (
                int(sx / scale), int(sy / scale),
                int(sw / scale), int(sh / scale)
            )

            # ── Non-blocking inference ────────────────────────────────────────
            # Collect result if previous inference finished
            if self._future is not None and self._future.done():
                try:
                    self._last = self._future.result()
                except Exception:
                    pass
                self._future = None

            # Submit new inference only if no job is running
            if self._future is None and drowsiness_model and emotion_model:
                face_roi = cv2.cvtColor(img[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
                face_in  = np.expand_dims(
                    np.expand_dims(cv2.resize(face_roi, (48, 48)), axis=-1),
                    axis=0
                ) / 255.0
                self._future = _executor.submit(_run_inference, face_in)

            d_label, e_label, safety, S = self._last

            # ── Annotate frame ────────────────────────────────────────────────
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, f"Drowsiness: {d_label}",
                        (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(img, f"Emotion: {e_label}",
                        (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 120, 0), 2)
            cv2.putText(img, f"Safety: {safety} (S={S:.2f})",
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
        media_stream_constraints={
            "video": {
                "width":  {"ideal": 640, "max": 640},
                "height": {"ideal": 480, "max": 480},
                "frameRate": {"ideal": 15, "max": 20},
            },
            "audio": False,
        },
        async_processing=True,
    )
else:
    st.markdown(
        "<h4 style='color:#909398; text-align:center; margin-top:60px;'>"
        "Click <b>Let's Start the Demo</b> to begin real-time monitoring.</h4>",
        unsafe_allow_html=True
    )