import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# ---------- CONFIG ----------
st.set_page_config(page_title="Driver State Monitor",
                   page_icon="🚗",
                   layout="wide")

drowsiness_model = load_model('models/fl3d_model_whts_3.h5')
emotion_model    = load_model('models/affectnet_model_whts_2.h5')

drowsiness_labels = ['alert', 'microsleep', 'yawning']
emotion_labels    = ['angry', 'happy', 'neutral', 'sad']

drowsiness_scores = {'alert': 1, 'yawning': 0, 'microsleep': -1}
emotion_scores    = {'neutral': 0, 'happy': 1, 'sad': -1, 'angry': -1}

safety_matrix = {
    'alert':     {'angry': 'Neutral', 'happy': 'Safe', 'neutral': 'Safe', 'sad': 'Neutral'},
    'yawning':   {'angry': 'Unsafe', 'happy': 'Neutral', 'neutral': 'Neutral', 'sad': 'Unsafe'},
    'microsleep':{'angry': 'Unsafe', 'happy': 'Unsafe', 'neutral': 'Unsafe', 'sad': 'Unsafe'}
}

MICROSLEEP_THRESHOLD = 0.6

# Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ---------- Helper to run both models on a face ----------
def analyze_face(gray_face):
    face_resized = cv2.resize(gray_face, (48, 48))
    face_resized = face_resized.astype("float32") / 255.0
    face_resized = np.expand_dims(face_resized, -1)   # (48,48,1)
    face_resized = np.expand_dims(face_resized, 0)    # (1,48,48,1)

    d_pred = drowsiness_model.predict(face_resized, verbose=0)[0]
    e_pred = emotion_model.predict(face_resized,    verbose=0)[0]

    d_idx = int(np.argmax(d_pred))
    e_idx = int(np.argmax(e_pred))

    d_label = drowsiness_labels[d_idx]
    e_label = emotion_labels[e_idx]

    microsleep_idx = drowsiness_labels.index('microsleep')
    if d_pred[microsleep_idx] > MICROSLEEP_THRESHOLD:
        d_label = 'microsleep'

    D = drowsiness_scores.get(d_label, 0)
    E = emotion_scores.get(e_label, 0)
    S = D + 0.7 * E
    safety_state = safety_matrix[d_label][e_label]
    unsafe_flag = (safety_state == 'Unsafe') or (S < 0)

    return d_label, e_label, S, safety_state, unsafe_flag

def color_for_state(safety_state):
    if safety_state == 'Safe':
        return (0, 255, 0)      # green
    elif safety_state == 'Neutral':
        return (0, 255, 255)    # yellow
    else:
        return (0, 0, 255)      # red

# ---------- UI ----------
st.title("🚗 Driver Drowsiness & Emotion Monitor")
st.markdown("""
Monitor driver state using **drowsiness** and **emotion** detection.
- Green = Safe  
- Yellow = Neutral  
- Red = Unsafe  
""")

uploaded_file = st.file_uploader("Upload a face / driver image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30,30))

    if len(faces) == 0:
        st.warning("No face detected.")
        st.image(img_rgb, caption="Uploaded image", use_column_width=True)
    else:
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            d_label, e_label, S, safety_state, unsafe_flag = analyze_face(face)

            color = color_for_state(safety_state)
            cv2.rectangle(img_rgb, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img_rgb, f"{d_label}, {e_label}, {safety_state}",
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            st.subheader("Detection Results")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Drowsiness", d_label)
            c2.metric("Emotion", e_label)
            c3.metric("Safety", safety_state)
            c4.metric("Score S", f"{S:.2f}")

        st.image(img_rgb, caption="Analysis", use_column_width=True)
