import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import time

# Safe model loading
def safe_load_model(model_path):
    try:
        return load_model(model_path)
    except Exception as e:
        return None

# Load models
drowsiness_model = safe_load_model('models/fl3d_model_whts_3.h5')
emotion_model = safe_load_model('affectnet_model_whts_2.h5')streamlit run app.py


# Labels
drowsiness_labels = ['alert', 'microsleep', 'yawning']
emotion_labels = ['angry', 'happy', 'neutral', 'sad']

# Risk scores
drowsiness_scores = {'alert': 1, 'yawning': 0, 'microsleep': -1}
emotion_scores = {'neutral': 0, 'happy': 1, 'sad': -1, 'angry': -1}

# Safety matrix
safety_matrix = {
    'alert':    {'angry': 'Neutral', 'happy': 'Safe',    'neutral': 'Safe',    'sad': 'Neutral'},
    'yawning':  {'angry': 'Unsafe',  'happy': 'Neutral', 'neutral': 'Neutral', 'sad': 'Unsafe'},
    'microsleep': {'angry': 'Unsafe', 'happy': 'Unsafe', 'neutral': 'Unsafe', 'sad': 'Unsafe'}
}

# Face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# UI styling
st.set_page_config(page_title="EMOTE-ION", layout="centered")
st.markdown("""
    <style>
        body { background-color: #03040B; color: #909398; }
        .title { font-size: 55px; color: #ffffff; font-weight: bold; display: flex; align-items: center; }
        .desc { font-size: 24px; color: #909398; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">EMOTE-ION</div>', unsafe_allow_html=True)
st.markdown('<div class="desc">Experience real-time drowsiness and emotion detection using advanced AI models. Click below to start the demo.</div>', unsafe_allow_html=True)

# Buttons
col1, col2 = st.columns(2)
with col1:
    start_demo = st.button("Let's Start the Demo")
with col2:
    stop_demo = st.button("Stop Demo")

# State tracking
if "camera_active" not in st.session_state:
    st.session_state.camera_active = False
    st.session_state.show_thanks = False

if start_demo:
    st.session_state.camera_active = True
    st.session_state.show_thanks = False
if stop_demo:
    st.session_state.camera_active = False
    st.session_state.show_thanks = True
    st.rerun()

frame_placeholder = st.empty()

if st.session_state.camera_active:
    cap = cv2.VideoCapture(0)
    while st.session_state.camera_active:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access camera")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (48, 48))
            face_resized = np.expand_dims(face_resized, axis=-1)
            face_resized = np.expand_dims(face_resized, axis=0) / 255.0

            if drowsiness_model and emotion_model:
                drowsiness_pred = drowsiness_model.predict(face_resized)
                drowsiness_label = drowsiness_labels[np.argmax(drowsiness_pred)]

                emotion_pred = emotion_model.predict(face_resized)
                emotion_label = emotion_labels[np.argmax(emotion_pred)]

                # Safety Score Calculation
                D = drowsiness_scores.get(drowsiness_label, 0)
                E = emotion_scores.get(emotion_label, 0)
                S = D + 0.7 * E
                safety_state = safety_matrix[drowsiness_label][emotion_label]

            else:
                drowsiness_label = emotion_label = safety_state = "N/A"
                S = 0

            # Drawing
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Drowsiness: {drowsiness_label}", (x, y-50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Emotion: {emotion_label}", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(frame, f"Safety: {safety_state} (S={S:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame, channels="RGB")
    
    cap.release()
    cv2.destroyAllWindows()
elif st.session_state.show_thanks:
    st.markdown("<h3 style='color:#ffffff;'>Thanks for Trying</h3>", unsafe_allow_html=True)