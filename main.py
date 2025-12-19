import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained models
drowsiness_model = load_model('fl3d_model_whts_3.h5')
emotion_model = load_model('affectnet_model_whts_2.h5')

# Define emotion and drowsiness labels
drowsiness_labels = ['alert', 'microsleep', 'yawning']
emotion_labels = ['angry', 'happy', 'neutral', 'sad']

# Define risk scores
drowsiness_scores = {'alert': 1, 'yawning': 0, 'microsleep': -1}
emotion_scores = {'neutral': 0, 'happy': 1, 'sad': -1, 'angry': -1}

# Safety Matrix: Drowsiness vs Emotion
safety_matrix = {
    'alert': {'angry': 'Neutral', 'happy': 'Safe', 'neutral': 'Safe', 'sad': 'Neutral'},
    'yawning': {'angry': 'Unsafe', 'happy': 'Neutral', 'neutral': 'Neutral', 'sad': 'Unsafe'},
    'microsleep': {'angry': 'Unsafe', 'happy': 'Unsafe', 'neutral': 'Unsafe', 'sad': 'Unsafe'}
}

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract face ROI
        face = gray[y:y+h, x:x+w]

        # Preprocess face
        face_resized = cv2.resize(face, (48, 48))
        face_resized = np.expand_dims(face_resized, axis=-1)
        face_resized = np.expand_dims(face_resized, axis=0)
        face_resized = face_resized / 255.0

        # Predictions
        drowsiness_pred = drowsiness_model.predict(face_resized)
        emotion_pred = emotion_model.predict(face_resized)

        drowsiness_label = drowsiness_labels[np.argmax(drowsiness_pred)]
        emotion_label = emotion_labels[np.argmax(emotion_pred)]

        # Compute Safety Score
        D = drowsiness_scores.get(drowsiness_label, 0)
        E = emotion_scores.get(emotion_label, 0)
        S = D + 0.7 * E

        # Get Safety State from Matrix
        safety_state = safety_matrix[drowsiness_label][emotion_label]

        # Draw results on frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Drowsiness: {drowsiness_label}", (x, y-60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Emotion: {emotion_label}", (x, y-40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Safety: {safety_state} (S={S:.2f})", (x, y-20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Display the frame
    cv2.imshow('Drowsiness & Emotion Detection with Safety Score', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
