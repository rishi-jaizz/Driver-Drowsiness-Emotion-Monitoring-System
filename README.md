<div align="center">

# 🚗 Driver Drowsiness & Emotion Monitoring System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenCV-4.10-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-1.35-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Keras-CNN-D00000?style=for-the-badge&logo=keras&logoColor=white"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Accuracy-~95%25-brightgreen?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Models-Dual--Pipeline-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Real--Time-Webcam-orange?style=for-the-badge"/>
</p>

> **A real-time AI-powered system that monitors driver fatigue and emotional state using dual CNN pipelines, providing live safety scoring and instant alerts to prevent road accidents.**

</div>

---

## 📌 Overview

The **Driver Drowsiness & Emotion Monitoring System** is a real-time computer vision application that detects driver fatigue and emotional distraction using live webcam input. Built with a **Dual-Pipeline CNN Architecture**, it simultaneously runs:

- 🔴 **Pipeline 1 — Drowsiness Detection**: Classifies driver state as Alert, Yawning, or Microsleep
- 🟡 **Pipeline 2 — Emotion Recognition**: Identifies driver emotion as Happy, Sad, Angry, or Neutral

Both outputs are fused into a unified **Safety Score** that determines whether the driver is in a safe, neutral, or unsafe state — providing real-time alerts to help prevent accidents.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎯 **Real-Time Face Detection** | Haar Cascade classifier for fast, lightweight face localization |
| 😴 **Drowsiness Detection** | Classifies Alert / Yawning / Microsleep states |
| 😊 **Emotion Recognition** | Detects Happy / Sad / Angry / Neutral emotions |
| 🧠 **Dual CNN Models** | Separate deep learning models for each detection task |
| 📊 **Safety Score** | Fused scoring formula: `S = D + 0.7 × E` |
| 🔔 **Real-Time Alerts** | Instant on-screen warnings for unsafe driving conditions |
| 🖥️ **Streamlit Interface** | Clean, interactive web UI with live webcam feed |
| ⚡ **Lightweight Deployment** | Optimized for real-time inference on standard hardware |

---

## 🛠️ Tech Stack

<table>
<tr>
  <td><b>Language</b></td>
  <td>Python 3.11</td>
</tr>
<tr>
  <td><b>Deep Learning</b></td>
  <td>TensorFlow 2.15 / Keras — CNN models</td>
</tr>
<tr>
  <td><b>Computer Vision</b></td>
  <td>OpenCV 4.10, Haar Cascade Classifier</td>
</tr>
<tr>
  <td><b>UI Framework</b></td>
  <td>Streamlit 1.35</td>
</tr>
<tr>
  <td><b>Data Processing</b></td>
  <td>NumPy, Pandas, Pillow</td>
</tr>
<tr>
  <td><b>Visualization</b></td>
  <td>Matplotlib, Seaborn</td>
</tr>
<tr>
  <td><b>Model Training</b></td>
  <td>Jupyter Notebook, Scikit-learn</td>
</tr>
</table>

---

## 📂 Project Structure

```
Driver Drowsiness & Emotion Monitoring System/
│
├── 📄 app.py                        # Streamlit application (main entry point)
├── 📄 main.py                       # Standalone OpenCV inference script
├── 📄 requirements.txt              # Python dependencies
├── 📄 runtime.txt                   # Python runtime specification
├── 📄 .python-version               # Python version pin (3.11)
├── 📄 .gitignore                    # Git ignore rules
│
├── 📁 models/                       # Trained model weights
│   ├── fl3d_model_whts_3.h5         # Drowsiness detection model (weights)
│   ├── fl3d_model_3.keras           # Drowsiness detection model (full)
│   ├── affectnet_model_whts_2.h5    # Emotion recognition model (weights)
│   └── affectnet_model_2.keras      # Emotion recognition model (full)
│
├── 📁 training/                     # Model training notebooks & scripts
│   ├── drowsiness/
│   │   └── model-training-fl3d.ipynb        # FL3D drowsiness training notebook
│   └── emotion/
│       ├── model-training-affectnet.ipynb   # AffectNet emotion training notebook
│       └── train_affectnet.py               # Emotion model training script
│
└── 📁 data/                         # Dataset directories (not tracked in git)
    ├── drowsiness_dataset/           # FL3D dataset images
    └── emotion_dataset/              # AffectNet subset images
```

---

## 📊 Datasets

### 1. Drowsiness Detection — FL3D Dataset *(derived from NITYMED)*

| Property | Details |
|---|---|
| **Total Images** | 53,331 |
| **Subjects** | 21 |
| **Alert** | 38,954 images |
| **Microsleep** | 8,880 images |
| **Yawning** | 5,497 images |

### 2. Emotion Recognition — AffectNet Dataset *(Subset)*

| Class | Description |
|---|---|
| 😊 **Happy** | Positive emotional state |
| 😢 **Sad** | Low mood, potential distraction |
| 😠 **Angry** | High arousal, aggressive driving risk |
| 😐 **Neutral** | Baseline emotional state |

---

## 🧠 Model Architecture

Both models follow a deep **Convolutional Neural Network (CNN)** architecture:

```
Input (48×48×1 Grayscale)
        │
    ┌───▼────┐
    │ Conv2D │  ← Feature extraction
    │  + BN  │
    │ + ReLU │
    └───┬────┘
        │
    ┌───▼────┐
    │MaxPool │  ← Spatial downsampling
    └───┬────┘
        │
   [Repeated blocks]
        │
    ┌───▼────┐
    │Dropout │  ← Regularization
    └───┬────┘
        │
    ┌───▼──────┐
    │  Dense   │  ← Classification head
    │ Softmax  │
    └──────────┘
         │
    Output Classes
```

| Layer Type | Purpose |
|---|---|
| **Convolution** | Spatial feature extraction |
| **Batch Normalization** | Training stability |
| **Max Pooling** | Dimensionality reduction |
| **Dropout** | Overfitting prevention |
| **Dense + Softmax** | Multi-class classification |

### Dual-Pipeline Architecture

```
 Webcam Feed
      │
  ┌───▼──────────────────┐
  │  Face Detection       │  ← Haar Cascade Classifier
  │  (Haar Cascade)       │
  └───┬──────────────────┘
      │
      ├─────────────────────────────────────┐
      │                                     │
  ┌───▼──────────────┐           ┌──────────▼──────────┐
  │ Pipeline 1       │           │ Pipeline 2           │
  │ Drowsiness CNN   │           │ Emotion CNN          │
  │                  │           │                      │
  │ Alert            │           │ Happy                │
  │ Yawning          │           │ Sad                  │
  │ Microsleep       │           │ Angry                │
  └───┬──────────────┘           │ Neutral              │
      │                          └──────────┬───────────┘
      │                                     │
      └──────────────┬──────────────────────┘
                     │
             ┌───────▼────────┐
             │ Safety Score   │
             │ S = D + 0.7×E  │
             └───────┬────────┘
                     │
          ┌──────────▼──────────┐
          │  Safety State Alert  │
          │  Safe / Neutral /    │
          │  Unsafe              │
          └─────────────────────┘
```

---

## 📊 Safety Score System

The system computes a unified safety score by combining both model outputs:

$$S = D + 0.7 \times E$$

Where:
- **D** = Drowsiness score
- **E** = Emotion score
- **0.7** = Emotion weighting factor

### Score Lookup Table

| Drowsiness | D Score | Emotion | E Score |
|---|---|---|---|
| Alert | +1 | Happy | +1 |
| Yawning | 0 | Neutral | 0 |
| Microsleep | -1 | Sad | -1 |
| — | — | Angry | -1 |

### Safety State Matrix

| | 😠 Angry | 😊 Happy | 😐 Neutral | 😢 Sad |
|---|---|---|---|---|
| **Alert** | 🟡 Neutral | 🟢 Safe | 🟢 Safe | 🟡 Neutral |
| **Yawning** | 🔴 Unsafe | 🟡 Neutral | 🟡 Neutral | 🔴 Unsafe |
| **Microsleep** | 🔴 Unsafe | 🔴 Unsafe | 🔴 Unsafe | 🔴 Unsafe |

---

## ⚙️ Preprocessing Pipeline

```
Raw Webcam Frame
       │
   ┌───▼──────────────────┐
   │ Grayscale Conversion  │  cv2.COLOR_BGR2GRAY
   └───┬──────────────────┘
       │
   ┌───▼──────────────────┐
   │ Face Detection        │  Haar Cascade (scaleFactor=1.3)
   └───┬──────────────────┘
       │
   ┌───▼──────────────────┐
   │ ROI Extraction        │  Crop face region
   └───┬──────────────────┘
       │
   ┌───▼──────────────────┐
   │ Resize to 48×48       │  cv2.resize()
   └───┬──────────────────┘
       │
   ┌───▼──────────────────┐
   │ Normalize [0–1]       │  pixel / 255.0
   └───┬──────────────────┘
       │
   CNN Inference
```

---

## 📈 Results

| Metric | Value |
|---|---|
| 🎯 **Drowsiness Detection Accuracy** | ~95% |
| ⚡ **Inference Mode** | Real-time (live webcam) |
| 🏋️ **Model Size** | ~6 MB per model |
| 📐 **Input Resolution** | 48 × 48 pixels |
| 🖼️ **Input Channels** | Grayscale (1 channel) |

---

## 🚀 Getting Started

### Prerequisites

- Python **3.11**
- A working **webcam**
- macOS / Linux / Windows

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Driver-Drowsiness-Emotion-Monitoring-System.git
cd Driver-Drowsiness-Emotion-Monitoring-System
```

### 2. Create & Activate Virtual Environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App

```bash
streamlit run app.py
```

Then open your browser at **http://localhost:8501**

### 5. (Optional) Run Standalone OpenCV Script

```bash
python main.py
```
> Press **`q`** to quit the OpenCV window.

---

## 🖥️ Usage

1. Launch the app with `streamlit run app.py`
2. Click **"Let's Start the Demo"** to activate the webcam
3. Allow camera permissions when prompted
4. The system will:
   - Detect your face in real-time
   - Display **Drowsiness** and **Emotion** labels
   - Calculate and show the **Safety Score**
   - Highlight safety state: 🟢 Safe / 🟡 Neutral / 🔴 Unsafe
5. Click **"Stop Demo"** to end the session

---

## 📦 Dependencies

```txt
streamlit==1.35.0
opencv-python-headless==4.10.0.84
numpy==1.26.4
pandas==2.2.2
matplotlib==3.8.4
seaborn==0.13.2
tensorflow==2.15.1
scikit-learn==1.4.2
pillow==10.3.0
```

---

## 🧪 Training

Training notebooks are available in the `training/` directory:

| Model | Notebook | Dataset |
|---|---|---|
| Drowsiness CNN | `training/drowsiness/model-training-fl3d.ipynb` | FL3D (NITYMED) |
| Emotion CNN | `training/emotion/model-training-affectnet.ipynb` | AffectNet (Subset) |

You can also retrain the emotion model via script:
```bash
python training/emotion/train_affectnet.py
```

---

## 🔮 Future Enhancements

- [ ] 🔊 Audio alerts (beeps/voice warnings) for unsafe states
- [ ] 📱 Mobile deployment via TensorFlow Lite
- [ ] 🌙 Night-mode support with IR camera compatibility
- [ ] 📊 Session logging and historical safety analytics
- [ ] 🚘 Integration with ADAS (Advanced Driver Assistance Systems)
- [ ] 🌍 Multi-face tracking for fleet/ride-share monitoring
- [ ] 📡 Cloud dashboard for fleet managers

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- **FL3D Dataset** — Derived from the NITYMED dataset for drowsiness classification
- **AffectNet** — Large-scale facial expression dataset for emotion recognition
- **OpenCV** — Open Source Computer Vision Library
- **TensorFlow / Keras** — Deep learning framework
- **Streamlit** — Interactive web app framework for ML applications

---

<div align="center">

**⭐ If you found this project useful, please give it a star!**

Made with ❤️ using Python, TensorFlow & OpenCV

</div>
