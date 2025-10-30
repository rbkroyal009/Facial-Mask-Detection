# 😷 Facial Mask Detection Streamlit App
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
import requests

# ---------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------
MODEL_PATH = "models/mask_detector.h5"
DRIVE_ID = "YOUR_GOOGLE_DRIVE_FILE_ID_HERE"  # 👈 Replace this
MODEL_URL = f"https://drive.google.com/uc?export=download&id={DRIVE_ID}"

st.set_page_config(page_title="😷 Facial Mask Detection", layout="centered")

# ---------------------------------------------------
# UI THEME / DESIGN
# ---------------------------------------------------
st.markdown("""
    <style>
    body { background-color: #f9fafb; }
    .main {
        background-color: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0px 0px 15px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

st.title("😷 Facial Mask Detection App")
st.markdown("Detect if a person is wearing a **mask** or **not** in real time or from images.")

st.write("✅ App loaded successfully!")  # test message

# ---------------------------------------------------
# GOOGLE DRIVE MODEL DOWNLOAD
# ---------------------------------------------------
def download_from_google_drive(file_id, destination):
    """Downloads file from Google Drive, even large ones."""
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.info("📥 Downloading model from Google Drive... please wait.")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        try:
            download_from_google_drive(DRIVE_ID, MODEL_PATH)
            st.success("✅ Model downloaded successfully!")
        except Exception as e:
            st.error(f"❌ Model download failed: {e}")
            return None
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()
if model is None:
    st.stop()

# ---------------------------------------------------
# DETECTION HELPERS
# ---------------------------------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_mask(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        resized = cv2.resize(face_img, (224, 224))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 224, 224, 3))
        prediction = model.predict(reshaped, verbose=0)[0][0]

        label = "Mask" if prediction > 0.5 else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{label}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return frame

# ---------------------------------------------------
# APP INTERFACE
# ---------------------------------------------------
mode = st.radio("Select Mode", ["📷 Upload Image", "🎥 Live Webcam"])

# ----- IMAGE UPLOAD -----
if mode == "📷 Upload Image":
    uploaded_file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = np.array(Image.open(uploaded_file))
        st.image(img, caption="Uploaded Image", use_column_width=True)
        processed_img = detect_mask(img.copy())
        st.image(processed_img, caption="Detection Result", use_column_width=True, channels="BGR")

# ----- LIVE WEBCAM -----
elif mode == "🎥 Live Webcam":
    run = st.checkbox("Start Webcam")
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    while run:
        ret, frame = camera.read()
        if not ret:
            st.error("Could not access webcam.")
            break
        frame = cv2.flip(frame, 1)
        detected = detect_mask(frame)
        FRAME_WINDOW.image(cv2.cvtColor(detected, cv2.COLOR_BGR2RGB))

    camera.release()
    st.success("Webcam stopped.")
