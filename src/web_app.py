import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os
import requests
from PIL import Image

# ==============================
# CONFIGURATION
# ==============================
MODEL_FILE_ID = "1Sqb6-kuClpMYhxbFas_PeobTaflw3-Sz"
MODEL_PATH = "models/mask_detector.h5"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# ==============================
# DOWNLOAD MODEL FUNCTION
# ==============================
def download_from_google_drive(file_id, destination):
    """Download file from Google Drive (handles large file confirmation)."""
    if os.path.exists(destination) and os.path.getsize(destination) > 1_000_000:
        return  # already downloaded

    st.info("📥 Downloading model from Google Drive...")
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

    content_type = response.headers.get("Content-Type", "")
    if "text/html" in content_type:
        st.error("❌ Google Drive returned an HTML page. Check file permissions or use a smaller file.")
        st.stop()

    CHUNK_SIZE = 32768
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

    if os.path.getsize(destination) < 1_000_000:
        st.error("❌ Downloaded file too small — likely not the real model.")
        st.stop()


# ==============================
# LOAD MODEL WITH CACHE
# ==============================
@st.cache_resource
def load_model():
    download_from_google_drive(MODEL_FILE_ID, MODEL_PATH)
    if not os.path.exists(MODEL_PATH):
        st.error("Model not found after download.")
        st.stop()
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


# ==============================
# MASK DETECTION FUNCTION
# ==============================
def detect_mask(image, model):
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))
        face = face / 255.0
        face = np.expand_dims(face, axis=0)
        pred = model.predict(face, verbose=0)[0][0]

        if pred < 0.5:
            label = "😷 Mask"
            color = (0, 255, 0)
            conf = f"{(1 - pred) * 100:.1f}%"
        else:
            label = "❌ No Mask"
            color = (0, 0, 255)
            conf = f"{pred * 100:.1f}%"

        cv2.putText(image, f"{label} ({conf})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    return image


# ==============================
# STREAMLIT UI
# ==============================
st.set_page_config(page_title="Face Mask Detection", page_icon="😷", layout="wide")

st.title("😷 Real-Time Face Mask Detection")
st.markdown("""
This app detects whether a person is wearing a face mask or not using **TensorFlow + OpenCV**.
""")

model = load_model()

tab1, tab2 = st.tabs(["📸 Upload Image", "🎥 Live Camera"])

# --- IMAGE UPLOAD ---
with tab1:
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image.convert("RGB"))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        result_img = detect_mask(image, model)
        st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="Result", use_container_width=True)

# --- LIVE CAMERA ---
with tab2:
    st.write("Click **Start** to open webcam.")
    run = st.checkbox("Start Camera")
    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)
    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Webcam not detected. Please check your camera.")
            break
        frame = cv2.flip(frame, 1)
        frame = detect_mask(frame, model)
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

st.success("✅ App loaded successfully!")
