import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os
import requests
import zipfile
from PIL import Image

# ==============================
# CONFIGURATION
# ==============================
MODEL_FILE_ID = "1Sqb6-kuClpMYhxbFas_PeobTaflw3-Sz"  # your Google Drive ID
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "mask_detector.h5")
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# ==============================
# PAGE STYLE
# ==============================
st.set_page_config(page_title="😷 Face Mask Detection", page_icon="🧠", layout="wide")
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
  background: linear-gradient(120deg, #89f7fe, #66a6ff);
}
.stButton>button {
  background: #6a11cb;
  background-image: linear-gradient(315deg, #6a11cb 0%, #2575fc 74%);
  color: white;
  border-radius: 10px;
  font-weight: bold;
  padding: 10px 25px;
  border: none;
}
.stButton>button:hover {
  transform: scale(1.05);
}
.main-title {text-align:center;font-size:2.3em;color:#222;font-weight:800;}
</style>
""", unsafe_allow_html=True)

# ==============================
# DOWNLOAD FROM GOOGLE DRIVE
# ==============================
def download_from_google_drive(file_id, destination):
    """Download large file from Google Drive and handle zip/unzip if needed."""
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    url = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(url, params={"id": file_id}, stream=True)
    token = None
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            token = v
    if token:
        response = session.get(url, params={"id": file_id, "confirm": token}, stream=True)

    # Detect if returned HTML instead of binary
    if "text/html" in response.headers.get("Content-Type", ""):
        st.error("❌ Google Drive returned an HTML page — file might be private or too large.")
        st.stop()

    tmp_path = destination
    with open(tmp_path, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

    # If zip, unzip
    if zipfile.is_zipfile(tmp_path):
        with zipfile.ZipFile(tmp_path, "r") as zip_ref:
            zip_ref.extractall(MODEL_DIR)
        os.remove(tmp_path)

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.info("📥 Downloading model from Google Drive...")
        download_from_google_drive(MODEL_FILE_ID, MODEL_PATH)
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

# ==============================
# DETECTION FUNCTION
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
            label, color, conf = "😷 Mask", (0, 255, 0), f"{(1-pred)*100:.1f}%"
        else:
            label, color, conf = "❌ No Mask", (255, 0, 0), f"{pred*100:.1f}%"
        cv2.putText(image, f"{label} ({conf})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    return image

# ==============================
# STREAMLIT UI
# ==============================
st.markdown('<h1 class="main-title">😷 Real-Time Face Mask Detection</h1>', unsafe_allow_html=True)
st.write("Detects whether a person is wearing a mask or not using TensorFlow + OpenCV.")

model = load_model()

tab1, tab2 = st.tabs(["📸 Upload Image", "🎥 Live Camera"])

# --- UPLOAD IMAGE TAB ---
with tab1:
    uploaded_file = st.file_uploader("Upload an image (JPG/PNG):", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = np.array(Image.open(uploaded_file).convert("RGB"))
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        result_img = detect_mask(image_bgr, model)
        st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="Detection Result", use_container_width=True)

# --- LIVE CAMERA TAB ---
with tab2:
    st.write("Toggle the checkbox below to start webcam.")
    run = st.checkbox("Start Camera")
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)
    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Webcam not detected. Please check camera permissions.")
            break
        frame = cv2.flip(frame, 1)
        frame = detect_mask(frame, model)
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
