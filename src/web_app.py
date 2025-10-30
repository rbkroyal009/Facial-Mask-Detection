import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import requests
import os
from PIL import Image

# ==============================
# CONFIGURATION
# ==============================
MODEL_URL = "https://huggingface.co/rbkroyal/facial-mask-model/resolve/main/mask_detector.h5"
MODEL_PATH = "models/mask_detector.h5"
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
.main-title {
  text-align:center;
  font-size:2.3em;
  color:#222;
  font-weight:800;
}
.result-text {
  text-align: center;
  font-size: 1.5em;
  font-weight: bold;
  color: #222;
  margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# DOWNLOAD AND LOAD MODEL
# ==============================
@st.cache_resource
def load_model():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        st.info("📥 Downloading model from Hugging Face...")
        r = requests.get(MODEL_URL)
        if r.status_code != 200:
            st.error("❌ Failed to download model. Check Hugging Face link.")
            st.stop()
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
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
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    label_text = "No face detected"
    color_text = "black"

    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))
        face = np.expand_dims(face / 255.0, axis=0)
        pred = model.predict(face, verbose=0)[0][0]

        if pred < 0.5:
            label = "Mask ✅"
            color = (0, 255, 0)
            conf = f"{(1 - pred) * 100:.1f}%"
            label_text = "🟢 Mask detected"
            color_text = "green"
        else:
            label = "No Mask ❌"
            color = (255, 0, 0)
            conf = f"{pred * 100:.1f}%"
            label_text = "🔴 No mask detected"
            color_text = "red"

        cv2.putText(image, f"{label} ({conf})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    return image, label_text, color_text

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
        result_img, label_text, color_text = detect_mask(image_bgr, model)
        st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="Detection Result", use_container_width=True)
        st.markdown(f'<p class="result-text" style="color:{color_text};">{label_text}</p>', unsafe_allow_html=True)

# --- LIVE CAMERA TAB ---
with tab2:
    st.write("Toggle the checkbox below to start webcam.")
    run = st.checkbox("Start Camera")
    FRAME_WINDOW = st.image([])
    MESSAGE = st.empty()
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Webcam not detected. Please check camera permissions.")
            break
        frame = cv2.flip(frame, 1)
        frame, label_text, color_text = detect_mask(frame, model)
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        MESSAGE.markdown(f'<p class="result-text" style="color:{color_text};">{label_text}</p>', unsafe_allow_html=True)
    cap.release()
