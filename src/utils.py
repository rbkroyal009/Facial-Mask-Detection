# src/utils.py
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

# load OpenCV Haar cascade (bundled with opencv)
def get_face_detector():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def preprocess_face(face, target_size=(224,224)):
    # face: BGR image (OpenCV)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, target_size)
    face = face.astype("float32") / 255.0
    face = img_to_array(face)
    return np.expand_dims(face, axis=0)

def draw_label_box(image, box, label, confidence, color):
    (x, y, w, h) = box
    cv2.rectangle(image, (x,y), (x+w, y+h), color, 2)
    text = f"{label}: {confidence:.2f}"
    cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return image
