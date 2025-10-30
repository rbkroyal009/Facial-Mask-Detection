import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load trained model and face detector
model = load_model(r"C:\Users\91834\OneDrive\Desktop\Facial-Mask-Detection\models\mask_detector.h5")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print("✅ Webcam started. Press 'q' to quit.")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Could not access webcam.")
        break

    # Flip camera for mirror effect
    frame = cv2.flip(frame, 1)

    # Create a copy for drawing
    output = frame.copy()

    # Add header bar
    cv2.rectangle(output, (0, 0), (output.shape[1], 60), (40, 40, 40), -1)
    cv2.putText(output, "FACIAL MASK DETECTOR - LIVE", (50, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face) / 255.0
        face = np.expand_dims(face, axis=0)

        # Get prediction
        mask_prob = float(model.predict(face, verbose=0)[0][0])

        label = "Mask" if mask_prob > 0.5 else "No Mask"
        confidence = mask_prob if label == "Mask" else 1 - mask_prob
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # Draw bounding box and text
        cv2.rectangle(output, (x, y), (x+w, y+h), color, 3)
        cv2.rectangle(output, (x, y-40), (x+w, y), color, -1)
        cv2.putText(output, f"{label} ({confidence*100:.1f}%)", (x+10, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Show output window
    cv2.imshow("Facial Mask Detection", output)

    # Exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
