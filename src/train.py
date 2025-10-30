import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os

# Paths
base_dir = r"C:\Users\91834\OneDrive\Desktop\Facial-Mask-Detection\data"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
model_path = r"C:\Users\91834\OneDrive\Desktop\Facial-Mask-Detection\models\mask_detector.h5"

# Parameters
IMG_SIZE = (224, 224)
BATCH = 32
EPOCHS = 10
LR = 1e-4

# Data generators with stronger augmentation
train_gen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True
)

val_gen = ImageDataGenerator(rescale=1.0/255)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode="binary"
)

val_data = val_gen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode="binary"
)

# Model (deeper and more accurate)
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(224,224,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=Adam(learning_rate=LR),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Training
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    model_path,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[checkpoint]
)

print(f"✅ Training complete. Model saved to: {model_path}")
