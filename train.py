import numpy as np
import os
import cv2

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# ---------------- PATHS ----------------
train_path = "Dataset/archive/skindisease/Train"
test_path = "Dataset/archive/skindisease/Test"

IMG_SIZE = 128

X_train, y_train = [], []
X_test, y_test = [], []

# AUTO CLASS LIST
classes = sorted(os.listdir(train_path))
print("Classes found:", classes)

# ---------------- LOAD TRAIN ----------------
for label in classes:
    folder = os.path.join(train_path, label)

    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)

        img = cv2.imread(img_path)

        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X_train.append(img)
            y_train.append(label)

# ---------------- LOAD TEST ----------------
for label in classes:
    folder = os.path.join(test_path, label)

    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)

        img = cv2.imread(img_path)

        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X_test.append(img)
            y_test.append(label)

# ---------------- CONVERT ----------------
X_train = np.array(X_train) / 255.0
X_test = np.array(X_test) / 255.0

# LABEL ENCODING
le = LabelEncoder()

y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

num_classes = len(classes)

# ---------------- MODEL ----------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ---------------- TRAIN ----------------
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=32
)

# ---------------- SAVE ----------------
os.makedirs("model", exist_ok=True)
model.save("model/skin_model.h5")

# SAVE CLASS NAMES (IMPORTANT FIX)
np.save("model/classes.npy", classes)

print("MODEL TRAINING COMPLETE")