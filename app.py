import os
import threading
import webbrowser
import numpy as np

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ---------------- APP ----------------
app = Flask(__name__)

# ---------------- PATHS ----------------
MODEL_PATH = "model/skin_model.h5"
CLASSES_PATH = "model/classes.npy"

UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ---------------- LOAD MODEL ----------------
model = load_model(MODEL_PATH)

# ---------------- LOAD CLASSES ----------------
CLASS_NAMES = np.load(CLASSES_PATH, allow_pickle=True)

print("Classes Loaded:")
print(CLASS_NAMES)

# ---------------- IMAGE PREPROCESS ----------------
def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ---------------- HOME ----------------
@app.route("/")
def home():
    return render_template("index.html")

# ---------------- PREDICT ----------------
@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return "No file uploaded"

    file = request.files["file"]

    if file.filename == "":
        return "No file selected"

    # Save uploaded image
    filename = file.filename

    file_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(file_path)

    print("Saved Image:", file_path)

    # Predict
    img = prepare_image(file_path)

    prediction = model.predict(img)

    class_index = np.argmax(prediction)

    result = CLASS_NAMES[class_index]

    confidence = round(
        float(np.max(prediction)) * 100,
        2
    )

    return render_template(
        "result.html",
        result=result,
        confidence=confidence,
        image_path=filename
    )

# ---------------- AUTO OPEN BROWSER ----------------
def open_browser():
    webbrowser.open_new(
        "http://127.0.0.1:5000/"
    )

# ---------------- RUN ----------------
if __name__ == "__main__":

    threading.Timer(
        1.5,
        open_browser
    ).start()

    app.run(
        debug=True,
        use_reloader=False
    )