from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from flask import Flask, render_template, request
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Brain_tumer.keras"
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMAGE_SIZE = (224, 224)
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

model = load_model(MODEL_PATH)


def prepare_image(image_bytes: bytes) -> np.ndarray:
    """Prepare an uploaded image using the notebook's BGR, 224x224, /255 pipeline."""
    with Image.open(BytesIO(image_bytes)) as image:
        rgb_image = image.convert("RGB").resize(
            IMAGE_SIZE, resample=Image.Resampling.BILINEAR
        )
        image_array = np.asarray(rgb_image, dtype=np.float32)

    bgr_image = image_array[:, :, ::-1]
    return np.expand_dims(bgr_image / 255.0, axis=0)


@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    prediction = None

    if request.method == "POST":
        uploaded_file = request.files.get("mri_image")
        if uploaded_file is None or not uploaded_file.filename:
            error = "Choose an MRI image to continue."
        elif Path(uploaded_file.filename).suffix.lower() not in ALLOWED_EXTENSIONS:
            error = "Upload a JPG, JPEG, PNG, or BMP image."
        else:
            image_bytes = uploaded_file.read(MAX_UPLOAD_BYTES + 1)
            if not image_bytes:
                error = "The selected image is empty."
            elif len(image_bytes) > MAX_UPLOAD_BYTES:
                error = "The image must be 10 MB or smaller."
            else:
                try:
                    input_batch = prepare_image(image_bytes)
                except (UnidentifiedImageError, OSError, ValueError):
                    error = "The uploaded file is not a valid image."
                else:
                    probabilities = np.asarray(model.predict(input_batch, verbose=0))
                    if probabilities.shape != (1, len(CLASS_NAMES)):
                        raise ValueError(
                            "Model output shape does not match the four configured classes: "
                            f"{probabilities.shape}"
                        )
                    if not np.isfinite(probabilities).all():
                        raise ValueError("Model returned non-finite prediction values.")

                    class_index = int(np.argmax(probabilities[0]))
                    prediction = {
                        "label": CLASS_NAMES[class_index],
                        "confidence": float(probabilities[0, class_index]) * 100,
                    }

    return render_template("index.html", error=error, prediction=prediction)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
