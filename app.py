"""Lokalna aplikacja WWW do klasyfikacji czynności w filmach.

Uruchomienie:
    python app.py

Następnie otwórz http://127.0.0.1:5000 w przeglądarce.
"""

import base64
import binascii
import json
import tempfile
import threading
from pathlib import Path

import cv2 as cv
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from model import VideoAugmentation
from video_loader import frames as FRAME_COUNT
from video_loader import load_video


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"
ALLOWED_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".webm"}
MAX_UPLOAD_BYTES = 250 * 1024 * 1024  # 250 MB

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

_model = None
_class_names = None
_model_lock = threading.Lock()


def get_model_and_classes():
    """Wczytuje model tylko przy pierwszym żądaniu klasyfikacji."""
    global _model, _class_names
    if _model is None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"Nie znaleziono modelu: {MODEL_PATH.name}")
        if not CLASS_NAMES_PATH.is_file():
            raise FileNotFoundError(f"Nie znaleziono pliku klas: {CLASS_NAMES_PATH.name}")

        _model = tf.keras.models.load_model(
            MODEL_PATH, custom_objects={"VideoAugmentation": VideoAugmentation}
        )
        _class_names = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
        if _model.output_shape[-1] != len(_class_names):
            raise ValueError("Liczba klas w modelu nie zgadza się z class_names.json.")
    return _model, _class_names


def predict(video: np.ndarray) -> list[dict[str, float | str]]:
    """Zwraca pięć najbardziej prawdopodobnych klas dla jednego filmu."""
    if video.shape != (FRAME_COUNT, 32, 32, 3):
        raise ValueError(
            f"Nieprawidłowy kształt danych {video.shape}; oczekiwano "
            f"({FRAME_COUNT}, 32, 32, 3)."
        )

    model, class_names = get_model_and_classes()
    # TensorFlow/Keras może obsługiwać równolegle więcej niż jedno żądanie;
    # blokada upraszcza stabilne działanie lokalnego serwera.
    with _model_lock:
        probabilities = model.predict(np.expand_dims(video, axis=0), verbose=0)[0]

    top_indices = np.argsort(probabilities)[::-1][:5]
    return [
        {"class_name": class_names[int(index)], "probability": float(probabilities[index])}
        for index in top_indices
    ]


def load_camera_frames(raw_frames: list[str]) -> np.ndarray:
    """Zamienia obrazy JPEG z przeglądarki w format identyczny z treningowym."""
    if len(raw_frames) != FRAME_COUNT:
        raise ValueError(f"Kamera musi dostarczyć dokładnie {FRAME_COUNT} klatki.")

    processed_frames = []
    for raw_frame in raw_frames:
        if not isinstance(raw_frame, str) or "," not in raw_frame:
            raise ValueError("Jedna z klatek kamery ma nieprawidłowy format.")
        try:
            encoded = raw_frame.split(",", 1)[1]
            image_bytes = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as error:
            raise ValueError("Nie udało się odczytać obrazu z kamery.") from error

        image = cv.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv.IMREAD_COLOR)
        if image is None:
            raise ValueError("Jedna z klatek kamery nie jest poprawnym obrazem.")
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = cv.resize(image, (32, 32)).astype(np.float32) / 255.0
        processed_frames.append(image)

    return np.asarray(processed_frames, dtype=np.float32)


@app.get("/")
def index():
    return render_template("index.html", frame_count=FRAME_COUNT)


@app.post("/api/predict-video")
def predict_video():
    uploaded_file = request.files.get("video")
    if uploaded_file is None or not uploaded_file.filename:
        return jsonify(error="Wybierz plik wideo."), 400

    suffix = Path(secure_filename(uploaded_file.filename)).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        formats = ", ".join(sorted(ALLOWED_EXTENSIONS))
        return jsonify(error=f"Nieobsługiwany format. Wybierz: {formats}."), 400

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
            temporary_path = Path(temporary_file.name)
            uploaded_file.save(temporary_path)
        result = predict(load_video(str(temporary_path)))
        return jsonify(predictions=result)
    except (ValueError, cv.error) as error:
        return jsonify(error=f"Nie udało się przetworzyć wideo: {error}"), 400
    except Exception as error:  # Błąd modelu lub nieoczekiwany błąd serwera.
        app.logger.exception("Błąd klasyfikacji pliku wideo")
        return jsonify(error=f"Klasyfikacja nie powiodła się: {error}"), 500
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


@app.post("/api/predict-camera")
def predict_camera():
    payload = request.get_json(silent=True) or {}
    try:
        result = predict(load_camera_frames(payload.get("frames", [])))
        return jsonify(predictions=result)
    except ValueError as error:
        return jsonify(error=str(error)), 400
    except Exception as error:
        app.logger.exception("Błąd klasyfikacji obrazu z kamery")
        return jsonify(error=f"Klasyfikacja nie powiodła się: {error}"), 500


@app.errorhandler(RequestEntityTooLarge)
def handle_large_upload(_error):
    return jsonify(error="Plik jest za duży. Maksymalny rozmiar to 250 MB."), 413


if __name__ == "__main__":
    # debug=False zapobiega dwukrotnemu wczytywaniu modelu przez automatyczny restart.
    app.run(host="127.0.0.1", port=5000, debug=False)
