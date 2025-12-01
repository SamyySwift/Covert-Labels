import os
import io
import json
import base64
import uuid
import datetime
import threading
import subprocess
from typing import List, Dict, Any, Optional
import numpy as np

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import cv2

from openai import OpenAI  # added for LLM verification
import tensorflow as tf
import tempfile

from train_autoencoder import (
    AnomalyDetector,
    detect_image_anomaly_sensitive,
    IMG_SIZE,  # use the same size as training
)


app = Flask(__name__)
CORS(app)

# Paths and constants
PROJECT_ROOT = os.getcwd()
GENUINE_IMAGE_DIR = os.getenv("GENUINE_IMAGE_DIR") or os.path.join(PROJECT_ROOT, "GENUINE_IMAGE_DIR")
OUTPUT_DIR = os.getenv("AUTOENCODER_OUTPUT_DIR") or os.path.join(PROJECT_ROOT, "autoencoder_outputs")
MODEL_PATH = os.getenv("MODEL_PATH") or os.path.join(PROJECT_ROOT, "autoencoder_genuine.keras")
THRESH_GLOBAL_PATH = os.path.join(OUTPUT_DIR, "anomaly_threshold.txt")
THRESH_PATCH_PATH = os.path.join(OUTPUT_DIR, "anomaly_threshold_patch.txt")
THRESH_PATCH_PATH_8 = os.path.join(OUTPUT_DIR, "anomaly_threshold_patch_ps8.txt")
THRESH_PATCH_PATH_32 = os.path.join(OUTPUT_DIR, "anomaly_threshold_patch_ps32.txt")
PATCH_THRESHOLD_DEFAULT = 0.02
RUNTIME_DIR = os.path.join(PROJECT_ROOT, "runtime")


# -------------------------- HELPERS --------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def decode_base64_image(data: str) -> Image.Image:
    """
    Accepts raw base64 or data URL (data:image/..;base64,...), returns PIL.Image
    Raises ValueError on invalid input.
    """
    if "," in data and data.strip().startswith("data:"):
        _, b64 = data.split(",", 1)
    else:
        b64 = data
    try:
        raw = base64.b64decode(b64)
    except Exception as e:
        raise ValueError(f"Invalid base64 input: {e}")
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return img
    except Exception as e:
        raise ValueError(f"Invalid image payload: {e}")


def save_pil_image(img: Image.Image, dest_dir: str, ext: str = "png") -> str:
    ensure_dir(dest_dir)
    fname = f"{uuid.uuid4().hex}.{ext.lower()}"
    path = os.path.join(dest_dir, fname)
    img.save(path, format=ext.upper())
    return path


def save_form_images(files: List[Any], dest_dir: str) -> List[str]:
    """
    Saves images from form-data to dest_dir after validation via PIL.
    Returns list of saved file paths.
    """
    saved = []
    for f in files:
        try:
            img = Image.open(f.stream).convert("RGB")
        except Exception as e:
            raise ValueError(f"Invalid form-data image: {e}")
        # Derive extension from original filename if possible
        ext = "png"
        if hasattr(f, "filename") and f.filename:
            lower = f.filename.lower()
            if lower.endswith(".jpg") or lower.endswith(".jpeg"):
                ext = "jpeg"
            elif lower.endswith(".png"):
                ext = "png"
        saved.append(save_pil_image(img, dest_dir, ext=ext))
    return saved


def load_threshold(path: str) -> float:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Threshold file not found: {path}. Run training first.")
    with open(path, "r") as f:
        return float(f.read().strip())


def pil_to_data_url(img: Image.Image, format: str = "JPEG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=format)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/jpeg" if format.upper() == "JPEG" else f"image/{format.lower()}"
    return f"data:{mime};base64,{b64}"

# Model cache to avoid repeated loads
_model_lock = threading.Lock()
_model_cached: Optional[tf.keras.Model] = None
_global_threshold: Optional[float] = None
_patch_threshold: Optional[float] = None
_patch_threshold_8: Optional[float] = None
_patch_threshold_32: Optional[float] = None


def get_model_and_thresholds() -> tuple[tf.keras.Model, float, float, float, float]:
    global _model_cached, _global_threshold, _patch_threshold, _patch_threshold_8, _patch_threshold_32
    with _model_lock:
        if _model_cached is None:
            if not os.path.isfile(MODEL_PATH):
                raise FileNotFoundError("Model not found. Train the autoencoder first.")
            _model_cached = tf.keras.models.load_model(
                MODEL_PATH,
                custom_objects={"AnomalyDetector": AnomalyDetector},
                compile=False,
            )
        if _global_threshold is None:
            _global_threshold = load_threshold(THRESH_GLOBAL_PATH)
        if _patch_threshold is None:
            try:
                _patch_threshold = load_threshold(THRESH_PATCH_PATH)
            except FileNotFoundError:
                _patch_threshold = PATCH_THRESHOLD_DEFAULT
        if _patch_threshold_8 is None:
            try:
                _patch_threshold_8 = load_threshold(THRESH_PATCH_PATH_8)
            except FileNotFoundError:
                _patch_threshold_8 = PATCH_THRESHOLD_DEFAULT
        if _patch_threshold_32 is None:
            try:
                _patch_threshold_32 = load_threshold(THRESH_PATCH_PATH_32)
            except FileNotFoundError:
                _patch_threshold_32 = PATCH_THRESHOLD_DEFAULT
        return _model_cached, _global_threshold, _patch_threshold, _patch_threshold_8, _patch_threshold_32

def _detect_image_anomaly_from_pil(model: tf.keras.Model, pil_img: Image.Image, global_threshold: float, patch_thr16: float, patch_thr8: float, patch_thr32: float) -> tuple[bool, float, float, float, float]:
    img = pil_img.resize((IMG_SIZE, IMG_SIZE))
    img_np = np.asarray(img, dtype=np.float32) / 255.0
    x_img = tf.expand_dims(img_np, axis=0)
    pred = model.predict(x_img, verbose=0)

    diff = x_img - pred
    global_loss = float(tf.reduce_mean(tf.square(diff), axis=(1, 2, 3))[0].numpy())

    sq = tf.square(diff)
    pools16 = tf.nn.avg_pool(sq, ksize=[1, 16, 16, 1], strides=[1, 16, 16, 1], padding="VALID")
    pools8 = tf.nn.avg_pool(sq, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding="VALID")
    pools32 = tf.nn.avg_pool(sq, ksize=[1, 32, 32, 1], strides=[1, 32, 32, 1], padding="VALID")
    local_max16 = float(tf.reduce_max(pools16).numpy())
    local_max8 = float(tf.reduce_max(pools8).numpy())
    local_max32 = float(tf.reduce_max(pools32).numpy())

    is_anom = (
        (global_loss > global_threshold) or
        (local_max16 > patch_thr16) or
        (local_max8 > patch_thr8) or
        (local_max32 > patch_thr32)
    )
    return is_anom, global_loss, local_max16, local_max8, local_max32


# Training manager for concurrency + status tracking
class TrainingRun:
    def __init__(self):
        self.id = uuid.uuid4().hex
        self.state = "running"  # running|finished|failed
        self.started_at = datetime.datetime.utcnow().isoformat() + "Z"
        self.finished_at = None
        self.returncode = None
        self.error = None
        self.log_path = os.path.join(RUNTIME_DIR, "training_runs", self.id, "train.log")
        self.status_path = os.path.join(RUNTIME_DIR, "training_runs", self.id, "status.json")
        self._process: Optional[subprocess.Popen] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "state": self.state,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "returncode": self.returncode,
            "error": self.error,
            "log_path": self.log_path,
        }

    def persist(self) -> None:
        ensure_dir(os.path.dirname(self.status_path))
        with open(self.status_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def start(self) -> None:
        ensure_dir(os.path.dirname(self.log_path))
        with open(self.log_path, "w") as log_f:
            # Use -u for unbuffered stdout to stream logs
            self._process = subprocess.Popen(
                ["python3", "-u", os.path.join(PROJECT_ROOT, "train_autoencoder.py")],
                stdout=log_f,
                stderr=subprocess.STDOUT,
                cwd=PROJECT_ROOT,
                env=os.environ.copy(),
            )
        self.persist()
        threading.Thread(target=self._monitor, daemon=True).start()

    def _monitor(self) -> None:
        try:
            rc = self._process.wait()
            self.returncode = rc
            self.finished_at = datetime.datetime.utcnow().isoformat() + "Z"
            if rc == 0:
                self.state = "finished"
                # Invalidate thresholds cache so new runs are picked up
                global _global_threshold, _patch_threshold
                _global_threshold = None
                _patch_threshold = None
            else:
                self.state = "failed"
        except Exception as e:
            self.state = "failed"
            self.error = str(e)
            self.finished_at = datetime.datetime.utcnow().isoformat() + "Z"
        finally:
            self.persist()

    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None


_current_training: Optional[TrainingRun] = None
_current_training_lock = threading.Lock()


def get_training_status(training_id: Optional[str] = None) -> Dict[str, Any]:
    # If training_id matches the current run, return live status
    with _current_training_lock:
        if training_id and _current_training and _current_training.id == training_id:
            return _current_training.to_dict()
        if not training_id and _current_training:
            return _current_training.to_dict()

    # Otherwise, look for persisted status
    if training_id:
        status_path = os.path.join(RUNTIME_DIR, "training_runs", training_id, "status.json")
        if os.path.isfile(status_path):
            with open(status_path, "r") as f:
                return json.load(f)

    # No active training and no persisted status found
    return {
        "state": "idle",
        "started_at": None,
        "finished_at": None,
        "returncode": None,
        "error": None,
        "id": training_id,
    }


# ------------------ API ENDPOINTS --------------------------
@app.route("/api", methods=["GET"])
def health():
    return jsonify({"status": "API is running..."}), 200


@app.route("/api/upload", methods=["POST"])
def upload_images():
    """
    Accepts multiple images via:
    - JSON: { "images": [ "data:image/png;base64,...", "..." ] }
    - form-data: key 'images' with multiple files, or any image file fields

    Saves to GENUINE_IMAGE_DIR. Returns list of saved filenames.
    """
    ensure_dir(GENUINE_IMAGE_DIR)

    try:
        saved_paths: List[str] = []

        # JSON base64
        if request.is_json:
            data = request.get_json(silent=True)
            if not isinstance(data, dict) or "images" not in data:
                return jsonify({"error": "Invalid JSON body. Expect { images: [...] }"}), 400
            images = data["images"]
            if not isinstance(images, list) or len(images) == 0:
                return jsonify({"error": "images must be a non-empty list"}), 400
            for b64 in images:
                img = decode_base64_image(str(b64))
                saved_paths.append(save_pil_image(img, GENUINE_IMAGE_DIR, ext="png"))

        # form-data
        elif request.files:
            files = []
            if "images" in request.files:
                files = request.files.getlist("images")
            else:
                # Accept all files if no 'images' key
                files = list(request.files.values())

            if len(files) == 0:
                return jsonify({"error": "No image files provided"}), 400

            saved_paths.extend(save_form_images(files, GENUINE_IMAGE_DIR))
        else:
            return jsonify({"error": "Unsupported content type. Use JSON or form-data"}), 415

        return jsonify({
            "status": "Images saved successfully",
            "Saved images": len(saved_paths),
        }), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/api/train", methods=["POST"])
def start_training():
    """
    Triggers training via train_autoencoder.py.
    Returns training ID and accepted status.
    Prevents concurrent runs; returns 409 if already running.
    """
    global _current_training
    with _current_training_lock:
        if _current_training and _current_training.is_running():
            return jsonify({
                "error": "Training already in progress",
                "training_status": _current_training.to_dict(),
            }), 409

        _current_training = TrainingRun()
        _current_training.start()
        return jsonify({
            "status": "started",
            "training_status": _current_training.to_dict(),
        }), 202


@app.route("/api/train-status", methods=["GET"])
def training_status():
    """
    Checks training status.
    Optional query param: ?id=<training_id>
    Returns state: idle|running|finished|failed + timestamps and return code.
    """
    training_id = request.args.get("id")
    status = get_training_status(training_id)
    return jsonify({"status": "ok", "training_status": status}), 200


@app.route("/api/verify", methods=["POST"])
def verify():
    """
    Accepts an image:
    - JSON: { "image": "data:image/png;base64,..." }
    - form-data: key 'image' (single file)

    Runs anomaly detection with the trained autoencoder and thresholds.
    Returns structured JSON:
    {
      "status": "normal|suspicious",
      "confidence": 0.0-1.0,
      "metrics": { "global_loss": float, "patch_max": float },
      "details": { "path": saved_path }
    }
    """
    try:
        # Load model and thresholds once (cached)
        model, global_thr, patch_thr16, patch_thr8, patch_thr32 = get_model_and_thresholds()
        # Decode image to PIL in-memory (no temp file)
        pil_img = None
        if request.is_json:
            data = request.get_json(silent=True)
            if not isinstance(data, dict) or "image" not in data:
                return jsonify({"error": "Invalid JSON body. Expect { image: base64 }"}), 400
            pil_img = decode_base64_image(str(data["image"]))
        elif request.files and "image" in request.files:
            img_file = request.files["image"]
            try:
                pil_img = Image.open(img_file.stream).convert("RGB")
            except Exception as e:
                return jsonify({"error": f"Invalid form-data image: {e}"}), 400
        else:
            return jsonify({"error": "Unsupported content type or missing 'image'"}), 415

        # Pre-gate to flag non-product/occluded/typo images (LLM + CV)
        gate_results = []
        triggered = False
        method = "AE"
        # LLM gate
        try:
            _ = _get_openrouter_api_key()
            data_url = pil_to_data_url(pil_img, format="JPEG")
            g_status, g_conf, g_details = llm_inspect_image(data_url)
            gate_results.append({"source": "LLM", "status": g_status, "confidence": g_conf, "details": g_details})
            if g_status == "suspicious" and g_conf >= 0.6:
                triggered = True
                method = "LLM_GATE"
        except Exception:
            pass
        # CV gate (paper/occlusion heuristic)
        try:
            arr = np.asarray(pil_img)
            h, w = arr.shape[:2]
            x1, x2 = int(0.2 * w), int(0.8 * w)
            y1, y2 = int(0.25 * h), int(0.75 * h)
            roi = arr[y1:y2, x1:x2]
            hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
            white_mask = (hsv[...,1] < 40) & (hsv[...,2] > 200)
            white_ratio = float(np.mean(white_mask))
            edges = cv2.Canny(cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY), 100, 200)
            edge_ratio = float(np.mean(edges > 0))
            cv_suspicious = (white_ratio > 0.35) or (edge_ratio < 0.01)
            cv_conf = float(min(1.0, max(white_ratio * 2.0, (0.02 - edge_ratio) * 50.0))) if cv_suspicious else float(max(0.0, 0.3 - white_ratio))
            gate_results.append({"source": "CV", "status": "suspicious" if cv_suspicious else "normal", "confidence": cv_conf, "details": {"white_ratio": white_ratio, "edge_ratio": edge_ratio}})
            if cv_suspicious and cv_conf >= 0.6 and not triggered:
                triggered = True
                method = "CV_GATE"
        except Exception:
            pass

        if triggered:
            return jsonify({
                "status": "suspicious",
                "confidence": max(r.get("confidence", 0.0) for r in gate_results) if gate_results else 1.0,
                "method": method,
                "gate": gate_results,
            }), 200

        # In-memory anomaly detection
        is_anom, g_loss, p16, p8, p32 = _detect_image_anomaly_from_pil(
            model, pil_img, global_threshold=global_thr, patch_thr16=patch_thr16, patch_thr8=patch_thr8, patch_thr32=patch_thr32
        )
        suspicious_score = float(max(
            g_loss / max(global_thr, 1e-8),
            p16 / max(patch_thr16, 1e-8),
            p8 / max(patch_thr8, 1e-8),
            p32 / max(patch_thr32, 1e-8)
        ))
        ae_status = "suspicious" if suspicious_score > 1.0 else "normal"
        ae_confidence = float(min(1.0, suspicious_score)) if ae_status == "suspicious" else float(1.0 - min(1.0, suspicious_score))

        final_status = ae_status
        final_confidence = ae_confidence
        source = "AE"

        return jsonify({
            "status": final_status,
            "confidence": final_confidence,
            "metrics": {"global_loss": g_loss, "patch_max": {"ps8": p8, "ps16": p16, "ps32": p32}},
            "thresholds": {"global": global_thr, "patch": {"ps8": patch_thr8, "ps16": patch_thr16, "ps32": patch_thr32}},
            "method": source,
            "ae": {"status": ae_status, "confidence": ae_confidence},
        }), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Server error: {e}"}), 500


# helper functions (place after get_model_and_thresholds)
def _get_openrouter_api_key() -> str:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key or not key.strip():
        raise ValueError("OPENROUTER_API_KEY is not set. Add it to environment or .env.")
    return key

def llm_inspect_image(image_input: str) -> tuple[str, float, dict | str]:
    """
    Calls OpenRouter to classify the image as 'normal' or 'suspicious'.
    Accepts either a file path or a base64 data URL.
    Returns (classification, confidence, raw_json_or_text).
    """
    api_key = _get_openrouter_api_key()
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    if isinstance(image_input, str) and image_input.startswith("data:image/"):
        data_url = image_input
    else:
        with open(image_input, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{b64}"

    prompt_text = (
        "Task: Gate images before autoencoder verification.\n"
        "Output STRICT JSON only (no prose): "
        '{"classification":"normal|suspicious","confidence":0.0-1.0,"evidence":["..."]}.\n'
        "Rules:\n"
        "- Return 'suspicious' if the image is not a real photo of a physical product label (e.g., website/app screenshot, graphic mockup, rendering).\n"
        "- Return 'suspicious' if the label is obscured or covered (paper overlay, tape, hand), severely cropped, unreadable due to heavy blur/glare).\n"
        "- Return 'suspicious' if you detect text typos or brand word mismatches on the primary label.\n"
        "- Otherwise return 'normal'. Do not guess.\n"
        "- Confidence: multiple clear indicators ≥0.9; one clear indicator ≈0.7–0.85; minor concerns ≤0.3.\n"
        "- Provide short evidence strings referencing visible cues.\n"
    )

    completion = client.chat.completions.create(
        model="google/gemini-2.5-flash-image",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
    )
    raw = completion.choices[0].message.content.strip()

    try:
        data = json.loads(raw)
        classification = str(data.get("classification", "normal")).lower().strip()
        confidence = float(data.get("confidence", 0.5))
        evidence = data.get("evidence", []) or []
        threshold = 0.7
        if classification == "suspicious" and (confidence < threshold or len(evidence) == 0):
            classification = "normal"
        return classification, confidence, data
    except Exception:
        low = raw.lower()
        confidence = 0.0
        if "suspicious" in low:
            return "suspicious", confidence, raw
        if "normal" in low:
            return "normal", confidence, raw
        return "normal", confidence, raw






if __name__ == "__main__":
    # Dev server
    app.run(host="0.0.0.0", port=8080, debug=True)



