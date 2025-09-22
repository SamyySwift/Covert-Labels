import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import preprocess_input
import base64
from openai import OpenAI
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import io
# import uuid
from werkzeug.utils import secure_filename  
import cv2
import sqlite3  
import random   
import threading  
import subprocess  
from datetime import datetime  


load_dotenv()
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Add these definitions so uploads and file validation work
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8MB limit
DB_PATH = os.getenv("DB_PATH", os.path.join(UPLOAD_DIR, "app.db"))

def init_db():
    """Initialize SQLite DB and uploads table if not present."""
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS uploads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                upload_id TEXT UNIQUE NOT NULL,
                product TEXT,
                batch TEXT,
                barcode TEXT,
                manufacturer TEXT,
                production_date TEXT,
                expiry_date TEXT,
                notes TEXT,
                original_path TEXT,
                processed_path TEXT,
                metadata_json TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_uploads_upload_id ON uploads(upload_id)")
        conn.commit()
    finally:
        conn.close()

init_db()  # <-- NEW

# --- Config ---
IMG_SIZE = (224, 224)
MODEL_PATH = "./batch_classifier_model.keras"
LABEL_MAP_PATH = "./batch_label_map.json"
METADATA_PATH = "./batch_metadata_map.json"
VARIATIONS_COUNT = 30

# Get API key from environment variable
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable is not set.")

# --- Load Model and Metadata ---
print(f"Loading model {MODEL_PATH} and metadata...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)  
   
    print("✅ Model loaded and compiled successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise

with open(LABEL_MAP_PATH, "r") as f:
    label_map = json.load(f)
    index_to_label = {int(k): v for k, v in label_map.items()}

with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

def reload_model_and_maps():
    """Reload the trained model and maps after re-training completes."""
    global model, label_map, index_to_label, metadata
    try:
        new_model = tf.keras.models.load_model(MODEL_PATH)
        with open(LABEL_MAP_PATH, "r") as f:
            lm = json.load(f)
        with open(METADATA_PATH, "r") as f:
            mm = json.load(f)
        model = new_model
        label_map = lm
        index_to_label = {int(k): v for k, v in lm.items()}
        metadata = mm
        print("🔄 Model and maps reloaded successfully")
    except Exception as e:
        print(f"❌ Failed to reload model/maps: {e}")

# --- OpenRouter Client ---
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# --- Dot Detection Function using OpenRouter ---
def detect_dots_with_ai(image_data):
    """Use OpenRouter's vision model to detect circular dot patterns"""
    try:
        completion = client.chat.completions.create(
            model="google/gemini-2.0-flash-exp:free",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Examine this product label image for authentication microdots. IMPORTANT: This label may be wrapped on a curved bottle, so some dots might be hidden or partially visible.\n\nLook for ANY of these authentication markers:\n\n1. Small circular dots (1-3mm diameter) - even if only a few are visible\n2. These may appear as:\n   - White or light-colored circles\n   - Slightly raised or embossed patterns\n   - Small round spots with subtle borders\n   - Partially visible dots at edges or curves\n\n3. Key points:\n   - Even 1-2 visible dots indicate authentication\n   - Dots may be scattered across visible areas\n   - Some may be cut off at label edges\n   - Look in corners, visible flat areas, and edges\n   - Ignore obvious printing defects or dust\n\n4. Real-world considerations:\n   - Label curvature may hide some dots\n   - Focus on clearly visible label areas\n   - Partial dots at edges still count\n\nRespond 'YES' if you can see ANY circular authentication dots (even just 1-2), or 'NO' if you see absolutely no circular dot patterns. Just YES or NO."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        }
                    ]
                }
            ]
        )
        
        response = completion.choices[0].message.content.strip().upper()
        
        # Simple and reliable detection
        if "YES" in response:
            has_dots = True
        elif "NO" in response:
            has_dots = False
        else:
            # If response is unclear, default to no dots (safer)
            has_dots = False
        
        return has_dots, response
        
    except Exception as e:
        print(f"❌ Error with AI dot detection: {e}")
        return False, f"Error: {e}"



# --- Preprocess Function ---
def preprocess_image(image):
    # Ensure image is RGB (3 channels)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img = image.resize(IMG_SIZE)
    img_array = np.array(img)
    
    # Ensure we have 3 channels
    if len(img_array.shape) == 2:  # Grayscale
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:  # RGBA
        img_array = img_array[:, :, :3]  # Remove alpha channel
    
    # Use ResNet50V2 preprocessing
    img_array = preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

# --- Upload Helpers ---
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def parse_json_field(value, field_name: str):
    """Accept dict or JSON string. Raise 400-friendly error on invalid JSON."""
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            raise ValueError(f"Invalid JSON for field '{field_name}'")
    raise ValueError(f"Invalid type for field '{field_name}'; expected JSON or object")

def save_image_bytes(image_bytes: bytes, original_filename: str, product: str | None = None, batch: str | None = None, label_id: str | None = None) -> dict:
   
    if not label_id or not isinstance(label_id, str) or not label_id.strip():
        raise ValueError("Missing or invalid label_id")
    safe_name = secure_filename(original_filename) or "upload.png"

    # Build folder as uploads/<product>/<batch>
    product_slug = secure_filename((product or "unknown_product").strip()) or "unknown_product"
    batch_slug = secure_filename((batch or "unknown_batch").strip()) or "unknown_batch"
    folder = os.path.join(UPLOAD_DIR, product_slug, batch_slug)
    os.makedirs(folder, exist_ok=True)

    # Do NOT save original image to disk anymore
    original_path = None

    return {"label_id": label_id, "folder": folder, "original_path": original_path}

def image_to_base64_png(pil_image: Image.Image) -> str:
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def apply_label_modifications(pil_image: Image.Image, modified_label: dict | None) -> tuple[Image.Image, dict]:
    """
    Apply label changes to the image.
    Current implementation: lightly add small visible microdots.
    Does NOT store or return coordinates to maintain security.
    """
    # Defaults can be overridden by modified_label fields
    params = {
        "dot_count": 7,
        "dot_radius": 6,
        "margin": 10,
    }
    if isinstance(modified_label, dict):
        params["dot_count"] = int(modified_label.get("dot_count", params["dot_count"]))
        params["dot_radius"] = int(modified_label.get("dot_radius", params["dot_radius"]))

    img_cv = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = img_cv.shape[:2]

    # Randomly place dots away from edges; no text mask to keep API fast
    rng = np.random.default_rng()
    xs = rng.integers(params["margin"] + params["dot_radius"], w - params["margin"] - params["dot_radius"], params["dot_count"])
    ys = rng.integers(params["margin"] + params["dot_radius"], h - params["margin"] - params["dot_radius"], params["dot_count"])
    coords = list(zip(xs.tolist(), ys.tolist()))

    # Draw small visible white-leaning dots with subtle halo (no coords saved)
    for (x, y) in coords:
        local = img_cv[max(0, y-3):min(h, y+4), max(0, x-3):min(w, x+4)]
        avg_color = np.mean(local, axis=(0, 1)) if local.size > 0 else np.array([200, 200, 200], dtype=np.float32)
        # White-leaning color
        dot_color = (0.75 * np.array([255, 255, 255]) + 0.25 * avg_color).clip(0, 255).astype(np.uint8)
        cv2.circle(img_cv, (x, y), params["dot_radius"], dot_color.tolist(), -1)
        # Subtle halo blended with background
        ring_color = (0.6 * avg_color + 0.4 * dot_color).clip(0, 255).astype(np.uint8)
        cv2.circle(img_cv, (x, y), params["dot_radius"] + 1, ring_color.tolist(), 1)

    processed = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    # Return a summary but no coordinates
    summary = {
        "applied": True,
        "method": "microdot_overlay",
        "dot_count": params["dot_count"],
        "dot_radius": params["dot_radius"]
    }
    return processed, summary

# --- Augmentations copied from generate_clg_batches.py ---
def apply_brightness_variation(img, factor_range=(0.85, 1.15)):
    factor = random.uniform(*factor_range)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def apply_contrast_variation(img, factor_range=(0.85, 1.15)):
    factor = random.uniform(*factor_range)
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

def apply_gaussian_blur(img, kernel_range=(1, 3)):
    kernel_candidates = [k for k in range(kernel_range[0], kernel_range[1] + 1) if k % 2 == 1]
    kernel_size = random.choice(kernel_candidates)
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def apply_rotation(img, angle_range=(-3, 3)):
    angle = random.uniform(*angle_range)
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, rotation_matrix, (w, h), borderValue=(255, 255, 255))

def apply_perspective_tilt(img, tilt_range=0.015):
    h, w = img.shape[:2]
    tilt = random.uniform(-tilt_range, tilt_range)
    src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst_points = np.float32([
        [tilt * w, tilt * h],
        [w - tilt * w, tilt * h],
        [w + tilt * w, h - tilt * h],
        [-tilt * w, h - tilt * h]
    ])
    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    return cv2.warpPerspective(img, perspective_matrix, (w, h), borderValue=(255, 255, 255))

def apply_crop_and_resize(img, crop_range=(0.92, 1.0)):
    h, w = img.shape[:2]
    crop_factor = random.uniform(*crop_range)
    new_h, new_w = int(h * crop_factor), int(w * crop_factor)
    start_y = random.randint(0, h - new_h)
    start_x = random.randint(0, w - new_w)
    cropped = img[start_y:start_y + new_h, start_x:start_x + new_w]
    return cv2.resize(cropped, (w, h))

def apply_noise(img, noise_factor=0.008):
    noise = np.random.normal(0, noise_factor * 255, img.shape)
    noisy_img = img.astype(np.float32) + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

def generate_variation(base_img, variation_id):
    img = base_img.copy()
    augmentations = [
        lambda x: apply_brightness_variation(x, (0.85, 1.15)),
        lambda x: apply_contrast_variation(x, (0.85, 1.15)),
        lambda x: apply_gaussian_blur(x, (1, 3)),
        lambda x: apply_rotation(x, (-3, 3)),
        lambda x: apply_perspective_tilt(x, 0.015),
        lambda x: apply_crop_and_resize(x, (0.92, 1.0)),
        lambda x: apply_noise(x, 0.008)
    ]
    num_augs = random.randint(2, 4)
    for aug in random.sample(augmentations, num_augs):
        img = aug(img)
    return img
def save_upload_record(label_id: str,
                       product: str | None,
                       batch: str | None,
                       sanitized_metadata: dict | None,
                       original_path: str | None,
                       processed_path: str) -> None:
    """Persist upload and its sanitized metadata to SQLite."""
    md = sanitized_metadata or {}
    row = (
        label_id,
        product,
        batch,
        md.get("barcode"),
        md.get("manufacturer"),
        md.get("production_date"),
        md.get("expiry_date"),
        md.get("notes"),
        original_path,
        processed_path,
        json.dumps(md)
    )
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """INSERT OR REPLACE INTO uploads
               (upload_id, product, batch, barcode, manufacturer, production_date, expiry_date, notes, original_path, processed_path, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            row
        )
        conn.commit()
    finally:
        conn.close()
# Helper and routes for reading uploads from SQLite
def _to_int(val, default: int, min_val: int, max_val: int) -> int:
    try:
        iv = int(val)
        return max(min_val, min(iv, max_val))
    except Exception:
        return default

def fetch_upload_by_id(upload_id: str) -> dict | None:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(
            """SELECT upload_id, product, batch, barcode, manufacturer, production_date, expiry_date, notes,
                      original_path, processed_path, metadata_json, created_at
               FROM uploads
               WHERE upload_id = ?""",
            (upload_id,)
        )
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()

def fetch_uploads_list(limit: int, offset: int) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(
            """SELECT upload_id, product, batch, barcode, manufacturer, production_date, expiry_date, notes,
                      original_path, processed_path, metadata_json, created_at
               FROM uploads
               ORDER BY datetime(created_at) DESC
               LIMIT ? OFFSET ?""",
            (limit, offset)
        )
        rows = cur.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()

@app.route('/api/uploads/<label_id>', methods=['GET'])
def get_upload(label_id):
    try:
        row = fetch_upload_by_id(label_id)
        if not row:
            return jsonify({"status": "error", "error": "Not found"}), 404
        # Parse metadata_json for convenience
        md = None
        if row.get("metadata_json"):
            try:
                md = json.loads(row["metadata_json"])
            except Exception:
                md = None
        row["metadata"] = md
        # Rename field in response
        row["label_id"] = row.pop("upload_id", None)
        return jsonify({"status": "success", "item": row}), 200
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route('/api/uploads', methods=['GET'])
def list_uploads():
    try:
        limit = _to_int(request.args.get("limit"), default=50, min_val=1, max_val=200)
        offset = _to_int(request.args.get("offset"), default=0, min_val=0, max_val=1_000_000)
        items = fetch_uploads_list(limit=limit, offset=offset)
        # Parse metadata_json for each item for convenience
        for it in items:
            md = None
            if it.get("metadata_json"):
                try:
                    md = json.loads(it["metadata_json"])
                except Exception:
                    md = None
            it["metadata"] = md
            # Rename field in response
            it["label_id"] = it.pop("upload_id", None)
        return jsonify({
            "status": "success",
            "count": len(items),
            "limit": limit,
            "offset": offset,
            "items": items
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

# --- Training management (use training script) ---
training_status = {
    "state": "idle",  # idle | running | finished | failed
    "started_at": None,
    "finished_at": None,
    "returncode": None,
    "error": None,
    "log": ""
}
_training_thread = None

def _run_training_subprocess():
    global _training_thread, training_status
    training_status.update({
        "state": "running",
        "started_at": datetime.utcnow().isoformat() + "Z",
        "finished_at": None,
        "returncode": None,
        "error": None,
        "log": ""
    })
    try:
        # Run the training script using only uploads (the script already set DATA_DIR='uploads')
        proc = subprocess.Popen(
            ["python3", "train_batch_classifier.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            text=True,
            bufsize=1
        )
        # Stream output to status log
        for line in proc.stdout:
            training_status["log"] += line
        proc.wait()
        training_status["returncode"] = proc.returncode

        if proc.returncode == 0:
            # Reload model and maps
            reload_model_and_maps()
            training_status["state"] = "finished"
        else:
            training_status["state"] = "failed"
            training_status["error"] = f"Training script exited with code {proc.returncode}"
    except Exception as e:
        training_status["state"] = "failed"
        training_status["error"] = str(e)
    finally:
        training_status["finished_at"] = datetime.utcnow().isoformat() + "Z"
        _training_thread = None

@app.route('/api/train', methods=['POST'])
def api_train():
    global _training_thread, training_status
    if _training_thread is not None and _training_thread.is_alive():
        return jsonify({"status": "error", "error": "Training already in progress", "training_status": training_status}), 409
    _training_thread = threading.Thread(target=_run_training_subprocess, daemon=True)
    _training_thread.start()
    return jsonify({"status": "started", "training_status": training_status}), 202

@app.route('/api/train-status', methods=['GET'])
def api_train_status():
    return jsonify({"status": "ok", "training_status": training_status}), 200

@app.route('/api/', methods=['GET'])
def home():
    return jsonify({
        "message": "Batch Verification API",
        "status": "active",
        "endpoints": {
            "/api/verify": "POST - Upload image for batch verification",
            "/api/upload_label": "POST - Upload image + metadata + modified label changes",
            "/api/uploads": "GET - List recent uploads (paginated)",
            "/api/uploads/<label_id>": "GET - Fetch single upload by label id",
            "/api/train": "POST - Trigger model training (uses uploads only)",
            "/api/train/status": "GET - Check current training job status"
        }
    })

@app.route('/api/verify', methods=['POST'])
def verify_batch():
    try:
        image_data = None
        image_base64 = None
        label_id = None
        # Check if it's a base64 image in JSON payload
        if request.is_json:
            data = request.get_json()
            label_id = data.get("label_id")
            if 'image' in data:
                # Handle base64 image
                image_base64_raw = data['image']
                # Remove data URL prefix if present
                if image_base64_raw.startswith('data:image/'):
                    image_base64_raw = image_base64_raw.split(',')[1]
                try:
                    image_data = base64.b64decode(image_base64_raw)
                    image_base64 = image_base64_raw
                except Exception as e:
                    return jsonify({
                        "error": f"Invalid base64 image data: {str(e)}",
                        "status": "error"
                    }), 400
            else:
                return jsonify({
                    "error": "No 'image' field found in JSON payload",
                    "status": "error"
                }), 400
        # Check if image file is provided (original file upload method)
        elif 'image' in request.files:
            file = request.files['image']
            label_id = request.form.get("label_id")
            if file.filename == '':
                return jsonify({
                    "error": "No image file selected",
                    "status": "error"
                }), 400
            # Read and process the image
            image_data = file.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')
        else:
            return jsonify({
                "error": "No image provided. Send either a file upload or JSON with base64 'image' field",
                "status": "error"
            }), 400

        # Convert to PIL Image for processing
        image = Image.open(io.BytesIO(image_data))

        # --- AI microdot detection gate (skip prediction if no dots) ---
        microdots_detected = None
        ai_detection_summary = None
        # if image_base64:
        #     try:
        #         microdots_detected, ai_detection_summary = detect_dots_with_ai(image_base64)
        #     except Exception as e:
        #         microdots_detected, ai_detection_summary = None, f"Error: {e}"

        # if microdots_detected is False:
        #     return jsonify({
        #         "status": "suspicious",
        #         "reason": "No microdots detected by AI",
        #         "predicted_batch": None,
        #         "confidence": 0.0,
        #         "label_id": None,
        #         "microdots_detected": False,
        #         "ai_detection_summary": ai_detection_summary
        #     })
    

        # Step 2: Predict batch using CNN
        print("🧠 Predicting batch...")
        img_tensor = preprocess_image(image)
        probs = model.predict(img_tensor)[0]
        top_idx = np.argmax(probs)
        top_batch = index_to_label[top_idx]
        batch_confidence = float(probs[top_idx])

        # Resolve label_id (from DB) based on predicted batch only
        resolved_label_id = resolve_label_id_for_batch(top_batch)

        # Suspicious if low confidence
        if batch_confidence < 0.55:
            return jsonify({
                "status": "suspicious",
                "reason": "Low batch classification confidence",
                "predicted_batch": top_batch,
                "confidence": batch_confidence,
                "label_id": resolved_label_id,
                "microdots_detected": microdots_detected,
                "ai_detection_summary": ai_detection_summary
            })
        # ... existing code ...

        # Successful verification: attach product metadata
        metadata_payload = None
        if resolved_label_id:
            try:
                row = fetch_upload_by_id(resolved_label_id)
                if row:
                    # Start with column fields
                    metadata_payload = {
                        "product": row.get("product"),
                        "batch": row.get("batch"),
                        "barcode": row.get("barcode"),
                        "manufacturer": row.get("manufacturer"),
                        "production_date": row.get("production_date"),
                        "expiry_date": row.get("expiry_date"),
                        "notes": row.get("notes"),
                    }
                    # Merge metadata_json without overwriting non-null column values
                    if row.get("metadata_json"):
                        try:
                            md_json = json.loads(row["metadata_json"])
                            if isinstance(md_json, dict):
                                for k, v in md_json.items():
                                    if metadata_payload.get(k) is None:
                                        metadata_payload[k] = v
                        except Exception:
                            pass
            except Exception:
                metadata_payload = None

        return jsonify({
            "status": "authentic",
            "predicted_batch": top_batch,
            "confidence": batch_confidence,
            "label_id": resolved_label_id,
            "microdots_detected": microdots_detected,
            "ai_detection_summary": ai_detection_summary,
            "metadata": metadata_payload
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/api/upload_label', methods=['POST'])
def upload_label():
    try:
        # Accept: multipart/form-data or JSON { label_id: "...", image: base64, metadata: {...}, modified_label: {...} }
        image_bytes = None
        original_filename = "upload.png"
        metadata_in = None
        modified_label = None
        label_id = None

        if request.is_json:
            data = request.get_json(silent=True) or {}
            label_id = data.get("label_id")
            if not label_id or not isinstance(label_id, str) or not label_id.strip():
                return jsonify({"status": "error", "error": "Missing 'label_id' in JSON payload"}), 400
            image_b64 = data.get("image")
            if not image_b64:
                return jsonify({"status": "error", "error": "Missing 'image' (base64) in JSON payload"}), 400
            if image_b64.startswith("data:image/"):
                image_b64 = image_b64.split(",", 1)[1]
            try:
                image_bytes = base64.b64decode(image_b64)
            except Exception as e:
                return jsonify({"status": "error", "error": f"Invalid base64 image: {e}"}), 400

            try:
                metadata_in = parse_json_field(data.get("metadata"), "metadata")
                modified_label = parse_json_field(data.get("modified_label"), "modified_label")
            except ValueError as ve:
                return jsonify({"status": "error", "error": str(ve)}), 400

        else:
            # Multipart form
            if "label_id" not in request.form:
                return jsonify({"status": "error", "error": "Missing 'label_id' form field"}), 400
            label_id = request.form.get("label_id")
            if not label_id or not isinstance(label_id, str) or not label_id.strip():
                return jsonify({"status": "error", "error": "Invalid 'label_id'"}), 400

            if "image" not in request.files:
                return jsonify({"status": "error", "error": "Missing 'image' file"}), 400
            file = request.files["image"]
            if file.filename == "":
                return jsonify({"status": "error", "error": "Empty filename"}), 400
            if not allowed_file(file.filename):
                return jsonify({"status": "error", "error": "Unsupported file type"}), 415

            original_filename = file.filename
            image_bytes = file.read()

            try:
                metadata_in = parse_json_field(request.form.get("metadata"), "metadata")
                modified_label = parse_json_field(request.form.get("modified_label"), "modified_label")
            except ValueError as ve:
                return jsonify({"status": "error", "error": str(ve)}), 400

        # Validate image
        if not image_bytes:
            return jsonify({"status": "error", "error": "No image data received"}), 400
        try:
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            return jsonify({"status": "error", "error": f"Invalid image content: {e}"}), 400

        # Determine product and batch (support both 'batch' and 'batch_id')
        product_name = None
        batch_id = None
        if isinstance(metadata_in, dict):
            product_name = metadata_in.get("product") or metadata_in.get("product_name")
            batch_id = metadata_in.get("batch") or metadata_in.get("batch_id")

        # Store only folder and label_id (do NOT save original to disk)
        saved = save_image_bytes(image_bytes, original_filename, product=product_name, batch=batch_id, label_id=label_id)
        folder = saved["folder"]
        original_path = saved["original_path"]  # remains None

        # Build sanitized metadata for DB only (no metadata.json file)
        sanitized_metadata = {}
        if metadata_in:
            allowed_keys = {"product", "batch", "barcode", "manufacturer", "production_date", "expiry_date", "notes"}
            sanitized_metadata = {k: v for k, v in metadata_in.items() if k in allowed_keys and v is not None}

        # Apply label modifications (visible microdots)
        processed_img, mod_summary = apply_label_modifications(pil_image, modified_label)

        # Use product name in filenames
        safe_product = secure_filename(product_name or "unknown_product") or "unknown_product"
        base_filename = f"{safe_product}_dot.png"
        base_path = os.path.join(folder, base_filename)

        # Save base processed image
        processed_img.save(base_path, format="PNG")

        # Create variations
        cv2_base = cv2.cvtColor(np.array(processed_img), cv2.COLOR_RGB2BGR)
        for i in range(VARIATIONS_COUNT):
            var_img = generate_variation(cv2_base, i)
            var_filename = f"{safe_product}_dot_var_{i+1:02d}.png"
            var_path = os.path.join(folder, var_filename)
            cv2.imwrite(var_path, var_img)

        # Return processed image as base64 (of the base processed image)
        processed_b64 = image_to_base64_png(processed_img)

        # Persist to SQLite (original_path is None; processed_path = base image)
        save_upload_record(
            label_id=label_id,
            product=product_name,
            batch=batch_id,
            sanitized_metadata=sanitized_metadata,
            original_path=original_path,
            processed_path=base_path
        )

        return jsonify({
            "label_id": label_id,
            "processed_image_base64": f"data:image/png;base64,{processed_b64}"
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

def resolve_label_id_for_prediction(predicted_label: str) -> str | None:
    """
    Resolve a label_id (stored as upload_id in DB) from the predicted label.
    If label format is 'PRODUCT::BATCH', match both; otherwise match by batch only.
    Returns the most recent matching record, or None if not found.
    """
    product = None
    batch = predicted_label
    if "::" in predicted_label:
        parts = predicted_label.split("::", 1)
        product = parts[0].strip() or None
        batch = parts[1].strip() if len(parts) > 1 else predicted_label

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        if product:
            cur = conn.execute(
                """SELECT upload_id FROM uploads
                   WHERE batch = ? AND product = ?
                   ORDER BY datetime(created_at) DESC
                   LIMIT 1""",
                (batch, product)
            )
            row = cur.fetchone()
            if row:
                return row["upload_id"]

        # Fallback: match by batch only
        cur = conn.execute(
            """SELECT upload_id FROM uploads
               WHERE batch = ?
               ORDER BY datetime(created_at) DESC
               LIMIT 1""",
            (batch,)
        )
        row = cur.fetchone()
        return row["upload_id"] if row else None
    finally:
        conn.close()

def resolve_label_id_for_batch(batch: str) -> str | None:
    """
    Resolve a label_id (stored as upload_id in DB) from the predicted batch.
    Returns the most recent matching record, or None if not found.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(
            """SELECT upload_id FROM uploads
               WHERE batch = ?
               ORDER BY datetime(created_at) DESC
               LIMIT 1""",
            (batch,)
        )
        row = cur.fetchone()
        return row["upload_id"] if row else None
    finally:
        conn.close()

if __name__ == "__main__":
    host = os.getenv("FLASK_RUN_HOST", "0.0.0.0")
    port = int(os.getenv("PORT", os.getenv("FLASK_RUN_PORT", "8080")))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    print(f"Starting Flask app on {host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug)
