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

    # Save original image to disk for record-keeping
    safe_name = secure_filename(original_filename) or "upload.png"
    original_path = os.path.join(folder, f"original_{safe_name}")
    with open(original_path, "wb") as f:
        f.write(image_bytes)

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
        "dot_count": 5,
        "dot_radius": 6,
        "margin": 10,
    }
    if isinstance(modified_label, dict):
        params["dot_count"] = int(modified_label.get("dot_count", params["dot_count"]))
        params["dot_radius"] = int(modified_label.get("dot_radius", params["dot_radius"]))

    img_cv = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = img_cv.shape[:2]

    rng = np.random.default_rng()
    xs = rng.integers(params["margin"] + params["dot_radius"], w - params["margin"] - params["dot_radius"], params["dot_count"])
    ys = rng.integers(params["margin"] + params["dot_radius"], h - params["margin"] - params["dot_radius"], params["dot_count"])
    coords = list(zip(xs.tolist(), ys.tolist()))

    # Draw small visible YELLOW dots (BGR: 0, 255, 255)
    for (x, y) in coords:
        dot_color = np.array([0, 255, 255], dtype=np.uint8)  # pure yellow
        cv2.circle(img_cv, (x, y), params["dot_radius"], dot_color.tolist(), -1)
        # Optional soft halo to blend edges slightly
        cv2.circle(img_cv, (x, y), params["dot_radius"] + 1, dot_color.tolist(), 1)

    processed = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    summary = {
        "applied": True,
        "method": "microdot_overlay_yellow",
        "dot_count": params["dot_count"],
        "dot_radius": params["dot_radius"]
    }
    return processed, summary

def generate_dot_mask(image: Image.Image, dot_coords: list[tuple[int, int]] | None = None, dot_count: int = 100, dot_radius: int = 3, margin: int = 5) -> Image.Image:
    """
    Create a black background + white dot mask of size IMG_SIZE.
    If dot_coords is None, random coordinates are generated within margins.
    """
    # Prepare black canvas
    w, h = IMG_SIZE
    mask = np.zeros((h, w), dtype=np.uint8)

    # Generate random coords if not provided
    if dot_coords is None:
        rng = np.random.default_rng()
        xs = rng.integers(margin + dot_radius, w - margin - dot_radius, dot_count)
        ys = rng.integers(margin + dot_radius, h - margin - dot_radius, dot_count)
        dot_coords = list(zip(xs.tolist(), ys.tolist()))

    # Draw white dots
    for (x, y) in dot_coords:
        cv2.circle(mask, (int(x), int(y)), int(dot_radius), 255, -1)

    # Convert single-channel mask to 3-channel RGB
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(mask_rgb)

def _apply_resolution_degradation(img_bgr: np.ndarray, min_scale=0.5, max_scale=0.85) -> np.ndarray:
    """Downsample and upsample to simulate resolution loss."""
    h, w = img_bgr.shape[:2]
    scale = random.uniform(min_scale, max_scale)
    small = cv2.resize(img_bgr, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return restored

def _add_salt_pepper_noise(gray: np.ndarray, amount=0.01) -> np.ndarray:
    """Add salt & pepper noise to a single-channel image."""
    noisy = gray.copy()
    num_pixels = gray.size
    num_salt = int(amount * num_pixels * 0.5)
    num_pepper = int(amount * num_pixels * 0.5)

    # Salt (white) noise
    coords = (np.random.randint(0, gray.shape[0], num_salt), np.random.randint(0, gray.shape[1], num_salt))
    noisy[coords] = 255
    # Pepper (black) noise
    coords = (np.random.randint(0, gray.shape[0], num_pepper), np.random.randint(0, gray.shape[1], num_pepper))
    noisy[coords] = 0
    return noisy

def apply_augmentations(mask_img: Image.Image) -> Image.Image:
    """
    Apply synthetic print–scan distortions to simulate real-world conditions:
    - Gaussian blur (ENABLED)
    - Brightness/contrast variation (ENABLED)
    - Perspective tilt (ENABLED)
    - Affine shear (ENABLED)
    - Random crop + resize (ENABLED)
    - Resolution degradation (ENABLED)
    - Additive noise (DISABLED)
    Ensures final output is a binary mask (black background + white dots).
    """
    img_bgr = cv2.cvtColor(np.array(mask_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    # 1) Gaussian blur (mild)
    kernel_candidates = [k for k in range(1, 5) if k % 2 == 1]  # 1 or 3
    k = random.choice(kernel_candidates)
    img_bgr = cv2.GaussianBlur(img_bgr, (k, k), 0)

    # 2) Brightness/contrast (small shifts)
    alpha = random.uniform(0.9, 1.1)  # contrast
    beta = random.uniform(-10, 10)    # brightness
    img_bgr = np.clip(img_bgr.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    # 3) Perspective tilt (small)
    tilt = random.uniform(-0.015, 0.015)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [tilt * w, tilt * h],
        [w - tilt * w, tilt * h],
        [w + tilt * w, h - tilt * h],
        [-tilt * w, h - tilt * h]
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    img_bgr = cv2.warpPerspective(img_bgr, M, (w, h), borderValue=(0, 0, 0))

    # 4) Affine shear (small)
    shear_x = random.uniform(-0.03, 0.03)
    shear_y = random.uniform(-0.02, 0.02)
    M_shear = np.float32([[1, shear_x, 0],
                          [shear_y, 1, 0]])
    img_bgr = cv2.warpAffine(img_bgr, M_shear, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))

    # 5) Random crop then resize back (simulate framing)
    crop_scale = random.uniform(0.92, 0.98)  # keep most of the image
    new_w = max(1, int(w * crop_scale))
    new_h = max(1, int(h * crop_scale))
    x0 = 0 if w == new_w else random.randint(0, w - new_w)
    y0 = 0 if h == new_h else random.randint(0, h - new_h)
    cropped = img_bgr[y0:y0 + new_h, x0:x0 + new_w]
    img_bgr = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    # 6) Resolution degradation
    img_bgr = _apply_resolution_degradation(img_bgr)

    # 7) Noise augmentations: DISABLED (keep off)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Re-threshold to enforce binary mask format
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    mask_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(mask_rgb)

def extract_dot_mask_from_scan(scan_img: Image.Image, use_blob: bool = True) -> Image.Image:
    """
    Extract a clean binary mask (black background + white dots) for yellow dots only.
    """
   

    img_bgr = cv2.cvtColor(np.array(scan_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    H, W = img_bgr.shape[:2]

    # Convert to HSV
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # --- Focus on yellow hue range ---
    # Yellow usually around 20–35 in HSV Hue (OpenCV hue range is 0–179)
    lower_yellow = np.array([18, 120, 180], dtype=np.uint8)  # tighter lower bound
    upper_yellow = np.array([40, 255, 255], dtype=np.uint8)  # tighter upper bound
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # --- Morphology to remove noise and close small gaps ---
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # --- Optional blob filtering ---
    if use_blob:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered = np.zeros_like(mask)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 5 or area > 300:  # discard too small/large areas
                continue
            per = cv2.arcLength(cnt, True)
            if per == 0:
                continue
            circularity = (4 * np.pi * area) / (per * per)
            if circularity > 0.4:  # keep roundish blobs
                cv2.drawContours(filtered, [cnt], -1, 255, -1)
        mask = filtered

    # Convert to binary RGB
    mask = (mask > 0).astype(np.uint8) * 255
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(mask_rgb)


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


@app.route('/api/verify', methods=['POST'])
def verify_batch():
    import io, base64
    try:
        image_data = None

        # Support JSON with base64 image
        if request.is_json:
            data = request.get_json(silent=True) or {}
            image_b64 = data.get("image") or data.get("image_base64")
            if not image_b64:
                return jsonify({"status": "error", "error": "Missing 'image' (base64) in JSON payload"}), 400
            if isinstance(image_b64, str) and image_b64.startswith("data:image/"):
                image_b64 = image_b64.split(",", 1)[1]
            try:
                image_data = base64.b64decode(image_b64)
            except Exception as e:
                return jsonify({"status": "error", "error": f"Invalid base64 image: {e}"}), 400

        # Support multipart form upload
        elif 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({"status": "error", "error": "No image file selected"}), 400
            image_data = file.read()
        else:
            return jsonify({"status": "error", "error": "No image provided. Use JSON with base64 'image' or multipart 'image' file"}), 400

        # Load scanned image
        scanned_image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Build dot-only binary mask (white dots on black) using the same training mask logic
        mask_img = extract_dot_mask_from_scan(scanned_image, use_blob=True)

        # Predict on the masked image
        print("🧠 Predicting batch (masked)...")
        img_tensor = preprocess_image(mask_img)
        probs = model.predict(img_tensor)[0]
        top_idx = int(np.argmax(probs))
        top_label = index_to_label[top_idx]
        confidence = float(probs[top_idx])

        return jsonify({
            "status": "ok",
            "predicted_label": top_label,
            "confidence": confidence
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500
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

        image_b64 = data.get("image") or data.get("image_base64")
        if not image_b64:
            return jsonify({"status": "error", "error": "Missing 'image' (base64) in JSON payload"}), 400
        if isinstance(image_b64, str) and image_b64.startswith("data:image/"):
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

        # Also accept product/batch at top-level JSON (your client example)
        top_product = (data.get("product") or data.get("product_name"))
        top_batch = (data.get("batch") or data.get("batch_id") or data.get("batch_name"))
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

        # Also accept product/batch at top-level form
        top_product = (request.form.get("product") or request.form.get("product_name"))
        top_batch = (request.form.get("batch") or request.form.get("batch_id") or request.form.get("batch_name"))

    # Validate image
    if not image_bytes:
        return jsonify({"status": "error", "error": "No image data received"}), 400
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"status": "error", "error": f"Invalid image content: {e}"}), 400

    # Determine product and batch (support both metadata and top-level variants)
    product_name = None
    batch_id = None
    if isinstance(metadata_in, dict):
        product_name = metadata_in.get("product") or metadata_in.get("product_name")
        batch_id = metadata_in.get("batch") or metadata_in.get("batch_id") or metadata_in.get("batch_name")
    # Fallback to top-level (JSON or form) if not present in metadata
    if not product_name:
        product_name = top_product
    if not batch_id:
        batch_id = top_batch

    # Persist only folder and label_id (original image not necessarily saved to disk here)
    saved = save_image_bytes(image_bytes, original_filename, product=product_name, batch=batch_id, label_id=label_id)
    folder = saved["folder"]
    original_path = saved["original_path"]  # may be None by design

    # Build sanitized metadata for DB only
    sanitized_metadata = {}
    if isinstance(metadata_in, dict):
        allowed_keys = {"product", "product_name", "batch", "batch_id", "batch_name", "barcode", "manufacturer", "production_date", "expiry_date", "notes"}
        sanitized_metadata = {k: v for k, v in metadata_in.items() if k in allowed_keys and v is not None}

    # 1) Apply dots to the original uploaded label and save this modified version
    dotted_image, _summary = apply_label_modifications(pil_image, modified_label)
    safe_product = secure_filename(product_name or "unknown_product") or "unknown_product"
    dotted_filename = f"{safe_product}_dotted.png"
    dotted_path = os.path.join(folder, dotted_filename)
    dotted_image.save(dotted_path, format="PNG")

    # 2) Create a base mask from the dotted image
    mask_img = extract_dot_mask_from_scan(dotted_image, use_blob=True)
    mask_filename = f"{safe_product}_mask.png"
    mask_path = os.path.join(folder, mask_filename)
    mask_img.save(mask_path, format="PNG")

    # 3) Generate 20 augmented variants (noise disabled inside apply_augmentations)
    num_variants = 20
    variant_filenames = []
    for i in range(1, num_variants + 1):
        var_img = apply_augmentations(mask_img)
        var_name = f"{safe_product}_mask_var_{i:02d}.png"
        var_path = os.path.join(folder, var_name)
        var_img.save(var_path, format="PNG")
        variant_filenames.append(var_name)

    saved_files = [dotted_filename, mask_filename] + variant_filenames

    # 4) Prepare response (return dotted preview and list of saved files)
    dotted_b64 = image_to_base64_png(dotted_image)
    return jsonify({
        "label_id": label_id,
        "processed_image_base64": f"data:image/png;base64,{dotted_b64}",
        "saved_folder": folder,
        "saved_files": saved_files
    }), 200


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
        # Run the training script using only uploads (the script already sets DATA_DIR='uploads')
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


if __name__ == "__main__":
    host = os.getenv("FLASK_RUN_HOST", "0.0.0.0")
    port = int(os.getenv("PORT", os.getenv("FLASK_RUN_PORT", "8080")))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    print(f"Starting Flask app on {host}:{port} (debug={debug})")
    print("Registered URL map:")
    for r in app.url_map.iter_rules():
        print(f" - {r} [{', '.join(sorted(list(r.methods)))}]")
    app.run(host=host, port=port, debug=debug)


 