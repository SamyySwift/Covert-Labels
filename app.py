from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import register_keras_serializable
from scipy.spatial import Delaunay
from scipy.spatial.distance import directed_hausdorff
import tempfile
import subprocess
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image
import threading
import time
import requests  # Add this import


app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Webhook configuration - Move to top
WEBHOOK_URL = "https://sartor-server-beta.onrender.com/api/v1/label/webhook"
WEBHOOK_TOKEN = "68543ef834267d53d9159dfa5fbfa997053152739e2519a50416742da4c4c31b"

def send_webhook_notification(label_id, status):
    """Send webhook notification when training completes or fails"""
    try:
        headers = {
            's-token': WEBHOOK_TOKEN,
            'Content-Type': 'application/json'
        }
        
        payload = {
            'label_id': label_id,
            'status': status
        }
        
        response = requests.post(
            WEBHOOK_URL,
            json=payload,
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            print(f"✅ Webhook notification sent successfully for label_id: {label_id}, status: {status}")
        else:
            print(f"❌ Webhook notification failed. Status: {response.status_code}, Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error sending webhook notification: {str(e)}")

# =====================
# Register Custom Parts
# =====================
@register_keras_serializable()
def contrastive_loss(y_true, y_pred):
    margin = 1.0
    y_true = tf.cast(y_true, tf.float32)
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)

@register_keras_serializable()
def l2_normalize(x):
    return tf.math.l2_normalize(x, axis=-1)

@register_keras_serializable()
def euclidean_distance(tensors):
    a, b = tensors
    return tf.norm(a - b, axis=1, keepdims=True)

# Configuration
IMG_SIZE = (224, 224)
AUGMENTED_AUTH_DIR = "dataset/authentic_augmented"
FINGERPRINT_DIR = "dataset/fingerprints"
DB_PATH = "product_db.json"
# Update model paths to use the persistent volume
MODEL_PATH = "app/models/siamese_contrastive.keras"
FALLBACK_MODEL_PATH = "siamese_contrastive.keras"  # Fallback to local if volume not available

# =====================
# Load Model and Database
# =====================
def load_model_and_db():
    global product_db
    
    # Try to load from persistent volume first
    model_path = MODEL_PATH if os.path.exists(MODEL_PATH) else FALLBACK_MODEL_PATH
    
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(
                model_path,
                custom_objects={
                    'contrastive_loss': contrastive_loss,
                    'l2_normalize': l2_normalize,
                    'euclidean_distance': euclidean_distance
                }
            )
            print(f"✅ Siamese model loaded from {model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            model = None
    else:
        print("⚠️ No trained model found")
        model = None
    
    # Load product database
    try:
        with open(DB_PATH, "r") as f:
            product_db = json.load(f)
    except Exception as e:
        print(f"Error loading database: {e}")
        product_db = {}
    
    return model, product_db

# Initialize model and database
model, product_db = load_model_and_db()

UPLOAD_FOLDER = "uploads"
AUTHENTIC_LABELS_DIR = "authentic_labels"
TRAINING_STATUS = {
    "status": "idle", 
    "progress": 0, 
    "message": "",
    "label_id": None
}

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)



# =====================
# Helper Functions
# =====================
def load_and_preprocess(image_data):
    """Preprocess image data for model input"""
    try:
        if isinstance(image_data, str):  # Base64 string
            image_data = base64.b64decode(image_data)
        
        # Convert bytes to PIL Image
        img = Image.open(BytesIO(image_data))
        img = img.convert('RGB')
        
        # Convert to numpy array and resize
        img_array = np.array(img)
        img_resized = cv2.resize(img_array, IMG_SIZE)
        
        # Normalize
        img_normalized = img_resized.astype(np.float32) / 255.0
        return np.expand_dims(img_normalized, axis=0)
        
    except Exception as e:
        print(f"Error in load_and_preprocess: {e}")
        raise

def extract_delaunay_fingerprint(img):
    """Extract Delaunay triangulation fingerprint"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=50, qualityLevel=0.01, minDistance=10)
    
    if corners is None or len(corners) < 4:
        return np.array([])
    
    points = corners.reshape(-1, 2)
    try:
        tri = Delaunay(points)
        return tri.simplices.flatten()
    except:
        return np.array([])

def compare_fingerprints(f1, f2):
    """Compare two fingerprints using Hausdorff distance"""
    if len(f1) == 0 or len(f2) == 0:
        return 0.0
    
    try:
        f1_2d = f1.reshape(-1, 1)
        f2_2d = f2.reshape(-1, 1)
        distance = directed_hausdorff(f1_2d, f2_2d)[0]
        return max(0, 1 - distance / 100)
    except:
        return 0.0

def preprocess_scanned_image(img):
    """Preprocess scanned image for better pattern detection"""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    return blurred

def detect_robust_watermark(img, expected_batch_id):
    """Detect robust watermark patterns"""
    try:
        preprocessed = preprocess_scanned_image(img)
        h, w = preprocessed.shape
        
        batch_hash = hash(expected_batch_id) % 1000000
        np.random.seed(batch_hash)
        
        block_size = 8  # Match the embedding block size
        
        expected_pattern = np.zeros((h, w), dtype=np.float32)
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                if np.random.random() > 0.5:  # Match embedding probability
                    variation = np.random.choice([-3, -2, 2, 3])  # REDUCED from ±12
                    expected_pattern[y:y+block_size, x:x+block_size] = variation
        
        # More sensitive correlation
        correlation = cv2.matchTemplate(preprocessed.astype(np.float32), expected_pattern, cv2.TM_CCOEFF_NORMED)
        max_corr = np.max(correlation)
        
        # Lower threshold for better detection
        score = max(0, min(1, (max_corr + 0.3) * 2))  # Adjusted scoring
        return score
        
    except Exception as e:
        print(f"Watermark detection error: {e}")
        return 0.0

def detect_texture_signature(img, expected_batch_id):
    """Detect texture-based signatures"""
    try:
        preprocessed = preprocess_scanned_image(img)
        
        # Generate batch-specific texture parameters
        batch_hash = hash(expected_batch_id) % 1000000
        np.random.seed(batch_hash)
        
        texture_scores = []
        
        # Method 1: Gabor filter bank
        angles = [0, 45, 90, 135]
        frequencies = [0.1, 0.3, 0.5]
        
        for angle in angles:
            for freq in frequencies:
                kernel = cv2.getGaborKernel((21, 21), 5, np.radians(angle), 2*np.pi*freq, 0.5, 0, ktype=cv2.CV_32F)
                filtered = cv2.filter2D(preprocessed, cv2.CV_8UC3, kernel)
                
                # Expected response based on batch
                expected_response = np.random.uniform(0.3, 0.7)
                actual_response = np.mean(filtered) / 255.0
                
                similarity = 1 - abs(expected_response - actual_response)
                texture_scores.append(similarity)
        
        # Method 2: Local Binary Pattern (LBP)
        def local_binary_pattern(img, radius=3, n_points=24):
            lbp = np.zeros_like(img)
            for i in range(radius, img.shape[0] - radius):
                for j in range(radius, img.shape[1] - radius):
                    center = img[i, j]
                    binary_string = ''
                    for k in range(n_points):
                        angle = 2 * np.pi * k / n_points
                        x = int(i + radius * np.cos(angle))
                        y = int(j + radius * np.sin(angle))
                        if 0 <= x < img.shape[0] and 0 <= y < img.shape[1]:
                            binary_string += '1' if img[x, y] >= center else '0'
                    lbp[i, j] = int(binary_string, 2) if binary_string else 0
            return lbp
        
        lbp = local_binary_pattern(preprocessed)
        lbp_hist = np.histogram(lbp, bins=256)[0]
        lbp_hist = lbp_hist / np.sum(lbp_hist)  # Normalize
        
        # Expected LBP signature
        expected_lbp = np.random.uniform(0, 1, 256)
        expected_lbp = expected_lbp / np.sum(expected_lbp)
        
        lbp_similarity = 1 - np.sum(np.abs(lbp_hist - expected_lbp)) / 2
        texture_scores.append(lbp_similarity)
        
        # Method 3: Edge density analysis
        edges = cv2.Canny(preprocessed, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        expected_edge_density = np.random.uniform(0.1, 0.3)
        edge_similarity = 1 - abs(edge_density - expected_edge_density) / max(edge_density, expected_edge_density)
        texture_scores.append(edge_similarity)
        
        # Combine all texture scores
        final_score = np.mean(texture_scores)
        
        # Amplify for better discrimination
        amplified_score = min(1.0, final_score * 1.8)
        
        return max(0.0, amplified_score)
        
    except Exception as e:
        print(f"Texture detection error: {e}")
        return 0.0

def add_invisible_microdots(img, batch_id, dot_count=10):
    """Add exactly 10 slightly visible microdots for authentication"""
    h, w = img.shape[:2]
    batch_hash = hash(batch_id) % 1000
    np.random.seed(batch_hash)
    
    # Generate exactly 10 dot positions
    coords = []
    for _ in range(dot_count):
        x = np.random.randint(20, w - 20)  # Keep dots away from edges
        y = np.random.randint(20, h - 20)
        coords.append((x, y))
    
    # Slightly visible dots that still blend reasonably with background
    dot_radius = 2
    
    # Draw dots that are slightly darker/lighter than background
    for (x, y) in coords:
        # Sample the local background color in a 5x5 area
        y1, y2 = max(0, y-2), min(h, y+3)
        x1, x2 = max(0, x-2), min(w, x+3)
        local_region = img[y1:y2, x1:x2]
        
        # Calculate average color of the local area
        if len(img.shape) == 3:  # Color image
            avg_color = np.mean(local_region, axis=(0, 1))
            # Add more noticeable variation (±8 to ±12) to make dots slightly visible
            variation = np.random.randint(-12, -8) if np.random.random() > 0.5 else np.random.randint(8, 13)
            dot_color = tuple(int(np.clip(c + variation, 0, 255)) for c in avg_color)
        else:  # Grayscale image
            avg_color = np.mean(local_region)
            variation = np.random.randint(-12, -8) if np.random.random() > 0.5 else np.random.randint(8, 13)
            dot_color = int(np.clip(avg_color + variation, 0, 255))
        
        cv2.circle(img, (x, y), dot_radius, dot_color, -1)
    
    return img

def add_print_resistant_patterns(img, batch_id):
    """Add patterns that survive print/camera lifecycle - microdots only"""
    
    # 1. Invisible watermark (most reliable)
    img = add_robust_watermark(img, batch_id)
    
    # 2. Removed texture signatures completely
    # img = add_texture_signature(img, batch_id)  # REMOVED
    
    # 3. Invisible microdots with exactly 10 dots
    img = add_invisible_microdots(img, batch_id, dot_count=10)
    
    return img

def add_robust_watermark(img, batch_id):
    """Watermark optimized for print/scan survival - REDUCED STRENGTH"""
    h, w = img.shape[:2]
    batch_hash = hash(batch_id) % 1000
    np.random.seed(batch_hash)
    
    # Smaller blocks for finer detail
    block_size = 8
    
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            if np.random.random() > 0.5:  # 50% of blocks
                # MUCH GENTLER variation - reduced from ±12 to ±3
                variation = np.random.choice([-3, -2, 2, 3])  # REDUCED STRENGTH
                
                img[y:y+block_size, x:x+block_size] = np.clip(
                    img[y:y+block_size, x:x+block_size].astype(np.int16) + variation,
                    0, 255
                ).astype(np.uint8)
    
    return img

def add_texture_signature(img, batch_id):
    """Texture patterns optimized for print/camera - REDUCED STRENGTH"""
    h, w = img.shape[:2]
    batch_hash = hash(batch_id) % 1000
    np.random.seed(batch_hash)
    
    # Smaller kernel for subtlety
    kernel_size = 3  # Reduced from 5
    # MUCH gentler texture kernel
    texture_kernel = np.random.uniform(-0.02, 0.02, (kernel_size, kernel_size))  # Reduced from ±0.08
    texture_kernel = texture_kernel / np.sum(np.abs(texture_kernel))
    
    # Fewer regions for subtlety
    num_regions = 10  # Reduced from 25
    for _ in range(num_regions):
        y = np.random.randint(0, h - kernel_size)
        x = np.random.randint(0, w - kernel_size)
        
        region = img[y:y+kernel_size, x:x+kernel_size].astype(np.float32)
        # Apply gentler texture effect
        textured = cv2.filter2D(region, -1, texture_kernel * 2)  # Much gentler effect
        # Ensure values stay within bounds and convert back to uint8
        result = np.clip(textured, 0, 255).astype(np.uint8)
        img[y:y+kernel_size, x:x+kernel_size] = result
    
    return img

def detect_print_resistant_patterns(img, expected_batch_id):
    """Combined print-resistant pattern detection"""
    watermark_score = detect_robust_watermark(img, expected_batch_id)
    texture_score = detect_texture_signature(img, expected_batch_id)
    
    # Weighted combination with print tolerance
    combined_score = 0.6 * watermark_score + 0.4 * texture_score
    
    # Lower threshold for print degradation tolerance
    authenticity_threshold = 0.4
    is_authentic = combined_score >= authenticity_threshold
    
    return {
        'watermark_score': float(watermark_score),  # Convert to Python float
        'texture_score': float(texture_score),  # Convert to Python float
        'combined_score': float(combined_score),  # Convert to Python float
        'is_authentic': bool(is_authentic),  # Ensure it's a Python bool
        'confidence': float(min(1.0, combined_score / authenticity_threshold))  # Convert to Python float
    }

def verify_images(image_files):
    """Main verification function"""
    if not model:
        return {"error": "Model not loaded"}, 500
    
    try:
        scanned_imgs = []
        scanned_fingerprints = []
        original_images = []  # Store original images for pattern detection
        
        # Process uploaded images
        for image_file in image_files:
            # Read image data once
            image_data = image_file.read()
            
            # Decode image
            img_array = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is None:
                return {"error": f"Failed to decode image: {image_file.filename}"}, 400
            
            img_resized = cv2.resize(img, IMG_SIZE)
            
            # Store original image for pattern detection
            original_images.append(img.copy())
            
            # Preprocess for model
            preprocessed = load_and_preprocess(image_data)
            scanned_imgs.append(preprocessed)
            
            # Extract fingerprint
            fingerprint = extract_delaunay_fingerprint(img_resized)
            scanned_fingerprints.append(fingerprint)
        
        best_score = 0
        best_batch = None
        best_pattern_result = None
        
        # Compare against all batches
        for batch_id in os.listdir(AUGMENTED_AUTH_DIR):
            batch_path = os.path.join(AUGMENTED_AUTH_DIR, batch_id)
            if not os.path.isdir(batch_path):
                continue
                
            ref_imgs = [f for f in os.listdir(batch_path) if f.endswith(".jpg")]
            if not ref_imgs:
                continue
            
            siamese_scores = []
            orb_scores = []
            pattern_scores = []
            
            for i, (scan_img, scan_fp, original_img) in enumerate(zip(scanned_imgs, scanned_fingerprints, original_images)):
                # Pattern detection on original image
                pattern_result = detect_print_resistant_patterns(original_img, batch_id)
                pattern_scores.append(float(pattern_result['combined_score']))  # Convert to Python float
                
                # Compare with reference images
                for ref_name in ref_imgs:
                    ref_path = os.path.join(batch_path, ref_name)
                    with open(ref_path, 'rb') as ref_file:
                        ref_data = ref_file.read()
                    ref_img = load_and_preprocess(ref_data)
                    
                    # Siamese similarity
                    distance = model.predict([scan_img, ref_img], verbose=0)[0][0]
                    similarity_score = float(1 / (1 + distance))  # Convert to Python float
                    siamese_scores.append(similarity_score)
                    
                    # ORB fingerprint comparison
                    fp_path = os.path.join(FINGERPRINT_DIR, f"{batch_id}_{ref_name[:-4]}.npy")
                    if os.path.exists(fp_path):
                        ref_fp = np.load(fp_path)
                        orb_score = float(compare_fingerprints(scan_fp, ref_fp))  # Convert to Python float
                        orb_scores.append(orb_score)
            
            if not siamese_scores or not orb_scores or not pattern_scores:
                continue
            
            # Calculate combined score
            siamese_avg = float(np.mean(siamese_scores))  # Convert to Python float
            orb_avg = float(np.mean(orb_scores))  # Convert to Python float
            pattern_avg = float(np.mean(pattern_scores))  # Convert to Python float
            
            combined_score = 0.6 * siamese_avg + 0.3 * orb_avg + 0.4 * pattern_avg
            
            if combined_score > best_score:
                best_score = combined_score
                best_batch = batch_id
                best_pattern_result = {
                    'watermark_score': pattern_avg,
                    'texture_score': pattern_avg,
                    'combined_score': pattern_avg
                }
        
        # Determine authenticity
        is_authentic = best_score >= 0.73
        product_info = product_db.get(best_batch, {}) if best_batch else {}
        
        result = {
            "is_authentic": is_authentic,
            "confidence_score": round(float(best_score), 3),  # Convert to Python float
            "matched_batch": best_batch,
            "product_info": product_info,
            "analysis": {
                "siamese_score": round(float(np.mean(siamese_scores)) if siamese_scores else 0, 3),  # Convert to Python float
                "orb_score": round(float(np.mean(orb_scores)) if orb_scores else 0, 3),  # Convert to Python float
                "pattern_score": round(float(best_pattern_result['combined_score']) if best_pattern_result else 0, 3)  # Convert to Python float
            },
            "threshold": 0.60,
            "timestamp": datetime.now().isoformat()
        }
        
        return result
        
    except Exception as e:
        return {"error": f"Verification failed: {str(e)}"}, 500

def train_model_async(label_id=None):
    """Train model in background thread with dynamic progress"""
    global TRAINING_STATUS
    
    try:
        TRAINING_STATUS = {
            "status": "training", 
            "progress": 0, 
            "message": "Starting dataset generation...", 
            "timestamp": datetime.now().isoformat(),
            "label_id": label_id
        }
        
        # Step 1: Generate dataset first
        print("🔄 Generating dataset...")
        dataset_process = subprocess.Popen(
            ["python", "generate_dataset.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        dataset_stdout, dataset_stderr = dataset_process.communicate()
        
        if dataset_process.returncode != 0:
            TRAINING_STATUS = {
                "status": "failed", 
                "progress": 0, 
                "message": f"Dataset generation failed: {dataset_stderr}",
                "timestamp": datetime.now().isoformat(),
                "label_id": label_id
            }
            # Send webhook notification for failure
            if label_id:
                send_webhook_notification(label_id, "failed")
            return
        
        TRAINING_STATUS = {
            "status": "training", 
            "progress": 20, 
            "message": "Dataset generated. Starting model training...",
            "timestamp": datetime.now().isoformat(),
            "label_id": label_id
        }
        
        # Step 2: Train the model with real-time progress
        print("🔄 Training model...")
        training_process = subprocess.Popen(
            ["python", "train_siamese.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stderr with stdout
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        total_epochs = 10  # From train_siamese.py line 132
        current_epoch = 0
        
        # Read output line by line for real-time progress
        while True:
            output = training_process.stdout.readline()
            if output == '' and training_process.poll() is not None:
                break
            
            if output:
                print(f"Training output: {output.strip()}")  # Debug log
                
                # Parse epoch information from TensorFlow output
                if "Epoch" in output and "/" in output:
                    try:
                        # Look for patterns like "Epoch 1/10" or "Epoch 3/10"
                        import re
                        epoch_match = re.search(r'Epoch (\d+)/(\d+)', output)
                        if epoch_match:
                            current_epoch = int(epoch_match.group(1))
                            total_epochs = int(epoch_match.group(2))
                            
                            # Calculate progress: 20% for dataset + 80% for training
                            training_progress = (current_epoch / total_epochs) * 80
                            total_progress = 20 + training_progress
                            
                            TRAINING_STATUS = {
                                "status": "training",
                                "progress": round(total_progress, 1),
                                "message": f"Training epoch {current_epoch}/{total_epochs}",
                                "timestamp": datetime.now().isoformat(),
                                "label_id": label_id
                            }
                    except (ValueError, AttributeError):
                        pass  # Skip if parsing fails
                
                # Look for completion indicators
                elif "model trained and saved" in output.lower() or "✅" in output:
                    TRAINING_STATUS = {
                        "status": "training",
                        "progress": 95,
                        "message": "Model saved. Finalizing...",
                        "timestamp": datetime.now().isoformat(),
                        "label_id": label_id
                    }
        
        # Wait for process to complete
        training_process.wait()
        
        if training_process.returncode == 0:
            TRAINING_STATUS = {
                "status": "completed", 
                "progress": 100, 
                "message": "Training completed successfully",
                "timestamp": datetime.now().isoformat(),
                "label_id": label_id
            }
            # Reload model
            global model
            model, _ = load_model_and_db()
            
            # Send webhook notification for success
            if label_id:
                send_webhook_notification(label_id, "completed")
        else:
            TRAINING_STATUS = {
                "status": "failed", 
                "progress": TRAINING_STATUS.get("progress", 20), 
                "message": "Training failed - check logs for details",
                "timestamp": datetime.now().isoformat(),
                "label_id": label_id
            }
            # Send webhook notification for failure
            if label_id:
                send_webhook_notification(label_id, "failed")
            
    except Exception as e:
        TRAINING_STATUS = {
            "status": "failed", 
            "progress": 0, 
            "message": f"Training error: {str(e)}",
            "timestamp": datetime.now().isoformat(),
            "label_id": label_id
        }
        # Send webhook notification for failure
        if label_id:
            send_webhook_notification(label_id, "failed")

# =====================
# API Endpoints
# =====================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/verify', methods=['POST'])
def verify_label():
    """Verify uploaded label images"""
    global TRAINING_STATUS
    
    try:
        if 'images' not in request.files:
            return jsonify({"error": "No images provided"}), 400
        
        files = request.files.getlist('images')
        if not files or len(files) == 0:
            return jsonify({"error": "No images provided"}), 400
        
        # Verify images
        result = verify_images(files)
        
        if isinstance(result, tuple):  # Error case
            return jsonify(result[0]), result[1]
        
        # If not authentic, remove sensitive metadata
        if not result.get('is_authentic', False):
            # Create a filtered result without metadata
            filtered_result = {
                "is_authentic": result.get('is_authentic', False),
                "confidence_score": result.get('confidence_score', 0),
                "analysis": result.get('analysis', {}),
                "threshold": result.get('threshold', 0.60),
                "timestamp": result.get('timestamp')
            }
            result = filtered_result
        
        # Automatically extract label_id from training status if available
        label_id = TRAINING_STATUS.get('label_id')
        if label_id:
            result['label_id'] = label_id
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": f"Verification failed: {str(e)}"}), 500


@app.route('/api/train', methods=['POST'])
def trigger_training():
    """Trigger model training"""
    global TRAINING_STATUS
    
    if TRAINING_STATUS["status"] == "training":
        return jsonify({"error": "Training already in progress"}), 400
    
    try:
        # Get label_id from request body
        data = request.get_json() or {}
        label_id = data.get('label_id')
        
        if not label_id:
            return jsonify({"error": "label_id is required"}), 400
        
        # Start training in background thread with label_id
        training_thread = threading.Thread(target=train_model_async, args=(label_id,))
        training_thread.daemon = True
        training_thread.start()
        
        return jsonify({
            "message": "Training started",
            "status": "training",
            "label_id": label_id,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": f"Failed to start training: {str(e)}"}), 500

@app.route('/api/training-status', methods=['GET'])
def get_training_status():
    """Get current training status"""
    return jsonify({
        **TRAINING_STATUS,
        "timestamp": datetime.now().isoformat()
    })


@app.route('/api/upload-labels', methods=['POST'])
def upload_labels():
    print("\n=== UPLOAD LABELS ENDPOINT CALLED ===")
    print(f"Request method: {request.method}")
    print(f"Content-Type: {request.content_type}")
    print(f"Request headers: {dict(request.headers)}")
    
    try:
        print("\n--- Checking for images in request ---")
        if 'images' not in request.files:
            print("ERROR: No 'images' key found in request.files")
            print(f"Available keys in request.files: {list(request.files.keys())}")
            return jsonify({'error': 'No images provided'}), 400
        
        print("✓ Found 'images' in request.files")
        
        files = request.files.getlist('images')
        print(f"Number of files received: {len(files)}")
        
        print("\n--- Extracting form data ---")
        batch_id = request.form.get('batch_id')
        product_name = request.form.get('product_name', 'unknown')
        sku = request.form.get('sku', 'unknown')
        
        print(f"batch_id: {batch_id}")
        print(f"product_name: {product_name}")
        print(f"sku: {sku}")
        print(f"All form data: {dict(request.form)}")
        
        if not batch_id:
            print("ERROR: batch_id is required but not provided")
            return jsonify({'error': 'batch_id is required'}), 400
        
        print("✓ batch_id validation passed")
        
        # Create the dataset/authentic structure
        AUTH_DIR = "dataset/authentic"
        batch_path = os.path.join(AUTH_DIR, batch_id)
        print(f"\n--- Creating directory structure ---")
        print(f"AUTH_DIR: {AUTH_DIR}")
        print(f"batch_path: {batch_path}")
        
        os.makedirs(batch_path, exist_ok=True)
        print(f"✓ Directory created/verified: {batch_path}")
        
        uploaded_files = []
        modified_images = []
        
        print(f"\n--- Processing {len(files)} files ---")
        
        for i, file in enumerate(files):
            print(f"\nProcessing file {i+1}/{len(files)}")
            print(f"File object: {file}")
            print(f"Filename: {file.filename}")
            
            if file and file.filename:
                print(f"✓ File {i+1} has valid filename: {file.filename}")
                
                try:
                    # Read file data
                    print(f"Reading file data for: {file.filename}")
                    file_data = file.read()
                    print(f"File data length: {len(file_data)} bytes")
                    
                    if len(file_data) == 0:
                        print(f"WARNING: File {file.filename} is empty, skipping")
                        continue
                    
                    # Convert to OpenCV image for processing
                    print(f"Converting to OpenCV image: {file.filename}")
                    nparr = np.frombuffer(file_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if img is None:
                        print(f"ERROR: Could not decode image: {file.filename}")
                        continue
                    
                    print(f"✓ Image decoded successfully: {img.shape}")
                    
                    # Apply badge-specific micro pattern embedding
                    print(f"Applying print-resistant patterns to: {file.filename}")
                    modified_img = add_print_resistant_patterns(img.copy(), batch_id)
                    print(f"✓ Patterns applied successfully")
                    
                    # Convert modified image to base64 for response
                    print(f"Converting to base64: {file.filename}")
                    _, buffer = cv2.imencode('.jpg', modified_img)
                    modified_img_base64 = base64.b64encode(buffer).decode('utf-8')
                    print(f"✓ Base64 conversion complete, length: {len(modified_img_base64)}")
                    
                    # Use original filename (no timestamp prefix needed)
                    filename = file.filename
                    file_path = os.path.join(batch_path, filename)
                    print(f"Saving to: {file_path}")
                    
                    # Save the modified image to the batch folder
                    cv2.imwrite(file_path, modified_img)
                    print(f"✓ File saved successfully: {file_path}")
                    
                    uploaded_files.append({
                        'filename': filename,
                        'file_path': file_path,
                        'batch_id': batch_id,
                        'product_name': product_name,
                        'sku': sku
                    })
                    
                    modified_images.append({
                        'filename': filename,
                        'modified_image': modified_img_base64
                    })
                    
                    print(f"✓ File {i+1} processed successfully: {filename}")
                    
                except Exception as file_error:
                    print(f"ERROR processing file {file.filename}: {str(file_error)}")
                    print(f"Error type: {type(file_error).__name__}")
                    import traceback
                    print(f"Traceback: {traceback.format_exc()}")
                    continue
            else:
                print(f"WARNING: File {i+1} is invalid or has no filename, skipping")
        
        print(f"\n--- File processing complete ---")
        print(f"Successfully processed: {len(uploaded_files)} files")
        print(f"Files in uploaded_files: {[f['filename'] for f in uploaded_files]}")
        
        if len(uploaded_files) == 0:
            print("ERROR: No files were successfully processed")
            return jsonify({'error': 'No valid images were processed'}), 400
        
        # Update product database to match generate_dataset.py format
        print(f"\n--- Updating product database ---")
        print(f"Calling update_product_database_local with:")
        print(f"  batch_id: {batch_id}")
        print(f"  product_name: {product_name}")
        print(f"  sku: {sku}")
        print(f"  files count: {len(uploaded_files)}")
        
        update_product_database_local(batch_id, product_name, sku, uploaded_files)
        print(f"✓ Product database updated")
        
        response_data = {
            'message': f'Successfully uploaded {len(uploaded_files)} images to batch {batch_id}',
            'batch_id': batch_id,
            'uploaded_files': len(uploaded_files),
            'batch_path': batch_path,
            'modified_images': modified_images
        }
        
        print(f"\n--- Preparing response ---")
        print(f"Response message: {response_data['message']}")
        print(f"Response batch_id: {response_data['batch_id']}")
        print(f"Response uploaded_files count: {response_data['uploaded_files']}")
        print(f"Response batch_path: {response_data['batch_path']}")
        print(f"Modified images count: {len(response_data['modified_images'])}")
        
        print("\n=== UPLOAD LABELS ENDPOINT SUCCESS ===")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"\n!!! CRITICAL ERROR in upload_labels !!!")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        print(f"Request method: {request.method}")
        print(f"Request content_type: {request.content_type}")
        print(f"Request form keys: {list(request.form.keys()) if hasattr(request, 'form') else 'No form'}")
        print(f"Request files keys: {list(request.files.keys()) if hasattr(request, 'files') else 'No files'}")
        print("=== END CRITICAL ERROR ===")
        return jsonify({'error': str(e)}), 500

def update_product_database_local(batch_id, product_name, sku, files):
    """Update product database to match generate_dataset.py format"""
    DB_PATH = "product_db.json"
    
    # Load existing database
    if os.path.exists(DB_PATH):
        try:
            with open(DB_PATH, 'r') as f:
                content = f.read().strip()
                if content:  # Check if file has content
                    product_db = json.loads(content)
                else:
                    print("Warning: product_db.json is empty, initializing new database")
                    product_db = {}
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Warning: Could not load product_db.json ({e}), initializing new database")
            product_db = {}
    else:
        product_db = {}
    
    # Add or update batch entry
    product_db[batch_id] = {
        "batch_id": batch_id,
        "product_name": product_name.replace("_", " ").title(),
        "sku": sku,
        "created_at": datetime.now().isoformat(),
        "image_count": len(files)
    }
    
    # Save updated database
    try:
        with open(DB_PATH, 'w') as f:
            json.dump(product_db, f, indent=2)
        print(f"✓ Product database updated successfully")
    except Exception as e:
        print(f"Error saving product database: {e}")
        raise

def update_product_database(batch_id, product_name, sku, files):
    """Update product database stored locally"""
    try:
        # Load existing database
        if os.path.exists(DB_PATH):
            with open(DB_PATH, 'r') as f:
                product_db = json.load(f)
        else:
            product_db = {}
        
        # Update database
        if batch_id not in product_db:
            product_db[batch_id] = {
                'product_name': product_name,
                'sku': sku,
                'created_at': datetime.now().isoformat(),
                'files': []
            }
        
        product_db[batch_id]['files'].extend(files)
        
        # Save updated database
        with open(DB_PATH, 'w') as f:
            json.dump(product_db, f, indent=2)
        
    except Exception as e:
        print(f"Error updating product database: {e}")

# Replace blob functions with local file storage
def save_uploaded_file(file_data, filename, folder="uploads"):
    """Save file data to local storage"""
    try:
        folder_path = os.path.join(folder)
        os.makedirs(folder_path, exist_ok=True)
        
        file_path = os.path.join(folder_path, filename)
        
        if isinstance(file_data, bytes):
            with open(file_path, 'wb') as f:
                f.write(file_data)
        else:
            with open(file_path, 'w') as f:
                f.write(file_data)
        
        return file_path
    except Exception as e:
        print(f"Error saving file: {e}")
        return None



if __name__ == '__main__':
    print("🚀 Starting Label Authentication API...")
    print(f"📊 Model loaded: {model is not None}")
    print(f"📦 Products in database: {len(product_db)}")
    
    # Use PORT environment variable (Fly.io sets this to 8080)
    port = int(os.environ.get('PORT', 3000))
    app.run(host='0.0.0.0', port=port, debug=False)

