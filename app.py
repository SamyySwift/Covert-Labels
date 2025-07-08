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


app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Configuration
IMG_SIZE = (224, 224)
AUGMENTED_AUTH_DIR = "dataset/authentic_augmented"
FINGERPRINT_DIR = "dataset/fingerprints"
DB_PATH = "product_db.json"
MODEL_PATH = "siamese_contrastive.keras"
UPLOAD_FOLDER = "uploads"
AUTHENTIC_LABELS_DIR = "authentic_labels"
TRAINING_STATUS = {"status": "idle", "progress": 0, "message": ""}

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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

# =====================
# Load Model and Database
# =====================
def load_model_and_db():
    try:
        model = load_model(MODEL_PATH, custom_objects={
            "contrastive_loss": contrastive_loss,
            "l2_normalize": l2_normalize,
            "euclidean_distance": euclidean_distance
        })
        
        with open(DB_PATH, "r") as f:
            product_db = json.load(f)
            
        return model, product_db
    except Exception as e:
        print(f"Error loading model or database: {e}")
        return None, {}

model, product_db = load_model_and_db()

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
        
        # Generate expected pattern based on batch ID
        batch_hash = hash(expected_batch_id) % 1000000
        np.random.seed(batch_hash)
        
        # Create smaller blocks for better print resistance
        block_size = 16  # Increased from 8 for better print durability
        pattern_strength = 15  # Increased strength
        
        expected_pattern = np.zeros((h, w), dtype=np.float32)
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                if np.random.random() > 0.5:
                    expected_pattern[y:y+block_size, x:x+block_size] = pattern_strength
        
        # Multiple correlation methods for robustness
        correlations = []
        
        # Method 1: Template matching
        result = cv2.matchTemplate(preprocessed.astype(np.float32), 
                                 expected_pattern.astype(np.float32), 
                                 cv2.TM_CCOEFF_NORMED)
        correlations.append(np.max(result))
        
        # Method 2: Frequency domain analysis
        f_img = np.fft.fft2(preprocessed)
        f_pattern = np.fft.fft2(expected_pattern)
        correlation_freq = np.fft.ifft2(f_img * np.conj(f_pattern))
        correlations.append(np.abs(correlation_freq).max() / (h * w))
        
        # Method 3: Adaptive thresholding correlation
        thresh_img = cv2.adaptiveThreshold(preprocessed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        thresh_pattern = cv2.adaptiveThreshold((expected_pattern * 255).astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        correlation_adaptive = cv2.matchTemplate(thresh_img.astype(np.float32), thresh_pattern.astype(np.float32), cv2.TM_CCOEFF_NORMED)
        correlations.append(np.max(correlation_adaptive))
        
        # Combine correlations with weights
        weights = [0.4, 0.3, 0.3]
        final_score = sum(w * c for w, c in zip(weights, correlations))
        
        # Amplify score for better discrimination
        amplified_score = min(1.0, final_score * 2.5)
        
        return max(0.0, amplified_score)
        
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
            
            combined_score = 0.4 * siamese_avg + 0.3 * orb_avg + 0.3 * pattern_avg
            
            if combined_score > best_score:
                best_score = combined_score
                best_batch = batch_id
                best_pattern_result = {
                    'watermark_score': pattern_avg,
                    'texture_score': pattern_avg,
                    'combined_score': pattern_avg
                }
        
        # Determine authenticity
        is_authentic = best_score >= 0.65
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
            "threshold": 0.65,
            "timestamp": datetime.now().isoformat()
        }
        
        return result
        
    except Exception as e:
        return {"error": f"Verification failed: {str(e)}"}, 500

def train_model_async():
    """Train model in background thread"""
    global TRAINING_STATUS
    
    try:
        TRAINING_STATUS = {"status": "training", "progress": 0, "message": "Starting dataset generation..."}
        
        # Step 1: Generate dataset first
        print("🔄 Generating dataset...")
        dataset_process = subprocess.Popen(
            ["python", "generate_dataset.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for dataset generation to complete
        dataset_stdout, dataset_stderr = dataset_process.communicate()
        
        if dataset_process.returncode != 0:
            TRAINING_STATUS = {
                "status": "failed", 
                "progress": 0, 
                "message": f"Dataset generation failed: {dataset_stderr}"
            }
            return
        
        TRAINING_STATUS = {"status": "training", "progress": 25, "message": "Dataset generated. Starting model training..."}
        
        # Step 2: Train the model
        print("🔄 Training model...")
        training_process = subprocess.Popen(
            ["python", "train_siamese.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Monitor training progress (simplified)
        for i in range(1, 16):  # Assuming 15 epochs after dataset generation
            time.sleep(10)  # Simulate training time
            progress = 25 + (i / 15) * 75  # 25% for dataset + 75% for training
            TRAINING_STATUS["progress"] = progress
            TRAINING_STATUS["message"] = f"Training epoch {i}/15"
            
            if training_process.poll() is not None:
                break
        
        training_stdout, training_stderr = training_process.communicate()
        
        if training_process.returncode == 0:
            TRAINING_STATUS = {
                "status": "completed", 
                "progress": 100, 
                "message": "Training completed successfully"
            }
            # Reload model
            global model
            model, _ = load_model_and_db()
        else:
            TRAINING_STATUS = {
                "status": "failed", 
                "progress": 0, 
                "message": f"Training failed: {training_stderr}"
            }
            
    except Exception as e:
        TRAINING_STATUS = {
            "status": "failed", 
            "progress": 0, 
            "message": f"Training error: {str(e)}"
        }

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
        # Start training in background thread
        training_thread = threading.Thread(target=train_model_async)
        training_thread.daemon = True
        training_thread.start()
        
        return jsonify({
            "message": "Training started",
            "status": "training",
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
    try:
        if 'images' not in request.files:
            return jsonify({'error': 'No images provided'}), 400
        
        files = request.files.getlist('images')
        batch_id = request.form.get('batch_id')
        product_name = request.form.get('product_name', 'unknown')
        sku = request.form.get('sku', 'unknown')
        
        if not batch_id:
            return jsonify({'error': 'batch_id is required'}), 400
        
        uploaded_files = []
        
        for file in files:
            if file and file.filename:
                # Read file data
                file_data = file.read()
                
                # Generate unique filename
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{batch_id}_{timestamp}_{file.filename}"
                
                # Save to local storage
                file_path = save_uploaded_file(file_data, filename, AUTHENTIC_LABELS_DIR)
                
                if file_path:
                    uploaded_files.append({
                        'filename': filename,
                        'file_path': file_path,
                        'batch_id': batch_id,
                        'product_name': product_name,
                        'sku': sku
                    })
        
        # Update product database
        update_product_database(batch_id, product_name, sku, uploaded_files)
        
        return jsonify({
            'message': f'Successfully uploaded {len(uploaded_files)} images',
            'batch_id': batch_id,
            'uploaded_files': len(uploaded_files)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
    # Use environment variable for port, default to 3000 for local dev
    port = int(os.environ.get('PORT', 3000))
    app.run(host='0.0.0.0', port=port, debug=False)