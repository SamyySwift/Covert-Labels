import os
import cv2
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import register_keras_serializable
from scipy.spatial import Delaunay
from scipy.spatial.distance import directed_hausdorff

# Paths
IMG_SIZE = (224, 224)
AUGMENTED_AUTH_DIR = "dataset/authentic_augmented"
FINGERPRINT_DIR = "dataset/fingerprints"
DB_PATH = "product_db.json"
MODEL_PATH = "siamese_contrastive.keras"

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
# Load Model
# =====================
model = load_model(MODEL_PATH, custom_objects={
    "contrastive_loss": contrastive_loss,
    "l2_normalize": l2_normalize,
    "euclidean_distance": euclidean_distance
})

# =====================
# Load Product Metadata
# =====================
with open(DB_PATH, "r") as f:
    product_db = json.load(f)

# =====================
# Utilities
# =====================
def load_and_preprocess(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"❌ Cannot load image: {path}")
    img = cv2.resize(img, IMG_SIZE)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img.astype("float32"))
    return np.expand_dims(img, axis=0)

def extract_delaunay_fingerprint(img):
    orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2)
    keypoints = orb.detect(img, None)
    keypoints, _ = orb.compute(img, keypoints)
    if keypoints is None or len(keypoints) < 3:
        return None
    pts = np.float32([kp.pt for kp in keypoints])
    tri = Delaunay(pts)
    return pts[tri.simplices]

def compare_fingerprints(f1, f2):
    if f1 is None or f2 is None:
        return 0.0
    f1 = f1 / np.max(f1)
    f2 = f2 / np.max(f2)
    d1 = directed_hausdorff(f1.reshape(-1, 2), f2.reshape(-1, 2))[0]
    d2 = directed_hausdorff(f2.reshape(-1, 2), f1.reshape(-1, 2))[0]
    return 1 / (1 + max(d1, d2))

# =====================
# Input Scanned Images
# =====================
scanned_imgs = []
scanned_fingerprints = []
scanned_paths = []  # Store original file paths
print("📷 Enter scanned image paths (multiple angles). Type 'done' to finish.")
while True:
    path = input("🔍 Image path: ").strip()
    if path.lower() == 'done':
        break
    if os.path.exists(path):
        try:
            img = cv2.imread(path)
            img_resized = cv2.resize(img, IMG_SIZE)
            scanned_imgs.append(load_and_preprocess(path))
            scanned_fingerprints.append(extract_delaunay_fingerprint(img_resized))
            scanned_paths.append(path)  # Store the original path
            print("✅ Image added.")
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print("❌ File not found.")

if len(scanned_imgs) < 1:
    print("⚠️ At least one image is required.")
    exit()

# =====================
# Print-Resistant Pattern Detection Functions
# =====================
def preprocess_scanned_image(img):
    """Pre-process scanned image to reduce print artifacts"""
    # Gaussian blur to reduce noise
    img = cv2.GaussianBlur(img, (3, 3), 0)
    
    # Histogram equalization for better contrast
    if len(img.shape) == 3:
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    else:
        img = cv2.equalizeHist(img)
    
    # Bilateral filter to preserve edges while reducing noise
    img = cv2.bilateralFilter(img, 9, 75, 75)
    
    return img

def detect_robust_watermark(img, expected_batch_id):
    """Enhanced watermark detection with stronger signal extraction"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        # Generate expected watermark pattern with stronger signal
        batch_hash = hash(expected_batch_id) % 1000
        np.random.seed(batch_hash)
        
        h, w = gray.shape
        block_size = 8  # Smaller blocks for finer detail
        watermark = np.zeros((h // block_size, w // block_size))
        
        # Create more distinctive pattern
        for i in range(watermark.shape[0]):
            for j in range(watermark.shape[1]):
                # Use batch-specific pattern with higher contrast
                pattern_val = (i + j + batch_hash) % 3
                watermark[i, j] = 1 if pattern_val == 0 else 0
        
        # Resize watermark to match image
        watermark_resized = cv2.resize(watermark, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Enhanced extraction using multiple methods
        extracted = np.zeros_like(watermark)
        
        # Method 1: Block averaging with adaptive threshold
        for i in range(watermark.shape[0]):
            for j in range(watermark.shape[1]):
                block = gray[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                if block.size > 0:
                    # Use local adaptive threshold instead of global 128
                    local_mean = np.mean(block)
                    extracted[i, j] = 1 if local_mean > np.mean(gray) else 0
        
        # Method 2: Frequency domain analysis
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # Look for periodic patterns in frequency domain
        freq_pattern = np.zeros_like(watermark)
        for i in range(watermark.shape[0]):
            for j in range(watermark.shape[1]):
                y_start = i * (magnitude_spectrum.shape[0] // watermark.shape[0])
                y_end = (i + 1) * (magnitude_spectrum.shape[0] // watermark.shape[0])
                x_start = j * (magnitude_spectrum.shape[1] // watermark.shape[1])
                x_end = (j + 1) * (magnitude_spectrum.shape[1] // watermark.shape[1])
                
                if y_end <= magnitude_spectrum.shape[0] and x_end <= magnitude_spectrum.shape[1]:
                    freq_block = magnitude_spectrum[y_start:y_end, x_start:x_end]
                    freq_pattern[i, j] = 1 if np.mean(freq_block) > np.mean(magnitude_spectrum) else 0
        
        # Combine both methods
        combined_extracted = (extracted + freq_pattern) / 2
        
        # Calculate enhanced correlation
        spatial_corr = np.corrcoef(watermark.flatten(), extracted.flatten())[0, 1]
        freq_corr = np.corrcoef(watermark.flatten(), freq_pattern.flatten())[0, 1]
        combined_corr = np.corrcoef(watermark.flatten(), combined_extracted.flatten())[0, 1]
        
        # Handle NaN cases
        spatial_corr = 0.0 if np.isnan(spatial_corr) else spatial_corr
        freq_corr = 0.0 if np.isnan(freq_corr) else freq_corr
        combined_corr = 0.0 if np.isnan(combined_corr) else combined_corr
        
        # Take the best correlation with amplification
        best_corr = max(spatial_corr, freq_corr, combined_corr)
        
        # Amplify positive correlations and suppress negative ones
        if best_corr > 0:
            score = min(1.0, best_corr * 2.0)  # Amplify positive correlations
        else:
            score = 0.0
        
        return score
        
    except Exception as e:
        print(f"Enhanced watermark detection error: {e}")
        return 0.0

def detect_texture_signature(img, expected_batch_id):
    """Enhanced texture detection with multiple filter responses"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        # Generate batch-specific texture patterns
        batch_hash = hash(expected_batch_id) % 1000
        np.random.seed(batch_hash + 100)
        
        # Multiple texture analysis methods
        texture_scores = []
        
        # Method 1: Gabor filter responses
        angles = [0, 45, 90, 135]  # Different orientations
        frequencies = [0.1, 0.3, 0.5]  # Different frequencies
        
        for angle in angles:
            for freq in frequencies:
                # Create Gabor kernel
                kernel = cv2.getGaborKernel((15, 15), 3, np.radians(angle), 2*np.pi*freq, 0.5, 0, ktype=cv2.CV_32F)
                
                # Apply filter
                filtered = cv2.filter2D(gray.astype(np.float32), cv2.CV_8UC3, kernel)
                
                # Calculate response strength
                response_strength = np.std(filtered)
                
                # Compare with expected pattern
                expected_strength = 20 + (batch_hash % 30)  # Batch-specific expected strength
                similarity = 1.0 / (1.0 + abs(response_strength - expected_strength) / expected_strength)
                texture_scores.append(similarity)
        
        # Method 2: Local Binary Pattern (LBP)
        def local_binary_pattern(image, radius=1, n_points=8):
            """Simple LBP implementation"""
            lbp = np.zeros_like(image)
            for i in range(radius, image.shape[0] - radius):
                for j in range(radius, image.shape[1] - radius):
                    center = image[i, j]
                    binary_string = ''
                    for k in range(n_points):
                        angle = 2 * np.pi * k / n_points
                        x = int(i + radius * np.cos(angle))
                        y = int(j + radius * np.sin(angle))
                        if 0 <= x < image.shape[0] and 0 <= y < image.shape[1]:
                            binary_string += '1' if image[x, y] >= center else '0'
                    lbp[i, j] = int(binary_string, 2) if binary_string else 0
            return lbp
        
        lbp_image = local_binary_pattern(gray)
        lbp_hist, _ = np.histogram(lbp_image.flatten(), bins=256, range=(0, 256))
        lbp_hist = lbp_hist.astype(float)
        lbp_hist /= (lbp_hist.sum() + 1e-7)  # Normalize
        
        # Generate expected LBP histogram for this batch
        np.random.seed(batch_hash + 200)
        expected_lbp = np.random.dirichlet(np.ones(256) * 0.1)  # Sparse distribution
        
        # Calculate histogram similarity
        lbp_similarity = 1.0 / (1.0 + np.sum(np.abs(lbp_hist - expected_lbp)))
        texture_scores.append(lbp_similarity)
        
        # Method 3: Edge density analysis
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Expected edge density for this batch
        expected_edge_density = 0.1 + (batch_hash % 100) / 1000.0
        edge_similarity = 1.0 / (1.0 + abs(edge_density - expected_edge_density) / expected_edge_density)
        texture_scores.append(edge_similarity)
        
        # Combine all texture scores
        final_score = np.mean(texture_scores)
        
        # Amplify the score for better discrimination
        amplified_score = min(1.0, final_score * 1.5)
        
        return amplified_score
        
    except Exception as e:
        print(f"Enhanced texture detection error: {e}")
        return 0.0

def detect_print_resistant_patterns(img, expected_batch_id):
    """Enhanced pattern detection with stronger algorithms"""
    
    # Pre-process for print artifacts
    img = preprocess_scanned_image(img)
    
    # Enhanced watermark detection
    watermark_score = detect_robust_watermark(img, expected_batch_id)
    
    # Enhanced texture detection
    texture_score = detect_texture_signature(img, expected_batch_id)
    
    # Adjusted weights to emphasize stronger signals
    combined_score = watermark_score * 0.6 + texture_score * 0.4
    
    # Apply non-linear amplification for better discrimination
    if combined_score > 0.3:
        combined_score = 0.3 + (combined_score - 0.3) * 2.0  # Amplify scores above threshold
    
    combined_score = min(1.0, combined_score)  # Cap at 1.0
    
    return {
        'watermark_score': watermark_score,
        'texture_score': texture_score,
        'combined_score': combined_score,
        'is_authentic': combined_score > 0.5  # Adjusted threshold
    }

# =====================
# Verification Loop (Updated)
# =====================
best_score = 0
best_batch = None
best_pattern_result = None

for batch_id in os.listdir(AUGMENTED_AUTH_DIR):
    batch_path = os.path.join(AUGMENTED_AUTH_DIR, batch_id)
    ref_imgs = [f for f in os.listdir(batch_path) if f.endswith(".jpg")]
    if not ref_imgs:
        continue

    siamese_scores = []
    orb_scores = []
    pattern_scores = []

    for i, (scan_img, scan_fp) in enumerate(zip(scanned_imgs, scanned_fingerprints)):
        # Get original image for pattern detection using stored path
        scan_img_original = cv2.imread(scanned_paths[i])
        
        # Detect print-resistant patterns
        pattern_result = detect_print_resistant_patterns(scan_img_original, batch_id)
        pattern_scores.append(pattern_result['combined_score'])
        
        for ref_name in ref_imgs:
            ref_path = os.path.join(batch_path, ref_name)
            ref_img = load_and_preprocess(ref_path)

            # Predict similarity score (distance: lower = more similar)
            distance = model.predict([scan_img, ref_img], verbose=0)[0][0]
            similarity_score = 1 / (1 + distance)  # Normalize to [0, 1]
            siamese_scores.append(similarity_score)

            # Fingerprint comparison
            fp_path = os.path.join(FINGERPRINT_DIR, f"{batch_id}_{ref_name[:-4]}.npy")
            if os.path.exists(fp_path):
                ref_fp = np.load(fp_path)
                orb_score = compare_fingerprints(scan_fp, ref_fp)
                orb_scores.append(orb_score)

    if not siamese_scores or not orb_scores or not pattern_scores:
        continue

    siamese_avg = np.mean(siamese_scores)
    orb_avg = np.mean(orb_scores)
    pattern_avg = np.mean(pattern_scores)
    
    # Updated combined score with pattern detection (weighted)
    combined_score = 0.4 * siamese_avg + 0.3 * orb_avg + 0.3 * pattern_avg

    print(f"[{batch_id}] Siamese: {siamese_avg:.2f}, ORB: {orb_avg:.2f}, Pattern: {pattern_avg:.2f} → Combined: {combined_score:.2f}")

    if combined_score > best_score:
        best_score = combined_score
        best_batch = batch_id
        # Calculate pattern results properly using stored paths
        watermark_scores = []
        texture_scores = []
        for path in scanned_paths:
            img = cv2.imread(path)
            result = detect_print_resistant_patterns(img, batch_id)
            watermark_scores.append(result['watermark_score'])
            texture_scores.append(result['texture_score'])
        
        best_pattern_result = {
            'watermark_score': np.mean(watermark_scores),
            'texture_score': np.mean(texture_scores),
            'combined_score': pattern_avg
        }

# =====================
# Final Decision (Updated)
# =====================
print("\n🔍 Final Decision")
print("------------------------")
if best_score >= 0.75:  # Slightly lower threshold due to additional validation
    product = product_db.get(best_batch, {})
    print(f"✅ Authentic — Match: {best_batch}")
    print(f"• Combined Score: {best_score:.2f}")
    if best_pattern_result:
        print(f"• Pattern Analysis:")
        print(f"  - Watermark: {best_pattern_result['watermark_score']:.2f}")
        print(f"  - Texture: {best_pattern_result['texture_score']:.2f}")
        print(f"  - Pattern Combined: {best_pattern_result['combined_score']:.2f}")
    if product:
        print(f"📦 Product: {product['product_name']} | SKU: {product['sku']} | Created: {product['created_at']}")
else:
    print("❌ Counterfeit — No strong match found.")
    print(f"• Best Score: {best_score:.2f} (threshold: 0.75)")
