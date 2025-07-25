import os
import cv2
import numpy as np
import json
import uuid
import random
from datetime import datetime
from scipy.spatial import Delaunay

AUTH_DIR = "dataset/authentic"
FAKE_DIR = "dataset/fake"
AUGMENTED_AUTH_DIR = "dataset/authentic_augmented"
FINGERPRINT_DIR = "dataset/fingerprints"
DB_PATH = "product_db.json"
IMG_SIZE = (224, 224)
# Increase views per image for more training data
VIEWS_PER_IMAGE = 15  # Increased from 10

# Add more aggressive augmentation
def batch_augment(img, seed=0):
    random.seed(seed)
    np.random.seed(seed)

    # More varied brightness/contrast
    alpha = random.uniform(0.8, 1.3)    # Wider range
    beta = random.randint(-25, 25)      # More brightness variation
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # More blur variation
    if random.random() < 0.7:  # More frequent blur
        k = random.choice([3, 5, 7])  # Varied blur kernels
        img = cv2.GaussianBlur(img, (k, k), 0)

    # More noise
    if random.random() < 0.7:
        noise = np.random.normal(0, 12, img.shape).astype(np.int16)  # Stronger noise
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img

os.makedirs(AUTH_DIR, exist_ok=True)
os.makedirs(FAKE_DIR, exist_ok=True)
os.makedirs(AUGMENTED_AUTH_DIR, exist_ok=True)
os.makedirs(FINGERPRINT_DIR, exist_ok=True)

# Load or initialize metadata DB
if os.path.exists(DB_PATH):
    with open(DB_PATH, 'r') as f:
        product_db = json.load(f)
else:
    product_db = {}

# --- Batch-specific augmentation functions ---
def batch_augment(img, seed=0):
    random.seed(seed)
    np.random.seed(seed)

    # Mild brightness/contrast changes
    alpha = random.uniform(0.97, 1.05)    # contrast
    beta = random.randint(-10, 10)        # brightness
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # Gentle blur
    if random.random() < 0.5:
        k = random.choice([3])  # keep blur minimal
        img = cv2.GaussianBlur(img, (k, k), 0)

    # Very light noise
    if random.random() < 0.5:
        noise = np.random.normal(0, 5, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


def random_perspective(img):
    h, w = img.shape[:2]
    delta = random.randint(5, 15)  # lower warping
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [random.randint(0, delta), random.randint(0, delta)],
        [w - random.randint(0, delta), random.randint(0, delta)],
        [w - random.randint(0, delta), h - random.randint(0, delta)],
        [random.randint(0, delta), h - random.randint(0, delta)],
    ])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)


def simulate_fake(img):
    """Enhanced fake generation with comprehensive color variations and realistic scenarios"""
    img = img.copy()
    
    # Choose a fake scenario type
    fake_type = random.choice([
        'cheap_reproduction', 'color_shift', 'poor_printing', 
        'digital_copy', 'worn_fake', 'home_printer', 'screen_photo'
    ])
    
    if fake_type == 'cheap_reproduction':
        # Low-quality reproduction with color degradation
        img = simulate_cheap_reproduction(img)
    elif fake_type == 'color_shift':
        # Wrong color profiles/inks
        img = simulate_color_shift(img)
    elif fake_type == 'poor_printing':
        # Bad printing equipment
        img = simulate_poor_print_quality(img)
    elif fake_type == 'digital_copy':
        # Screenshot/digital reproduction
        img = simulate_digital_copy(img)
    elif fake_type == 'worn_fake':
        # Aged/damaged counterfeit
        img = simulate_worn_fake(img)
    elif fake_type == 'home_printer':
        # Home inkjet printer reproduction
        img = simulate_home_printer(img)
    elif fake_type == 'screen_photo':
        # Photo of screen/monitor
        img = simulate_screen_photo(img)
    
    # Apply additional random degradations
    img = apply_random_degradations(img)
    
    return img

def simulate_cheap_reproduction(img):
    """Simulate cheap reproduction with color variations"""
    # Reduce color depth (cheaper printing)
    img = (img // 16) * 16  # Quantize colors
    
    # Color cast variations
    cast_type = random.choice(['cyan', 'magenta', 'yellow', 'red', 'blue', 'green'])
    intensity = random.uniform(0.1, 0.3)
    
    if cast_type == 'cyan':
        img[:,:,0] = np.clip(img[:,:,0] * (1 + intensity), 0, 255)  # More blue
        img[:,:,1] = np.clip(img[:,:,1] * (1 + intensity), 0, 255)  # More green
    elif cast_type == 'magenta':
        img[:,:,0] = np.clip(img[:,:,0] * (1 + intensity), 0, 255)  # More blue
        img[:,:,2] = np.clip(img[:,:,2] * (1 + intensity), 0, 255)  # More red
    elif cast_type == 'yellow':
        img[:,:,1] = np.clip(img[:,:,1] * (1 + intensity), 0, 255)  # More green
        img[:,:,2] = np.clip(img[:,:,2] * (1 + intensity), 0, 255)  # More red
        img[:,:,0] = np.clip(img[:,:,0] * (1 - intensity), 0, 255)  # Less blue
    elif cast_type == 'red':
        img[:,:,2] = np.clip(img[:,:,2] * (1 + intensity), 0, 255)
    elif cast_type == 'blue':
        img[:,:,0] = np.clip(img[:,:,0] * (1 + intensity), 0, 255)
    elif cast_type == 'green':
        img[:,:,1] = np.clip(img[:,:,1] * (1 + intensity), 0, 255)
    
    return img.astype(np.uint8)

def simulate_color_shift(img):
    """Simulate wrong color profiles or ink variations"""
    # HSV color space manipulation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Hue shift (wrong color reproduction)
    hue_shift = random.randint(-30, 30)
    hsv[:,:,0] = (hsv[:,:,0].astype(np.int16) + hue_shift) % 180
    
    # Saturation changes (faded or oversaturated)
    sat_factor = random.uniform(0.6, 1.4)
    hsv[:,:,1] = np.clip(hsv[:,:,1] * sat_factor, 0, 255)
    
    # Value/brightness changes
    val_factor = random.uniform(0.7, 1.3)
    hsv[:,:,2] = np.clip(hsv[:,:,2] * val_factor, 0, 255)
    
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Add color channel imbalance
    channel_factors = [random.uniform(0.8, 1.2) for _ in range(3)]
    for i in range(3):
        img[:,:,i] = np.clip(img[:,:,i] * channel_factors[i], 0, 255)
    
    return img.astype(np.uint8)

def simulate_digital_copy(img):
    """Simulate screenshot or digital reproduction"""
    # JPEG compression artifacts
    quality = random.randint(60, 85)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', img, encode_param)
    img = cv2.imdecode(encimg, 1)
    
    # Screen resolution simulation
    if random.random() < 0.5:
        scale = random.uniform(0.7, 0.9)
        h, w = img.shape[:2]
        small = cv2.resize(img, (int(w*scale), int(h*scale)))
        img = cv2.resize(small, (w, h))
    
    # Monitor color profile differences
    gamma = random.uniform(0.8, 1.2)
    img = np.power(img / 255.0, gamma) * 255
    
    return img.astype(np.uint8)

def simulate_worn_fake(img):
    """Simulate aged/damaged counterfeit"""
    # Fading
    fade_factor = random.uniform(0.7, 0.9)
    img = img * fade_factor
    
    # Stains and discoloration
    num_stains = random.randint(2, 8)
    for _ in range(num_stains):
        center_x = random.randint(0, img.shape[1])
        center_y = random.randint(0, img.shape[0])
        radius = random.randint(10, 30)
        
        # Create circular stain
        y, x = np.ogrid[:img.shape[0], :img.shape[1]]
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        
        # Random stain color
        stain_color = random.choice([
            [0.9, 0.8, 0.6],  # Yellow stain
            [0.8, 0.7, 0.6],  # Brown stain
            [0.9, 0.9, 0.9],  # Light stain
            [0.7, 0.7, 0.7]   # Gray stain
        ])
        
        for i in range(3):
            img[:,:,i][mask] = img[:,:,i][mask] * stain_color[i]
    
    # Scratches (thin lines)
    num_scratches = random.randint(1, 4)
    for _ in range(num_scratches):
        pt1 = (random.randint(0, img.shape[1]), random.randint(0, img.shape[0]))
        pt2 = (random.randint(0, img.shape[1]), random.randint(0, img.shape[0]))
        cv2.line(img, pt1, pt2, (200, 200, 200), random.randint(1, 3))
    
    return img.astype(np.uint8)

def simulate_home_printer(img):
    """Simulate home inkjet printer reproduction"""
    # Ink bleeding
    kernel = np.ones((2,2), np.float32) / 4
    img = cv2.filter2D(img, -1, kernel)
    
    # Banding (horizontal lines from printer head)
    band_spacing = random.randint(8, 16)
    band_intensity = random.uniform(0.02, 0.08)
    
    for y in range(0, img.shape[0], band_spacing):
        if random.random() < 0.7:  # Not every band
            variation = random.uniform(-band_intensity, band_intensity)
            img[y:y+2, :] = np.clip(img[y:y+2, :] * (1 + variation), 0, 255)
    
    # Color separation (CMYK simulation)
    # Simulate slight misalignment of color layers
    shift_r = random.randint(-1, 1)
    shift_b = random.randint(-1, 1)
    
    if shift_r != 0:
        img[:,:,2] = np.roll(img[:,:,2], shift_r, axis=1)
    if shift_b != 0:
        img[:,:,0] = np.roll(img[:,:,0], shift_b, axis=1)
    
    return img.astype(np.uint8)

def simulate_screen_photo(img):
    """Simulate photo taken of a screen/monitor"""
    # Moiré patterns
    if random.random() < 0.6:
        # Create subtle grid pattern
        grid_size = random.randint(3, 6)
        grid_intensity = random.uniform(0.02, 0.05)
        
        for y in range(0, img.shape[0], grid_size):
            img[y, :] = np.clip(img[y, :] * (1 + grid_intensity), 0, 255)
        for x in range(0, img.shape[1], grid_size):
            img[:, x] = np.clip(img[:, x] * (1 + grid_intensity), 0, 255)
    
    # Screen curvature (slight barrel distortion)
    if random.random() < 0.4:
        h, w = img.shape[:2]
        # Create barrel distortion
        camera_matrix = np.array([[w, 0, w/2], [0, h, h/2], [0, 0, 1]], dtype=np.float32)
        dist_coeffs = np.array([random.uniform(-0.1, -0.05), 0, 0, 0], dtype=np.float32)
        img = cv2.undistort(img, camera_matrix, dist_coeffs)
    
    # Screen refresh lines
    if random.random() < 0.3:
        line_spacing = random.randint(4, 8)
        line_intensity = random.uniform(0.01, 0.03)
        
        for y in range(0, img.shape[0], line_spacing):
            img[y, :] = np.clip(img[y, :] * (1 + line_intensity), 0, 255)
    
    return img.astype(np.uint8)

def simulate_poor_print_quality(img):
    """Most common counterfeit issue"""
    technique = random.choice(['low_res', 'dot_gain', 'misregistration', 'cheap_ink'])
    
    if technique == 'low_res':
        # Simulate lower resolution printing
        scale = random.uniform(0.6, 0.8)
        h, w = img.shape[:2]
        small = cv2.resize(img, (int(w*scale), int(h*scale)))
        img = cv2.resize(small, (w, h))
    
    elif technique == 'dot_gain':
        # Ink bleeding effect
        kernel = np.ones((2,2), np.float32) / 4
        img = cv2.filter2D(img, -1, kernel)
    
    elif technique == 'misregistration':
        # Color separation issues
        shift = random.randint(1, 3)
        img[:,:,0] = np.roll(img[:,:,0], shift, axis=1)  # Shift red
        img[:,:,2] = np.roll(img[:,:,2], -shift, axis=1)  # Shift blue opposite
    
    elif technique == 'cheap_ink':
        # Faded/wrong colors
        img = cv2.convertScaleAbs(img, alpha=0.85, beta=0)
        # Slight color shift
        img[:,:,1] = np.clip(img[:,:,1] * 0.9, 0, 255)  # Reduce green
    
    return img

def simulate_cheap_reproduction(img):
    """Simulate cheap reproduction with color variations"""
    # Reduce color depth (cheaper printing)
    img = (img // 16) * 16  # Quantize colors
    
    # Color cast variations
    cast_type = random.choice(['cyan', 'magenta', 'yellow', 'red', 'blue', 'green'])
    intensity = random.uniform(0.1, 0.3)
    
    if cast_type == 'cyan':
        img[:,:,0] = np.clip(img[:,:,0] * (1 + intensity), 0, 255)  # More blue
        img[:,:,1] = np.clip(img[:,:,1] * (1 + intensity), 0, 255)  # More green
    elif cast_type == 'magenta':
        img[:,:,0] = np.clip(img[:,:,0] * (1 + intensity), 0, 255)  # More blue
        img[:,:,2] = np.clip(img[:,:,2] * (1 + intensity), 0, 255)  # More red
    elif cast_type == 'yellow':
        img[:,:,1] = np.clip(img[:,:,1] * (1 + intensity), 0, 255)  # More green
        img[:,:,2] = np.clip(img[:,:,2] * (1 + intensity), 0, 255)  # More red
        img[:,:,0] = np.clip(img[:,:,0] * (1 - intensity), 0, 255)  # Less blue
    elif cast_type == 'red':
        img[:,:,2] = np.clip(img[:,:,2] * (1 + intensity), 0, 255)
    elif cast_type == 'blue':
        img[:,:,0] = np.clip(img[:,:,0] * (1 + intensity), 0, 255)
    elif cast_type == 'green':
        img[:,:,1] = np.clip(img[:,:,1] * (1 + intensity), 0, 255)
    
    return img.astype(np.uint8)

def simulate_color_shift(img):
    """Simulate wrong color profiles or ink variations"""
    # HSV color space manipulation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Hue shift (wrong color reproduction)
    hue_shift = random.randint(-30, 30)
    hsv[:,:,0] = (hsv[:,:,0].astype(np.int16) + hue_shift) % 180
    
    # Saturation changes (faded or oversaturated)
    sat_factor = random.uniform(0.6, 1.4)
    hsv[:,:,1] = np.clip(hsv[:,:,1] * sat_factor, 0, 255)
    
    # Value/brightness changes
    val_factor = random.uniform(0.7, 1.3)
    hsv[:,:,2] = np.clip(hsv[:,:,2] * val_factor, 0, 255)
    
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Add color channel imbalance
    channel_factors = [random.uniform(0.8, 1.2) for _ in range(3)]
    for i in range(3):
        img[:,:,i] = np.clip(img[:,:,i] * channel_factors[i], 0, 255)
    
    return img.astype(np.uint8)

def simulate_digital_copy(img):
    """Simulate screenshot or digital reproduction"""
    # JPEG compression artifacts
    quality = random.randint(60, 85)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', img, encode_param)
    img = cv2.imdecode(encimg, 1)
    
    # Screen resolution simulation
    if random.random() < 0.5:
        scale = random.uniform(0.7, 0.9)
        h, w = img.shape[:2]
        small = cv2.resize(img, (int(w*scale), int(h*scale)))
        img = cv2.resize(small, (w, h))
    
    # Monitor color profile differences
    gamma = random.uniform(0.8, 1.2)
    img = np.power(img / 255.0, gamma) * 255
    
    return img.astype(np.uint8)

def simulate_worn_fake(img):
    """Simulate aged/damaged counterfeit"""
    # Fading
    fade_factor = random.uniform(0.7, 0.9)
    img = img * fade_factor
    
    # Stains and discoloration
    num_stains = random.randint(2, 8)
    for _ in range(num_stains):
        center_x = random.randint(0, img.shape[1])
        center_y = random.randint(0, img.shape[0])
        radius = random.randint(10, 30)
        
        # Create circular stain
        y, x = np.ogrid[:img.shape[0], :img.shape[1]]
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        
        # Random stain color
        stain_color = random.choice([
            [0.9, 0.8, 0.6],  # Yellow stain
            [0.8, 0.7, 0.6],  # Brown stain
            [0.9, 0.9, 0.9],  # Light stain
            [0.7, 0.7, 0.7]   # Gray stain
        ])
        
        for i in range(3):
            img[:,:,i][mask] = img[:,:,i][mask] * stain_color[i]
    
    # Scratches (thin lines)
    num_scratches = random.randint(1, 4)
    for _ in range(num_scratches):
        pt1 = (random.randint(0, img.shape[1]), random.randint(0, img.shape[0]))
        pt2 = (random.randint(0, img.shape[1]), random.randint(0, img.shape[0]))
        cv2.line(img, pt1, pt2, (200, 200, 200), random.randint(1, 3))
    
    return img.astype(np.uint8)

def simulate_home_printer(img):
    """Simulate home inkjet printer reproduction"""
    # Ink bleeding
    kernel = np.ones((2,2), np.float32) / 4
    img = cv2.filter2D(img, -1, kernel)
    
    # Banding (horizontal lines from printer head)
    band_spacing = random.randint(8, 16)
    band_intensity = random.uniform(0.02, 0.08)
    
    for y in range(0, img.shape[0], band_spacing):
        if random.random() < 0.7:  # Not every band
            variation = random.uniform(-band_intensity, band_intensity)
            img[y:y+2, :] = np.clip(img[y:y+2, :] * (1 + variation), 0, 255)
    
    # Color separation (CMYK simulation)
    # Simulate slight misalignment of color layers
    shift_r = random.randint(-1, 1)
    shift_b = random.randint(-1, 1)
    
    if shift_r != 0:
        img[:,:,2] = np.roll(img[:,:,2], shift_r, axis=1)
    if shift_b != 0:
        img[:,:,0] = np.roll(img[:,:,0], shift_b, axis=1)
    
    return img.astype(np.uint8)

def simulate_screen_photo(img):
    """Simulate photo taken of a screen/monitor"""
    # Moiré patterns
    if random.random() < 0.6:
        # Create subtle grid pattern
        grid_size = random.randint(3, 6)
        grid_intensity = random.uniform(0.02, 0.05)
        
        for y in range(0, img.shape[0], grid_size):
            img[y, :] = np.clip(img[y, :] * (1 + grid_intensity), 0, 255)
        for x in range(0, img.shape[1], grid_size):
            img[:, x] = np.clip(img[:, x] * (1 + grid_intensity), 0, 255)
    
    # Screen curvature (slight barrel distortion)
    if random.random() < 0.4:
        h, w = img.shape[:2]
        # Create barrel distortion
        camera_matrix = np.array([[w, 0, w/2], [0, h, h/2], [0, 0, 1]], dtype=np.float32)
        dist_coeffs = np.array([random.uniform(-0.1, -0.05), 0, 0, 0], dtype=np.float32)
        img = cv2.undistort(img, camera_matrix, dist_coeffs)
    
    # Screen refresh lines
    if random.random() < 0.3:
        line_spacing = random.randint(4, 8)
        line_intensity = random.uniform(0.01, 0.03)
        
        for y in range(0, img.shape[0], line_spacing):
            img[y, :] = np.clip(img[y, :] * (1 + line_intensity), 0, 255)
    
    return img.astype(np.uint8)

def apply_random_degradations(img):
    """Apply additional random degradations"""
    # Random brightness/contrast (more extreme for fakes)
    alpha = random.uniform(0.7, 1.4)  # Wider range than authentic
    beta = random.randint(-40, 40)    # More extreme brightness
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    
    # Random blur (poor focus/movement)
    if random.random() < 0.6:
        blur_type = random.choice(['gaussian', 'motion', 'defocus'])
        
        if blur_type == 'gaussian':
            kernel_size = random.choice([3, 5, 7])
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        elif blur_type == 'motion':
            # Motion blur
            size = random.randint(5, 15)
            kernel = np.zeros((size, size))
            kernel[int((size-1)/2), :] = np.ones(size)
            kernel = kernel / size
            img = cv2.filter2D(img, -1, kernel)
        elif blur_type == 'defocus':
            # Defocus blur (circular)
            radius = random.randint(2, 5)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius*2+1, radius*2+1))
            kernel = kernel.astype(np.float32) / np.sum(kernel)
            img = cv2.filter2D(img, -1, kernel)
    
    # Noise (various types)
    if random.random() < 0.7:
        noise_type = random.choice(['gaussian', 'salt_pepper', 'speckle'])
        
        if noise_type == 'gaussian':
            var = random.randint(10, 30)
            sigma = var ** 0.5
            noise = np.random.normal(0, sigma, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        elif noise_type == 'salt_pepper':
            prob = random.uniform(0.01, 0.03)
            mask = np.random.choice((0, 1, 2), img.shape[:2], p=[1-prob, prob/2, prob/2])
            img[mask == 1] = 255  # Salt
            img[mask == 2] = 0    # Pepper
        elif noise_type == 'speckle':
            noise = np.random.randn(*img.shape) * random.uniform(0.1, 0.3)
            img = np.clip(img + img * noise, 0, 255).astype(np.uint8)
    
    return img

def extract_fingerprint(img, out_path):
    orb = cv2.ORB_create(500)
    keypoints = orb.detect(img, None)
    keypoints, _ = orb.compute(img, keypoints)
    if keypoints is None or len(keypoints) < 3: return None
    pts = np.float32([kp.pt for kp in keypoints])
    tri = Delaunay(pts)
    triangles = pts[tri.simplices]
    np.save(out_path, triangles)
    return triangles

auth_images = []
fake_images = []

for batch in os.listdir(AUTH_DIR):
    batch_path = os.path.join(AUTH_DIR, batch)
    if not os.path.isdir(batch_path): continue

    out_batch_path = os.path.join(AUGMENTED_AUTH_DIR, batch)
    os.makedirs(out_batch_path, exist_ok=True)

    if batch not in product_db:
        sample_files = [f for f in os.listdir(batch_path) if f.lower().endswith(('.png', '.jpg'))]
        product_name = sample_files[0].rsplit(".", 1)[0] if sample_files else f"Unnamed_{batch}"
        product_db[batch] = {
            "batch_id": batch,
            "product_name": product_name.replace("_", " ").title(),
            "sku": str(uuid.uuid4())[:8],
            "created_at": datetime.now().isoformat(),
            "image_count": 0
        }

    count = 0
    for fname in os.listdir(batch_path):
        img_path = os.path.join(batch_path, fname)
        img = cv2.imread(img_path)
        if img is None: continue
        img = cv2.resize(img, IMG_SIZE)

        base_name = os.path.splitext(fname)[0]
        fingerprint_base = f"{batch}_{base_name}"

        # Enhanced version with consistent seeding
        for i in range(VIEWS_PER_IMAGE):
        # Set seed for both perspective and augmentation
            view_seed = i + hash(batch) % 1000
        
            # Seed the perspective transformation
            random.seed(view_seed)
            np.random.seed(view_seed)
            transformed = random_perspective(img)
            
            # Apply batch-specific augmentation
            augmented = batch_augment(transformed, seed=view_seed)

            view_name = f"{base_name}_view{i}"
            view_path = os.path.join(out_batch_path, f"{view_name}.jpg")
            cv2.imwrite(view_path, augmented)
            auth_images.append(augmented)

            extract_fingerprint(augmented, os.path.join(FINGERPRINT_DIR, f"{fingerprint_base}_view{i}.npy"))

            fake = simulate_fake(augmented)
            fake_out = os.path.join(FAKE_DIR, f"{fingerprint_base}_view{i}_fake.jpg")
            cv2.imwrite(fake_out, fake)
            fake_images.append(fake)
            count += 1

    product_db[batch]["image_count"] = count

with open(DB_PATH, 'w') as f:
    json.dump(product_db, f, indent=2)

print("✅ Authentic, augmented views + triangulated fake images saved.")

# Replace the current pair generation section (lines 165-226) with:
print("🔄 Generating Siamese training pairs...")

pairs, labels = [], []
auth_by_batch = {}
fake_by_batch = {}

# Organize images by batch
for batch in os.listdir(AUGMENTED_AUTH_DIR):
    batch_path = os.path.join(AUGMENTED_AUTH_DIR, batch)
    if not os.path.isdir(batch_path): continue
    
    auth_by_batch[batch] = []
    fake_by_batch[batch] = []
    
    # Collect authentic images for this batch
    for fname in os.listdir(batch_path):
        if fname.endswith('.jpg'):
            auth_by_batch[batch].append(os.path.join(batch_path, fname))
    
    # Collect fake images for this batch
    for fname in os.listdir(FAKE_DIR):
        if fname.startswith(batch) and fname.endswith('_fake.jpg'):
            fake_by_batch[batch].append(os.path.join(FAKE_DIR, fname))

# Generate positive pairs (same batch = similar)
for batch, batch_images in auth_by_batch.items():
    for i in range(len(batch_images)):
        for j in range(i + 1, len(batch_images)):
            pairs.append([batch_images[i], batch_images[j]])  # Remove label from here
            labels.append(1)  # Same batch = similar

# Generate negative pairs (different batches)
batch_list = list(auth_by_batch.keys())
for i in range(len(batch_list)):
    for j in range(i + 1, len(batch_list)):
        batch1, batch2 = batch_list[i], batch_list[j]
        # Cross-batch negatives (different products)
        for _ in range(min(len(auth_by_batch[batch1]), len(auth_by_batch[batch2]))):
            img1 = random.choice(auth_by_batch[batch1])
            img2 = random.choice(auth_by_batch[batch2])
            pairs.append([img1, img2])  # Remove label from here
            labels.append(0)  # Different batches = different products

# Generate authentic vs fake pairs (negatives)
for batch in auth_by_batch.keys():
    auth_imgs = auth_by_batch[batch]
    fake_imgs = fake_by_batch[batch]
    for _ in range(len(auth_imgs)):
        auth_img = random.choice(auth_imgs)
        fake_img = random.choice(fake_imgs)
        pairs.append([auth_img, fake_img])  # Remove label from here
        labels.append(0)  # Authentic vs fake = dissimilar

print(f"Generated {len(pairs)} training pairs")

# Remove the conflicting duplicate pair generation (lines 211-214)
# This was causing the inhomogeneous array issue

np.save("pairs.npy", pairs)  # Save as list, not np.array()
np.save("labels.npy", labels)  # Save as list, not np.array()
print(f"✅ Saved {len(pairs)} training pairs (balanced)")


def add_lighting_variations(img):
    """Critical for real-world robustness"""
    lighting_type = random.choice(['indoor', 'outdoor', 'fluorescent', 'led', 'dim'])
    
    if lighting_type == 'indoor':
        # Warm indoor lighting
        img = cv2.convertScaleAbs(img, alpha=0.9, beta=10)
        img[:,:,0] = np.clip(img[:,:,0] * 0.95, 0, 255)  # Reduce blue
        img[:,:,2] = np.clip(img[:,:,2] * 1.05, 0, 255)  # Increase red
    elif lighting_type == 'outdoor':
        # Bright outdoor lighting
        img = cv2.convertScaleAbs(img, alpha=1.2, beta=15)
    elif lighting_type == 'fluorescent':
        # Cool fluorescent
        img[:,:,0] = np.clip(img[:,:,0] * 1.1, 0, 255)  # Increase blue
        img[:,:,1] = np.clip(img[:,:,1] * 1.05, 0, 255)  # Slight green
    elif lighting_type == 'dim':
        # Low light conditions
        img = cv2.convertScaleAbs(img, alpha=0.7, beta=-20)
        # Add noise in low light
        noise = np.random.normal(0, 8, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return img


def simulate_phone_capture(img):
    """Simulate different phone cameras"""
    phone_type = random.choice(['iphone', 'android_high', 'android_mid', 'old_phone'])
    
    if phone_type == 'iphone':
        # iPhone tends to oversharpen
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        img = cv2.filter2D(img, -1, kernel * 0.1)
    elif phone_type == 'android_high':
        # High-end Android - good quality
        img = cv2.GaussianBlur(img, (1, 1), 0.3)
    elif phone_type == 'android_mid':
        # Mid-range - slight compression
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        _, encimg = cv2.imencode('.jpg', img, encode_param)
        img = cv2.imdecode(encimg, 1)
    elif phone_type == 'old_phone':
        # Older phone - lower quality
        img = cv2.resize(img, (180, 180))
        img = cv2.resize(img, (224, 224))
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
        _, encimg = cv2.imencode('.jpg', img, encode_param)
        img = cv2.imdecode(encimg, 1)
    
    return img


def add_print_resistant_patterns(img, batch_id):
    """Add patterns that survive print/camera lifecycle"""
    
    # 1. Invisible watermark (most reliable)
    img = add_robust_watermark(img, batch_id)
    
    # 2. Texture signatures (backup method)
    img = add_texture_signature(img, batch_id)
    
    return img

def add_robust_watermark(img, batch_id):
    """Watermark optimized for print/scan survival"""
    h, w = img.shape[:2]
    batch_hash = hash(batch_id) % 1000
    np.random.seed(batch_hash)
    
    # Larger blocks for print durability
    block_size = 12  # Increased from 8 for better print survival
    
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            if np.random.random() > 0.6:  # 40% of blocks
                # Stronger variation for print survival
                variation = np.random.choice([-4, -3, 3, 4])  # Increased from ±2
                
                img[y:y+block_size, x:x+block_size] = np.clip(
                    img[y:y+block_size, x:x+block_size].astype(np.int16) + variation,
                    0, 255
                ).astype(np.uint8)
    
    return img

def add_texture_signature(img, batch_id):
    """Texture patterns optimized for print/camera"""
    h, w = img.shape[:2]
    batch_hash = hash(batch_id) % 1000
    np.random.seed(batch_hash)
    
    # Larger kernel for print durability
    kernel_size = 5  # Increased from 3
    texture_kernel = np.random.uniform(-0.15, 0.15, (kernel_size, kernel_size))
    texture_kernel = texture_kernel / np.sum(np.abs(texture_kernel))
    
    # More regions with stronger effect
    num_regions = 30  # Increased from 20
    for _ in range(num_regions):
        y = np.random.randint(0, h - kernel_size)
        x = np.random.randint(0, w - kernel_size)
        
        region = img[y:y+kernel_size, x:x+kernel_size].astype(np.float32)
        textured = cv2.filter2D(region, -1, texture_kernel * 2)  # Stronger effect
        img[y:y+kernel_size, x:x+kernel_size] = np.clip(textured, 0, 255).astype(np.uint8)
    
    return img