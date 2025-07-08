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
VIEWS_PER_IMAGE = 10

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
    img = img.copy()

    # Mild contrast and brightness
    alpha = np.random.uniform(0.95, 1.1)
    beta = np.random.randint(-15, 20)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # Optional blur
    if np.random.rand() < 0.5:
        img = cv2.GaussianBlur(img, (3, 3), 0)

    # Light Gaussian or salt-and-pepper noise
    if np.random.rand() < 0.5:
        var = np.random.randint(5, 20)
        sigma = var ** 0.5
        noise = np.random.normal(0, sigma, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    else:
        prob = 0.005 + np.random.rand() * 0.01
        mask = np.random.choice((0, 255), img.shape, p=[1 - prob, prob])
        img = np.where(mask == 255, 255, img)

    # Simulate print alignment issues
    if np.random.rand() < 0.5:
        shift_x = random.randint(-5, 5)
        shift_y = random.randint(-5, 5)
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)

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