import os
import cv2
import json
import hashlib
import numpy as np
from datetime import datetime
from scipy.spatial import Delaunay
import random

# --- Configuration ---
IMG_DIR = "images"
OUTPUT_DIR = "clg_output_images"
METADATA_PATH = "clg_microdot_metadata.json"
DOT_COUNT = 8
JITTER = 5
DOT_RADIUS = 2  
IMG_SIZE = (600, 600)
MIN_DISTANCE_FROM_TEXT = 20
VARIATIONS_COUNT = 30

# Create output folder
os.makedirs(OUTPUT_DIR, exist_ok=True)
metadata = {}

# --- Detect likely text areas ---
def detect_text_areas(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    text_mask = cv2.dilate(edges, kernel, iterations=2)

    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 15, 10)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if 100 < cv2.contourArea(c) < 10000:
            cv2.rectangle(text_mask,
                          (max(0, x - MIN_DISTANCE_FROM_TEXT), max(0, y - MIN_DISTANCE_FROM_TEXT)),
                          (min(img.shape[1], x + w + MIN_DISTANCE_FROM_TEXT), min(img.shape[0], y + h + MIN_DISTANCE_FROM_TEXT)),
                          255, -1)
    return text_mask

def is_valid_dot_position(x, y, text_mask, existing, min_dist=15):
    if text_mask[y, x] > 0:
        return False
    return all(np.sqrt((x - ex)**2 + (y - ey)**2) >= min_dist for ex, ey in existing)

def generate_structured_microdots(img, batch_id, count=12):
    """Generate structured dot patterns unique to each batch"""
    h, w = img.shape[:2]
    text_mask = detect_text_areas(img)
    
    # Use batch_id to seed the pattern generation
    batch_seed = sum(ord(c) for c in batch_id)
    np.random.seed(batch_seed)
    
    # Create a grid-based pattern with batch-specific offsets
    grid_size = int(np.sqrt(count)) + 1
    cell_w, cell_h = w // grid_size, h // grid_size
    
    coords = []
    for i in range(grid_size):
        for j in range(grid_size):
            if len(coords) >= count:
                break
                
            # Add batch-specific offsets within each cell
            offset_x = np.random.randint(cell_w // 4, 3 * cell_w // 4)
            offset_y = np.random.randint(cell_h // 4, 3 * cell_h // 4)
            
            x = i * cell_w + offset_x
            y = j * cell_h + offset_y
            
            # Ensure dot is in valid position
            if 0 <= x < w and 0 <= y < h and text_mask[y, x] == 0:
                coords.append([x, y])
    
    # If we don't have enough dots, add some random ones
    while len(coords) < count:
        x = np.random.randint(DOT_RADIUS + MIN_DISTANCE_FROM_TEXT, w - DOT_RADIUS)
        y = np.random.randint(DOT_RADIUS + MIN_DISTANCE_FROM_TEXT, h - DOT_RADIUS)
        if text_mask[y, x] == 0:
            coords.append([x, y])
    
    return np.array(coords)

def apply_jitter(coords):
    jittered = coords + np.random.randint(-JITTER, JITTER + 1, coords.shape)
    return np.clip(jittered, DOT_RADIUS, IMG_SIZE[0] - DOT_RADIUS)

def draw_invisible_dots(img, coords):
    """Draw very faint microdots that blend with the image colors for minimal visibility while maintaining detectability"""
    img_copy = img.copy()
    
    for (x, y) in coords:
        # Enhanced color sampling - use larger area for better color representation
        sample_radius = 8
        local_area = img[max(0, y-sample_radius):min(img.shape[0], y+sample_radius+1), 
                        max(0, x-sample_radius):min(img.shape[1], x+sample_radius+1)]
        
        # Get multiple color statistics for better blending
        avg_color = np.mean(local_area, axis=(0, 1))
        median_color = np.median(local_area.reshape(-1, 3), axis=0)
        
        # Use weighted combination of average and median for more natural color
        base_color = 0.7 * avg_color + 0.3 * median_color
        
        # Sample a few pixels around the dot position for color variation
        sample_points = [
            (max(0, min(img.shape[1]-1, x + dx)), max(0, min(img.shape[0]-1, y + dy)))
            for dx, dy in [(-3, -3), (3, -3), (-3, 3), (3, 3), (0, 0)]
        ]
        
        sampled_colors = [img[py, px] for px, py in sample_points]
        dominant_color = np.mean(sampled_colors, axis=0)
        
        # Blend the base color with dominant color for natural appearance
        final_base_color = 0.6 * base_color + 0.4 * dominant_color
        
        # Use even smaller brightness shift for more subtle appearance
        brightness_shift = random.randint(1, 3)  # Reduced from (2, 6) to (1, 3)
        
        # Randomly decide to make it darker or lighter based on image brightness
        local_brightness = np.mean(final_base_color)
        if local_brightness > 127:  # Bright area - make dot darker
            dot_color = np.clip(final_base_color - brightness_shift, 0, 255)
        else:  # Dark area - make dot lighter
            dot_color = np.clip(final_base_color + brightness_shift, 0, 255)
        
        # Make the dot a bit whiter (very subtle)
        white_bias = 0.15
        dot_color = ((1 - white_bias) * dot_color.astype(np.float32) + white_bias * 255.0).astype(np.uint8)
        
        # Draw the main dot with adaptive color
        cv2.circle(img_copy, (x, y), DOT_RADIUS, dot_color.tolist(), -1)
        
        # Add an extremely subtle outer ring that blends with surrounding colors
        ring_samples = [
            img[max(0, min(img.shape[0]-1, y + dy)), max(0, min(img.shape[1]-1, x + dx))]
            for dx, dy in [(-5, 0), (5, 0), (0, -5), (0, 5)]
        ]
        ring_color = np.mean(ring_samples, axis=0)
        outer_color = 0.9 * ring_color + 0.1 * dot_color  # Even more subtle blend (was 0.8/0.2)
        cv2.circle(img_copy, (x, y), DOT_RADIUS + 1, outer_color.tolist(), 1)
    
    return img_copy

def draw_visible_small_dots(img, coords):
    """Draw small visible microdots that are clearly visible but minimal in size"""
    img_copy = img.copy()
    
    for (x, y) in coords:
        # Get the local background color for contrast
        local_area = img[max(0, y-3):min(img.shape[0], y+4), 
                        max(0, x-3):min(img.shape[1], x+4)]
        avg_color = np.mean(local_area, axis=(0, 1))
        
        # Choose contrasting color based on brightness
        local_brightness = np.mean(avg_color)
        if local_brightness > 127:  # Bright area - use dark dot
            dot_color = (0, 0, 0)  # Black
        else:  # Dark area - use light dot
            dot_color = (255, 255, 255)  # White
        
        # Increase contrast - remove white bias for more distinct dots
        
        # Draw slightly larger visible dot (2px instead of 1px)
        # small visible variant (use configured DOT_RADIUS instead of hardcoded 2)
        cv2.circle(img_copy, (x, y), DOT_RADIUS, dot_color, -1)
        cv2.circle(img_copy, (x, y), DOT_RADIUS, border_color, 1)
    
    return img_copy

def draw_visible_dots(img, coords):
    """Original visible dots function - kept for comparison"""
    for (x, y) in coords:
        cv2.circle(img, (x, y), DOT_RADIUS + 1, (255, 255, 255), -1)
        cv2.circle(img, (x, y), DOT_RADIUS, (0, 0, 0), -1)
    return img

def hash_dot_geometry(coords):
    rel = coords - coords.mean(axis=0)
    rel = np.round(rel / np.linalg.norm(rel))
    return hashlib.sha256(rel.astype(np.int16).tobytes()).hexdigest()

# --- Image Augmentation Functions ---
def apply_brightness_variation(img, factor_range=(0.8, 1.2)):
    """Apply brightness variation while preserving dot visibility"""
    factor = random.uniform(*factor_range)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def apply_contrast_variation(img, factor_range=(0.8, 1.2)):
    """Apply contrast variation"""
    factor = random.uniform(*factor_range)
    return np.clip(img * factor, 0, 255).astype(np.uint8)

def apply_gaussian_blur(img, kernel_range=(1, 3)):
    """Apply slight gaussian blur"""
    kernel_size = random.choice(range(kernel_range[0], kernel_range[1] + 1, 2))  # Odd numbers only
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def apply_rotation(img, angle_range=(-5, 5)):
    """Apply slight rotation"""
    angle = random.uniform(*angle_range)
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, rotation_matrix, (w, h), borderValue=(255, 255, 255))

def apply_perspective_tilt(img, tilt_range=0.02):
    """Apply slight perspective tilt"""
    h, w = img.shape[:2]
    tilt = random.uniform(-tilt_range, tilt_range)
    
    # Define source and destination points for perspective transform
    src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst_points = np.float32([
        [tilt * w, tilt * h],
        [w - tilt * w, tilt * h],
        [w + tilt * w, h - tilt * h],
        [-tilt * w, h - tilt * h]
    ])
    
    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    return cv2.warpPerspective(img, perspective_matrix, (w, h), borderValue=(255, 255, 255))

def apply_crop_and_resize(img, crop_range=(0.9, 1.0)):
    """Apply random crop and resize back to original size"""
    h, w = img.shape[:2]
    crop_factor = random.uniform(*crop_range)
    
    new_h, new_w = int(h * crop_factor), int(w * crop_factor)
    start_y = random.randint(0, h - new_h)
    start_x = random.randint(0, w - new_w)
    
    cropped = img[start_y:start_y + new_h, start_x:start_x + new_w]
    return cv2.resize(cropped, (w, h))

def apply_noise(img, noise_factor=0.01):
    """Apply slight noise"""
    noise = np.random.normal(0, noise_factor * 255, img.shape)
    noisy_img = img.astype(np.float32) + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

def generate_variation(base_img, variation_id):
    """Generate a single variation of the base image"""
    img = base_img.copy()
    
    # Apply random combination of augmentations
    augmentations = [
        lambda x: apply_brightness_variation(x, (0.85, 1.15)),
        lambda x: apply_contrast_variation(x, (0.85, 1.15)),
        lambda x: apply_gaussian_blur(x, (1, 3)),
        lambda x: apply_rotation(x, (-3, 3)),
        lambda x: apply_perspective_tilt(x, 0.015),
        lambda x: apply_crop_and_resize(x, (0.92, 1.0)),
        lambda x: apply_noise(x, 0.008)
    ]
    
    # Randomly select 2-4 augmentations to apply
    num_augs = random.randint(2, 4)
    selected_augs = random.sample(augmentations, num_augs)
    
    for aug in selected_augs:
        img = aug(img)
    
    return img

records = [
  {
    "product": "Sartor Activated Charcoal Face Wash 500ml",
    "batches": [
      {"batch": "06250101", "prod": "01/06/2025", "exp": "01/06/2027"},
      {"batch": "06250102", "prod": "01/06/2026", "exp": "01/06/2027"},
    ],
    "barcode": "5191102015196500",
    "manufacturer": "Sartor Health Company Ltd."
  },
  {
    "product": "Sartor Disinfectant Solution 70ml",
    "batches": [
      {"batch": "05250101", "prod": "01/06/2025", "exp": "01/06/2025"},
      {"batch": "05250102", "prod": "01/06/2026", "exp": "01/06/2027"},
    ],
    "barcode": "51911820151800700",
    "manufacturer": "Sartor Health Company Ltd."
  },
  {
    "product": "Sartor Aleo Vera Face Cleanser 500ml",
    "batches": [
      {"batch": "02250101", "prod": "01/06/2025", "exp": "01/06/2025"},
      {"batch": "02250102", "prod": "01/06/2026", "exp": "01/06/2027"},
    ],
    "barcode": "19118201814115500",
    "manufacturer": "Sartor Health Company Ltd."
  },
  {
    "product": "Sartor Instant Hand Sanitizer 30ml",
    "batches": [
      {"batch": "03250101", "prod": "01/06/2025", "exp": "01/06/2025"},
      {"batch": "03250102", "prod": "01/06/2026", "exp": "01/06/2027"},
    ],
    "barcode": "519118201518030",
    "manufacturer": "Sartor Health Company Ltd."
  },
  {
    "product": "Sartor Instant Hand Sanitizer 70ml",
    "batches": [
      {"batch": "01250101", "prod": "01/06/2025", "exp": "01/06/2025"},
      {"batch": "01250102", "prod": "01/06/2026", "exp": "01/06/2027"},
    ],
    "barcode": "5191182015180070",
    "manufacturer": "Sartor Health Company Ltd."
  },
   {
    "product": "Sartor Instant Hand Sanitizer 500ml",
    "batches": [
      {"batch": "04250101", "prod": "01/06/2025", "exp": "01/06/2025"},
      {"batch": "04250102", "prod": "01/06/2026", "exp": "01/06/2027"},
    ],
    "barcode": "19118201814115500",
    "manufacturer": "Sartor Health Company Ltd."
  },
  
  {
    "product": "Sartor Intense Moisturizing Lotion 500ml",
    "batches": [
      {"batch": "00250101", "prod": "01/06/2025", "exp": "01/06/2025"},
      {"batch": "00250102", "prod": "01/06/2026", "exp": "01/06/2027"},
    ],
    "barcode": "191182019913500",
    "manufacturer": "Sartor Health Company Ltd."
  }
]

# --- Generate microdot variants ---
for entry in records:
    product_name = entry["product"]
    base_img_name = product_name.lower().replace(" ", "_") + ".png"

    if not os.path.exists(base_img_path):
        print(f"⚠️ Image not found for {product_name}")
        continue

    original = cv2.imread(base_img_path)
    original = cv2.resize(original, IMG_SIZE)

    for batch in entry["batches"]:
        coords = apply_jitter(generate_microdots(original))
        triang = Delaunay(coords)
        h = hash_dot_geometry(coords)

        # Draw image with SMALL VISIBLE microdots
        dotted = draw_visible_small_dots(original.copy(), coords)

        # Folder structure
        subfolder = os.path.join(OUTPUT_DIR, product_name, batch["batch"])
        os.makedirs(subfolder, exist_ok=True)
        
        # Save original dotted image
        filename = f"{product_name}_dot.png"
        output_path = os.path.join(subfolder, filename)
        cv2.imwrite(output_path, dotted)
        
        # Generate and save variations
        print(f"🔄 Generating {VARIATIONS_COUNT} invisible dot variations for {product_name} - {batch['batch']}")
        for i in range(VARIATIONS_COUNT):
            variation = generate_variation(dotted, i)
            var_filename = f"{product_name}_dot_var_{i+1:02d}.png"
            var_output_path = os.path.join(subfolder, var_filename)
            cv2.imwrite(var_output_path, variation)

        # Metadata - without coordinates for security
        key = f"{product_name}_{batch['batch']}"
        metadata[key] = {
            "product": product_name,
            "batch": batch["batch"],
            "production_date": batch["prod"],
            "expiry_date": batch["exp"],
            "barcode": entry["barcode"],
            "manufacturer": entry["manufacturer"]
         
        }

# Save metadata
with open(METADATA_PATH, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"✅ CLG invisible micro-dot variants generated with {VARIATIONS_COUNT} variations per batch and metadata saved.")