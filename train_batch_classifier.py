import json
import os
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping
from collections import Counter
import random
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# --- Config ---
IMG_SIZE = (224, 224)
DATA_DIR = "clg_output_images"
METADATA_FILE = "./clg_microdot_metadata.json"
BATCH = 8
EPOCHS = 30

# --- Augmentation Functions ---
# Increase minimum samples and add more aggressive augmentation
MIN_SAMPLES_PER_CLASS = 10  # Increase from 5

def augment_image(img_array):
    """Apply more aggressive augmentations"""
    img = Image.fromarray((img_array * 255).astype(np.uint8))
    
    # More rotation range
    if random.random() > 0.3:  # More frequent
        angle = random.uniform(-25, 25)  # Wider range
        img = img.rotate(angle, fillcolor=(255, 255, 255))
    
    # More brightness/contrast variation
    if random.random() > 0.3:
        enhancer = ImageEnhance.Brightness(img)
        factor = random.uniform(0.7, 1.3)  # Wider range
        img = enhancer.enhance(factor)
    
    if random.random() > 0.3:
        enhancer = ImageEnhance.Contrast(img)
        factor = random.uniform(0.7, 1.3)
        img = enhancer.enhance(factor)
    
    # Add sharpness variation
    if random.random() > 0.5:
        enhancer = ImageEnhance.Sharpness(img)
        factor = random.uniform(0.8, 1.2)
        img = enhancer.enhance(factor)
    
    # Add color variation
    if random.random() > 0.5:
        enhancer = ImageEnhance.Color(img)
        factor = random.uniform(0.9, 1.1)
        img = enhancer.enhance(factor)
    
    # Random horizontal flip
    if random.random() > 0.5:
        img = ImageOps.mirror(img)
    
    # More noise
    if random.random() > 0.3:
        img_array = np.array(img) / 255.0
        noise = np.random.normal(0, 0.03, img_array.shape)  # Slightly more noise
        img_array = np.clip(img_array + noise, 0, 1)
        return img_array
    
    return np.array(img.resize(IMG_SIZE)) / 255.0

# --- Load Images and Labels ---
images = []
labels = []
batch_metadata = {}

with open(METADATA_FILE, "r") as f:
    metadata = json.load(f)

# Load all images (original + variations)
# Remove lines 23-65 (augmentation functions)
# Remove lines 95-115 (MIN_SAMPLES_PER_CLASS logic)

# Keep only the image loading section:
for key, product_data in tqdm(metadata.items(), desc="Loading images"):
    product_name = product_data["product"]
    batch_id = product_data["batch"]
    batch_folder = os.path.join(DATA_DIR, product_name, batch_id)
    
    if os.path.exists(batch_folder):
        # Load all images in the batch folder (original + 10 variations)
        for filename in os.listdir(batch_folder):
            if filename.endswith('.png'):
                img_path = os.path.join(batch_folder, filename)
                try:
                    img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
                    img_array = np.array(img) / 255.0
                    images.append(img_array)
                    labels.append(batch_id)
                except Exception as e:
                    print(f"⚠️ Could not load {img_path}: {e}")
        
        batch_metadata[batch_id] = {
            "product": product_name,
            "barcode": product_data["barcode"],
            "manufacturer": product_data["manufacturer"],
            "prod_date": product_data["production_date"],
            "exp_date": product_data["expiry_date"]
        }
    else:
        print(f"⚠️ Missing folder: {batch_folder}")

print(f"\n✅ Loaded dataset: {len(images)} images across {len(set(labels))} classes")

# Remove all augmentation code (lines 23-65)
# Remove MIN_SAMPLES_PER_CLASS logic (lines 95-115)

# --- Prepare Data ---
X = np.array(images)
# Preprocess for EfficientNet (scales to [-1, 1])
X = preprocess_input(X * 255.0)  # Convert back to [0, 255] then preprocess

le = LabelEncoder()
y = le.fit_transform(labels)
class_names = le.classes_

# Check final class distribution
class_counts = Counter(y)
print(f"\n📊 Final class distribution: {dict(class_counts)}")
print(f"Minimum class count: {min(class_counts.values())}")

# Now we can safely use stratified split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
print(f"✅ Using stratified split: {len(X_train)} train, {len(X_test)} test samples")

# --- Build Model with Transfer Learning ---
def create_model():
    """Create and compile the CNN model using EfficientNetB0 transfer learning"""
    # Load pre-trained EfficientNetB0 without top layers
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    
    # Keep base model trainable from the start (since we achieved 100% accuracy)
    base_model.trainable = True
    
    # Add custom classification head
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.3),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(len(class_names), activation='softmax')
    ])
    
    # Use a lower learning rate since we're training all layers
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Create initial model
model = create_model()
model.summary()



callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
    ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7, monitor='val_loss'),
    ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy')
]

# --- Train ---
print("\n🚀 Training with end-to-end fine-tuning...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH,
    callbacks=callbacks,
    verbose=1
)

# --- Save Model & Label Map ---
model.save("batch_classifier_model.h5")
with open("batch_label_map.json", "w") as f:
    json.dump({i: class_names[i] for i in range(len(class_names))}, f, indent=2)

with open("batch_metadata_map.json", "w") as f:
    json.dump(batch_metadata, f, indent=2)

# --- Evaluate Final Model ---
final_loss, final_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n📊 Final Test Results:")
print(f"Test Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
print(f"Test Loss: {final_loss:.4f}")

print("\n✅ Transfer learning training complete! Model and metadata saved.")


