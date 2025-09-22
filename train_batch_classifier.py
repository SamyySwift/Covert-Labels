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
from tensorflow.keras.callbacks import EarlyStopping
from collections import Counter
import random
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input
import sqlite3


IMG_SIZE = (224, 224)
DATA_DIR = os.getenv("UPLOAD_DIR", "uploads")
DB_PATH = os.getenv("DB_PATH", os.path.join(DATA_DIR, "app.db"))
METADATA_FILE = "./clg_microdot_metadata.json"
BATCH = 8
EPOCHS = 30

# --- Augmentation Functions ---
MIN_SAMPLES_PER_CLASS = 10  

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

# Optionally enrich metadata from SQLite DB if present
# db_path = os.path.join(DATA_DIR, "app.db")
db_metadata_by_batch = {}
if os.path.exists(DB_PATH):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT batch, product, barcode, manufacturer, production_date, expiry_date, created_at
            FROM uploads
            ORDER BY datetime(created_at) DESC
        """).fetchall()
        for r in rows:
            b = r["batch"] or "unknown_batch"
            if b not in db_metadata_by_batch:
                db_metadata_by_batch[b] = {
                    "product": r["product"],
                    "barcode": r["barcode"],
                    "manufacturer": r["manufacturer"],
                    "production_date": r["production_date"],
                    "expiry_date": r["expiry_date"],
                }
    except Exception as e:
        print(f"⚠️ Could not read DB metadata from {DB_PATH}: {e}")
    finally:
        try:
            conn.close()
        except Exception:
            pass

# Scan uploads/<product>/<batch>/ for images; use batch folder name as label
if not os.path.isdir(DATA_DIR):
    raise RuntimeError(f"DATA_DIR '{DATA_DIR}' does not exist. Set UPLOAD_DIR to your uploads path (e.g., /data/uploads).")

total_folders = 0
for product_name in sorted(os.listdir(DATA_DIR)):
    prod_path = os.path.join(DATA_DIR, product_name)
    if not os.path.isdir(prod_path) or product_name == "app.db":
        continue

    for batch_id in sorted(os.listdir(prod_path)):
        batch_folder = os.path.join(prod_path, batch_id)
        if not os.path.isdir(batch_folder):
            continue

        total_folders += 1
        found_any = False
        for filename in os.listdir(batch_folder):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(batch_folder, filename)
                try:
                    img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
                    img_array = np.array(img) / 255.0
                    images.append(img_array)
                    labels.append(batch_id)
                    found_any = True
                except Exception as e:
                    print(f"⚠️ Could not load {img_path}: {e}")

        if found_any:
            md = db_metadata_by_batch.get(batch_id, {}) or {}
            md.setdefault("product", product_name)
            batch_metadata[batch_id] = {
                "product": md.get("product", product_name),
                "barcode": md.get("barcode"),
                "manufacturer": md.get("manufacturer"),
                "prod_date": md.get("production_date"),
                "exp_date": md.get("expiry_date"),
            }

print(f"\n✅ Loaded dataset from uploads: {len(images)} images across {len(set(labels))} classes (in {total_folders} batch folders)")


# --- Prepare Data ---
X = np.array(images)
X = preprocess_input(X * 255.0)  # Convert back to [0, 255] then preprocess
print(f"✅ Dataset shape: {X.shape}")

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
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input

def create_model():
    base_model = ResNet50V2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    # Phase 1: freeze backbone, train head only
    base_model.trainable = False

    inputs = base_model.input
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(len(class_names), activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    # Head-only compile: higher LR is okay since backbone is frozen
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Create initial model
model = create_model()
model.summary()



callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
    ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7, monitor='val_loss')
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
model.save("batch_classifier_model.keras")

with open("batch_label_map.json", "w") as f:
    json.dump({i: class_names[i] for i in range(len(class_names))}, f, indent=2)

with open("batch_metadata_map.json", "w") as f:
    json.dump(batch_metadata, f, indent=2)

# --- Evaluate Final Model ---
final_loss, final_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n📊 Final Test Results:")
print(f"Test Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
print(f"Test Loss: {final_loss:.4f}")
