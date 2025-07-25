import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input


# --- Config ---
IMG_SIZE = (224, 224)
MODEL_PATH = "best_model.h5"
LABEL_MAP_PATH = "batch_label_map.json"
METADATA_PATH = "batch_metadata_map.json"
CLG_METADATA_FILE = "clg_microdot_metadata.json"

# --- Load Model and Metadata ---
print("🔍 Loading model and metadata...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(LABEL_MAP_PATH, "r") as f:
    label_map = json.load(f)
    index_to_label = {int(k): v for k, v in label_map.items()}

with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

# --- Preprocess Function ---
def preprocess_image(path):
    img = Image.open(path).convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img)
    # Use EfficientNet preprocessing (same as training)
    img_array = preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

# --- Verification ---

while True:
    img_path = input("📸 Enter path to scanned label image: ").strip()
    if img_path.lower() == "exit":
        print("👋 Exiting...")
        break
    if not os.path.exists(img_path):
        print("❌ Image not found.")
        continue

    # Predict batch using CNN
    print("🧠 Predicting batch...")
    img_tensor = preprocess_image(img_path)
    probs = model.predict(img_tensor)[0]
    top_idx = np.argmax(probs)
    top_batch = index_to_label[top_idx]
    batch_confidence = float(probs[top_idx])
    
    print(f"\n🔍 Predicted Batch: {top_batch} ({batch_confidence*100:.2f}% confidence)")
    
    # Authentication decision based on batch classification only
    if batch_confidence < 0.6:
        print("\n⚠️ SUSPICIOUS: Low batch classification confidence.")
        print("   This product may not be authentic.")
        continue
    
    print("\n✅ AUTHENTIC: High confidence batch classification passed!")
    
    # --- Display Metadata ---
    if top_batch in metadata:
        info = metadata[top_batch]
        print("\n📦 Batch Metadata:")
        print(f"  Product Name : {info['product']}")
        print(f"  Batch Number : {top_batch}")
        print(f"  Barcode      : {info['barcode']}")
        print(f"  Manufacturer : {info['manufacturer']}")
        print(f"  Production   : {info['prod_date']}")
        print(f"  Expiry       : {info['exp_date']}")
    else:
        print("⚠️ No metadata found for this batch.")