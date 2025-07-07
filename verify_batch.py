import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf

# --- Config ---
IMG_SIZE = (224, 224)
MODEL_PATH = "best_model.h5"
LABEL_MAP_PATH = "batch_label_map.json"
METADATA_PATH = "batch_metadata_map.json"

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
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# --- Verification ---

while True:
    img_path = input("📸 Enter path to scanned label image: ").strip()
    if img_path.lower() == "exit":
        print("👋 Exiting...")
        break
    if not os.path.exists(img_path):
        print("❌ Image not found.")
        exit()

    print("🧠 Predicting batch...")
    img_tensor = preprocess_image(img_path)
    probs = model.predict(img_tensor)[0]
    top_idx = np.argmax(probs)
    top_batch = index_to_label[top_idx]
    confidence = float(probs[top_idx])

    print(f"\n🔍 Predicted Batch: {top_batch} ({confidence*100:.2f}% confidence)")

    if confidence*100 < 50:
        print("⚠️ Low confidence. This product may not be authentic.")
        continue

    # --- Display Metadata ---
    if top_batch in metadata:
        info = metadata[top_batch]
        print("✅ High confidence. This product is authentic.")
        print("\n📦 Batch Metadata:")
        print(f"  Product Name : {info['product']}")
        print(f"  Batch Number : {top_batch}")
        print(f"  Barcode      : {info['barcode']}")
        print(f"  Manufacturer : {info['manufacturer']}")
        print(f"  Production   : {info['prod_date']}")
        print(f"  Expiry       : {info['exp_date']}")

    else:
        print("⚠️ No metadata found for this batch.")