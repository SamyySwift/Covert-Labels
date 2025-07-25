import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
import base64
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# --- Config ---
IMG_SIZE = (224, 224)
MODEL_PATH = "best_model.h5"
LABEL_MAP_PATH = "batch_label_map.json"
METADATA_PATH = "batch_metadata_map.json"
CLG_METADATA_FILE = "clg_microdot_metadata.json"

# Get API key from environment variable
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable is not set. Please set it before running the script.")

# --- Load Model and Metadata ---
print("🔍 Loading model and metadata...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(LABEL_MAP_PATH, "r") as f:
    label_map = json.load(f)
    index_to_label = {int(k): v for k, v in label_map.items()}

with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

# --- OpenRouter Client ---
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# --- Dot Detection Function using OpenRouter ---
def detect_dots_with_ai(img_path):
    """Use OpenRouter's vision model to detect circular dot patterns"""
    try:
        # Convert image to base64
        with open(img_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://your-site.com",  # Optional
                "X-Title": "Batch Verification System",  # Optional
            },
            model="moonshotai/kimi-vl-a3b-thinking:free",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze this image carefully. Look for small circular dots or microdots that could be authentication markers. These would typically be small, round, white or light-colored dots scattered across the image. Respond with 'YES' if you can clearly identify circular dot patterns that could be authentication microdots, or 'NO' if you cannot see any such patterns. Be very specific about what you observe."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        }
                    ]
                }
            ]
        )
        
        response = completion.choices[0].message.content.strip().upper()
        
        # Check if the response indicates dots are present
        has_dots = "YES" in response or "CIRCULAR" in response or "DOTS" in response or "MICRODOTS" in response
        
        return has_dots, response
        
    except Exception as e:
        print(f"❌ Error with AI dot detection: {e}")
        return False, f"Error: {e}"

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

    # Step 1: Check for circular dot patterns using AI
    print("🤖 Analyzing image for circular dot patterns...")
    has_dots, ai_response = detect_dots_with_ai(img_path)
    
    print(f"AI Analysis: {ai_response}")
    
    if not has_dots:
        print("❌ COUNTERFEIT: No circular dot patterns detected by AI!")
        print("   This product appears to lack authentication microdots.")
        continue
    
    print("✅ Circular dot patterns detected by AI - proceeding with batch verification")
    
    # Step 2: Predict batch using CNN
    print("🧠 Predicting batch...")
    img_tensor = preprocess_image(img_path)
    probs = model.predict(img_tensor)[0]
    top_idx = np.argmax(probs)
    top_batch = index_to_label[top_idx]
    batch_confidence = float(probs[top_idx])
    
    print(f"\n🔍 Predicted Batch: {top_batch} ({batch_confidence*100:.2f}% confidence)")
    
    # Authentication decision based on batch classification
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