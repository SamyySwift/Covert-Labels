import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import preprocess_input
import base64
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()


# --- Config ---
IMG_SIZE = (224, 224)
MODEL_PATH = "batch_classifier_model.keras" 
LABEL_MAP_PATH = "batch_label_map.json"
METADATA_PATH = "batch_metadata_map.json"


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
            model="google/gemini-2.0-flash-exp:free",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Examine this product label image very carefully for authentication microdots. Look for:\n\n1. Small circular dots (typically 1-3mm in diameter)\n2. These dots may appear as:\n   - White or light-colored circles\n   - Slightly raised or embossed circular patterns\n   - Small round spots that stand out from the background\n   - Dots that may have a subtle border or rim\n\n3. They are usually scattered across the label surface\n4. May appear on colored backgrounds (blue, purple, etc.)\n5. Look at ALL areas of the label, including corners and edges\n\nPay special attention to any small circular elements that look intentionally placed rather than printing artifacts or dust specks.\n\nRespond with ONLY 'YES' if you can see ANY circular dot patterns that could be authentication markers, or ONLY 'NO' if you see absolutely no such circular patterns. Do not provide explanations - just YES or NO."
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
        
        # Simple and reliable detection
        if "YES" in response:
            has_dots = True
        elif "NO" in response:
            has_dots = False
        else:
            # If response is unclear, default to no dots (safer)
            has_dots = False
            
        print(f"🤖 AI Response: {response}")
        print(f"🔍 Dots detected: {has_dots}")
        
        return has_dots, response
        
    except Exception as e:
        print(f"❌ Error with AI dot detection: {e}")
        return False, f"Error: {e}"

# --- Preprocess Function ---
def preprocess_image(path):
    img = Image.open(path).convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img)
    # Use ResNet50V2 preprocessing (same as training)
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


    # print("🤖 Analyzing image for circular dot patterns...")
    # has_dots, ai_response = detect_dots_with_ai(img_path)
    
    # print(f"🔍 Final decision: has_dots = {has_dots}")
    
    # if not has_dots:
    #     print("❌ COUNTERFEIT: No circular dot patterns detected by AI!")
    #     print("   This product appears to lack authentication microdots.")
    #     continue
    
    # print("✅ Circular dot patterns detected by AI - proceeding with batch verification")
    
    # Step 2: Predict batch using CNN
    print("🧠 Predicting batch...")
    img_tensor = preprocess_image(img_path)
    probs = model.predict(img_tensor)[0]
    top_idx = np.argmax(probs)
    top_batch = index_to_label[top_idx]
    batch_confidence = float(probs[top_idx])
    
    print(f"\n🔍 Predicted Batch: {top_batch} ({batch_confidence*100:.2f}% confidence)")
    
    # Authentication decision based on batch classification
    if batch_confidence < 0.50:
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