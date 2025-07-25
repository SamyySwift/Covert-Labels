import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
import base64
from openai import OpenAI
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import io

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# --- Config ---
IMG_SIZE = (224, 224)
MODEL_PATH = "best_model.h5"
LABEL_MAP_PATH = "batch_label_map.json"
METADATA_PATH = "batch_metadata_map.json"

# Get API key from environment variable
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable is not set.")

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
def detect_dots_with_ai(image_data):
    """Use OpenRouter's vision model to detect circular dot patterns"""
    try:
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://batch-verifier.onrender.com",
                "X-Title": "Batch Verification System",
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
def preprocess_image(image):
    img = image.convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img)
    # Use EfficientNet preprocessing (same as training)
    img_array = preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Batch Verification API",
        "status": "active",
        "endpoints": {
            "/verify": "POST - Upload image for batch verification"
        }
    })

@app.route('/verify', methods=['POST'])
def verify_batch():
    try:
        # Check if image file is provided
        if 'image' not in request.files:
            return jsonify({
                "error": "No image file provided",
                "status": "error"
            }), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({
                "error": "No image file selected",
                "status": "error"
            }), 400
        
        # Read and process the image
        image_data = file.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # Convert to PIL Image for processing
        image = Image.open(io.BytesIO(image_data))
        
        # Step 1: Check for circular dot patterns using AI
        print("🤖 Analyzing image for circular dot patterns...")
        has_dots, ai_response = detect_dots_with_ai(image_base64)
        
        if not has_dots:
            return jsonify({
                "status": "counterfeit",
                "reason": "No circular dot patterns detected",
                "ai_analysis": ai_response,
                "confidence": 0.0
            })
        
        # Step 2: Predict batch using CNN
        print("🧠 Predicting batch...")
        img_tensor = preprocess_image(image)
        probs = model.predict(img_tensor)[0]
        top_idx = np.argmax(probs)
        top_batch = index_to_label[top_idx]
        batch_confidence = float(probs[top_idx])
        
        # Authentication decision
        if batch_confidence < 0.6:
            return jsonify({
                "status": "suspicious",
                "reason": "Low batch classification confidence",
                "predicted_batch": top_batch,
                "confidence": batch_confidence,
                "ai_analysis": ai_response
            })
        
        # Get metadata if available
        batch_metadata = None
        if top_batch in metadata:
            batch_metadata = metadata[top_batch]
        
        return jsonify({
            "status": "authentic",
            "predicted_batch": top_batch,
            "confidence": batch_confidence,
            "ai_analysis": ai_response,
            "metadata": batch_metadata
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))
    app.run(host='0.0.0.0', port=port, debug=False)