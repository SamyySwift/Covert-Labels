import os
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
import threading
from dotenv import load_dotenv

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
from openai import OpenAI


load_dotenv()


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'reference_images'
app.config['TEMP_FOLDER'] = 'temp_uploads'
app.config['DATABASE'] = 'product_auth.db'
app.config['OPENROUTER_API_KEY'] = os.getenv('OPENROUTER_API_KEY', '')



os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['TEMP_FOLDER'], exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('product_auth.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

db_lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=4)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

PRODUCT_NAME_EXTRACTION_PROMPT = """Extract ONLY the textual product name from this image.

Rules:
1. Return JUST the product name words (brand + product), no symbols or graphics
2. IGNORE decorative glyphs and logos: +, crosses, emojis, icons, shapes
3. EXCLUDE trademark/copyright marks: TM, ®, ©
4. Do not include volume/size, batch codes, barcodes, prices, slogans
5. Preserve normal spelling and capitalization of letters; do not substitute symbols into letters
6. If multiple products are visible, choose the primary/largest packaging text
7. Output strictly as JSON: {"product_name": "..."}
"""

IMAGE_COMPARISON_PROMPT = """You are an expert counterfeit detection system. You will receive TWO images:
1. REFERENCE IMAGE: The authentic product (this is the standard)
2. TEST IMAGE: The product being verified (check this against the reference)

Compare the TEST image against the REFERENCE image and identify ALL discrepancies.

Analyze these specific aspects:
1. TEXT ACCURACY:
   - Spelling errors in product name, ingredients, or any text
   - Font type, size, and weight differences
   - Text alignment and spacing issues
   - Character kerning inconsistencies
   - Missing or extra text elements

2. PRINT QUALITY:
   - Blurry or pixelated text
   - Ink bleeding or smudging
   - Color registration errors
   - Print resolution differences
   - Faded or oversaturated colors

3. PACKAGING DESIGN:
   - Logo placement and proportions
   - Color accuracy (hue, saturation, brightness)
   - Graphics quality and sharpness
   - Barcode quality and placement
   - Design element positioning

4. PHYSICAL ATTRIBUTES:
   - Material texture differences
   - Finish quality (matte vs glossy)
   - Edge quality and cutting precision
   - Overall build quality

5. AUTHENTICITY INDICATORS:
   - Security features (holograms, watermarks)
   - Official seals or certifications
   - Batch codes and serial numbers

IMPORTANT:
- If images are identical or nearly identical: is_authentic = true, confidence_score > 0.9
- If minor differences exist: is_authentic = true, confidence_score 0.7-0.9
- If significant differences exist: is_authentic = false, confidence_score < 0.7
- List ALL discrepancies found, even minor ones

Return a JSON object with this exact structure:
{
  "is_authentic": boolean,
  "confidence_score": float (0.0 to 1.0),
  "discrepancies": [
    {
      "category": "text|print_quality|design|physical|security",
      "severity": "critical|high|medium|low",
      "description": "detailed description",
      "location": "where on the product"
    }
  ],
  "overall_assessment": "brief summary"
}"""


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def init_database():
    with db_lock:
        conn = sqlite3.connect(app.config['DATABASE'])
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_name TEXT NOT NULL,
                image_path TEXT NOT NULL,
                date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                additional_metadata TEXT,
                variant_key TEXT DEFAULT 'default'
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS verification_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_name TEXT NOT NULL,
                is_authentic BOOLEAN,
                confidence_score REAL,
                discrepancies TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_image_path TEXT
            )
        ''')
        cursor.execute("PRAGMA table_info(products)")
        cols = [r[1] for r in cursor.fetchall()]
        if 'variant_key' not in cols:
            cursor.execute("ALTER TABLE products ADD COLUMN variant_key TEXT DEFAULT 'default'")
        cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_products_name_variant ON products(product_name, variant_key)")
        conn.commit()
        conn.close()
    logger.info("Database initialized successfully")

try:
    init_database()
except Exception as e:
    logger.error(f"Database init error: {str(e)}")

def normalize_product_name(name: str) -> str:
    """
    Normalize product name by removing special characters and standardizing format.
    Handles: ™, ®, ©, extra spaces
    """
    import re
    
    normalized = name
    normalized = normalized.replace('™', '').replace('®', '').replace('©', '')
    normalized = normalized.replace('–', '-').replace('—', '-')
    normalized = re.sub(r'[^a-zA-Z0-9\s\-]', '', normalized)
    normalized = re.sub(r'\s+', ' ', normalized)
    normalized = normalized.strip().lower()
    
    return normalized


def normalize_color(key: str) -> str:
    import re
    s = (key or '').lower()
    s = re.sub(r'[^a-z\s-]', '', s)
    if 'blue' in s: return 'blue'
    if 'pink' in s or 'rose' in s: return 'pink'
    if 'red' in s or 'maroon' in s: return 'red'
    if 'green' in s: return 'green'
    if 'purple' in s or 'violet' in s: return 'purple'
    if 'yellow' in s: return 'yellow'
    if 'orange' in s: return 'orange'
    if 'black' in s: return 'black'
    if 'white' in s: return 'white'
    if 'clear' in s or 'transparent' in s: return 'clear'
    if 'grey' in s or 'gray' in s: return 'gray'
    if 'gold' in s: return 'gold'
    if 'silver' in s: return 'silver'
    if 'brown' in s: return 'brown'
    return 'default'


def call_llm_api(prompt: str, image_paths: list, response_format: str = "text") -> str:
    try:
        import base64
        
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=app.config['OPENROUTER_API_KEY'],
            default_headers={
                "HTTP-Referer": "http://localhost:8080",
                "X-Title": "Product Authentication System"
            }
        )
        content = [{"type": "text", "text": prompt}]
        
        for image_path in image_paths:
            with open(image_path, 'rb') as img_file:
                image_data = base64.b64encode(img_file.read()).decode('utf-8')
            
            ext = Path(image_path).suffix.lower()
            mime_type = f"image/{ext[1:]}" if ext in ['.jpg', '.jpeg', '.png', '.webp'] else "image/jpeg"
            
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{image_data}"
                }
            })
        
        extra_params = {}
        if response_format == "json":
            extra_params["response_format"] = {"type": "json_object"}
        
        completion = client.chat.completions.create(
            model="google/gemini-3-flash-preview",
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ],
            **extra_params
        )
        
        result_content = completion.choices[0].message.content
        
        logger.info(f"LLM API call successful for {len(image_paths)} image(s)")
        return result_content.strip()
    
    except Exception as e:
        logger.error(f"LLM API call failed: {str(e)}")
        raise


def extract_variant_color(image_path: str) -> str:
    try:
        prompt = "Return the dominant product color as one of: blue, pink, red, green, purple, yellow, orange, black, white, clear, gray, gold, silver, brown. JSON: {\"color\": \"...\"}."
        result = call_llm_api(prompt, [image_path], response_format="json")
        try:
            data = json.loads(result)
            raw = data.get("color", "")
        except json.JSONDecodeError:
            raw = ""
        return normalize_color(raw)
    except Exception:
        return "default"


def normalize_volume(s: str) -> str:
    import re
    if not s:
        return "default"
    txt = s.lower().strip()
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(ml|mL|l|L)?", txt)
    if not m:
        return "default"
    num = float(m.group(1))
    unit = (m.group(2) or "ml").lower()
    if unit.startswith("l"):
        num = num * 1000
    return f"{int(round(num))}ml"


def extract_volume_ml(image_path: str) -> str:
    try:
        prompt = "Return the package volume in milliliters as an integer string like '70ml'. JSON: {\"volume_ml\": \"...\"}."
        result = call_llm_api(prompt, [image_path], response_format="json")
        try:
            data = json.loads(result)
            raw = data.get("volume_ml", "")
        except json.JSONDecodeError:
            raw = ""
        return normalize_volume(raw)
    except Exception:
        return "default"


def parse_variant_fields(v: str):
    import re
    v = (v or "").lower()
    color = normalize_color(v)
    m = re.search(r"([0-9]+)\s*ml", v)
    vol = f"{m.group(1)}ml" if m else "default"
    return color, vol


def extract_product_name(image_path: str) -> str:
    try:
        result = call_llm_api(PRODUCT_NAME_EXTRACTION_PROMPT, [image_path], response_format="json")
        name = ""
        try:
            data = json.loads(result)
            name = (data.get("product_name", "") or "").strip()
        except json.JSONDecodeError:
            name = result.strip().splitlines()[0].strip()
        logger.info(f"Extracted product name: {name}")
        return name
    except Exception as e:
        logger.error(f"Failed to extract product name: {str(e)}")
        raise


def save_reference_product(product_name: str, image_path: str, metadata: Optional[Dict] = None, variant_key: str = "default"):
    with db_lock:
        conn = sqlite3.connect(app.config['DATABASE'])
        cursor = conn.cursor()
        try:
            cursor.execute(
                'INSERT INTO products (product_name, image_path, additional_metadata, variant_key) VALUES (?, ?, ?, ?)',
                (product_name, image_path, json.dumps(metadata or {}), variant_key)
            )
            conn.commit()
            logger.info(f"Saved reference product: {product_name} [{variant_key}]")
        except sqlite3.IntegrityError:
            cursor.execute(
                'UPDATE products SET image_path = ?, additional_metadata = ? WHERE product_name = ? AND variant_key = ?',
                (image_path, json.dumps(metadata or {}), product_name, variant_key)
            )
            conn.commit()
            logger.info(f"Updated reference product: {product_name} [{variant_key}]")
        finally:
            conn.close()


def get_reference_image(product_name: str, variant_key: Optional[str] = None) -> Optional[str]:
    with db_lock:
        conn = sqlite3.connect(app.config['DATABASE'])
        cursor = conn.cursor()
        cursor.execute('SELECT product_name, variant_key, image_path FROM products')
        rows = cursor.fetchall()
        norm_pn = normalize_product_name(product_name)
        target_color, target_vol = (None, None)
        if variant_key:
            c, v = parse_variant_fields(variant_key)
            target_color, target_vol = c, v
        exact = None
        color_match = None
        for pn, vk, ip in rows:
            if normalize_product_name(pn) != norm_pn:
                continue
            c, v = parse_variant_fields(vk)
            if target_color and target_vol and c == target_color and v == target_vol:
                exact = ip
                break
            if target_color and c == target_color and color_match is None:
                color_match = ip
            if not target_color and vk == 'default':
                color_match = ip
        conn.close()
        if exact:
            return exact
        return color_match


def validate_image(image_path: str) -> bool:
    try:
        img = Image.open(image_path)
        img.verify()
        return True
    except Exception as e:
        logger.error(f"Image validation failed: {str(e)}")
        return False


def perform_llm_comparison(reference_path: str, test_path: str) -> Dict:
    try:
        if not validate_image(reference_path) or not validate_image(test_path):
            raise ValueError("Invalid image file(s)")
        
        enhanced_prompt = f"{IMAGE_COMPARISON_PROMPT}\n\nThe FIRST image below is the REFERENCE (authentic product).\nThe SECOND image below is the TEST image (product being verified)."
        
        result = call_llm_api(enhanced_prompt, [reference_path, test_path], response_format="json")
        
        try:
            comparison_result = json.loads(result)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                comparison_result = json.loads(json_match.group())
            else:
                raise ValueError("Could not parse JSON from LLM response")
        
        if 'is_authentic' not in comparison_result:
            comparison_result['is_authentic'] = False
        if 'confidence_score' not in comparison_result:
            comparison_result['confidence_score'] = 0.0
        if 'discrepancies' not in comparison_result:
            comparison_result['discrepancies'] = []
        if 'overall_assessment' not in comparison_result:
            comparison_result['overall_assessment'] = "Analysis completed"
        
        return comparison_result
    
    except Exception as e:
        logger.error(f"LLM comparison failed: {str(e)}")
        return {
            "is_authentic": False,
            "confidence_score": 0.0,
            "discrepancies": [{
                "category": "system_error",
                "severity": "critical",
                "description": f"LLM analysis failed: {str(e)}",
                "location": "N/A"
            }],
            "overall_assessment": "Analysis failed due to system error"
        }


def log_verification(product_name: str, is_authentic: bool, confidence_score: float, 
                     discrepancies: List[Dict], user_image_path: str):
    with db_lock:
        conn = sqlite3.connect(app.config['DATABASE'])
        cursor = conn.cursor()
        cursor.execute(
            '''INSERT INTO verification_logs 
               (product_name, is_authentic, confidence_score, discrepancies, user_image_path)
               VALUES (?, ?, ?, ?, ?)''',
            (product_name, is_authentic, confidence_score, json.dumps(discrepancies), user_image_path)
        )
        conn.commit()
        conn.close()


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})


@app.route('/api/reference/upload', methods=['POST'])
def upload_reference():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type. Allowed: png, jpg, jpeg, webp"}), 400
        
        temp_filename = secure_filename(f"temp_{datetime.now().timestamp()}_{file.filename}")
        temp_path = os.path.join(app.config['TEMP_FOLDER'], temp_filename)
        file.save(temp_path)
        
        logger.info(f"Processing reference image: {temp_filename}")
        
        product_name = extract_product_name(temp_path)
        
        if not product_name:
            os.remove(temp_path)
            return jsonify({"error": "Could not extract product name from image"}), 400
        
        color = request.form.get('color') or extract_variant_color(temp_path)
        volume = request.form.get('volume_ml') or extract_volume_ml(temp_path)
        variant_key = f"{color}_{volume}" if (color != 'default' or volume != 'default') else 'default'
        extension = Path(file.filename).suffix
        safe_product_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in product_name)
        safe_product_name = safe_product_name.replace(' ', '_')
        safe_variant = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in (variant_key or 'default')).replace(' ', '-')
        final_filename = f"{safe_product_name}__{safe_variant}{extension}"
        final_path = os.path.join(app.config['UPLOAD_FOLDER'], final_filename)
        
        os.rename(temp_path, final_path)
        
        metadata = request.form.get('metadata', '{}')
        try:
            metadata_dict = json.loads(metadata)
        except:
            metadata_dict = {}
        
        save_reference_product(product_name, final_path, metadata_dict, variant_key)
        
        logger.info(f"Successfully saved reference product: {product_name} [{variant_key}]")
        
        return jsonify({
            "success": True,
            "product_name": product_name,
            "image_path": final_path,
            "message": "Reference product saved successfully"
        }), 201
    
    except Exception as e:
        logger.error(f"Reference upload failed: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/verify', methods=['POST'])
def verify_product():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type"}), 400
        
        temp_filename = secure_filename(f"verify_{datetime.now().timestamp()}_{file.filename}")
        temp_path = os.path.join(app.config['TEMP_FOLDER'], temp_filename)
        file.save(temp_path)
        
        logger.info(f"Verifying product image: {temp_filename}")
        
        product_name = extract_product_name(temp_path)
        
        if not product_name:
            os.remove(temp_path)
            return jsonify({"error": "Could not extract product name from image"}), 400
        
        color = extract_variant_color(temp_path)
        volume = extract_volume_ml(temp_path)
        variant_key = f"{color}_{volume}" if (color != 'default' or volume != 'default') else 'default'
        reference_path = get_reference_image(product_name, variant_key)
        
        if not reference_path:
            with db_lock:
                conn = sqlite3.connect(app.config['DATABASE'])
                cursor = conn.cursor()
                cursor.execute('SELECT product_name, variant_key FROM products')
                rows = cursor.fetchall()
                conn.close()
            norm_in = normalize_product_name(product_name)
            known = [vk for pn, vk in rows if normalize_product_name(pn) == norm_in]
            os.remove(temp_path)
            if known:
                return jsonify({
                    "error": "Variant not found for this product",
                    "product_name": product_name,
                    "variant_key": variant_key,
                    "known_variants": known,
                    "suggestion": "Upload this variant as a reference or choose one of the known variants"
                }), 404
            return jsonify({
                "error": "This product is not in our database",
                "product_name": product_name,
                "in_database": False,
                "suggestion": "Please upload a reference image first to train the model on this product."
            }), 404
        
        if not os.path.exists(reference_path):
            os.remove(temp_path)
            return jsonify({"error": "Reference image file not found"}), 500
        
        llm_results = perform_llm_comparison(reference_path, temp_path)
        
        is_authentic = llm_results.get('is_authentic', False)
        confidence_score = llm_results.get('confidence_score', 0.0)
        discrepancies = llm_results.get('discrepancies', [])
        
        log_verification(product_name, is_authentic, confidence_score, discrepancies, temp_path)
        
        response = {
            "product_name": product_name,
            "is_authentic": is_authentic,
            "confidence_score": round(confidence_score, 3),
            "analysis": {
                "llm_analysis": llm_results,
                "assessment": {
                    "verdict": "AUTHENTIC" if is_authentic else "COUNTERFEIT DETECTED",
                    "confidence_percentage": round(confidence_score * 100, 1),
                    "discrepancies_found": len(discrepancies),
                    "critical_issues": len([d for d in discrepancies if d.get('severity') == 'critical']),
                    "high_issues": len([d for d in discrepancies if d.get('severity') == 'high']),
                    "medium_issues": len([d for d in discrepancies if d.get('severity') == 'medium']),
                    "low_issues": len([d for d in discrepancies if d.get('severity') == 'low'])
                }
            },
            "discrepancies": discrepancies,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Verification complete: {product_name} - {'AUTHENTIC' if is_authentic else 'COUNTERFEIT'}")
        
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Verification failed: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/products', methods=['GET'])
def list_products():
    try:
        with db_lock:
            conn = sqlite3.connect(app.config['DATABASE'])
            cursor = conn.cursor()
            cursor.execute('SELECT product_name, image_path, date_added, additional_metadata FROM products')
            products = cursor.fetchall()
            conn.close()
        
        result = []
        for product in products:
            result.append({
                "product_name": product[0],
                "image_path": product[1],
                "date_added": product[2],
                "metadata": json.loads(product[3]) if product[3] else {}
            })
        
        return jsonify({"products": result, "count": len(result)}), 200
    
    except Exception as e:
        logger.error(f"Failed to list products: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/products/<product_name>', methods=['DELETE'])
def delete_product(product_name):
    try:
        reference_path = get_reference_image(product_name)
        
        if not reference_path:
            return jsonify({"error": "Product not found"}), 404
        
        with db_lock:
            conn = sqlite3.connect(app.config['DATABASE'])
            cursor = conn.cursor()
            cursor.execute('DELETE FROM products WHERE product_name = ?', (product_name,))
            conn.commit()
            conn.close()
        
        if os.path.exists(reference_path):
            os.remove(reference_path)
        
        logger.info(f"Deleted product: {product_name}")
        
        return jsonify({"success": True, "message": f"Product '{product_name}' deleted"}), 200
    
    except Exception as e:
        logger.error(f"Failed to delete product: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/verification-history', methods=['GET'])
def get_verification_history():
    try:
        limit = request.args.get('limit', 50, type=int)
        product_name = request.args.get('product_name', None)
        
        with db_lock:
            conn = sqlite3.connect(app.config['DATABASE'])
            cursor = conn.cursor()
            
            if product_name:
                cursor.execute(
                    '''SELECT product_name, is_authentic, confidence_score, discrepancies, timestamp
                       FROM verification_logs WHERE product_name = ?
                       ORDER BY timestamp DESC LIMIT ?''',
                    (product_name, limit)
                )
            else:
                cursor.execute(
                    '''SELECT product_name, is_authentic, confidence_score, discrepancies, timestamp
                       FROM verification_logs ORDER BY timestamp DESC LIMIT ?''',
                    (limit,)
                )
            
            logs = cursor.fetchall()
            conn.close()
        
        result = []
        for log in logs:
            result.append({
                "product_name": log[0],
                "is_authentic": bool(log[1]),
                "confidence_score": log[2],
                "discrepancies": json.loads(log[3]) if log[3] else [],
                "timestamp": log[4]
            })
        
        return jsonify({"history": result, "count": len(result)}), 200
    
    except Exception as e:
        logger.error(f"Failed to retrieve history: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    init_database()
    logger.info("Product Authentication System started (LLM-only mode)")
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)