import os
import base64
from openai import OpenAI
import tensorflow as tf
from dotenv import load_dotenv
import json

load_dotenv()

# Reuse the exact model and preprocessing from training
from train_autoencoder import (
    AnomalyDetector,
    detect_image_anomaly_sensitive,
    IMG_SIZE,
    CHANNELS,
)

MODEL_PATH = "autoencoder_genuine.keras"
THRESH_PATH = os.path.join("autoencoder_outputs", "anomaly_threshold.txt")
IMAGE_TO_VERIFY_DIR = os.path.join("autoencoder_outputs", "augmented_dataset")
IMAGE_TO_VERIFY = ""
PROMPT_FOR_PATH = True  

# Default patch threshold when file is missing (tuned for 224x224)
PATCH_THRESHOLD_DEFAULT = 0.02


def load_threshold(threshold_path: str) -> float:
    if not os.path.isfile(threshold_path):
        raise FileNotFoundError(f"Threshold file not found: {threshold_path}. Run training first.")
    with open(threshold_path, "r") as f:
        return float(f.read().strip())

def pick_image_to_verify() -> str:
    # Manual entry or code-defined override
    path = IMAGE_TO_VERIFY.strip()
    if PROMPT_FOR_PATH and not path:
        try:
            path = input("Enter image path (or press Enter to use default folder): ").strip()
        except EOFError:
            path = ""
    if path:
        if os.path.isfile(path):
            return path
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for fname in files:
                    if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                        return os.path.join(root, fname)
            raise FileNotFoundError(f"No images found under {path}")
        raise FileNotFoundError(f"Path not found: {path}")

    # Fallback to default folder
    if not os.path.isdir(IMAGE_TO_VERIFY_DIR):
        raise FileNotFoundError(f"Directory not found: {IMAGE_TO_VERIFY_DIR}")
    for root, _, files in os.walk(IMAGE_TO_VERIFY_DIR):
        for fname in files:
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                return os.path.join(root, fname)
    raise FileNotFoundError(f"No images found under {IMAGE_TO_VERIFY_DIR}")


def _get_openrouter_api_key() -> str:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key or not key.strip():
        raise ValueError("OPENROUTER_API_KEY is not set. Add it to .env or export in your shell.")
    return key


# llm_inspect_image(image_path)
def llm_inspect_image(image_path: str) -> tuple[str, float, str]:
    api_key = _get_openrouter_api_key()
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key)

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64}"

    prompt_text = (
        "Task: Verify product label authenticity.\n"
        "Output STRICT JSON only (no prose): "
        '{"classification":"normal|suspicious","confidence":0.0-1.0,"evidence":["..."]}.\n'
        "Rules:\n"
        "- Return 'suspicious' ONLY if you identify at least one concrete, localized indicator visible in the image "
        "(e.g., misspelling, glyph/kerning mismatch, logo shape inconsistency, color code mismatch, misalignment, "
        "obvious manipulation artifacts like warping/ghosting).\n"
        "- If uncertain or evidence is not clearly visible, return 'normal'. Do not guess.\n"
        "- Confidence: multiple clear indicators ≥0.9; one clear indicator ≈0.7–0.85; minor noise/compression ≤0.3.\n"
        "- Include short evidence strings; avoid generic statements.\n"
    )

    completion = client.chat.completions.create(
        model="meta-llama/llama-4-maverick:free",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
    )
    raw = completion.choices[0].message.content.strip()

    try:
        data = json.loads(raw)
        classification = str(data.get("classification", "normal")).lower().strip()
        confidence = float(data.get("confidence", 0.5))
        evidence = data.get("evidence", []) or []

        threshold = float(os.getenv("LLM_SUSPICIOUS_THRESHOLD", "0.7"))
        if classification == "suspicious" and (confidence < threshold or len(evidence) == 0):
            classification = "normal"
        return classification, confidence, raw
    except Exception:
        low = raw.lower()
        confidence = 0.0
        if "suspicious" in low:
            return "suspicious", confidence, raw
        if "normal" in low:
            return "normal", confidence, raw
        return "normal", confidence, raw


def main():
    image_path = pick_image_to_verify()
    print(f"Verifying image: {image_path}")

    # Load trained model (include custom objects and avoid compile-time loss deserialization)
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "AnomalyDetector": AnomalyDetector,
        },
        compile=False,
    )

    global_thr = load_threshold(THRESH_PATH)

    # Try loading patch threshold; fall back to default if missing
    patch_thr_path = os.path.join("autoencoder_outputs", "anomaly_threshold_patch.txt")
    try:
        patch_thr = load_threshold(patch_thr_path)
    except FileNotFoundError:
        print(f"⚠️ Patch threshold not found at {patch_thr_path}. Using default={PATCH_THRESHOLD_DEFAULT}.")
        patch_thr = PATCH_THRESHOLD_DEFAULT

    # Autoencoder decision
    is_anom, g_loss, p_max = detect_image_anomaly_sensitive(
        model, image_path, global_thr, patch_thr, patch_size=16
    )
    ae_status = "suspicious" if is_anom else "normal"

    # LLM decision with confidence
    llm_status, llm_conf, llm_raw = llm_inspect_image(image_path)

    # Only let LLM override when confidence >= min_conf
    min_conf = float(os.getenv("LLM_OVERRIDE_MIN_CONF", "0.6"))
    if ae_status != llm_status:
        if llm_conf >= min_conf:
            final_status = llm_status
            source = "LLM (override)"
        else:
            final_status = ae_status
            source = "AE (retained)"
    else:
        final_status = ae_status
        source = "Consensus"

    print(
        # f"AE={ae_status}\tglobal_loss={g_loss:.6f}\tpatch_max={p_max:.6f}\n"
        # f"LLM={llm_status}\tconfidence={llm_conf:.2f}\n"
        f"Status = {final_status}\n"
        # f"LLM_details={llm_raw}\npath={image_path}"
    )


if __name__ == "__main__":
    main()