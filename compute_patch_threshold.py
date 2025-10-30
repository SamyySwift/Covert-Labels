import os
import tensorflow as tf
from train_autoencoder_genuine_noargs import (
    AnomalyDetector,
    compute_threshold_patch,
    resolve_image_root,
    build_datasets,
    IMG_SIZE,
    BATCH_SIZE,
    VAL_SPLIT,
    LIMIT,
)

MODEL_PATH = "autoencoder_genuine.keras"

def main():
    # Load model with custom class, bypass compile to avoid custom loss deserialization
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"AnomalyDetector": AnomalyDetector},
        compile=False,
    )

    # Use same data source as training
    image_root = resolve_image_root()
    _, val_img_ds, _, _ = build_datasets(image_root, IMG_SIZE, BATCH_SIZE, VAL_SPLIT, LIMIT)

    percentile = float(os.getenv("ANOMALY_PERCENTILE_PATCH", "95"))
    patch_size = int(os.getenv("PATCH_SIZE", "16"))
    thr = compute_threshold_patch(model, val_img_ds, percentile=percentile, patch_size=patch_size)
    print(f"Saved patch threshold={thr:.6f} at autoencoder_outputs/anomaly_threshold_patch.txt")

if __name__ == "__main__":
    main()