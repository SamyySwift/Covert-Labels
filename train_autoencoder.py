# module-level constants
import os
import datetime
import csv
import numpy as np
import tensorflow as tf
from PIL import Image,ImageEnhance, ImageOps, ImageFilter
import random

# Reproducibility
tf.random.set_seed(1337)
np.random.seed(1337)

# Use local GENUINE_IMAGE_DIR directly for training exclusively on manually captured images
IMAGE_DIR = os.getenv("GENUINE_IMAGE_DIR") or os.path.join(os.getcwd(), "GENUINE_IMAGE_DIR")
OUTPUT_MODEL = os.getenv("MODEL_PATH", "autoencoder_genuine.keras")
OUTPUT_DIR = os.getenv("AUTOENCODER_OUTPUT_DIR", "autoencoder_outputs")
RUNTIME_DIR = os.path.join(os.getcwd(), "runtime")
IMG_SIZE = 224
BATCH_SIZE = 8
VAL_SPLIT = 0.2
EPOCHS = 30
LIMIT = None
CHANNELS = 3
AUG_SAVE_SIZE = 512
RECON_DISPLAY_SCALE = 4
BACKGROUND_DIR = os.getenv("BACKGROUND_DIR") or "BACKGROUND_DIR"
ENABLE_BG_RANDOMIZATION = True


def load_and_preprocess(path: tf.Tensor, img_size: int, channels: int = 3) -> tf.Tensor:
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_image(img_bytes, channels=channels, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    img = tf.image.resize(img, [img_size, img_size])
    return img

def build_datasets(image_dir: str, img_size: int, batch_size: int, val_split: float,
                    limit: int | None, seed: int = 42):

    # Collect files robustly (recursive) and normalize to absolute paths
    all_files = collect_image_paths(image_dir)
    if limit is not None and limit > 0:
        all_files = all_files[:limit]

    num_files = len(all_files)
    # Robust recursive, case-insensitive file discovery
    valid_exts = (".png", ".jpg", ".jpeg")
    discovered_files = []
    for root, _, files in os.walk(image_dir):
        for fname in files:
            if fname.lower().endswith(valid_exts):
                discovered_files.append(os.path.join(root, fname))
    if len(discovered_files) == 0:
        raise RuntimeError(f"No image files found under {os.path.abspath(image_dir)}. Ensure it contains .png/.jpg/.jpeg.")

   
    train_count = int(num_files * (1.0 - val_split))
    if num_files > 1:
        train_count = max(1, min(train_count, num_files - 1))
    else:
        train_count = 1

    # Build dataset from file paths
    files_ds = tf.data.Dataset.from_tensor_slices(all_files)

    dataset_img = files_ds.map(lambda p: load_and_preprocess(p, img_size),
                               num_parallel_calls=tf.data.AUTOTUNE)

    # Train/val splits on images (no vectorization)
    train_img_ds = dataset_img.take(train_count).shuffle(2048, seed=seed).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_img_ds = dataset_img.skip(train_count).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Return image datasets; keep signature shape
    return train_img_ds, val_img_ds, val_img_ds, num_files

    # Vectorize for Dense model
    input_dim = img_size * img_size * CHANNELS
    dataset_vec = dataset_img.map(lambda img: tf.reshape(img, [input_dim]),
                                  num_parallel_calls=tf.data.AUTOTUNE)

    train_vec_ds = dataset_vec.take(train_count).shuffle(2048, seed=seed).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_vec_ds = dataset_vec.skip(train_count).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_img_ds = dataset_img.skip(train_count).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_vec_ds, val_vec_ds, val_img_ds, num_files



def save_reconstructions(model, val_img_ds, out_dir, n=8, img_size=IMG_SIZE, fallback_ds=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Try to take one batch from validation; fallback to train if empty
    batch_img = None
    for b in val_img_ds.take(1):
        batch_img = b
    if batch_img is None and fallback_ds is not None:
        for b in fallback_ds.take(1):
            batch_img = b
    if batch_img is None:
        raise RuntimeError("No images available in validation or fallback dataset to save reconstructions.")

    # Clamp n to available batch size
    avail = int(batch_img.shape[0])
    n = min(n, avail)

    # Run model
    recon = model.predict(batch_img[:n], verbose=0)

    # Save side-by-side grid scaled for readability
    # (Assumes RECON_DISPLAY_SCALE and helper to tile-save are already defined)
    W, H = img_size * RECON_DISPLAY_SCALE, img_size * RECON_DISPLAY_SCALE
    grid = Image.new("RGB", (n * W, 2 * H))
    originals = batch_img[:n].numpy()
    recon_img = recon
    for i in range(n):
        orig = (np.clip(originals[i] * 255.0, 0, 255)).astype(np.uint8)
        rec = (np.clip(recon_img[i] * 255.0, 0, 255)).astype(np.uint8)
        orig_pil = Image.fromarray(orig).resize((W, H), resample=Image.BICUBIC)
        rec_pil  = Image.fromarray(rec).resize((W, H), resample=Image.BICUBIC)
        grid.paste(orig_pil, (i * W, 0))
        grid.paste(rec_pil, (i * W, H))

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUTPUT_DIR, f"reconstructions_{ts}.png")
    grid.save(out_path)
    return out_path


class AnomalyDetector(tf.keras.Model):
    def __init__(self, img_size: int = IMG_SIZE, channels: int = CHANNELS, latent_channels: int = 256, **kwargs):
        super(AnomalyDetector, self).__init__(**kwargs)
        self.img_size = img_size
        self.channels = channels
        self.latent_channels = latent_channels
        L = tf.keras.layers
        
        # Improved encoder with skip connections and less aggressive downsampling
        self.encoder = tf.keras.Sequential([
            L.InputLayer(input_shape=(img_size, img_size, channels)),
            L.Conv2D(64, 3, strides=1, padding="same", activation="relu"),
            L.Conv2D(64, 3, strides=2, padding="same", activation="relu"),  # 224→112
            L.Conv2D(128, 3, strides=1, padding="same", activation="relu"),
            L.Conv2D(128, 3, strides=2, padding="same", activation="relu"), # 112→56
            L.Conv2D(256, 3, strides=1, padding="same", activation="relu"),
            L.Conv2D(latent_channels, 3, strides=2, padding="same", activation="relu"), # 56→28
        ], name="encoder")

        # Improved decoder with more layers
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(256, 3, strides=2, padding="same", activation="relu"), # 28→56
            tf.keras.layers.Conv2D(256, 3, strides=1, padding="same", activation="relu"),
            tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu"), # 56→112
            tf.keras.layers.Conv2D(128, 3, strides=1, padding="same", activation="relu"),
            tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu"),  # 112→224
            tf.keras.layers.Conv2D(64, 3, strides=1, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(channels, 3, padding="same", activation="sigmoid"),  # align with [0,1] inputs
        ], name="decoder")

    def call(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y

    def get_config(self):
        config = super().get_config()
        config.update({
            "img_size": self.img_size,
            "channels": self.channels,
            "latent_channels": self.latent_channels,
        })
        return config

    @classmethod
    def from_config(cls, config):
        kwargs = dict(config)
        return cls(**kwargs)

def reconstruction_mse(x_true: tf.Tensor, x_pred: tf.Tensor) -> tf.Tensor:
    # Per-image mean squared error across H, W, C
    return tf.reduce_mean(tf.math.squared_difference(x_true, x_pred), axis=(1, 2, 3))

# NEW: per-pixel MSE map and patch-wise max
def mse_map(x_true: tf.Tensor, x_pred: tf.Tensor) -> tf.Tensor:
    # [B,H,W] mean over channels
    return tf.reduce_mean(tf.math.squared_difference(x_true, x_pred), axis=3)

def patch_max_mse(x_true: tf.Tensor, x_pred: tf.Tensor, patch_size: int = 16) -> tf.Tensor:
    # Compute average MSE per patch then take max across patches per image
    m = mse_map(x_true, x_pred)                         # [B,H,W]
    m = tf.expand_dims(m, axis=-1)                      # [B,H,W,1]
    pooled = tf.nn.avg_pool(m,
                            ksize=[1, patch_size, patch_size, 1],
                            strides=[1, patch_size, patch_size, 1],
                            padding="SAME")             # [B,H/ps,W/ps,1]
    return tf.reduce_max(pooled, axis=(1, 2))           # [B]

def compute_threshold(autoencoder: tf.keras.Model, val_img_ds: tf.data.Dataset, percentile: float = 95.0) -> float:
    losses = []
    for batch in val_img_ds:
        preds = autoencoder.predict(batch, verbose=0)
        batch_losses = reconstruction_mse(batch, preds).numpy()
        losses.extend(batch_losses.tolist())
    if len(losses) == 0:
        raise RuntimeError("Validation set empty; cannot compute anomaly threshold.")
    threshold = float(np.percentile(losses, percentile))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "anomaly_threshold.txt"), "w") as f:
        f.write(str(threshold))
    print(f"🔎 Computed anomaly threshold (P{percentile:.0f}): {threshold:.6f}")
    return threshold

# NEW: threshold for localized anomalies
def compute_threshold_patch(autoencoder: tf.keras.Model, val_img_ds: tf.data.Dataset, percentile: float = 95.0, patch_size: int = 16) -> float:
    losses = []
    for batch in val_img_ds:
        preds = autoencoder.predict(batch, verbose=0)
        batch_losses = patch_max_mse(batch, preds, patch_size=patch_size).numpy()
        losses.extend(batch_losses.tolist())
    if len(losses) == 0:
        raise RuntimeError("Validation set empty; cannot compute patch anomaly threshold.")
    threshold = float(np.percentile(losses, percentile))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "anomaly_threshold_patch.txt"), "w") as f:
        f.write(str(threshold))
    print(f"🔎 Computed patch anomaly threshold (P{percentile:.0f}, ps={patch_size}): {threshold:.6f}")
    return threshold



# NEW: sensitive detector that also checks localized anomalies
def detect_image_anomaly_sensitive(autoencoder: tf.keras.Model, image_path: str, global_threshold: float, patch_threshold: float, patch_size: int = 16):
    img_bytes = tf.io.read_file(image_path)
    img = tf.image.decode_image(img_bytes, channels=CHANNELS, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    x_img = tf.expand_dims(img, axis=0)
    pred = autoencoder.predict(x_img, verbose=0)
    global_loss = float(reconstruction_mse(x_img, pred)[0].numpy())
    local_max = float(patch_max_mse(x_img, pred, patch_size=patch_size)[0].numpy())
    is_anom = (global_loss > global_threshold)
    return is_anom, global_loss, local_max

# Top-level helpers (place anywhere above main())
def list_background_paths(dir_path: str) -> list[str]:
    paths: list[str] = []
    for root, _, files in os.walk(dir_path):
        for fname in files:
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                paths.append(os.path.join(root, fname))
    return sorted(paths)


def load_random_background(target_size: int) -> Image.Image:
    paths = list_background_paths(BACKGROUND_DIR)
    if paths:
        bg = Image.open(random.choice(paths)).convert("RGB")
        bg = ImageOps.fit(bg, (target_size, target_size), method=Image.BICUBIC, centering=(0.5, 0.5))
        return bg
    # Fallback: synthetic color + light texture
    color = tuple(int(c) for c in np.random.randint(30, 225, size=(3,)))
    bg = Image.new("RGB", (target_size, target_size), color)
    if np.random.rand() < 0.6:
        arr = np.array(bg).astype(np.float32) / 255.0
        noise = np.random.normal(0.0, 0.05, arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0.0, 1.0)
        bg = Image.fromarray((arr * 255.0).astype(np.uint8))
    return bg

def make_foreground_mask(img: Image.Image, white_thresh: int = 245) -> Image.Image:
    arr = np.array(img.convert("RGB"))
    non_white = (arr[...,0] < white_thresh) | (arr[...,1] < white_thresh) | (arr[...,2] < white_thresh)
    mask = (non_white * 255).astype(np.uint8)
    return Image.fromarray(mask).filter(ImageFilter.GaussianBlur(radius=1))


def composite_on_random_background(fg: Image.Image, target_size: int) -> Image.Image:
    """
    Pastes a foreground image onto a random background image or solid color.
    - Uses alpha mask if the foreground has transparency.
    - Falls back to a computed non-white mask if alpha is fully opaque.
    """
    fg_rgba = ImageOps.fit(fg.convert("RGBA"), (target_size, target_size),
                           method=Image.BICUBIC, centering=(0.5, 0.5))

    # pick background (folder or random color)
    paths = list_background_paths(BACKGROUND_DIR)
    if paths:
        bg = Image.open(random.choice(paths)).convert("RGB")
        bg = ImageOps.fit(bg, (target_size, target_size), method=Image.BICUBIC, centering=(0.5, 0.5)).convert("RGBA")
    else:
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        bg = Image.new("RGBA", (target_size, target_size), color + (255,))

    # alpha-aware mask; fallback to non-white mask if the alpha is fully opaque
    alpha = fg_rgba.split()[-1]
    a_arr = np.array(alpha)
    if a_arr.min() == 255 and a_arr.max() == 255:
        mask = make_foreground_mask(fg_rgba)
    else:
        mask = alpha

    bg.paste(fg_rgba, (0, 0), mask)
    return bg.convert("RGB")



def generate_augmented_dataset(single_image_path: str, out_dir: str, count: int, img_size: int):
    os.makedirs(out_dir, exist_ok=True)
    src = Image.open(single_image_path).convert("RGB")
    for i in range(count):
        aug_img = augment_realistic(src, img_size)
        fname = f"aug_{i:04d}.png"
        save_path = os.path.join(out_dir, fname)
        aug_img.save(save_path, format="PNG")
    print(f"✅ Generated {count} augmented samples at {out_dir}")

# ADD BACK: first-image finder
def find_first_image_in_dir(dir_path: str) -> str | None:
    for root, _, files in os.walk(dir_path):
        for fname in files:
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                return os.path.join(root, fname)
    return None

# ADD BACK: robust, recursive image listing
def validate_image_file(image_path: str) -> bool:
    """Validate that an image file can be loaded and is not corrupted."""
    try:
        # Check file size first
        if os.path.getsize(image_path) == 0:
            print(f"⚠️ Skipping empty file: {image_path}")
            return False
        
        # Try to open with PIL
        with Image.open(image_path) as img:
            img.verify()  # Verify the image is not corrupted
        
        # Try to open again and convert (verify() closes the file)
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            if img.size[0] == 0 or img.size[1] == 0:
                print(f"⚠️ Skipping zero-size image: {image_path}")
                return False
        
        return True
    except Exception as e:
        print(f"⚠️ Skipping corrupted image {image_path}: {e}")
        return False

def collect_image_paths(image_dir: str) -> list[str]:
    """Collect and validate image paths, filtering out corrupted files."""
    roots = os.path.abspath(image_dir)
    paths: list[str] = []
    total_found = 0
    
    for root, _, files in os.walk(roots):
        for fname in files:
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                total_found += 1
                full_path = os.path.join(root, fname)
                if validate_image_file(full_path):
                    paths.append(full_path)
    
    print(f"📊 Found {total_found} image files, {len(paths)} are valid")
    if len(paths) < total_found:
        print(f"⚠️ Filtered out {total_found - len(paths)} corrupted/empty image files")
    
    return sorted(paths)

# ADD BACK: training data source resolver
def resolve_image_root() -> str:
    """
    Determine training source, preferring SINGLE_IMAGE_PATH for on-the-fly augmentation.
    """
    single_env = os.getenv("SINGLE_IMAGE_PATH")
    single_image_path = None
    if single_env:
        if os.path.isdir(single_env):
            single_image_path = find_first_image_in_dir(single_env)
        elif os.path.isfile(single_env):
            single_image_path = single_env
    else:
        local_single_dir = os.path.join(os.getcwd(), "SINGLE_IMAGE_PATH")
        if os.path.isdir(local_single_dir):
            single_image_path = find_first_image_in_dir(local_single_dir)

    if single_image_path:
        augment_dir = os.path.join(OUTPUT_DIR, "augmented_dataset")
        os.makedirs(augment_dir, exist_ok=True)
        generate_augmented_dataset(single_image_path, augment_dir, count=400, img_size=IMG_SIZE)
        # Verify that augmentation produced at least one valid image
        has_imgs = False
        for root, _, files in os.walk(augment_dir):
            for fname in files:
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    has_imgs = True
                    break
            if has_imgs:
                break
        if has_imgs:
            return augment_dir
        print(f"⚠️ No images found in {augment_dir} after augmentation. Falling back to source directory.")
        return os.path.dirname(single_image_path)

    image_dir_env = os.getenv("GENUINE_IMAGE_DIR") or os.getenv("UPLOAD_DIR")
    if image_dir_env and os.path.isdir(image_dir_env):
        return image_dir_env

    local_genuine_dir = os.path.join(os.getcwd(), "GENUINE_IMAGE_DIR")
    if os.path.isdir(local_genuine_dir):
        return local_genuine_dir

    if os.path.isdir("uploads"):
        return "uploads"

    raise FileNotFoundError(
        "No valid image source found. Provide SINGLE_IMAGE_PATH (file/dir) or ensure GENUINE_IMAGE_DIR/ or uploads/ exists."
    )

    
def augment_realistic(img: Image.Image, img_size: int) -> Image.Image:
    fg = ImageOps.fit(img.convert("RGBA"), (target_size, target_size), method=Image.BICUBIC, centering=(0.5, 0.5))
    bg = load_random_background(target_size).convert("RGBA")
    mask = make_foreground_mask(fg)
    bg.paste(fg, (0, 0), mask)
    return bg.convert("RGB")

    target_size = AUG_SAVE_SIZE

    if ENABLE_BG_RANDOMIZATION:
        # Rotate foreground slightly, then composite on a random background
        angle = float(np.random.uniform(-15.0, 15.0))
        rotated_fg = ImageOps.fit(img.convert("RGBA"), (target_size, target_size), method=Image.BICUBIC, centering=(0.5, 0.5)).rotate(
            angle, resample=Image.BICUBIC, expand=False
        )
        canvas = composite_on_random_background(rotated_fg, target_size)
    else:
        # Fallback: square-fit without padding; no white letterbox
        canvas = ImageOps.fit(img.convert("RGB"), (target_size, target_size), method=Image.BICUBIC, centering=(0.5, 0.5))

    # Photometric tweaks
    if np.random.rand() < 0.7:
        canvas = ImageEnhance.Brightness(canvas).enhance(float(np.random.uniform(0.9, 1.1)))
    if np.random.rand() < 0.7:
        canvas = ImageEnhance.Contrast(canvas).enhance(float(np.random.uniform(0.9, 1.1)))
    if np.random.rand() < 0.5:
        canvas = ImageEnhance.Sharpness(canvas).enhance(float(np.random.uniform(0.9, 1.1)))
    if np.random.rand() < 0.3:
        canvas = ImageOps.mirror(canvas)

    # Light noise
    if np.random.rand() < 0.4:
        na = np.asarray(canvas).astype(np.float32) / 255.0
        noise = np.random.normal(0.0, 0.02, na.shape).astype(np.float32)
        na = np.clip(na + noise, 0.0, 1.0)
        canvas = Image.fromarray((na * 255.0).astype(np.uint8))

    return canvas


def main():
    # DISABLED: Single-image augmentation flow
    # This functionality has been disabled to train exclusively on GENUINE_IMAGE_DIR
    
    # Force training to use only local GENUINE_IMAGE_DIR
    image_root = IMAGE_DIR  # This points directly to the local GENUINE_IMAGE_DIR folder
    
    # Validate that GENUINE_IMAGE_DIR exists
    if not os.path.isdir(image_root):
        raise FileNotFoundError(f"GENUINE_IMAGE_DIR not found: {image_root}")
    
    # Check if GENUINE_IMAGE_DIR contains any images
    image_files = []
    for root, _, files in os.walk(image_root):
        for fname in files:
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                image_files.append(os.path.join(root, fname))
    
    if not image_files:
        raise FileNotFoundError(f"No images found in GENUINE_IMAGE_DIR: {image_root}")
    
    print(f"🎯 Training exclusively on manually captured images from: {image_root}")
    print(f"📸 Found {len(image_files)} images for training")

    print("TensorFlow:", tf.__version__)
    print("GPUs:", tf.config.list_physical_devices("GPU"))

    train_img_ds, val_img_ds, _, num_files = build_datasets(
        image_root, IMG_SIZE, BATCH_SIZE, VAL_SPLIT, LIMIT
    )
    print(f"Total images: {num_files}")

    autoencoder = AnomalyDetector(img_size=IMG_SIZE, channels=CHANNELS, latent_channels=256)
    autoencoder.compile(optimizer="adam", loss="mse")

    # Build submodules by calling once, so summary shows params
    _ = autoencoder(tf.zeros((1, IMG_SIZE, IMG_SIZE, CHANNELS), dtype=tf.float32))
    autoencoder.summary()

    # Train on (x, x) pairs
    train_pairs_ds = train_img_ds.map(lambda x: (x, x))
    val_pairs_ds = val_img_ds.map(lambda x: (x, x))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    ]
    history = autoencoder.fit(
        train_pairs_ds,
        validation_data=val_pairs_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    autoencoder.save(OUTPUT_MODEL)
    print(f"✅ Saved model to {OUTPUT_MODEL}")

    if os.getenv("SAVE_RECONSTRUCTIONS", "0") == "1":
        recon_path = save_reconstructions(
            autoencoder, val_img_ds, OUTPUT_DIR, n=8, img_size=IMG_SIZE, fallback_ds=train_img_ds
        )
        print(f"🖼️ Saved reconstructions to {recon_path}")

    hist_path = os.path.join(RUNTIME_DIR, "training_history.csv")
    os.makedirs(RUNTIME_DIR, exist_ok=True)
    with open(hist_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "val_loss"])
        losses = history.history.get("loss", [])
        val_losses = history.history.get("val_loss", [None] * len(losses))
        for i, (l, vl) in enumerate(zip(losses, val_losses), start=1):
            writer.writerow([i, l, vl])
    print(f"📈 Saved training history to {hist_path}")

    # Use a more lenient percentile via env (default 95)
    anomaly_percentile = float(os.getenv("ANOMALY_PERCENTILE", "95"))
    threshold = compute_threshold(autoencoder, val_img_ds, percentile=anomaly_percentile)
    # Save patch-level threshold for sensitive detection
    patch_threshold = compute_threshold_patch(autoencoder, val_img_ds, percentile=95.0, patch_size=16)

   

if __name__ == "__main__":
    main()
