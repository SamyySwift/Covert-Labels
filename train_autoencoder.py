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
EPOCHS = 50
LIMIT = None
CHANNELS = 3
AUG_SAVE_SIZE = 512
RECON_DISPLAY_SCALE = 4
BACKGROUND_DIR = os.getenv("BACKGROUND_DIR") or "BACKGROUND_DIR"
ENABLE_BG_RANDOMIZATION = False
LEARNING_RATE = 0.0001
DROPOUT_RATE = 0.15


def load_and_preprocess(path: tf.Tensor, img_size: int, channels: int = 3) -> tf.Tensor:
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_image(img_bytes, channels=channels, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    img = tf.image.resize(img, [img_size, img_size])
    return img

def build_datasets(image_dir: str, img_size: int, batch_size: int, val_split: float, limit: int | None, seed: int = 42):
    all_files = collect_image_paths(image_dir)
    if limit is not None and limit > 0:
        all_files = all_files[:limit]
    num_files = len(all_files)
    train_count = int(num_files * (1.0 - val_split))
    if num_files > 1:
        train_count = max(1, min(train_count, num_files - 1))
    else:
        train_count = 1
    def augment_tf(img: tf.Tensor) -> tf.Tensor:
        img = tf.image.random_brightness(img, max_delta=0.05)
        img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
        img = tf.image.random_saturation(img, lower=0.9, upper=1.1)
        noise = tf.random.normal(tf.shape(img), mean=0.0, stddev=0.02, dtype=img.dtype)
        img = tf.clip_by_value(img + noise, 0.0, 1.0)
        img = tf.image.random_flip_left_right(img)
        return img
    files_ds = tf.data.Dataset.from_tensor_slices(all_files)
    dataset_img = files_ds.map(lambda p: load_and_preprocess(p, img_size), num_parallel_calls=tf.data.AUTOTUNE)
    train_img_ds = dataset_img.take(train_count).map(augment_tf, num_parallel_calls=tf.data.AUTOTUNE).shuffle(2048, seed=seed).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_img_ds = dataset_img.skip(train_count).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_img_ds, val_img_ds, val_img_ds, num_files



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


class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.conv = tf.keras.layers.Conv2D(1, 7, padding="same", activation="sigmoid")
    
    def call(self, x):
        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        attention = self.conv(concat)
        return x * attention

class AnomalyDetector(tf.keras.Model):
    def __init__(self, img_size: int = IMG_SIZE, channels: int = CHANNELS, latent_channels: int = 256, **kwargs):
        super(AnomalyDetector, self).__init__(**kwargs)
        self.img_size = img_size
        self.channels = channels
        self.latent_channels = latent_channels
        L = tf.keras.layers
        
        self.conv1a = L.Conv2D(64, 3, strides=1, padding="same", use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.bn1a = L.BatchNormalization()
        self.act1a = L.ReLU()
        self.conv1b = L.Conv2D(64, 3, strides=2, padding="same", use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.bn1b = L.BatchNormalization()
        self.act1b = L.ReLU()
        self.drop1 = L.Dropout(DROPOUT_RATE)
        
        self.conv2a = L.Conv2D(128, 3, strides=1, padding="same", use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.bn2a = L.BatchNormalization()
        self.act2a = L.ReLU()
        self.conv2b = L.Conv2D(128, 3, strides=2, padding="same", use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.bn2b = L.BatchNormalization()
        self.act2b = L.ReLU()
        self.drop2 = L.Dropout(DROPOUT_RATE)
        
        self.conv3a = L.Conv2D(256, 3, strides=1, padding="same", use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.bn3a = L.BatchNormalization()
        self.act3a = L.ReLU()
        self.conv3b = L.Conv2D(latent_channels, 3, strides=2, padding="same", use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.bn3b = L.BatchNormalization()
        self.act3b = L.ReLU()
        self.drop3 = L.Dropout(DROPOUT_RATE)
        
        self.attention = SpatialAttention()
        self.up1 = L.Conv2DTranspose(256, 3, strides=2, padding="same", use_bias=False)
        self.bn_up1 = L.BatchNormalization()
        self.act_up1 = L.ReLU()
        self.merge1 = L.Concatenate()
        self.conv_up1b = L.Conv2D(256, 3, strides=1, padding="same", activation="relu")
        self.drop_up1 = L.Dropout(DROPOUT_RATE * 0.5)
        
        self.up2 = L.Conv2DTranspose(128, 3, strides=2, padding="same", use_bias=False)
        self.bn_up2 = L.BatchNormalization()
        self.act_up2 = L.ReLU()
        self.merge2 = L.Concatenate()
        self.conv_up2b = L.Conv2D(128, 3, strides=1, padding="same", activation="relu")
        self.drop_up2 = L.Dropout(DROPOUT_RATE * 0.5)
        
        self.up3 = L.Conv2DTranspose(64, 3, strides=2, padding="same", use_bias=False)
        self.bn_up3 = L.BatchNormalization()
        self.act_up3 = L.ReLU()
        self.merge3 = L.Concatenate()
        self.conv_up3b = L.Conv2D(64, 3, strides=1, padding="same", activation="relu")
        self.out_conv = L.Conv2D(channels, 3, padding="same", activation="sigmoid")

    def call(self, x, training=None):
        c1 = self.act1a(self.bn1a(self.conv1a(x), training=training))
        d1 = self.act1b(self.bn1b(self.conv1b(c1), training=training))
        d1 = self.drop1(d1, training=training)
        
        c2 = self.act2a(self.bn2a(self.conv2a(d1), training=training))
        d2 = self.act2b(self.bn2b(self.conv2b(c2), training=training))
        d2 = self.drop2(d2, training=training)
        
        c3 = self.act3a(self.bn3a(self.conv3a(d2), training=training))
        z = self.act3b(self.bn3b(self.conv3b(c3), training=training))
        z = self.drop3(z, training=training)
        z = self.attention(z)
        
        u1 = self.act_up1(self.bn_up1(self.up1(z), training=training))
        u1 = self.merge1([u1, c3])
        u1 = self.conv_up1b(u1)
        u1 = self.drop_up1(u1, training=training)
        
        u2 = self.act_up2(self.bn_up2(self.up2(u1), training=training))
        u2 = self.merge2([u2, c2])
        u2 = self.conv_up2b(u2)
        u2 = self.drop_up2(u2, training=training)
        
        u3 = self.act_up3(self.bn_up3(self.up3(u2), training=training))
        u3 = self.merge3([u3, c1])
        u3 = self.conv_up3b(u3)
        y = self.out_conv(u3)
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
    return tf.reduce_mean(tf.math.squared_difference(x_true, x_pred), axis=(1, 2, 3))

def mse_map(x_true: tf.Tensor, x_pred: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(tf.math.squared_difference(x_true, x_pred), axis=3)

def patch_max_mse(x_true: tf.Tensor, x_pred: tf.Tensor, patch_size: int = 16) -> tf.Tensor:
    m = mse_map(x_true, x_pred)
    m = tf.expand_dims(m, axis=-1)
    pooled = tf.nn.avg_pool(m, ksize=[1, patch_size, patch_size, 1], strides=[1, patch_size, patch_size, 1], padding="SAME")
    return tf.reduce_max(pooled, axis=(1, 2))

perceptual_model = None

def build_perceptual_model():
    vgg = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
    vgg.trainable = False
    layer_names = ['block1_conv2', 'block2_conv2', 'block3_conv3']
    outputs = [vgg.get_layer(name).output for name in layer_names]
    return tf.keras.Model(inputs=vgg.input, outputs=outputs)

def perceptual_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    global perceptual_model
    if perceptual_model is None:
        perceptual_model = build_perceptual_model()
    
    true_features = perceptual_model(y_true)
    pred_features = perceptual_model(y_pred)
    
    loss = 0.0
    for true_feat, pred_feat in zip(true_features, pred_features):
        loss += tf.reduce_mean(tf.abs(true_feat - pred_feat))
    return loss / len(true_features)

def mixed_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    mse = tf.reduce_mean(tf.math.squared_difference(y_true, y_pred), axis=(1, 2, 3))
    ssim_val = tf.image.ssim(y_true, y_pred, max_val=1.0)
    ssim_loss = 1.0 - ssim_val
    
    perc_loss = perceptual_loss(y_true, y_pred)
    
    total_loss = 0.3 * mse + 0.4 * ssim_loss + 0.3 * perc_loss
    return tf.reduce_mean(total_loss)

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
    fname = "anomaly_threshold_patch.txt" if patch_size == 16 else f"anomaly_threshold_patch_ps{patch_size}.txt"
    with open(os.path.join(OUTPUT_DIR, fname), "w") as f:
        f.write(str(threshold))
    print(f"🔎 Computed patch anomaly threshold (P{percentile:.0f}, ps={patch_size}): {threshold:.6f}")
    return threshold



# NEW: sensitive detector that also checks localized anomalies
def multi_scale_anomaly_score(x_true: tf.Tensor, x_pred: tf.Tensor) -> dict:
    global_mse = float(reconstruction_mse(x_true, x_pred)[0].numpy())
    
    patch_8 = float(patch_max_mse(x_true, x_pred, patch_size=8)[0].numpy())
    patch_16 = float(patch_max_mse(x_true, x_pred, patch_size=16)[0].numpy())
    patch_32 = float(patch_max_mse(x_true, x_pred, patch_size=32)[0].numpy())
    
    ssim_val = float(tf.image.ssim(x_true, x_pred, max_val=1.0)[0].numpy())
    
    mse_map_val = mse_map(x_true, x_pred)[0].numpy()
    spatial_variance = float(np.std(mse_map_val))
    
    return {
        'global_mse': global_mse,
        'patch_8': patch_8,
        'patch_16': patch_16,
        'patch_32': patch_32,
        'ssim': ssim_val,
        'spatial_variance': spatial_variance
    }

def detect_image_anomaly_sensitive(autoencoder: tf.keras.Model, image_path: str, global_threshold: float, patch_threshold: float, patch_size: int = 16):
    img_bytes = tf.io.read_file(image_path)
    img = tf.image.decode_image(img_bytes, channels=CHANNELS, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    x_img = tf.expand_dims(img, axis=0)
    pred = autoencoder.predict(x_img, verbose=0)
    
    scores = multi_scale_anomaly_score(x_img, pred)
    global_loss = scores['global_mse']
    local_max = scores['patch_16']
    
    is_anom_global = (global_loss > global_threshold)
    is_anom_patch = (local_max > patch_threshold)
    is_anom_ssim = (scores['ssim'] < 0.85)
    
    is_anom = is_anom_global or is_anom_patch or is_anom_ssim
    
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
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
    autoencoder.compile(optimizer=optimizer, loss=mixed_loss)

    # Build submodules by calling once, so summary shows params
    _ = autoencoder(tf.zeros((1, IMG_SIZE, IMG_SIZE, CHANNELS), dtype=tf.float32))
    autoencoder.summary()

    # Train on (x, x) pairs
    train_pairs_ds = train_img_ds.map(lambda x: (x, x))
    val_pairs_ds = val_img_ds.map(lambda x: (x, x))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, min_delta=1e-5),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=OUTPUT_MODEL.replace('.keras', '_best.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
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

    # Use a more lenient percentile via env (default 98)
    anomaly_percentile = float(os.getenv("ANOMALY_PERCENTILE", "95"))
    threshold = compute_threshold(autoencoder, val_img_ds, percentile=anomaly_percentile)
    patch_threshold_16 = compute_threshold_patch(autoencoder, val_img_ds, percentile=95.0, patch_size=16)
    patch_threshold_8 = compute_threshold_patch(autoencoder, val_img_ds, percentile=95.0, patch_size=8)
    patch_threshold_32 = compute_threshold_patch(autoencoder, val_img_ds, percentile=95.0, patch_size=32)
    
    print("\n" + "="*60)
    print("📊 ANOMALY DETECTION THRESHOLDS")
    print("="*60)
    print(f"Global MSE (P{anomaly_percentile}): {threshold:.6f}")
    print(f"Patch-8 (P95): {patch_threshold_8:.6f}")
    print(f"Patch-16 (P95): {patch_threshold_16:.6f}")
    print(f"Patch-32 (P95): {patch_threshold_32:.6f}")
    print("="*60 + "\n")



if __name__ == "__main__":
    main()