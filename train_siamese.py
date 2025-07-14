import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Dense, Dropout
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import register_keras_serializable
from sklearn.model_selection import train_test_split
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Custom functions
@register_keras_serializable()
def contrastive_loss(y_true, y_pred):
    margin = 1.0
    return tf.reduce_mean(
        y_true * tf.square(y_pred) + 
        (1 - y_true) * tf.square(tf.maximum(margin - y_pred, 0))
    )

@register_keras_serializable()
def l2_normalize(x):
    return tf.nn.l2_normalize(x, axis=1)

@register_keras_serializable()
def euclidean_distance(tensors):
    x, y = tensors
    return tf.sqrt(tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True))

# =========================
# Data Loading Functions
# =========================
def load_and_preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot load image: {img_path}")
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_input(img.astype('float32'))
    return img

def create_dataset_from_pairs(pairs, labels, batch_size=4):  # Reduced from 16
    def data_generator():
        for i in range(len(pairs)):
            pair = pairs[i]
            label = labels[i]
            
            img1 = load_and_preprocess_image(pair[0])
            img2 = load_and_preprocess_image(pair[1])
            
            yield (img1, img2), label
    
    # Create dataset with proper output signature
    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=(
            (tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
             tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32)),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )
    )
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# =========================
# Load Data
# =========================
pairs = np.load("pairs.npy", allow_pickle=True)
labels = np.load("labels.npy").astype("float32")

print(f"Loaded {len(pairs)} pairs with {len(labels)} labels")

# Split data for training and validation
train_pairs, val_pairs, train_labels, val_labels = train_test_split(
    pairs, labels, test_size=0.3, random_state=42
)

print(f"Training: {len(train_pairs)} pairs, Validation: {len(val_pairs)} pairs")


# GPU optimized:
train_dataset = create_dataset_from_pairs(train_pairs, train_labels, batch_size=8)
val_dataset = create_dataset_from_pairs(val_pairs, val_labels, batch_size=8)

# =========================
# Build Siamese Network
# =========================
def build_siamese_model():
    input_shape = (224, 224, 3)
    
    # Shared CNN backbone
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet', pooling='avg')
    base_model.trainable = False
    
    # Input layers
    input_a = Input(shape=input_shape, name='input_a')
    input_b = Input(shape=input_shape, name='input_b')
    
    # Shared feature extraction
    features_a = base_model(input_a)
    features_b = base_model(input_b)
    
    # L2 normalization
    norm_a = Lambda(l2_normalize, name='norm_a')(features_a)
    norm_b = Lambda(l2_normalize, name='norm_b')(features_b)
    
    # Euclidean distance
    distance = Lambda(euclidean_distance, name='distance')([norm_a, norm_b])
    
    model = Model(inputs=[input_a, input_b], outputs=distance)
    return model

model = build_siamese_model()

lr_schedule = ExponentialDecay(initial_learning_rate=1e-3, decay_steps=500, decay_rate=0.95)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss=contrastive_loss
)

model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10, 
    callbacks=[early_stop]
)

model.save("siamese_contrastive.keras")
print("✅ Contrastive model trained and saved to siamese_contrastive.keras")