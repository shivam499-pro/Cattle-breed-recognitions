"""
MobileNetV2 High-Accuracy Classifier Training
============================================
Trains a MobileNetV2-based classifier for cattle breed recognition.
Optimized for production with fine-tuning and class weights.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from collections import Counter

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ================== CONFIG ==================
DATA_DIR = 'data/train'
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 60  # Extended to 60 breeds
EPOCHS_PHASE1 = 10  # Train head only
EPOCHS_PHASE2 = 40  # Fine-tune
LEARNING_RATE_PHASE1 = 0.001
LEARNING_RATE_PHASE2 = 1e-5  # Very low for fine-tuning
MODEL_OUTPUT = 'models/cattle_breed_pro_v1.h5'
TFLITE_OUTPUT = 'models/tflite/cattle_breed_pro_v1_int8.tflite'


# ================== DATA AUGMENTATION ==================
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
], name="data_augmentation")


# ================== LOAD DATA ==================
def load_data():
    """Load training and validation data."""
    print("Loading data...")
    
    train_ds = keras.utils.image_dataset_from_directory(
        DATA_DIR,
        labels='inferred',
        label_mode='categorical',
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        validation_split=0.2,
        subset='training',
        seed=42
    )
    
    val_ds = keras.utils.image_dataset_from_directory(
        DATA_DIR,
        labels='inferred',
        label_mode='categorical',
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        validation_split=0.2,
        subset='validation',
        seed=42
    )
    
    # Get class names
    class_names = train_ds.class_names
    num_classes = len(class_names)
    
    print(f"Found {num_classes} classes")
    print(f"Classes: {class_names}")
    
    return train_ds, val_ds, class_names


# ================== COMPUTE CLASS WEIGHTS ==================
def compute_class_weights(train_ds):
    """Compute class weights for imbalanced dataset."""
    print("\nComputing class weights...")
    
    # Collect all labels
    all_labels = []
    for _, labels in train_ds:
        all_labels.extend(np.argmax(labels.numpy(), axis=1))
    
    # Count occurrences
    class_counts = Counter(all_labels)
    total = len(all_labels)
    num_classes = len(class_counts)
    
    # Compute weights using inverse sqrt
    max_count = max(class_counts.values())
    class_weights = {}
    
    for cls_id, count in class_counts.items():
        weight = np.sqrt(max_count / count)
        class_weights[cls_id] = weight
    
    # Normalize
    mean_weight = np.mean(list(class_weights.values()))
    class_weights = {k: v / mean_weight for k, v in class_weights.items()}
    
    print(f"Class weights computed for {num_classes} classes")
    print(f"Weight range: {min(class_weights.values()):.2f} - {max(class_weights.values()):.2f}")
    
    return class_weights


# ================== BUILD MODEL ==================
def build_model(num_classes):
    """Build MobileNetV2 model with custom head."""
    print("\nBuilding MobileNetV2 model...")
    
    # Load MobileNetV2 without top
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMAGE_SIZE, 3)
    )
    
    # Freeze base model
    base_model.trainable = False
    
    # Build model
    inputs = keras.Input(shape=(*IMAGE_SIZE, 3))
    
    # Data augmentation
    x = data_augmentation(inputs)
    
    # Preprocess for MobileNetV2
    x = layers.Rescaling(1./127.5, offset=-1)(x)
    
    # Base model
    x = base_model(x, training=False)
    
    # Custom head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs, name="cattle_breed_mobilenetv2")
    
    print(f"Total parameters: {model.count_params():,}")
    
    return model, base_model


# ================== TRAIN ==================
def train_model():
    """Train the model."""
    # Load data
    train_ds, val_ds, class_names = load_data()
    
    # Compute class weights
    class_weights = compute_class_weights(train_ds)
    
    # Build model
    model, base_model = build_model(len(class_names))
    
    # Phase 1: Train head only
    print("\n" + "="*60)
    print("PHASE 1: Training custom head (frozen base)")
    print("="*60)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE_PHASE1),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(MODEL_OUTPUT, monitor='val_accuracy', save_best_only=True),
        keras.callbacks.TensorBoard(log_dir='models/logs')
    ]
    
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_PHASE1,
        class_weight=class_weights,
        callbacks=callbacks
    )
    
    # Phase 2: Fine-tune
    print("\n" + "="*60)
    print("PHASE 2: Fine-tuning top 20 layers")
    print("="*60)
    
    # Unfreeze top 20 layers
    for layer in base_model.layers[-20:]:
        layer.trainable = True
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE_PHASE2),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_PHASE1 + EPOCHS_PHASE2,
        initial_epoch=EPOCHS_PHASE1,
        class_weight=class_weights,
        callbacks=callbacks
    )
    
    # Save final model
    model.save(MODEL_OUTPUT)
    print(f"\nModel saved to: {MODEL_OUTPUT}")
    
    return model, history2, class_names


# ================== CONVERT TO TFLITE ==================
def convert_to_tflite(model):
    """Convert model to TFLite with INT8 quantization."""
    print("\n" + "="*60)
    print("Converting to TFLite INT8")
    print("="*60)
    
    # Create representative dataset
    train_ds, _, _ = load_data()
    representative_data = train_ds.take(100)
    
    def representative_dataset_gen():
        for images, _ in representative_data:
            yield [images]
    
    # Convert
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_model = converter.convert()
    
    # Save
    os.makedirs(os.path.dirname(TFLITE_OUTPUT), exist_ok=True)
    with open(TFLITE_OUTPUT, 'wb') as f:
        f.write(tflite_model)
    
    size_mb = len(tflite_model) / (1024 * 1024)
    print(f"TFLite model saved to: {TFLITE_OUTPUT}")
    print(f"Size: {size_mb:.2f} MB")
    
    return TFLITE_OUTPUT


# ================== MAIN ==================
def main():
    print("="*60)
    print("MobileNetV2 Cattle Breed Classifier Training")
    print("="*60)
    
    # Train
    model, history, class_names = train_model()
    
    # Convert
    tflite_path = convert_to_tflite(model)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Model: {MODEL_OUTPUT}")
    print(f"TFLite: {tflite_path}")
    print(f"Classes: {len(class_names)}")


if __name__ == '__main__':
    main()
