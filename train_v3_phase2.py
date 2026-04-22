"""
Phase 2 Training - Fine-tuning on V3 model
"""
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Config
NUM_CLASSES = 22
BATCH_SIZE = 32
PHASE2_EPOCHS = 50
PHASE2_LR = 1e-5

def spares_categorical_crossentropy_with_smoothing(y_true, y_pred, label_smoothing=0.1):
    num_classes = tf.shape(y_pred)[-1]
    y_true = tf.cast(y_true, tf.int32)
    y_true = tf.one_hot(y_true, num_classes)
    if label_smoothing > 0:
        y_true = y_true * (1 - label_smoothing) + label_smoothing / tf.cast(num_classes, tf.float32)
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

# Load saved Phase 1 model
print("Loading Phase 1 model...")
model = keras.models.load_model('models/saved/v3_phase1_best.keras', safe_mode=False)

# Unfreeze last 80 layers
for layer in model.layers:
    if layer.name.startswith('efficientnetb0'):
        trainable = [l for l in layer.layers if l.trainable]
        for l in trainable[-80:]:
            l.trainable = True
        print(f"Unfroze {len(trainable[-80:])} layers")
        break

# Compile
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=PHASE2_LR),
    loss=lambda y_true, y_pred: spares_categorical_crossentropy_with_smoothing(y_true, y_pred, 0.1),
    metrics=['accuracy']
)

# Load datasets
train_ds = image_dataset_from_directory(
    'data/train_final_v3',
    labels='inferred',
    label_mode='int',
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    image_size=(224, 224),
    shuffle=True,
    seed=42
).prefetch(tf.data.AUTOTUNE)

val_ds = image_dataset_from_directory(
    'data/val_final_v3',
    labels='inferred',
    label_mode='int',
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    image_size=(224, 224),
    shuffle=False,
    seed=42
).prefetch(tf.data.AUTOTUNE)

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, verbose=1),
    ModelCheckpoint('models/saved/v3_phase2_best.keras', monitor='val_accuracy', save_best_only=True, verbose=1),
    tf.keras.callbacks.CSVLogger('results/v3_phase2_log.csv')
]

print(f"\n=== Phase 2: Fine-tuning ({PHASE2_EPOCHS} epochs, lr={PHASE2_LR}) ===")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=PHASE2_EPOCHS,
    callbacks=callbacks,
    verbose=1
)

print(f"\nBest val accuracy: {max(history.history['val_accuracy']):.4f}")

# Save final
model.save('models/saved/v3_final_model.keras')
print("Done!")