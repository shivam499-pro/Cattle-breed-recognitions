#!/usr/bin/env python3
"""
Train final cattle breed classification model using EfficientNetB0
Dataset: 35 classes (train_final/val_final/test_final)
"""

import os
import json
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# ====================
# Configuration
# ====================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 35
EPOCHS_PHASE1 = 10
EPOCHS_PHASE2 = 20

# Paths
TRAIN_DIR = 'data/train_final'
VAL_DIR = 'data/val_final'
TEST_DIR = 'data/test_final'
MODEL_PATH = 'models/cattle_breed_final.keras'
MAPPING_FILE = 'models/breed_mapping_final.json'

print("="*60)
print("CATTLE BREED CLASSIFICATION - FINAL MODEL TRAINING")
print("="*60)
print(f"Classes: {NUM_CLASSES}")
print(f"Image size: {IMG_SIZE}")
print(f"Batch size: {BATCH_SIZE}")

# ====================
# Load class mapping
# ====================
with open(MAPPING_FILE, 'r') as f:
    mapping = json.load(f)
class_names = mapping['classes']
print(f"\nLoaded {len(class_names)} classes from mapping")

# ====================
# Data generators with augmentation
# ====================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.15,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

print("\nCreating data generators...")
train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Verify class order matches mapping
print(f"\nTrain classes found: {len(train_gen.class_indices)}")
print(f"Validation classes found: {len(val_gen.class_indices)}")
print(f"Test classes found: {len(test_gen.class_indices)}")

# ====================
# Compute class weights
# ====================
train_labels = train_gen.classes
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weight_dict = dict(enumerate(class_weights))
print(f"\nClass weights computed: {len(class_weight_dict)} classes")

# ====================
# Build model
# ====================
def build_model():
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    # Freeze base initially
    base_model.trainable = False
    
    inputs = layers.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    return model, base_model

print("\nBuilding EfficientNetB0 model...")
model, base_model = build_model()
model.summary()

# ====================
# Phase 1: Train top layers
# ====================
print("\n" + "="*60)
print("PHASE 1: Training Top Layers")
print("="*60)

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

cb = [
    callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
]

history1 = model.fit(
    train_gen,
    epochs=EPOCHS_PHASE1,
    validation_data=val_gen,
    class_weight=class_weight_dict,
    callbacks=cb,
    verbose=1
)

# Phase 1 results
val_loss, val_acc = model.evaluate(val_gen, verbose=0)
print(f"\nPhase 1 - Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

# ====================
# Phase 2: Fine-tune
# ====================
print("\n" + "="*60)
print("PHASE 2: Fine-tuning Last 40 Layers")
print("="*60)

# Unfreeze last 40 layers
for layer in base_model.layers[-40:]:
    layer.trainable = True

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

cb2 = [
    callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
]

history2 = model.fit(
    train_gen,
    epochs=EPOCHS_PHASE2,
    validation_data=val_gen,
    class_weight=class_weight_dict,
    callbacks=cb2,
    verbose=1
)

# ====================
# Evaluation
# ====================
print("\n" + "="*60)
print("EVALUATION")
print("="*60)

# Training metrics
train_loss, train_acc = model.evaluate(train_gen, verbose=0)
val_loss, val_acc = model.evaluate(val_gen, verbose=0)
test_loss, test_acc = model.evaluate(test_gen, verbose=0)

print(f"\nFinal Results:")
print(f"  Train Accuracy: {train_acc:.4f}")
print(f"  Val Accuracy:   {val_acc:.4f}")
print(f"  Test Accuracy:  {test_acc:.4f}")

# ====================
# Predictions analysis
# ====================
test_gen.reset()
y_pred = model.predict(test_gen, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_gen.classes

# Top-1 accuracy
top1_acc = accuracy_score(y_true, y_pred_classes)
print(f"\nTop-1 Accuracy: {top1_acc:.4f}")

# Top-3 accuracy
top3_acc = 0
for i in range(len(y_true)):
    top3_preds = np.argsort(y_pred[i])[-3:]
    if y_true[i] in top3_preds:
        top3_acc += 1
top3_acc /= len(y_true)
print(f"Top-3 Accuracy: {top3_acc:.4f}")

# ====================
# Per-class accuracy
# ====================
from collections import defaultdict
class_correct = defaultdict(int)
class_total = defaultdict(int)

for i in range(len(y_true)):
    class_total[y_true[i]] += 1
    if y_true[i] == y_pred_classes[i]:
        class_correct[y_true[i]] += 1

class_acc = {}
for cls in class_total:
    class_acc[cls] = class_correct[cls] / class_total[cls]

# Sort by accuracy
sorted_acc = sorted(class_acc.items(), key=lambda x: x[1], reverse=True)

print("\n" + "="*60)
print("TOP 5 BEST CLASSES")
print("="*60)
for cls_idx, acc in sorted_acc[:5]:
    print(f"  {class_names[cls_idx]}: {acc:.4f}")

print("\n" + "="*60)
print("TOP 5 WORST CLASSES")
print("="*60)
for cls_idx, acc in sorted_acc[-5:]:
    print(f"  {class_names[cls_idx]}: {acc:.4f}")

# ====================
# Confusion matrix
# ====================
cm = confusion_matrix(y_true, y_pred_classes)
print(f"\nConfusion matrix shape: {cm.shape}")
print(f"Test samples: {len(y_true)}")

# ====================
# Save model
# ====================
model.save(MODEL_PATH)
print(f"\nModel saved to {MODEL_PATH}")

# ====================
# Summary report
# ====================
print("\n" + "="*60)
print("FINAL TRAINING REPORT")
print("="*60)
print(f"Classes: {NUM_CLASSES}")
print(f"Train samples: {train_gen.samples}")
print(f"Val samples: {val_gen.samples}")
print(f"Test samples: {test_gen.samples}")
print(f"\nAccuracy:")
print(f"  Train: {train_acc:.4f}")
print(f"  Val:   {val_acc:.4f}")
print(f"  Test:  {test_acc:.4f}")
print(f"\nTop-1 Test: {top1_acc:.4f}")
print(f"Top-3 Test: {top3_acc:.4f}")
print("="*60)