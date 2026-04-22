"""
Evaluate V2 Model
================
Simple evaluation to complete the report.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ================== CONFIG ==================
TEST_DIR = 'data/test_41'
MAPPING_FILE = 'models/breed_mapping_v2.json'
MODEL_PATH = 'models/breed_classifier_41_v2.keras'
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 41

np.random.seed(42)
tf.random.set_seed(42)

# ================== CUSTOM METRICS ==================
def top_k_accuracy(k=3):
    def metric(y_true, y_pred):
        return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=k)
    metric.__name__ = f'top_{k}_accuracy'
    return metric

CUSTOM_OBJECTS = {'top_3_accuracy': top_k_accuracy(3)}

# ================== LOAD MAPPING ==================
with open(MAPPING_FILE, 'r') as f:
    mapping = json.load(f)
class_names = [mapping[str(i)] for i in range(NUM_CLASSES)]

# ================== LOAD DATA ==================
test_ds = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels='inferred',
    label_mode='categorical',
    class_names=class_names,
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    shuffle=False,
    seed=42
)

# ================== LOAD MODEL ==================
print("Loading model...")
model = keras.models.load_model(MODEL_PATH, custom_objects=CUSTOM_OBJECTS)

# ================== EVALUATE ==================
y_true = []
y_pred = []
all_preds = []
for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    all_preds.extend(preds)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)
all_preds = np.array(all_preds)

from sklearn.metrics import accuracy_score

top1 = accuracy_score(y_true, y_pred)
print(f"\nTop-1 Accuracy: {top1:.4f} ({top1*100:.2f}%)")

top3_correct = sum(1 for i, t in enumerate(y_true) if t in np.argsort(all_preds[i])[-3:])
top3 = top3_correct / len(y_true)
print(f"Top-3 Accuracy: {top3:.4f} ({top3*100:.2f}%)")

# Class-wise accuracy
print("\nClass-wise accuracy:")
for idx, name in enumerate(class_names):
    mask = y_true == idx
    if mask.sum() > 0:
        acc = (y_pred[mask] == idx).mean()
        print(f"  {idx:2d}. {name:20s}: {acc:.4f} ({mask.sum()} samples)")