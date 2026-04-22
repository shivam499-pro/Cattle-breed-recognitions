"""
Evaluate V3 model on test set
"""
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Configuration
TEST_DIR = "data/test_final_v3"
MAPPING_FILE = "models/breed_mapping_final_v3.json"
MODEL_PATH = "models/saved/v3_phase2_best.keras"

print("Loading...")
model = keras.models.load_model(MODEL_PATH, safe_mode=False)

# Load mapping
with open(MAPPING_FILE, 'r') as f:
    mapping = json.load(f)
breed_mapping = mapping['classes']

# Load test
test_ds = image_dataset_from_directory(
    TEST_DIR,
    labels='inferred',
    label_mode='int',
    color_mode='rgb',
    batch_size=32,
    image_size=(224, 224),
    shuffle=False
).prefetch(tf.data.AUTOTUNE)

# Evaluate
y_true, y_pred, y_pred_proba = [], [], []
for images, labels in test_ds.unbatch():
    y_true.append(labels.numpy())
    pred = model.predict(images[tf.newaxis, ...], verbose=0)
    y_pred.append(np.argmax(pred))
    y_pred_proba.append(pred[0])

y_true, y_pred, y_pred_proba = np.array(y_true), np.array(y_pred), np.array(y_pred_proba)

# Metrics
top1 = accuracy_score(y_true, y_pred)
top3 = np.mean([y_true[i] in np.argsort(y_pred_proba, axis=1)[i, -3:] for i in range(len(y_true))])

print(f"\n=== V3 Phase 1 Results ===")
print(f"Top-1: {top1:.4f}")
print(f"Top-3: {top3:.4f}")

# Per-class
class_names = [breed_mapping[str(i)] for i in range(len(breed_mapping))]
cm = confusion_matrix(y_true, y_pred)
per_class = list(zip(class_names, cm.diagonal() / cm.sum(axis=1)))
per_class.sort(key=lambda x: x[1], reverse=True)

print("\nTop 5 Best:")
for n, a in per_class[:5]:
    print(f"  {n}: {a:.4f}")
print("\nTop 5 Worst:")
for n, a in per_class[-5:]:
    print(f"  {n}: {a:.4f}")

# Confusion pairs
confused = []
for i in range(len(cm)):
    for j in range(len(cm)):
        if i != j and cm[i, j] > 0:
            confused.append((class_names[i], class_names[j], cm[i, j]))
confused.sort(key=lambda x: x[2], reverse=True)

print("\nTop 10 Confused:")
for t, p, c in confused[:10]:
    print(f"  {t} -> {p}: {c}")

# Save
report = {
    'test_accuracy_top1': float(top1),
    'test_accuracy_top3': float(top3),
    'top5_best': [{'name': n, 'accuracy': float(a)} for n, a in per_class[:5]],
    'top5_worst': [{'name': n, 'accuracy': float(a)} for n, a in per_class[-5:]],
    'top10_confused': [{'true': t, 'pred': p, 'count': c} for t, p, c in confused[:10]]
}
with open('results/v3_phase1_evaluation.json', 'w') as f:
    json.dump(report, f, indent=2)

print("\nSaved!")