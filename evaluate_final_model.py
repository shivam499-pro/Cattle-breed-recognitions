"""
Quick evaluation script to generate final report from saved model
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Configuration
TEST_DIR = "data/test_final_v2"
MAPPING_FILE = "models/breed_mapping_final_v2.json"
MODEL_PATH = "models/saved/phase2_best_model.keras"
OUTPUT_DIR = "results"

# Load breed mapping
with open(MAPPING_FILE, 'r') as f:
    mapping = json.load(f)
breed_mapping = mapping['classes']

# Load model
print("Loading model...")
model = keras.models.load_model(MODEL_PATH)

# Load test dataset
print("Loading test dataset...")
test_ds = image_dataset_from_directory(
    TEST_DIR,
    labels='inferred',
    label_mode='int',
    color_mode='rgb',
    batch_size=32,
    image_size=(224, 224),
    shuffle=False
)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

# Get predictions
print("Evaluating...")
y_true = []
y_pred = []
y_pred_proba = []

for images, labels in test_ds.unbatch():
    y_true.append(labels.numpy())
    pred = model.predict(images[tf.newaxis, ...], verbose=0)
    y_pred.append(np.argmax(pred))
    y_pred_proba.append(pred[0])

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_pred_proba = np.array(y_pred_proba)

# Calculate metrics
top1_acc = accuracy_score(y_true, y_pred)
top3_preds = np.argsort(y_pred_proba, axis=1)[:, -3:]
top3_acc = np.mean([y_true[i] in top3_preds[i] for i in range(len(y_true))])

class_names = [breed_mapping[str(i)] for i in range(len(breed_mapping))]
cm = confusion_matrix(y_true, y_pred)
per_class_acc = cm.diagonal() / cm.sum(axis=1)

# Generate report
print("\n" + "="*60)
print("FINAL EVALUATION REPORT")
print("="*60)

print(f"\nTest Accuracy (Top-1): {top1_acc:.4f}")
print(f"Test Accuracy (Top-3): {top3_acc:.4f}")

# Classification report
report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print(f"\nClassification Report:\n{report}")

# Per-class performance
class_performance = list(zip(class_names, per_class_acc))
class_performance.sort(key=lambda x: x[1], reverse=True)

print("\nTop 5 Best Classes:")
for name, acc in class_performance[:5]:
    print(f"  - {name}: {acc:.4f}")

print("\nTop 5 Worst Classes:")
for name, acc in class_performance[-5:][::-1]:
    print(f"  - {name}: {acc:.4f}")

# Most confused pairs
confused_pairs = []
for i in range(len(cm)):
    for j in range(len(cm)):
        if i != j and cm[i, j] > 0:
            confused_pairs.append((class_names[i], class_names[j], cm[i, j]))
confused_pairs.sort(key=lambda x: x[2], reverse=True)

print("\nMost Confused Class Pairs:")
for true, pred, count in confused_pairs[:5]:
    print(f"  - {true} -> {pred}: {count}")

# Save final report
report_data = {
    'test_accuracy_top1': float(top1_acc),
    'test_accuracy_top3': float(top3_acc),
    'top5_best_classes': [{'name': name, 'accuracy': float(acc)} for name, acc in class_performance[:5]],
    'top5_worst_classes': [{'name': name, 'accuracy': float(acc)} for name, acc in class_performance[-5:][::-1]],
    'most_confused_pairs': [{'true': true, 'pred': pred, 'count': int(count)} for true, pred, count in confused_pairs[:5]]
}

report_file = os.path.join(OUTPUT_DIR, 'final_training_report.json')
with open(report_file, 'w') as f:
    json.dump(report_data, f, indent=2)

print(f"\nReport saved to: {report_file}")