"""
Evaluate Saved EfficientNetB0 Model
====================================
Loads the saved model and generates full evaluation report.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ================== CUSTOM METRICS ==================
def top_k_accuracy(k=3):
    """Create top-k accuracy metric."""
    def metric(y_true, y_pred):
        return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=k)
    metric.__name__ = f'top_{k}_accuracy'
    return metric

# Register custom objects
CUSTOM_OBJECTS = {
    'top_3_accuracy': top_k_accuracy(3),
}

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# ================== CONFIG ==================
VAL_DIR = 'data/val_41'
TEST_DIR = 'data/test_41'
MAPPING_FILE = 'models/breed_mapping_v2.json'
MODEL_PATH = 'models/breed_classifier_41.keras'

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 41

# ================== LOAD BREED MAPPING ==================
def load_breed_mapping():
    """Load breed mapping from JSON file."""
    with open(MAPPING_FILE, 'r') as f:
        mapping = json.load(f)
    class_names = [mapping[str(i)] for i in range(NUM_CLASSES)]
    return class_names

# ================== LOAD DATA ==================
def load_data_from_directory(directory, class_names, batch_size=32, shuffle=False):
    """Load images from directory with explicit class order."""
    ds = keras.utils.image_dataset_from_directory(
        directory,
        labels='inferred',
        label_mode='categorical',
        class_names=class_names,
        batch_size=batch_size,
        image_size=IMAGE_SIZE,
        shuffle=shuffle,
        seed=42
    )
    return ds

# ================== EVALUATE MODEL ==================
def evaluate_model(model, test_ds, class_names):
    """Evaluate model and generate comprehensive report."""
    print("="*60)
    print("EVALUATION ON TEST SET")
    print("="*60)
    
    # Get predictions
    y_true = []
    y_pred = []
    
    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(predictions, axis=1))
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, confusion_matrix
    
    # Top-1 accuracy
    top1_accuracy = accuracy_score(y_true, y_pred)
    print(f"\nTop-1 Accuracy: {top1_accuracy:.4f} ({top1_accuracy*100:.2f}%)")
    
    # Top-3 accuracy
    all_predictions = []
    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)
        all_predictions.extend(predictions)
    all_predictions = np.array(all_predictions)
    
    top3_correct = 0
    for i, true_label in enumerate(y_true):
        top3_preds = np.argsort(all_predictions[i])[-3:]
        if true_label in top3_preds:
            top3_correct += 1
    top3_accuracy = top3_correct / len(y_true)
    print(f"Top-3 Accuracy: {top3_accuracy:.4f} ({top3_accuracy*100:.2f}%)")
    
    # Class-wise accuracy
    print("\n" + "="*60)
    print("CLASS-WISE ACCURACY")
    print("="*60)
    
    class_accuracies = {}
    for class_idx, class_name in enumerate(class_names):
        mask = y_true == class_idx
        if np.sum(mask) > 0:
            class_acc = np.mean(y_pred[mask] == class_idx)
            class_accuracies[class_name] = {
                'accuracy': float(class_acc),
                'samples': int(np.sum(mask))
            }
            print(f"  {class_idx:2d}. {class_name:20s}: {class_acc:.4f} ({np.sum(mask)} samples)")
    
    # Best and worst performing breeds
    sorted_classes = sorted(class_accuracies.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    print("\n" + "="*60)
    print("BEST PERFORMING BREEDS (Top 5)")
    print("="*60)
    for breed, data in sorted_classes[:5]:
        print(f"  {breed:20s}: {data['accuracy']:.4f} ({data['samples']} samples)")
    
    print("\n" + "="*60)
    print("WORST PERFORMING BREEDS (Bottom 5)")
    print("="*60)
    for breed, data in sorted_classes[-5:]:
        print(f"  {breed:20s}: {data['accuracy']:.4f} ({data['samples']} samples)")
    
    # Confusion matrix
    print("\n" + "="*60)
    print("CONFUSION MATRIX ANALYSIS")
    print("="*60)
    
    conf_matrix = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    
    # Find most confused pairs
    confusions = []
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i != j and conf_matrix[i, j] > 0:
                confusions.append((i, j, conf_matrix[i, j]))
    
    confusions.sort(key=lambda x: x[2], reverse=True)
    
    print("\nMost confused breed pairs:")
    for true_idx, pred_idx, count in confusions[:10]:
        true_name = class_names[true_idx]
        pred_name = class_names[pred_idx]
        print(f"  {true_name:20s} --> {pred_name:20s}: {count} times")
    
    return {
        'top1_accuracy': top1_accuracy,
        'top3_accuracy': top3_accuracy,
        'class_accuracies': class_accuracies,
        'best_breeds': sorted_classes[:5],
        'worst_breeds': sorted_classes[-5:],
        'confusions': confusions[:10]
    }

# ================== MAIN ==================
def main():
    print("Loading trained model...")
    
    # Load breed mapping
    class_names = load_breed_mapping()
    print(f"Loaded {len(class_names)} classes from mapping file")
    
    # Load saved model
    model = keras.models.load_model(MODEL_PATH, custom_objects=CUSTOM_OBJECTS)
    print(f"Model loaded from: {MODEL_PATH}")
    
    # Load test dataset
    test_ds = load_data_from_directory(TEST_DIR, class_names, BATCH_SIZE, shuffle=False)
    print(f"Test dataset loaded from: {TEST_DIR}")
    
    # Evaluate
    evaluation = evaluate_model(model, test_ds, class_names)
    
    # Generate text report
    print("\n" + "="*60)
    print("GENERATING FINAL REPORT")
    print("="*60)
    
    # Load history for training summary
    try:
        with open('models/training_history_41.json', 'r') as f:
            history = json.load(f)
        final_train_acc = history['accuracy'][-1] if history.get('accuracy') else 0
        final_val_acc = history['val_accuracy'][-1] if history.get('val_accuracy') else 0
        epochs_run = len(history.get('accuracy', []))
    except:
        final_train_acc = 0
        final_val_acc = 0
        epochs_run = 0
    
    # Check for overfitting
    overfitting = (final_train_acc - final_val_acc) > 0.15
    underfitting = final_val_acc < 0.5
    
    # Build report
    report = f"""
================================================================================
CATTLE BREED CLASSIFICATION MODEL - TRAINING REPORT
================================================================================

1. CLASS ORDER USED
   The following 41 classes were used in the order defined by breed_mapping_v2.json:

"""
    for i, name in enumerate(class_names):
        report += f"   {i:2d}. {name}\n"
    
    report += f"""
2. TRAINING SUMMARY
   Epochs run: {epochs_run}
   Final training accuracy: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)
   Final validation accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)
   Model saved to: {MODEL_PATH}

3. TEST PERFORMANCE
   Top-1 Accuracy: {evaluation['top1_accuracy']:.4f} ({evaluation['top1_accuracy']*100:.2f}%)
   Top-3 Accuracy: {evaluation['top3_accuracy']:.4f} ({evaluation['top3_accuracy']*100:.2f}%)

4. CLASS-WISE ACCURACY
   Best performing breeds:
"""
    for breed, data in evaluation['best_breeds']:
        report += f"   - {breed}: {data['accuracy']:.4f} ({data['samples']} samples)\n"
    
    report += "\n   Worst performing breeds:\n"
    for breed, data in evaluation['worst_breeds']:
        report += f"   - {breed}: {data['accuracy']:.4f} ({data['samples']} samples)\n"
    
    report += """
5. CONFUSION INSIGHTS
   Most confused breed pairs:
"""
    for true_idx, pred_idx, count in evaluation['confusions']:
        true_name = class_names[true_idx]
        pred_name = class_names[pred_idx]
        report += f"   - {true_name} --> {pred_name}: {count} times\n"
    
    report += """
6. WARNINGS
"""
    if overfitting:
        report += "   - OVERFITTING DETECTED: Train accuracy significantly higher than validation\n"
    if underfitting:
        report += "   - UNDERFITTING DETECTED: Validation accuracy is low (<50%)\n"
    
    report += """
================================================================================
"""
    
    print(report)
    
    # Save report
    report_path = 'models/training_report_41.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")

if __name__ == '__main__':
    main()