"""
EfficientNetB0 41-Class Cattle Breed Classifier Training
================================================
Trains an EfficientNetB0-based classifier for 41 cattle breed recognition.
Features:
- Directory-based loading matching breed_mapping_v2.json order
- Data augmentation (flip, rotation, zoom, brightness/contrast)
- Class weights for imbalanced dataset handling
- Top-1 and Top-3 accuracy metrics
- Comprehensive evaluation report
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ================== CONFIG ==================
TRAIN_DIR = 'data/train_41'
VAL_DIR = 'data/val_41'
TEST_DIR = 'data/test_41'
MAPPING_FILE = 'models/breed_mapping_v2.json'

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 41
EPOCHS = 20  # 15-25 range
LEARNING_RATE = 0.001
PATIENCE = 5  # Early stopping patience

MODEL_OUTPUT = 'models/breed_classifier_41.keras'
HISTORY_OUTPUT = 'models/training_history_41.json'

# ================== LOAD BREED MAPPING ==================
def load_breed_mapping():
    """Load breed mapping from JSON file to ensure consistent class order."""
    with open(MAPPING_FILE, 'r') as f:
        mapping = json.load(f)
    
    # Extract class names in order
    class_names = [mapping[str(i)] for i in range(NUM_CLASSES)]
    return class_names

# ================== DATA AUGMENTATION ==================
def get_data_augmentation():
    """Create data augmentation layers for training."""
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomBrightness(0.15),
        layers.RandomContrast(0.15),
    ], name="data_augmentation")

# ================== LOAD DATA FROM DIRECTORY ==================
def load_data_from_directory(directory, class_names, batch_size=32, shuffle=False):
    """Load images from directory with explicit class order matching breed_mapping_v2.json."""
    
    # Create image dataset from directory
    ds = keras.utils.image_dataset_from_directory(
        directory,
        labels='inferred',
        label_mode='categorical',
        class_names=class_names,  # Force order from mapping
        batch_size=batch_size,
        image_size=IMAGE_SIZE,
        shuffle=shuffle,
        seed=42
    )
    
    return ds

# ================== COMPUTE CLASS WEIGHTS ==================
def compute_class_weights(train_ds, class_names):
    """Compute class weights based on inverse frequency."""
    print("\nComputing class weights...")
    
    # Collect all labels
    all_labels = []
    for _, labels in train_ds:
        all_labels.extend(np.argmax(labels.numpy(), axis=1))
    
    # Count occurrences
    class_counts = Counter(all_labels)
    total = len(all_labels)
    
    # Compute weights using inverse frequency
    class_weights = {}
    for class_idx in range(NUM_CLASSES):
        count = class_counts.get(class_idx, 1)
        # Use inverse square root for balancing
        weight = np.sqrt(total / (NUM_CLASSES * count))
        class_weights[class_idx] = weight
    
    # Normalize weights
    mean_weight = np.mean(list(class_weights.values()))
    class_weights = {k: v / mean_weight for k, v in class_weights.items()}
    
    # Print class distribution
    print("\nTraining class distribution:")
    for class_idx, class_name in enumerate(class_names):
        count = class_counts.get(class_idx, 0)
        weight = class_weights[class_idx]
        print(f"  {class_idx:2d}. {class_name:20s}: {count:4d} samples, weight: {weight:.3f}")
    
    min_count = min(class_counts.values())
    max_count = max(class_counts.values())
    print(f"\nClass balance ratio: {max_count/min_count:.2f}:1")
    
    return class_weights

# ================== BUILD MODEL ==================
def build_model(num_classes):
    """Build EfficientNetB0 model with custom classification head."""
    print("\nBuilding EfficientNetB0 model...")
    
    # Load EfficientNetB0 without top
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMAGE_SIZE, 3)
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Build model
    inputs = keras.Input(shape=(*IMAGE_SIZE, 3))
    
    # Data augmentation (training only)
    x = get_data_augmentation()(inputs) if hasattr(get_data_augmentation(), '__call__') else inputs
    
    # Preprocess for EfficientNetB0
    x = layers.Rescaling(1./255.0)(x)
    x = layers.Normalization(
        mean=(0.485, 0.456, 0.406),
        variance=(0.229, 0.224, 0.225)
    )(x)
    
    # Base model
    x = base_model(x, training=False)
    
    # Custom classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs, name="cattle_breed_efficientnet")
    
    print(f"Total parameters: {model.count_params():,}")
    print(f"Trainable parameters: {sum([tf.reduce_prod(v.shape).numpy() for v in model.trainable_variables]):,}")
    
    return model, base_model

# ================== CUSTOM METRICS ==================
def top_k_accuracy(k=3):
    """Create top-k accuracy metric."""
    def metric(y_true, y_pred):
        return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=k)
    metric.__name__ = f'top_{k}_accuracy'
    return metric

# ================== TRAIN MODEL ==================
def train_model():
    """Train the model with all configurations."""
    # Load breed mapping
    class_names = load_breed_mapping()
    print(f"\nClass order from {MAPPING_FILE}:")
    for i, name in enumerate(class_names):
        print(f"  {i:2d}. {name}")
    
    # Load datasets
    print(f"\nLoading datasets...")
    print(f"  Training: {TRAIN_DIR}")
    train_ds = load_data_from_directory(TRAIN_DIR, class_names, BATCH_SIZE, shuffle=True)
    
    print(f"  Validation: {VAL_DIR}")
    val_ds = load_data_from_directory(VAL_DIR, class_names, BATCH_SIZE, shuffle=False)
    
    print(f"  Test: {TEST_DIR}")
    test_ds = load_data_from_directory(TEST_DIR, class_names, BATCH_SIZE, shuffle=False)
    
    # Compute class weights
    class_weights = compute_class_weights(train_ds, class_names)
    
    # Build model
    model, base_model = build_model(NUM_CLASSES)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', top_k_accuracy(3)]
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            MODEL_OUTPUT,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Training
    print("\n" + "="*60)
    print("PHASE 1: Training classification head (frozen base)")
    print("="*60)
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # Fine-tuning phase
    print("\n" + "="*60)
    print("PHASE 2: Fine-tuning (unfreezing top layers)")
    print("="*60)
    
    # Unfreeze top layers of base model
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE * 0.1),
        loss='categorical_crossentropy',
        metrics=['accuracy', top_k_accuracy(3)]
    )
    
    # Continue training
    history_finetune = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,  # Additional epochs for fine-tuning
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # Merge histories
    for key in history.history:
        history.history[key].extend(history_finetune.history[key])
    
    # Save final model
    model.save(MODEL_OUTPUT)
    print(f"\nModel saved to: {MODEL_OUTPUT}")
    
    # Save training history
    history_dict = {k: [float(v) for v in arr] for k, arr in history.history.items()}
    with open(HISTORY_OUTPUT, 'w') as f:
        json.dump(history_dict, f, indent=2)
    print(f"Training history saved to: {HISTORY_OUTPUT}")
    
    return model, history, class_names, test_ds, class_weights

# ================== EVALUATE MODEL ==================
def evaluate_model(model, test_ds, class_names):
    """Evaluate model and generate comprehensive report."""
    print("\n" + "="*60)
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
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    
    # Top-1 accuracy
    top1_accuracy = accuracy_score(y_true, y_pred)
    print(f"\nTop-1 Accuracy: {top1_accuracy:.4f} ({top1_accuracy*100:.2f}%)")
    
    # Top-3 accuracy
    # Get all predictions
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
    
    # Sort by confusion count
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

# ================== GENERATE FINAL REPORT ==================
def generate_report(history, evaluation, class_names):
    """Generate comprehensive training and evaluation report."""
    
    # Extract training metrics
    final_train_acc = history.history['accuracy'][-1] if history.history['accuracy'] else 0
    final_val_acc = history.history['val_accuracy'][-1] if history.history['val_accuracy'] else 0
    epochs_run = len(history.history['accuracy'])
    
    # Check for overfitting
    overfitting = (final_train_acc - final_val_acc) > 0.15
    underfitting = final_val_acc < 0.5
    
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
   Model saved to: {MODEL_OUTPUT}

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
        report += f"   - {true_name} → {pred_name}: {count} times\n"
    
    report += """
6. WARNINGS
"""
    if overfitting:
        report += "   ⚠️ OVERFITTING DETECTED: Train accuracy significantly higher than validation\n"
    if underfitting:
        report += "   ⚠️ UNDERFITTING DETECTED: Validation accuracy is low (<50%)\n"
    
    report += """
================================================================================
"""
    
    print(report)
    
    # Save report
    report_path = 'models/training_report_41.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")
    
    return report

# ================== MAIN ==================
def main():
    print("="*60)
    print("EfficientNetB0 41-Class Cattle Breed Classifier")
    print("="*60)
    print(f"Training data: {TRAIN_DIR}")
    print(f"Validation data: {VAL_DIR}")
    print(f"Test data: {TEST_DIR}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Model output: {MODEL_OUTPUT}")
    
    # Train
    model, history, class_names, test_ds, class_weights = train_model()
    
    # Evaluate
    evaluation = evaluate_model(model, test_ds, class_names)
    
    # Generate report
    generate_report(history, evaluation, class_names)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)

if __name__ == '__main__':
    main()