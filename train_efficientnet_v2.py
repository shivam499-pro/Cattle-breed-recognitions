"""
EfficientNetB0 41-Class Cattle Breed Classifier - V2
===================================================
Improved transfer learning with proper two-phase training.
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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ================== CONFIG ==================
TRAIN_DIR = 'data/train_41'
VAL_DIR = 'data/val_41'
TEST_DIR = 'data/test_41'
MAPPING_FILE = 'models/breed_mapping_v2.json'

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 41
PATIENCE = 5

MODEL_OUTPUT = 'models/breed_classifier_41_v2.keras'
HISTORY_OUTPUT = 'models/training_history_41_v2.json'

# Phase 1: Train only top layers (frozen base)
EPOCHS_PHASE1 = 10
LR_PHASE1 = 0.001

# Phase 2: Fine-tuning
EPOCHS_PHASE2 = 20
LR_PHASE2 = 1e-5
UNFREEZE_LAYERS = 40  # Last 40 layers to unfreeze

# ================== CUSTOM METRICS ==================
def top_k_accuracy(k=3):
    """Create top-k accuracy metric."""
    def metric(y_true, y_pred):
        return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=k)
    metric.__name__ = f'top_{k}_accuracy'
    return metric

CUSTOM_OBJECTS = {
    'top_3_accuracy': top_k_accuracy(3),
    'top_k_accuracy': top_k_accuracy,
}

# ================== LOAD BREED MAPPING ==================
def load_breed_mapping():
    """Load breed mapping from JSON file."""
    with open(MAPPING_FILE, 'r') as f:
        mapping = json.load(f)
    class_names = [mapping[str(i)] for i in range(NUM_CLASSES)]
    return class_names

# ================== DATA AUGMENTATION ==================
def get_data_augmentation():
    """Create data augmentation layers."""
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.15),
        layers.RandomBrightness(0.1),
        layers.RandomContrast(0.1),
    ], name="data_augmentation")

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

# ================== COMPUTE CLASS WEIGHTS ==================
def compute_class_weights(train_ds, class_names):
    """Compute class weights based on inverse frequency."""
    print("\nComputing class weights...")
    
    all_labels = []
    for _, labels in train_ds:
        all_labels.extend(np.argmax(labels.numpy(), axis=1))
    
    class_counts = Counter(all_labels)
    total = len(all_labels)
    
    class_weights = {}
    for class_idx in range(NUM_CLASSES):
        count = class_counts.get(class_idx, 1)
        weight = np.sqrt(total / (NUM_CLASSES * count))
        class_weights[class_idx] = weight
    
    mean_weight = np.mean(list(class_weights.values()))
    class_weights = {k: v / mean_weight for k, v in class_weights.items()}
    
    print(f"Class balance ratio: {max(class_counts.values())/min(class_counts.values()):.2f}:1")
    
    return class_weights

# ================== BUILD MODEL ==================
def build_model(num_classes):
    """Build EfficientNetB0 model with custom classification head."""
    print("\nBuilding EfficientNetB0 model...")
    
    # Load with pretrained ImageNet weights
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMAGE_SIZE, 3)
    )
    
    # Freeze ALL base layers initially
    base_model.trainable = False
    
    # Build model with custom classification head
    inputs = keras.Input(shape=(*IMAGE_SIZE, 3))
    
    # Data augmentation
    x = get_data_augmentation()(inputs)
    
    # Preprocess for EfficientNetB0
    x = layers.Rescaling(1./255.0)(x)
    
    # Base model
    x = base_model(x, training=False)
    
    # Custom classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs, name="cattle_breed_efficientnet_v2")
    
    print(f"Total parameters: {model.count_params():,}")
    print(f"Trainable parameters: {sum([tf.reduce_prod(v.shape).numpy() for v in model.trainable_variables]):,}")
    
    return model, base_model

# ================== TRAIN MODEL ==================
def train_model():
    """Train the model with two phases."""
    # Load breed mapping
    class_names = load_breed_mapping()
    print(f"\nClass order from {MAPPING_FILE}: {len(class_names)} classes")
    
    # Load datasets
    print(f"\nLoading datasets...")
    train_ds = load_data_from_directory(TRAIN_DIR, class_names, BATCH_SIZE, shuffle=True)
    val_ds = load_data_from_directory(VAL_DIR, class_names, BATCH_SIZE, shuffle=False)
    test_ds = load_data_from_directory(TEST_DIR, class_names, BATCH_SIZE, shuffle=False)
    
    print(f"Train: {len(train_ds.file_paths)} files")
    print(f"Val: {len(val_ds.file_paths)} files")
    print(f"Test: {len(test_ds.file_paths)} files")
    
    # Compute class weights
    class_weights = compute_class_weights(train_ds, class_names)
    
    # Build model
    model, base_model = build_model(NUM_CLASSES)
    
    # ================== PHASE 1: Train only top layers ==================
    print("\n" + "="*60)
    print("PHASE 1: Training classification head (frozen base)")
    print("="*60)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR_PHASE1),
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
    
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_PHASE1,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    phase1_acc = history1.history['val_accuracy'][-1]
    print(f"\nPhase 1 validation accuracy: {phase1_acc:.4f}")
    
    # ================== PHASE 2: Fine-tuning ==================
    print("\n" + "="*60)
    print("PHASE 2: Fine-tuning (unfreezing top layers)")
    print("="*60)
    
    # Unfreeze last layers
    base_model.trainable = True
    for i, layer in enumerate(base_model.layers):
        if i >= len(base_model.layers) - UNFREEZE_LAYERS:
            layer.trainable = True
        else:
            layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR_PHASE2),
        loss='categorical_crossentropy',
        metrics=['accuracy', top_k_accuracy(3)]
    )
    
    trainable_count = sum([tf.reduce_prod(v.shape).numpy() for v in model.trainable_variables])
    print(f"Trainable parameters after unfreeze: {trainable_count:,}")
    
    # Continue training
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_PHASE2,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    phase2_acc = history2.history['val_accuracy'][-1] if history2.history['val_accuracy'] else phase1_acc
    print(f"\nPhase 2 validation accuracy: {phase2_acc:.4f}")
    
    # Merge histories
    for key in history1.history:
        if key in history2.history:
            history1.history[key].extend(history2.history[key])
    
    # Save final model
    model.save(MODEL_OUTPUT, custom_objects=CUSTOM_OBJECTS)
    print(f"\nModel saved to: {MODEL_OUTPUT}")
    
    # Save training history
    history_dict = {k: [float(v) for v in arr] for k, arr in history1.history.items()}
    with open(HISTORY_OUTPUT, 'w') as f:
        json.dump(history_dict, f, indent=2)
    print(f"Training history saved to: {HISTORY_OUTPUT}")
    
    return model, history1, class_names, test_ds, class_weights, phase1_acc, phase2_acc

# ================== EVALUATE MODEL ==================
def evaluate_model(model, test_ds, class_names):
    """Evaluate model and generate report."""
    print("\n" + "="*60)
    print("EVALUATION ON TEST SET")
    print("="*60)
    
    y_true = []
    y_pred = []
    all_predictions = []
    
    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)
        all_predictions.extend(predictions)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(predictions, axis=1))
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    all_predictions = np.array(all_predictions)
    
    from sklearn.metrics import accuracy_score, confusion_matrix
    
    # Top-1 accuracy
    top1_accuracy = accuracy_score(y_true, y_pred)
    print(f"\nTop-1 Accuracy: {top1_accuracy:.4f} ({top1_accuracy*100:.2f}%)")
    
    # Top-3 accuracy
    top3_correct = 0
    for i, true_label in enumerate(y_true):
        top3_preds = np.argsort(all_predictions[i])[-3:]
        if true_label in top3_preds:
            top3_correct += 1
    top3_accuracy = top3_correct / len(y_true)
    print(f"Top-3 Accuracy: {top3_accuracy:.4f} ({top3_accuracy*100:.2f}%)")
    
    # Class-wise accuracy
    class_accuracies = {}
    for class_idx, class_name in enumerate(class_names):
        mask = y_true == class_idx
        if np.sum(mask) > 0:
            class_acc = np.mean(y_pred[mask] == class_idx)
            class_accuracies[class_name] = {
                'accuracy': float(class_acc),
                'samples': int(np.sum(mask))
            }
    
    sorted_classes = sorted(class_accuracies.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    print("\nBest performing breeds:")
    for breed, data in sorted_classes[:5]:
        print(f"  {breed}: {data['accuracy']:.4f} ({data['samples']} samples)")
    
    print("\nWorst performing breeds:")
    for breed, data in sorted_classes[-5:]:
        print(f"  {breed}: {data.get('accuracy', 0):.4f} ({data['samples']} samples)")
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
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
        print(f"  {true_name} --> {pred_name}: {count} times")
    
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
    print("="*60)
    print("EfficientNetB0 41-Class Cattle Breed Classifier - V2")
    print("="*60)
    
    # Train
    model, history, class_names, test_ds, class_weights, phase1_acc, phase2_acc = train_model()
    
    # Evaluate
    evaluation = evaluate_model(model, test_ds, class_names)
    
    # Generate report
    final_train_acc = history.history['accuracy'][-1] if history.history['accuracy'] else 0
    final_val_acc = history.history['val_accuracy'][-1] if history.history['val_accuracy'] else 0
    epochs_run = len(history.history['accuracy'])
    
    # Check overfitting
    overfitting = (final_train_acc - final_val_acc) > 0.15
    underfitting = final_val_acc < 0.5
    
    report = f"""
================================================================================
CATTLE BREED CLASSIFICATION MODEL - V2 TRAINING REPORT
================================================================================

1. CLASS ORDER USED
   41 classes from breed_mapping_v2.json:

"""
    for i, name in enumerate(class_names):
        report += f"   {i:2d}. {name}\n"
    
    report += f"""
2. TRAINING SUMMARY
   Phase 1 validation accuracy: {phase1_acc:.4f} ({phase1_acc*100:.2f}%)
   Phase 2 validation accuracy: {phase2_acc:.4f} ({phase2_acc*100:.2f}%)
   Total epochs run: {epochs_run}
   Final train accuracy: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)
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
        report += f"   - {true_name} --> {pred_name}: {count} times\n"
    
    report += """
6. WARNINGS
"""
    if overfitting:
        report += "   - OVERFITTING: Train accuracy significantly higher than validation\n"
    if underfitting:
        report += "   - UNDERFITTING: Validation accuracy is low (<50%)\n"
    
    report += """
================================================================================
"""
    
    print(report)
    
    # Save report
    with open('models/training_report_41_v2.txt', 'w') as f:
        f.write(report)
    print(f"Report saved to: models/training_report_41_v2.txt")

if __name__ == '__main__':
    main()