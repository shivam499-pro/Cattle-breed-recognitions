"""
Final Optimized Cattle Breed Classification Model Training

Architecture: EfficientNetB0 with custom classification head
- Phase 1: Warmup training with frozen base (15 epochs, lr=1e-3)
- Phase 2: Fine-tuning with unfrozen last 80 layers (40 epochs, lr=1e-5)
"""

import os
import json
import time
import warnings
import numpy as np
from datetime import datetime

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

# ============== Configuration ==============
class Config:
    # Paths
    TRAIN_DIR = "data/train_final_v2"
    VAL_DIR = "data/val_final_v2"
    TEST_DIR = "data/test_final_v2"
    MAPPING_FILE = "models/breed_mapping_final_v2.json"
    MODEL_SAVE_DIR = "models/saved"
    OUTPUT_DIR = "results"
    
    # Model
    IMG_SIZE = (224, 224)
    INPUT_SHAPE = (224, 224, 3)
    NUM_CLASSES = 25
    
    # Training Phase 1 (Warmup)
    PHASE1_EPOCHS = 15
    PHASE1_LR = 1e-3
    
    # Training Phase 2 (Fine-tuning)
    PHASE2_EPOCHS = 40
    PHASE2_LR = 1e-5
    UNFREEZE_LAYERS = 80
    
    # Data Augmentation
    AUGMENTATION = {
        'random_flip': True,
        'random_rotation': 0.2,
        'random_zoom': 0.2,
        'random_contrast': 0.2,
        'random_brightness': 0.2
    }
    
    # Callbacks
    EARLY_STOPPING_PATIENCE = 8
    REDUCE_LR_PATIENCE = 4
    REDUCE_LR_FACTOR = 0.3
    
    # Other
    BATCH_SIZE = 32
    SEED = 42
    VERBOSE = 1


config = Config()

# Create directories
os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

# Set random seeds
tf.random.set_seed(config.SEED)
np.random.seed(config.SEED)


def load_breed_mapping():
    """Load breed class mapping from JSON file."""
    with open(config.MAPPING_FILE, 'r') as f:
        mapping = json.load(f)
    return mapping['classes']


def create_data_augmentation():
    """Create data augmentation layers."""
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2),
    ], name="data_augmentation")
    return data_augmentation


def create_model():
    """Create EfficientNetB0 model with custom classification head."""
    # Load base model
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=config.INPUT_SHAPE
    )
    
    # Build classification head
    inputs = keras.Input(shape=config.INPUT_SHAPE)
    
    # Data augmentation (only during training)
    x = create_data_augmentation()(inputs, training=True)
    
    # Pass through base model
    x = base_model(x, training=True)
    
    # Classification head
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dense(512, activation='relu', name="dense_512")(x)
    x = layers.Dropout(0.6, name="dropout")(x)
    outputs = layers.Dense(config.NUM_CLASSES, activation='softmax', name="output")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="EfficientNetB0_Classifier")
    
    return model


def load_datasets():
    """Load train, validation, and test datasets."""
    # Load training data
    train_ds = image_dataset_from_directory(
        config.TRAIN_DIR,
        labels='inferred',
        label_mode='int',
        class_names=None,
        color_mode='rgb',
        batch_size=config.BATCH_SIZE,
        image_size=config.IMG_SIZE,
        shuffle=True,
        seed=config.SEED,
        validation_split=None,
        subset=None,
        interpolation='bilinear'
    )
    
    # Load validation data
    val_ds = image_dataset_from_directory(
        config.VAL_DIR,
        labels='inferred',
        label_mode='int',
        class_names=None,
        color_mode='rgb',
        batch_size=config.BATCH_SIZE,
        image_size=config.IMG_SIZE,
        shuffle=False,
        seed=config.SEED,
        validation_split=None,
        subset=None,
        interpolation='bilinear'
    )
    
    # Load test data
    test_ds = image_dataset_from_directory(
        config.TEST_DIR,
        labels='inferred',
        label_mode='int',
        class_names=None,
        color_mode='rgb',
        batch_size=config.BATCH_SIZE,
        image_size=config.IMG_SIZE,
        shuffle=False,
        seed=config.SEED,
        validation_split=None,
        subset=None,
        interpolation='bilinear'
    )
    
    # Get class names from training set
    class_names = train_ds.class_names
    print(f"Found {len(class_names)} classes: {class_names}")
    
    # Configure datasets for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)
    
    return train_ds, val_ds, test_ds, class_names


def compute_class_weights(train_ds):
    """Compute inverse frequency class weights."""
    # Get all labels from training set
    all_labels = []
    for images, labels in train_ds.unbatch():
        all_labels.append(labels.numpy())
    
    all_labels = np.array(all_labels)
    
    # Compute class weights manually using inverse frequency
    classes, counts = np.unique(all_labels, return_counts=True)
    total_samples = len(all_labels)
    
    # Inverse frequency weighting
    class_weight_dict = {}
    for cls, count in zip(classes, counts):
        # Weight = total_samples / (num_classes * count)
        weight = total_samples / (len(classes) * count)
        class_weight_dict[cls] = weight
    
    # Normalize weights to sum to number of classes
    total_weight = sum(class_weight_dict.values())
    for cls in class_weight_dict:
        class_weight_dict[cls] = class_weight_dict[cls] * len(classes) / total_weight
    
    print(f"Class weights: {class_weight_dict}")
    return class_weight_dict


def freeze_base_model(model):
    """Freeze entire base model."""
    for layer in model.layers:
        if layer.name != 'output' and layer.name != 'dropout' and \
           layer.name != 'dense_512' and layer.name != 'batch_norm' and \
           layer.name != 'global_avg_pool' and layer.name != 'data_augmentation':
            layer.trainable = False


def unfreeze_last_layers(model, num_layers=80):
    """Unfreeze the last N layers of the base model."""
    # Find base model layers
    base_model = None
    for layer in model.layers:
        if layer.name.startswith('efficientnetb0'):
            base_model = layer
            break
    
    if base_model is None:
        print("Warning: Could not find base model to unfreeze")
        return
    
    # Unfreeze last N layers
    trainable_layers = [l for l in base_model.layers if l.trainable]
    layers_to_unfreeze = trainable_layers[-num_layers:] if len(trainable_layers) >= num_layers else trainable_layers
    
    for layer in layers_to_unfreeze:
        layer.trainable = True
    
    print(f"Unfroze {len(layers_to_unfreeze)} layers")


def print_model_summary(model):
    """Print model summary."""
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    model.summary()
    print("\n")


def train_phase1(model, train_ds, val_ds, class_weight_dict, breed_mapping):
    """Phase 1: Warmup training with frozen base model."""
    print("\n" + "="*60)
    print("PHASE 1: WARMUP TRAINING (Frozen Base)")
    print("="*60)
    
    # Freeze base model
    freeze_base_model(model)
    print_model_summary(model)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.PHASE1_LR),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(config.MODEL_SAVE_DIR, 'phase1_best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        CSVLogger(os.path.join(config.OUTPUT_DIR, 'phase1_training_log.csv'))
    ]
    
    # Train
    print(f"\nTraining Phase 1: {config.PHASE1_EPOCHS} epochs, lr={config.PHASE1_LR}")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.PHASE1_EPOCHS,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=config.VERBOSE
    )
    
    # Results
    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    best_val_acc = max(history.history['val_accuracy'])
    
    print(f"\nPhase 1 Results:")
    print(f"  Training Accuracy: {train_acc:.4f}")
    print(f"  Validation Accuracy: {val_acc:.4f}")
    print(f"  Best Validation Accuracy: {best_val_acc:.4f}")
    
    return history, best_val_acc


def train_phase2(model, train_ds, val_ds, class_weight_dict, breed_mapping):
    """Phase 2: Fine-tuning with unfrozen layers."""
    print("\n" + "="*60)
    print("PHASE 2: FINE-TUNING (Unfrozen Layers)")
    print("="*60)
    
    # Unfreeze last layers
    unfreeze_last_layers(model, config.UNFREEZE_LAYERS)
    print_model_summary(model)
    
    # Compile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.PHASE2_LR),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(config.MODEL_SAVE_DIR, 'phase2_best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        CSVLogger(os.path.join(config.OUTPUT_DIR, 'phase2_training_log.csv'))
    ]
    
    # Train
    print(f"\nTraining Phase 2: {config.PHASE2_EPOCHS} epochs, lr={config.PHASE2_LR}")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.PHASE2_EPOCHS,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=config.VERBOSE
    )
    
    # Results
    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    best_val_acc = max(history.history['val_accuracy'])
    
    print(f"\nPhase 2 Results:")
    print(f"  Training Accuracy: {train_acc:.4f}")
    print(f"  Validation Accuracy: {val_acc:.4f}")
    print(f"  Best Validation Accuracy: {best_val_acc:.4f}")
    
    return history, best_val_acc


def evaluate_model(model, test_ds, breed_mapping):
    """Evaluate model on test set."""
    print("\n" + "="*60)
    print("EVALUATION ON TEST SET")
    print("="*60)
    
    # Get predictions
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
    
    # Top-1 Accuracy
    top1_acc = accuracy_score(y_true, y_pred)
    print(f"\nTop-1 Accuracy: {top1_acc:.4f}")
    
    # Top-3 Accuracy
    top3_preds = np.argsort(y_pred_proba, axis=1)[:, -3:]
    top3_acc = np.mean([y_true[i] in top3_preds[i] for i in range(len(y_true))])
    print(f"Top-3 Accuracy: {top3_acc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix Shape: {cm.shape}")
    
    # Classification Report
    class_names = [breed_mapping[str(i)] for i in range(len(breed_mapping))]
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(f"\nClassification Report:\n{report}")
    
    # Per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    return {
        'top1_accuracy': top1_acc,
        'top3_accuracy': top3_acc,
        'confusion_matrix': cm,
        'classification_report': report,
        'per_class_accuracy': per_class_acc,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }


def generate_report(results, train_history, phase1_best_val_acc, phase2_best_val_acc, 
                  training_duration, breed_mapping):
    """Generate final training report."""
    print("\n" + "="*60)
    print("FINAL TRAINING REPORT")
    print("="*60)
    
    # Get class names
    class_names = [breed_mapping[str(i)] for i in range(len(breed_mapping))]
    
    # Get final accuracies from training history
    final_train_acc = train_history.history['accuracy'][-1]
    final_val_acc = train_history.history['val_accuracy'][-1]
    
    # Per-class accuracy
    per_class_acc = results['per_class_accuracy']
    class_performance = list(zip(class_names, per_class_acc))
    class_performance.sort(key=lambda x: x[1], reverse=True)
    
    # Top 5 best classes
    top5_best = class_performance[:5]
    
    # Top 5 worst classes
    top5_worst = class_performance[-5:][::-1]
    
    # Most confused class pairs
    cm = results['confusion_matrix']
    confused_pairs = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm[i, j] > 0:
                confused_pairs.append((class_names[i], class_names[j], cm[i, j]))
    confused_pairs.sort(key=lambda x: x[2], reverse=True)
    most_confused = confused_pairs[:5]
    
    # Print report
    print(f"\n1. Final Train Accuracy: {final_train_acc:.4f}")
    print(f"2. Final Validation Accuracy: {final_val_acc:.4f}")
    print(f"3. Test Accuracy (Top-1): {results['top1_accuracy']:.4f}")
    print(f"4. Test Accuracy (Top-3): {results['top3_accuracy']:.4f}")
    print(f"5. Top 5 Best Classes:")
    for name, acc in top5_best:
        print(f"   - {name}: {acc:.4f}")
    print(f"6. Top 5 Worst Classes:")
    for name, acc in top5_worst:
        print(f"   - {name}: {acc:.4f}")
    print(f"7. Most Confused Class Pairs:")
    for true, pred, count in most_confused:
        print(f"   - {true} -> {pred}: {count}")
    print(f"8. Training Duration: {training_duration:.2f} seconds ({training_duration/60:.2f} minutes)")
    
    # Save report to file
    report = {
        'final_train_accuracy': float(final_train_acc),
        'final_validation_accuracy': float(final_val_acc),
        'test_accuracy_top1': float(results['top1_accuracy']),
        'test_accuracy_top3': float(results['top3_accuracy']),
        'phase1_best_val_accuracy': float(phase1_best_val_acc),
        'phase2_best_val_accuracy': float(phase2_best_val_acc),
        'top5_best_classes': [{'name': name, 'accuracy': float(acc)} for name, acc in top5_best],
        'top5_worst_classes': [{'name': name, 'accuracy': float(acc)} for name, acc in top5_worst],
        'most_confused_pairs': [{'true': true, 'pred': pred, 'count': int(count)} for true, pred, count in most_confused],
        'training_duration_seconds': float(training_duration)
    }
    
    report_file = os.path.join(config.OUTPUT_DIR, 'final_training_report.json')
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReport saved to: {report_file}")
    
    return report


def main():
    """Main training function."""
    start_time = time.time()
    
    print("="*60)
    print("CATTLE BREED CLASSIFICATION - FINAL OPTIMIZED MODEL")
    print("="*60)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load breed mapping
    breed_mapping = load_breed_mapping()
    print(f"Number of classes: {len(breed_mapping)}")
    
    # Load datasets
    print("\nLoading datasets...")
    train_ds, val_ds, test_ds, class_names = load_datasets()
    
    # Print dataset info
    train_count = sum(1 for _ in train_ds.unbatch())
    val_count = sum(1 for _ in val_ds.unbatch())
    test_count = sum(1 for _ in test_ds.unbatch())
    print(f"Train samples: {train_count}")
    print(f"Validation samples: {val_count}")
    print(f"Test samples: {test_count}")
    
    # Compute class weights
    print("\nComputing class weights...")
    class_weight_dict = compute_class_weights(train_ds)
    
    # Create model
    print("\nCreating model...")
    model = create_model()
    
    # Phase 1: Warmup training
    phase1_history, phase1_best_val_acc = train_phase1(
        model, train_ds, val_ds, class_weight_dict, breed_mapping
    )
    
    # Phase 2: Fine-tuning
    phase2_history, phase2_best_val_acc = train_phase2(
        model, train_ds, val_ds, class_weight_dict, breed_mapping
    )
    
    # Evaluate on test set
    results = evaluate_model(model, test_ds, breed_mapping)
    
    # Calculate training duration
    training_duration = time.time() - start_time
    
    # Generate final report
    report = generate_report(
        results, phase2_history, 
        phase1_best_val_acc, phase2_best_val_acc,
        training_duration, breed_mapping
    )
    
    # Save final model
    model.save(os.path.join(config.MODEL_SAVE_DIR, 'final_model.keras'))
    
    print(f"\n" + "="*60)
    print(f"Training completed in {training_duration:.2f} seconds ({training_duration/60:.2f} minutes)")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)


if __name__ == "__main__":
    main()