"""
Final V3 Optimized Training Script (22 classes)

Training Improvements:
- Phase 1: 20 epochs (warmup)
- Phase 2: 40-60 epochs (fine-tuning)
- Improved data augmentation
- Label smoothing
- Class weights
- LR scheduling
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
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

# ============== Configuration ==============
class Config:
    # Paths - V3 DATASET
    TRAIN_DIR = "data/train_final_v3"
    VAL_DIR = "data/val_final_v3"
    TEST_DIR = "data/test_final_v3"
    MAPPING_FILE = "models/breed_mapping_final_v3.json"
    MODEL_SAVE_DIR = "models/saved"
    OUTPUT_DIR = "results"
    
    # Model
    IMG_SIZE = (224, 224)
    INPUT_SHAPE = (224, 224, 3)
    NUM_CLASSES = 22
    
    # Training Phase 1 (Warmup) - INCREASED
    PHASE1_EPOCHS = 20
    PHASE1_LR = 1e-3
    
    # Training Phase 2 (Fine-tuning) - INCREASED
    PHASE2_EPOCHS = 50  # 40-60 range
    PHASE2_LR = 1e-5
    UNFREEZE_LAYERS = 80
    
    # Data Augmentation - IMPROVED
    AUGMENTATION = {
        'rotation_range': 25,  # degrees
        'zoom_range': 0.2,
        'width_shift': 0.1,
        'height_shift': 0.1,
        'horizontal_flip': True,
    }
    
    # Callbacks - IMPROVED
    EARLY_STOPPING_PATIENCE = 10  # increased
    REDUCE_LR_PATIENCE = 3  # more responsive
    REDUCE_LR_FACTOR = 0.3
    
    # Label smoothing
    LABEL_SMOOTHING = 0.1
    
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
    """Create improved data augmentation layers."""
    data_augmentation = keras.Sequential([
        layers.RandomRotation(config.AUGMENTATION['rotation_range'] / 180 * np.pi),
        layers.RandomZoom(config.AUGMENTATION['zoom_range']),
        layers.RandomTranslation(
            config.AUGMENTATION['width_shift'],
            config.AUGMENTATION['height_shift']
        ),
        layers.RandomFlip("horizontal" if config.AUGMENTATION['horizontal_flip'] else None),
        layers.RandomContrast(0.15),
        layers.RandomBrightness(0.15),
    ], name="data_augmentation")
    return data_augmentation


def create_model():
    """Create EfficientNetB0 model with custom classification head."""
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=config.INPUT_SHAPE
    )
    
    inputs = keras.Input(shape=config.INPUT_SHAPE)
    x = create_data_augmentation()(inputs, training=True)
    x = base_model(x, training=True)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dense(512, activation='relu', name="dense_512")(x)
    x = layers.Dropout(0.5, name="dropout")(x)  # reduced from 0.6
    outputs = layers.Dense(config.NUM_CLASSES, activation='softmax', name="output")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="EfficientNetB0_Classifier")
    return model


def spares_categorical_crossentropy_with_smoothing(y_true, y_pred, label_smoothing=0.1):
    """Sparse categorical crossentropy with label smoothing."""
    # Convert sparse labels to one-hot
    num_classes = tf.shape(y_pred)[-1]
    y_true = tf.cast(y_true, tf.int32)
    y_true = tf.one_hot(y_true, num_classes)
    
    # Apply label smoothing
    if label_smoothing > 0:
        y_true = y_true * (1 - label_smoothing) + label_smoothing / tf.cast(num_classes, tf.float32)
    
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred)


def load_datasets():
    """Load train, validation, and test datasets."""
    train_ds = image_dataset_from_directory(
        config.TRAIN_DIR,
        labels='inferred',
        label_mode='int',
        class_names=None,
        color_mode='rgb',
        batch_size=config.BATCH_SIZE,
        image_size=config.IMG_SIZE,
        shuffle=True,
        seed=config.SEED
    )
    
    val_ds = image_dataset_from_directory(
        config.VAL_DIR,
        labels='inferred',
        label_mode='int',
        class_names=None,
        color_mode='rgb',
        batch_size=config.BATCH_SIZE,
        image_size=config.IMG_SIZE,
        shuffle=False,
        seed=config.SEED
    )
    
    test_ds = image_dataset_from_directory(
        config.TEST_DIR,
        labels='inferred',
        label_mode='int',
        class_names=None,
        color_mode='rgb',
        batch_size=config.BATCH_SIZE,
        image_size=config.IMG_SIZE,
        shuffle=False,
        seed=config.SEED
    )
    
    class_names = train_ds.class_names
    print(f"Found {len(class_names)} classes: {class_names}")
    
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)
    
    return train_ds, val_ds, test_ds, class_names


def compute_class_weights(train_ds):
    """Compute inverse frequency class weights."""
    all_labels = []
    for images, labels in train_ds.unbatch():
        all_labels.append(labels.numpy())
    
    all_labels = np.array(all_labels)
    classes, counts = np.unique(all_labels, return_counts=True)
    total_samples = len(all_labels)
    
    class_weight_dict = {}
    for cls, count in zip(classes, counts):
        weight = total_samples / (len(classes) * count)
        class_weight_dict[cls] = weight
    
    # Normalize
    total_weight = sum(class_weight_dict.values())
    for cls in class_weight_dict:
        class_weight_dict[cls] = class_weight_dict[cls] * len(classes) / total_weight
    
    return class_weight_dict


def freeze_base_model(model):
    """Freeze entire base model."""
    for layer in model.layers:
        if layer.name not in ['output', 'dropout', 'dense_512', 'batch_norm', 'global_avg_pool', 'data_augmentation']:
            layer.trainable = False


def unfreeze_last_layers(model, num_layers=80):
    """Unfreeze the last N layers of the base model."""
    base_model = None
    for layer in model.layers:
        if layer.name.startswith('efficientnetb0'):
            base_model = layer
            break
    
    if base_model is None:
        print("Warning: Could not find base model")
        return
    
    trainable_layers = [l for l in base_model.layers if l.trainable]
    layers_to_unfreeze = trainable_layers[-num_layers:] if len(trainable_layers) >= num_layers else trainable_layers
    
    for layer in layers_to_unfreeze:
        layer.trainable = True
    
    print(f"Unfroze {len(layers_to_unfreeze)} layers")


def train_phase1(model, train_ds, val_ds, class_weight_dict):
    """Phase 1: Warmup training."""
    print("\n" + "="*60)
    print("PHASE 1: WARMUP (20 epochs, lr=0.001)")
    print("="*60)
    
    freeze_base_model(model)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.PHASE1_LR),
        loss=lambda y_true, y_pred: spares_categorical_crossentropy_with_smoothing(
            y_true, y_pred, config.LABEL_SMOOTHING
        ),
        metrics=['accuracy']
    )
    
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
            os.path.join(config.MODEL_SAVE_DIR, 'v3_phase1_best.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        CSVLogger(os.path.join(config.OUTPUT_DIR, 'v3_phase1_log.csv'))
    ]
    
    print(f"\nTraining Phase 1: {config.PHASE1_EPOCHS} epochs, lr={config.PHASE1_LR}")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.PHASE1_EPOCHS,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=config.VERBOSE
    )
    
    return history


def train_phase2(model, train_ds, val_ds, class_weight_dict):
    """Phase 2: Fine-tuning."""
    print("\n" + "="*60)
    print("PHASE 2: FINE-TUNING (50 epochs, lr=0.00001)")
    print("="*60)
    
    unfreeze_last_layers(model, config.UNFREEZE_LAYERS)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.PHASE2_LR),
        loss=lambda y_true, y_pred: spares_categorical_crossentropy_with_smoothing(
            y_true, y_pred, config.LABEL_SMOOTHING
        ),
        metrics=['accuracy']
    )
    
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
            os.path.join(config.MODEL_SAVE_DIR, 'v3_phase2_best.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        CSVLogger(os.path.join(config.OUTPUT_DIR, 'v3_phase2_log.csv'))
    ]
    
    print(f"\nTraining Phase 2: {config.PHASE2_EPOCHS} epochs, lr={config.PHASE2_LR}")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.PHASE2_EPOCHS,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=config.VERBOSE
    )
    
    return history


def evaluate_model(model, test_ds, breed_mapping):
    """Evaluate model on test set."""
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    
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
    
    # Top-1
    top1_acc = accuracy_score(y_true, y_pred)
    print(f"\nTop-1 Accuracy: {top1_acc:.4f}")
    
    # Top-3
    top3_preds = np.argsort(y_pred_proba, axis=1)[:, -3:]
    top3_acc = np.mean([y_true[i] in top3_preds[i] for i in range(len(y_true))])
    print(f"Top-3 Accuracy: {top3_acc:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    # Classification report
    class_names = [breed_mapping[str(i)] for i in range(len(breed_mapping))]
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(f"\nClassification Report:\n{report}")
    
    return {
        'top1_accuracy': top1_acc,
        'top3_accuracy': top3_acc,
        'confusion_matrix': cm,
        'per_class_accuracy': per_class_acc,
        'y_true': y_true,
        'y_pred': y_pred,
        'class_names': class_names
    }


def main():
    """Main training function."""
    start_time = time.time()
    
    print("="*60)
    print("FINAL V3 TRAINING (22 CLASSES)")
    print("="*60)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load mapping
    breed_mapping = load_breed_mapping()
    print(f"Number of classes: {len(breed_mapping)}")
    
    # Load datasets
    print("\nLoading datasets...")
    train_ds, val_ds, test_ds, class_names = load_datasets()
    
    train_count = sum(1 for _ in train_ds.unbatch())
    val_count = sum(1 for _ in val_ds.unbatch())
    test_count = sum(1 for _ in test_ds.unbatch())
    print(f"Train: {train_count}, Val: {val_count}, Test: {test_count}")
    
    # Compute class weights
    print("\nComputing class weights...")
    class_weight_dict = compute_class_weights(train_ds)
    
    # Create model
    print("\nCreating model...")
    model = create_model()
    
    # Phase 1
    phase1_history = train_phase1(model, train_ds, val_ds, class_weight_dict)
    phase1_best = max(phase1_history.history['val_accuracy'])
    
    # Phase 2
    phase2_history = train_phase2(model, train_ds, val_ds, class_weight_dict)
    phase2_best = max(phase2_history.history['val_accuracy'])
    
    # Evaluate
    results = evaluate_model(model, test_ds, breed_mapping)
    
    # Save final model
    model.save(os.path.join(config.MODEL_SAVE_DIR, 'v3_final_model.keras'))
    
    # Generate report
    duration = time.time() - start_time
    
    print("\n" + "="*60)
    print("FINAL REPORT")
    print("="*60)
    print(f"\nPhase 1 Best Val Accuracy: {phase1_best:.4f}")
    print(f"Phase 2 Best Val Accuracy: {phase2_best:.4f}")
    print(f"\nTest Accuracy (Top-1): {results['top1_accuracy']:.4f}")
    print(f"Test Accuracy (Top-3): {results['top3_accuracy']:.4f}")
    
    # Per-class accuracy
    class_names = results['class_names']
    per_class_acc = list(zip(class_names, results['per_class_accuracy']))
    per_class_acc.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 5 Best Classes:")
    for name, acc in per_class_acc[:5]:
        print(f"  {name}: {acc:.4f}")
    
    print("\nTop 5 Worst Classes:")
    for name, acc in per_class_acc[-5:]:
        print(f"  {name}: {acc:.4f}")
    
    # Confusion matrix
    cm = results['confusion_matrix']
    confused = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm[i, j] > 0:
                confused.append((class_names[i], class_names[j], cm[i, j]))
    confused.sort(key=lambda x: x[2], reverse=True)
    
    print("\nTop 10 Most Confused Pairs:")
    for true, pred, count in confused[:10]:
        print(f"  {true} -> {pred}: {count}")
    
    # Save report
    report = {
        'phase1_best_val_accuracy': float(phase1_best),
        'phase2_best_val_accuracy': float(phase2_best),
        'test_accuracy_top1': float(results['top1_accuracy']),
        'test_accuracy_top3': float(results['top3_accuracy']),
        'top5_best': [{'name': name, 'accuracy': float(acc)} for name, acc in per_class_acc[:5]],
        'top5_worst': [{'name': name, 'accuracy': float(acc)} for name, acc in per_class_acc[-5:]],
        'top10_confused': [{'true': t, 'pred': p, 'count': int(c)} for t, p, c in confused[:10]],
        'training_duration': float(duration)
    }
    
    with open(os.path.join(config.OUTPUT_DIR, 'v3_final_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nTraining completed in {duration/60:.1f} minutes")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()