"""
Model Trainer Module
====================
Handles training pipeline for breed classification models.
"""

import os
import json
import numpy as np
from collections import Counter
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

from ..models.efficientnet_classifier import EfficientNetClassifier
from ..data.dataset import BreedDataset


def compute_class_weights(dataset_labels: list, 
                          method: str = 'inverse_sqrt') -> Dict[int, float]:
    """
    Compute class weights to handle imbalanced dataset.
    
    This is the 'Goldilocks' solution for minority breeds - using inverse square root
    scaling to balance attention between common breeds (Sahiwal, Gir) and rare breeds
    (Vechur, Punganur) without being too aggressive.
    
    Args:
        dataset_labels: List of integer labels
        method: 'inverse_frequency', 'inverse_sqrt', or 'balanced'
    
    Returns:
        Dictionary mapping class index to weight
    
    Example:
        >>> labels = [0, 0, 0, 1, 1, 2]  # Class 2 is minority
        >>> weights = compute_class_weights(labels, 'inverse_sqrt')
        >>> print(weights)
        {0: 0.5, 1: 0.7, 2: 1.5}  # Class 2 gets 3x more weight
    """
    # Count samples per class
    class_counts = Counter(dataset_labels)
    total_samples = len(dataset_labels)
    num_classes = len(class_counts)
    
    # Find max count for normalization
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    
    class_weights = {}
    
    if method == 'inverse_frequency':
        # Standard inverse frequency: weight = N / (n_classes * n_samples)
        for cls_idx in range(num_classes):
            count = class_counts.get(cls_idx, 1)  # Avoid division by zero
            class_weights[cls_idx] = total_samples / (num_classes * count)
    
    elif method == 'inverse_sqrt':
        # Square root scaling - 'Goldilocks' solution
        # Less aggressive than pure inverse frequency, but effective
        for cls_idx in range(num_classes):
            count = class_counts.get(cls_idx, 1)
            # weight = sqrt(max_count / count)
            # This gives: minority class (10 samples) with max (1000) = sqrt(100) = 10x weight
            #             majority class (1000 samples) = sqrt(1) = 1x weight
            class_weights[cls_idx] = np.sqrt(max_count / count)
    
    elif method == 'balanced':
        # Keras balanced method
        for cls_idx in range(num_classes):
            count = class_counts.get(cls_idx, 1)
            class_weights[cls_idx] = (total_samples / num_classes) / count
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'inverse_frequency', 'inverse_sqrt', or 'balanced'")
    
    # Normalize weights to have mean = 1 (recommended for Keras)
    mean_weight = np.mean(list(class_weights.values()))
    class_weights = {k: float(v / mean_weight) for k, v in class_weights.items()}
    
    return class_weights


def get_class_weights_from_directory(data_dir: str, method: str = 'inverse_sqrt') -> Dict[int, float]:
    """
    Generate class weights directly from directory structure.
    
    Args:
        data_dir: Path to training data directory (with breed subdirectories)
        method: Weighting method to use
    
    Returns:
        Dictionary mapping class index to weight
    
    Raises:
        ValueError: If directory doesn't exist or is empty
    """
    import os
    from ..data.preprocessing import BREED_TO_IDX
    
    # Use absolute path for reliability
    data_dir = os.path.abspath(data_dir)
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory does not exist: {data_dir}")
    
    if not os.path.isdir(data_dir):
        raise ValueError(f"Path is not a directory: {data_dir}")
    
    labels = []
    breed_counts = {}
    
    # Count images per class
    valid_breeds = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    if not valid_breeds:
        raise ValueError(f"No subdirectories found in: {data_dir}")
    
    for breed_name in valid_breeds:
        breed_dir = os.path.join(data_dir, breed_name)
        if breed_name in BREED_TO_IDX:
            breed_label = BREED_TO_IDX[breed_name]
            image_files = [f for f in os.listdir(breed_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            image_count = len(image_files)
            if image_count > 0:
                labels.extend([breed_label] * image_count)
                breed_counts[breed_name] = image_count
    
    if not labels:
        raise ValueError(f"No valid images found in: {data_dir}")
    
    print(f"Found {len(labels)} images across {len(breed_counts)} breeds")
    
    return compute_class_weights(labels, method)


def print_class_weights(weights: Dict[int, float], data_dir: str = None):
    """
    Pretty print class weights with breed names.
    
    Args:
        weights: Dictionary of class weights
        data_dir: Optional data directory for breed names
    """
    import os
    from ..data.preprocessing import IDX_TO_BREED
    
    print("\n" + "=" * 60)
    print("Class Weights for Minority Breed Handling")
    print("=" * 60)
    
    if data_dir:
        # Use absolute path
        data_dir = os.path.abspath(data_dir)
        
        # Show with counts
        from ..data.preprocessing import BREED_TO_IDX
        
        for breed_name in sorted(os.listdir(data_dir)):
            breed_dir = os.path.join(data_dir, breed_name)
            if os.path.isdir(breed_dir) and breed_name in BREED_TO_IDX:
                idx = BREED_TO_IDX[breed_name]
                if idx in weights:
                    image_count = len([f for f in os.listdir(breed_dir) 
                                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                    print(f"{breed_name:20s}: weight={weights[idx]:.3f}, samples={image_count}")
    else:
        # Show with indices
        for idx, weight in sorted(weights.items(), key=lambda x: -x[1]):
            breed_name = IDX_TO_BREED.get(idx, f'Unknown_{idx}')
            print(f"{breed_name:20s}: {weight:.3f}")
    
    print("=" * 60 + "\n")


class ModelTrainer:
    """
    Training pipeline for breed classification models.
    
    Handles:
    - Model initialization
    - Training with callbacks
    - Evaluation
    - Checkpointing
    - Logging
    """
    
    def __init__(self,
                 model: EfficientNetClassifier,
                 output_dir: str = 'models',
                 experiment_name: str = 'breed_classification'):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            output_dir: Directory for outputs
            experiment_name: Name for this experiment
        """
        self.model = model
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        
        # Create output directories
        self.checkpoint_dir = os.path.join(output_dir, 'checkpoints', experiment_name)
        self.log_dir = os.path.join(output_dir, 'logs', experiment_name)
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.history = None
        self.best_metrics = {}
    
    def train(self,
              train_data: tf.data.Dataset,
              val_data: tf.data.Dataset,
              epochs: int = 20,
              initial_epoch: int = 0,
              learning_rate: float = 0.001,
              fine_tune: bool = False,
              fine_tune_layers: int = 20,
              fine_tune_lr: float = 0.0001,
              class_weights: Dict[int, float] = None) -> Dict:
        """
        Train the model.
        
        Args:
            train_data: Training dataset
            val_data: Validation dataset
            epochs: Number of epochs
            initial_epoch: Starting epoch (for resuming)
            learning_rate: Initial learning rate
            fine_tune: Whether to fine-tune after initial training
            fine_tune_layers: Number of layers to unfreeze for fine-tuning
            fine_tune_lr: Learning rate for fine-tuning
            class_weights: Optional class weights for imbalanced data
                         
        Returns:
            Training history
        """
        print(f"\n{'='*60}")
        print(f"Starting Training: {self.experiment_name}")
        if class_weights:
            print(f"Using Class Weights: YES ({len(class_weights)} classes)")
        else:
            print(f"Using Class Weights: NO")
        print(f"{'='*60}")
        
        # Compile model
        self.model.compile_model(learning_rate=learning_rate)
        
        # Get callbacks
        callbacks = self._get_callbacks()
        
        # Phase 1: Train with frozen backbone
        print("\nPhase 1: Training classification head...")
        
        # Prepare fit kwargs
        fit_kwargs = {
            'validation_data': val_data,
            'epochs': epochs,
            'initial_epoch': initial_epoch,
            'callbacks': callbacks
        }
        
        # Add class weights if provided
        if class_weights is not None:
            fit_kwargs['class_weight'] = class_weights
            print(f"✓ Class weights applied: min={min(class_weights.values()):.2f}, max={max(class_weights.values()):.2f}")
        
        history = self.model.model.fit(
            train_data,
            **fit_kwargs
        )
        
        # Phase 2: Fine-tuning (optional)
        if fine_tune:
            print(f"\nPhase 2: Fine-tuning last {fine_tune_layers} layers...")
            self.model.unfreeze_backbone(fine_tune_layers)
            
            # Recompile with lower learning rate
            self.model.compile_model(learning_rate=fine_tune_lr)
            
            fine_tune_history = self.model.model.fit(
                train_data,
                validation_data=val_data,
                epochs=epochs + 10,
                initial_epoch=epochs,
                callbacks=callbacks
            )
            
            # Merge histories
            for key in fine_tune_history.history:
                history.history[key].extend(fine_tune_history.history[key])
        
        self.history = history.history
        
        # Save training history
        self._save_history()
        
        # Print summary
        self._print_training_summary()
        
        return self.history
    
    def train_with_class_weights(self,
                                  train_data: tf.data.Dataset,
                                  val_data: tf.data.Dataset,
                                  epochs: int = 20,
                                  learning_rate: float = 0.001,
                                  class_weights: Dict[int, float] = None,
                                  data_dir: str = None) -> Dict:
        """
        Train with class weights for handling imbalanced datasets.
        
        This is the recommended method for Phase 2 training as it helps
        the model learn minority breeds better (Vechur, Punganur, etc.).
        
        Args:
            train_data: Training dataset
            val_data: Validation dataset
            epochs: Number of epochs
            learning_rate: Initial learning rate
            class_weights: Pre-computed class weights (optional)
            data_dir: Data directory to compute weights from (optional)
        
        Returns:
            Training history
        
        Example:
            >>> # Option 1: Compute weights from data directory
            >>> history = trainer.train_with_class_weights(
            ...     train_data, val_data, epochs=20,
            ...     data_dir='data/train'
            ... )
            >>> 
            >>> # Option 2: Provide pre-computed weights
            >>> weights = get_class_weights_from_directory('data/train')
            >>> history = trainer.train_with_class_weights(
            ...     train_data, val_data, epochs=20,
            ...     class_weights=weights
            ... )
        """
        # Compute class weights if data_dir provided
        if class_weights is None and data_dir is not None:
            print("\nComputing class weights from data directory...")
            class_weights = get_class_weights_from_directory(data_dir, method='inverse_sqrt')
            print_class_weights(class_weights, data_dir)
        
        if class_weights is None:
            print("\n⚠ Warning: No class weights provided. Using standard training.")
            print("   Consider providing class_weights or data_dir for better minority breed accuracy.")
        
        return self.train(
            train_data=train_data,
            val_data=val_data,
            epochs=epochs,
            learning_rate=learning_rate,
            class_weights=class_weights
        )
    
    def _get_callbacks(self) -> List[Callback]:
        """
        Get training callbacks.
        
        Returns:
            List of callbacks
        """
        callbacks = [
            # Early stopping
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=7,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Model checkpoint
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.checkpoint_dir, 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            
            # CSV logger
            tf.keras.callbacks.CSVLogger(
                os.path.join(self.log_dir, 'training_log.csv')
            ),
            
            # TensorBoard
            tf.keras.callbacks.TensorBoard(
                log_dir=self.log_dir,
                histogram_freq=1
            )
        ]
        
        return callbacks
    
    def _save_history(self):
        """Save training history to JSON."""
        history_path = os.path.join(self.log_dir, 'history.json')
        
        # Convert numpy arrays to lists
        history_serializable = {}
        for key, value in self.history.items():
            history_serializable[key] = [float(v) for v in value]
        
        with open(history_path, 'w') as f:
            json.dump(history_serializable, f, indent=2)
        
        print(f"Saved training history to {history_path}")
    
    def _print_training_summary(self):
        """Print training summary."""
        print(f"\n{'='*60}")
        print("Training Summary")
        print(f"{'='*60}")
        
        if self.history:
            best_epoch = np.argmax(self.history['val_accuracy'])
            
            print(f"Best Epoch: {best_epoch + 1}")
            print(f"Best Validation Accuracy: {self.history['val_accuracy'][best_epoch]:.4f}")
            print(f"Best Validation Loss: {self.history['val_loss'][best_epoch]:.4f}")
            print(f"Training Accuracy: {self.history['accuracy'][best_epoch]:.4f}")
            print(f"Training Loss: {self.history['loss'][best_epoch]:.4f}")
            
            self.best_metrics = {
                'best_epoch': int(best_epoch + 1),
                'val_accuracy': float(self.history['val_accuracy'][best_epoch]),
                'val_loss': float(self.history['val_loss'][best_epoch]),
                'train_accuracy': float(self.history['accuracy'][best_epoch]),
                'train_loss': float(self.history['loss'][best_epoch])
            }
    
    def evaluate(self, test_data: tf.data.Dataset) -> Dict:
        """
        Evaluate model on test data.
        
        Args:
            test_data: Test dataset
            
        Returns:
            Evaluation metrics
        """
        print(f"\n{'='*60}")
        print("Evaluating on Test Data")
        print(f"{'='*60}")
        
        results = self.model.evaluate(test_data)
        
        print(f"Test Loss: {results['loss']:.4f}")
        print(f"Test Accuracy: {results['accuracy']:.4f}")
        if results['top_k_accuracy']:
            print(f"Test Top-K Accuracy: {results['top_k_accuracy']:.4f}")
        
        # Save results
        results_path = os.path.join(self.log_dir, 'test_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def export_model(self, output_path: str, quantize: bool = True):
        """
        Export model to TFLite format.
        
        Args:
            output_path: Output file path
            quantize: Apply quantization
        """
        self.model.export_to_tflite(output_path, quantize)
    
    def save_training_config(self, config: Dict):
        """
        Save training configuration.
        
        Args:
            config: Configuration dictionary
        """
        config_path = os.path.join(self.log_dir, 'config.json')
        
        config['experiment_name'] = self.experiment_name
        config['timestamp'] = datetime.now().isoformat()
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Saved training config to {config_path}")


class TrainingProgressCallback(Callback):
    """Custom callback for training progress tracking."""
    
    def __init__(self, total_epochs: int):
        super().__init__()
        self.total_epochs = total_epochs
    
    def on_epoch_end(self, epoch, logs=None):
        """Print progress at end of each epoch."""
        progress = (epoch + 1) / self.total_epochs * 100
        
        print(f"\nEpoch {epoch + 1}/{self.total_epochs} ({progress:.1f}%)")
        print(f"  Loss: {logs['loss']:.4f}")
        print(f"  Accuracy: {logs['accuracy']:.4f}")
        print(f"  Val Loss: {logs['val_loss']:.4f}")
        print(f"  Val Accuracy: {logs['val_accuracy']:.4f}")


def create_trainer(model: EfficientNetClassifier,
                   output_dir: str = 'models',
                   experiment_name: str = None) -> ModelTrainer:
    """
    Create a trainer instance.
    
    Args:
        model: Model to train
        output_dir: Output directory
        experiment_name: Experiment name (auto-generated if None)
        
    Returns:
        ModelTrainer instance
    """
    if experiment_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_name = f'breed_cls_{timestamp}'
    
    return ModelTrainer(
        model=model,
        output_dir=output_dir,
        experiment_name=experiment_name
    )
