"""
Model Trainer Module
====================
Handles training pipeline for breed classification models.
"""

import os
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

from ..models.efficientnet_classifier import EfficientNetClassifier
from ..data.dataset import BreedDataset


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
              fine_tune_lr: float = 0.0001) -> Dict:
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
            
        Returns:
            Training history
        """
        print(f"\n{'='*60}")
        print(f"Starting Training: {self.experiment_name}")
        print(f"{'='*60}")
        
        # Compile model
        self.model.compile_model(learning_rate=learning_rate)
        
        # Get callbacks
        callbacks = self._get_callbacks()
        
        # Phase 1: Train with frozen backbone
        print("\nPhase 1: Training classification head...")
        history = self.model.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            initial_epoch=initial_epoch,
            callbacks=callbacks
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
