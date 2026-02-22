"""
Training Callbacks Module
=========================
Custom callbacks for training.
"""

import os
import numpy as np
from typing import Dict, Optional
import tensorflow as tf
from tensorflow.keras.callbacks import Callback


def get_training_callbacks(checkpoint_dir: str = 'checkpoints',
                           log_dir: str = 'logs',
                           patience: int = 7) -> list:
    """
    Get standard training callbacks.
    
    Args:
        checkpoint_dir: Directory for model checkpoints
        log_dir: Directory for training logs
        patience: Early stopping patience
        
    Returns:
        List of callbacks
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    callbacks = [
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
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
            filepath=os.path.join(checkpoint_dir, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        
        # CSV logger
        tf.keras.callbacks.CSVLogger(
            os.path.join(log_dir, 'training_log.csv')
        ),
        
        # TensorBoard
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1
        )
    ]
    
    return callbacks


class ConfusionMatrixCallback(Callback):
    """
    Callback to generate confusion matrix at end of each epoch.
    """
    
    def __init__(self, 
                 validation_data: tf.data.Dataset,
                 class_names: list,
                 output_dir: str = 'logs'):
        """
        Initialize callback.
        
        Args:
            validation_data: Validation dataset
            class_names: List of class names
            output_dir: Directory to save confusion matrices
        """
        super().__init__()
        self.validation_data = validation_data
        self.class_names = class_names
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
    
    def on_epoch_end(self, epoch, logs=None):
        """Generate confusion matrix at end of epoch."""
        # Get predictions
        y_true = []
        y_pred = []
        
        for images, labels in self.validation_data:
            predictions = self.model.predict(images, verbose=0)
            
            y_true.extend(np.argmax(labels, axis=1))
            y_pred.extend(np.argmax(predictions, axis=1))
        
        # Generate confusion matrix
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title(f'Confusion Matrix - Epoch {epoch + 1}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        
        # Save
        plt.savefig(os.path.join(self.output_dir, f'confusion_matrix_epoch_{epoch + 1}.png'))
        plt.close()


class MetricsLoggerCallback(Callback):
    """
    Callback to log detailed metrics during training.
    """
    
    def __init__(self, output_file: str = 'logs/metrics.json'):
        """
        Initialize callback.
        
        Args:
            output_file: Path to metrics log file
        """
        super().__init__()
        self.output_file = output_file
        self.metrics_history = []
    
    def on_epoch_end(self, epoch, logs=None):
        """Log metrics at end of epoch."""
        import json
        
        metrics = {
            'epoch': epoch + 1,
            'train_loss': float(logs.get('loss', 0)),
            'train_accuracy': float(logs.get('accuracy', 0)),
            'val_loss': float(logs.get('val_loss', 0)),
            'val_accuracy': float(logs.get('val_accuracy', 0)),
            'learning_rate': float(self.model.optimizer.learning_rate.numpy())
        }
        
        self.metrics_history.append(metrics)
        
        # Save to file
        with open(self.output_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)


class LearningRateFinderCallback(Callback):
    """
    Callback to find optimal learning rate range.
    """
    
    def __init__(self, 
                 min_lr: float = 1e-7,
                 max_lr: float = 1.0,
                 num_steps: int = 100):
        """
        Initialize callback.
        
        Args:
            min_lr: Minimum learning rate
            max_lr: Maximum learning rate
            num_steps: Number of steps to increase LR
        """
        super().__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.num_steps = num_steps
        
        self.learning_rates = []
        self.losses = []
        self.step = 0
    
    def on_train_batch_end(self, batch, logs=None):
        """Record loss at each batch."""
        if self.step >= self.num_steps:
            self.model.stop_training = True
            return
        
        # Calculate current learning rate
        lr = self.min_lr * (self.max_lr / self.min_lr) ** (self.step / self.num_steps)
        
        # Set learning rate
        self.model.optimizer.learning_rate = lr
        
        # Record
        self.learning_rates.append(lr)
        self.losses.append(logs.get('loss', 0))
        
        self.step += 1
    
    def plot_lr_finder(self, output_path: str = 'lr_finder.png'):
        """Plot learning rate finder results."""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.learning_rates, self.losses)
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('Learning Rate Finder')
        plt.savefig(output_path)
        plt.close()
        
        print(f"Learning rate finder plot saved to {output_path}")


class GradientClippingCallback(Callback):
    """
    Callback to monitor gradient norms during training.
    """
    
    def __init__(self, clip_norm: float = 1.0):
        """
        Initialize callback.
        
        Args:
            clip_norm: Maximum gradient norm
        """
        super().__init__()
        self.clip_norm = clip_norm
        self.gradient_norms = []
    
    def on_train_batch_end(self, batch, logs=None):
        """Monitor gradients."""
        # This is a placeholder - actual gradient monitoring
        # requires custom training loop
        pass


class ModelCheckpointWithMetadata(Callback):
    """
    Enhanced model checkpoint that saves metadata.
    """
    
    def __init__(self, 
                 filepath: str,
                 monitor: str = 'val_accuracy',
                 mode: str = 'max'):
        """
        Initialize callback.
        
        Args:
            filepath: Path template for checkpoints
            monitor: Metric to monitor
            mode: 'max' or 'min'
        """
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.best_value = float('-inf') if mode == 'max' else float('inf')
    
    def on_epoch_end(self, epoch, logs=None):
        """Save checkpoint if improved."""
        current_value = logs.get(self.monitor)
        
        if current_value is None:
            return
        
        improved = False
        if self.mode == 'max' and current_value > self.best_value:
            improved = True
        elif self.mode == 'min' and current_value < self.best_value:
            improved = True
        
        if improved:
            self.best_value = current_value
            
            # Save model
            import json
            from datetime import datetime
            
            # Save weights
            self.model.save_weights(self.filepath)
            
            # Save metadata
            metadata = {
                'epoch': epoch + 1,
                'monitor': self.monitor,
                'value': float(current_value),
                'timestamp': datetime.now().isoformat(),
                'all_metrics': {k: float(v) for k, v in logs.items()}
            }
            
            metadata_path = self.filepath.replace('.h5', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"\nSaved checkpoint at epoch {epoch + 1} with {self.monitor} = {current_value:.4f}")
