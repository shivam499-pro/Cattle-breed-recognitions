"""
Visualization Utilities
=======================
Functions for visualizing training results and model performance.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from sklearn.metrics import confusion_matrix, classification_report


def plot_training_history(history: Dict,
                          output_path: Optional[str] = None,
                          figsize: tuple = (12, 5)):
    """
    Plot training history.
    
    Args:
        history: Training history dictionary
        output_path: Path to save plot (optional)
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot loss
    axes[0].plot(history['loss'], label='Training Loss')
    axes[0].plot(history['val_loss'], label='Validation Loss')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot accuracy
    axes[1].plot(history['accuracy'], label='Training Accuracy')
    axes[1].plot(history['val_accuracy'], label='Validation Accuracy')
    axes[1].set_title('Model Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved training history plot to {output_path}")
    
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          class_names: List[str],
                          output_path: Optional[str] = None,
                          figsize: tuple = (12, 10),
                          normalize: bool = False):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        output_path: Path to save plot (optional)
        figsize: Figure size
        normalize: Whether to normalize the confusion matrix
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    # Plot
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix to {output_path}")
    
    plt.show()


def plot_per_class_accuracy(y_true: np.ndarray,
                            y_pred: np.ndarray,
                            class_names: List[str],
                            output_path: Optional[str] = None,
                            figsize: tuple = (14, 6)):
    """
    Plot per-class accuracy.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        output_path: Path to save plot (optional)
        figsize: Figure size
    """
    # Compute per-class accuracy
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    # Sort by accuracy
    sorted_indices = np.argsort(per_class_acc)
    sorted_names = [class_names[i] for i in sorted_indices]
    sorted_acc = per_class_acc[sorted_indices]
    
    # Plot
    plt.figure(figsize=figsize)
    bars = plt.barh(range(len(sorted_names)), sorted_acc)
    plt.yticks(range(len(sorted_names)), sorted_names)
    plt.xlabel('Accuracy')
    plt.title('Per-Class Accuracy')
    plt.xlim(0, 1)
    
    # Add value labels
    for bar, acc in zip(bars, sorted_acc):
        plt.text(acc + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{acc:.2%}', va='center')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved per-class accuracy plot to {output_path}")
    
    plt.show()


def plot_confidence_distribution(confidences: np.ndarray,
                                 output_path: Optional[str] = None,
                                 figsize: tuple = (10, 6)):
    """
    Plot distribution of prediction confidences.
    
    Args:
        confidences: Array of confidence scores
        output_path: Path to save plot (optional)
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    plt.hist(confidences, bins=50, edgecolor='black', alpha=0.7)
    
    # Add threshold lines
    plt.axvline(x=0.85, color='green', linestyle='--', label='High Confidence (85%)')
    plt.axvline(x=0.60, color='orange', linestyle='--', label='Medium Confidence (60%)')
    
    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    plt.title('Distribution of Prediction Confidences')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved confidence distribution plot to {output_path}")
    
    plt.show()


def plot_sample_predictions(images: np.ndarray,
                            true_labels: np.ndarray,
                            pred_labels: np.ndarray,
                            confidences: np.ndarray,
                            class_names: List[str],
                            output_path: Optional[str] = None,
                            num_samples: int = 12,
                            figsize: tuple = (15, 10)):
    """
    Plot sample predictions with true and predicted labels.
    
    Args:
        images: Array of images
        true_labels: True labels
        pred_labels: Predicted labels
        confidences: Prediction confidences
        class_names: List of class names
        output_path: Path to save plot (optional)
        num_samples: Number of samples to display
        figsize: Figure size
    """
    num_samples = min(num_samples, len(images))
    indices = np.random.choice(len(images), num_samples, replace=False)
    
    rows = (num_samples + 3) // 4
    fig, axes = plt.subplots(rows, 4, figsize=figsize)
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        ax = axes[i]
        
        # Display image
        img = images[idx]
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        
        ax.imshow(img)
        ax.axis('off')
        
        # Set title
        true_name = class_names[true_labels[idx]]
        pred_name = class_names[pred_labels[idx]]
        conf = confidences[idx]
        
        correct = true_labels[idx] == pred_labels[idx]
        color = 'green' if correct else 'red'
        
        ax.set_title(f'True: {true_name}\nPred: {pred_name}\nConf: {conf:.2%}',
                     fontsize=8, color=color)
    
    # Hide empty subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved sample predictions plot to {output_path}")
    
    plt.show()


def generate_classification_report(y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   class_names: List[str],
                                   output_path: Optional[str] = None) -> str:
    """
    Generate classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        output_path: Path to save report (optional)
        
    Returns:
        Classification report string
    """
    report = classification_report(y_true, y_pred, target_names=class_names)
    
    print("\nClassification Report:")
    print("=" * 60)
    print(report)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"Saved classification report to {output_path}")
    
    return report
