"""
Dataset Module
==============
Handles dataset loading and management for breed classification.
"""

import os
import numpy as np
from typing import List, Tuple, Optional, Dict
import cv2
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from .preprocessing import ImagePreprocessor, BREED_TO_IDX, IDX_TO_BREED, NUM_CLASSES
from .augmentation import DataAugmenter


class BreedDataset:
    """
    Dataset class for cattle breed classification.
    
    Handles:
    - Loading images from directory structure
    - Train/val/test splitting
    - Batch generation
    - On-the-fly augmentation
    """
    
    def __init__(self, 
                 data_dir: str,
                 target_size: Tuple[int, int] = (224, 224),
                 augmentation: bool = True):
        """
        Initialize dataset.
        
        Args:
            data_dir: Root directory containing images
            target_size: Target image size
            augmentation: Whether to apply augmentation
        """
        self.data_dir = data_dir
        self.target_size = target_size
        self.augmentation = augmentation
        
        self.preprocessor = ImagePreprocessor(target_size)
        self.augmenter = DataAugmenter() if augmentation else None
        
        self.classes = list(BREED_TO_IDX.keys())
        self.num_classes = NUM_CLASSES
        
        self.image_paths = []
        self.labels = []
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Load all image paths and labels from directory."""
        if not os.path.exists(self.data_dir):
            print(f"Warning: Data directory not found: {self.data_dir}")
            return
        
        for breed_name in os.listdir(self.data_dir):
            breed_dir = os.path.join(self.data_dir, breed_name)
            
            if not os.path.isdir(breed_dir):
                continue
            
            if breed_name not in BREED_TO_IDX:
                print(f"Warning: Unknown breed '{breed_name}', skipping...")
                continue
            
            label = BREED_TO_IDX[breed_name]
            
            for image_name in os.listdir(breed_dir):
                if image_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_path = os.path.join(breed_dir, image_name)
                    self.image_paths.append(image_path)
                    self.labels.append(label)
        
        print(f"Loaded {len(self.image_paths)} images from {self.data_dir}")
    
    def split(self, 
              train_ratio: float = 0.7,
              val_ratio: float = 0.15,
              test_ratio: float = 0.15,
              shuffle: bool = True,
              seed: int = 42) -> Tuple['BreedDataset', 'BreedDataset', 'BreedDataset']:
        """
        Split dataset into train, validation, and test sets.
        
        Args:
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            shuffle: Whether to shuffle before splitting
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1"
        
        indices = np.arange(len(self.image_paths))
        
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(indices)
        
        n_total = len(indices)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        # Create split datasets
        train_dataset = self._create_subset(train_indices)
        val_dataset = self._create_subset(val_indices)
        test_dataset = self._create_subset(test_indices)
        
        print(f"Split: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")
        
        return train_dataset, val_dataset, test_dataset
    
    def _create_subset(self, indices: np.ndarray) -> 'BreedDataset':
        """Create a subset of the dataset with given indices."""
        subset = BreedDataset.__new__(BreedDataset)
        subset.data_dir = self.data_dir
        subset.target_size = self.target_size
        subset.augmentation = self.augmentation
        subset.preprocessor = self.preprocessor
        subset.augmenter = self.augmenter
        subset.classes = self.classes
        subset.num_classes = self.num_classes
        subset.image_paths = [self.image_paths[i] for i in indices]
        subset.labels = [self.labels[i] for i in indices]
        
        return subset
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        """Get a single sample."""
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        image = self.preprocessor.preprocess(image)
        
        # Augment if enabled
        if self.augmentation and self.augmenter:
            image = self.augmenter.augment((image * 255).astype(np.uint8))
            image = image.astype(np.float32) / 255.0
        
        return image, label
    
    def get_batch(self, 
                  batch_size: int,
                  shuffle: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a batch of images and labels.
        
        Args:
            batch_size: Number of samples in batch
            shuffle: Whether to shuffle indices
            
        Returns:
            Tuple of (images_batch, labels_batch)
        """
        indices = np.arange(len(self.image_paths))
        
        if shuffle:
            np.random.shuffle(indices)
        
        batch_indices = indices[:batch_size]
        
        images = []
        labels = []
        
        for idx in batch_indices:
            image, label = self[idx]
            images.append(image)
            labels.append(label)
        
        images = np.array(images)
        labels = to_categorical(labels, num_classes=self.num_classes)
        
        return images, labels
    
    def create_tf_dataset(self, 
                          batch_size: int = 32,
                          shuffle: bool = True) -> tf.data.Dataset:
        """
        Create a TensorFlow dataset for training.
        
        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle
            
        Returns:
            TensorFlow dataset
        """
        def generator():
            indices = np.arange(len(self.image_paths))
            if shuffle:
                np.random.shuffle(indices)
            
            for idx in indices:
                image, label = self[idx]
                yield image, to_categorical(label, num_classes=self.num_classes)
        
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(*self.target_size, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(self.num_classes,), dtype=tf.float32)
            )
        )
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of samples across classes."""
        distribution = {}
        
        for label in self.labels:
            breed_name = IDX_TO_BREED[label]
            distribution[breed_name] = distribution.get(breed_name, 0) + 1
        
        return distribution
    
    def print_class_distribution(self):
        """Print distribution of samples across classes."""
        distribution = self.get_class_distribution()
        
        print("\nClass Distribution:")
        print("-" * 40)
        
        for breed, count in sorted(distribution.items(), key=lambda x: -x[1]):
            print(f"{breed:25s}: {count:5d}")
        
        print("-" * 40)
        print(f"{'Total':25s}: {len(self.labels):5d}")


def create_data_generators(data_dir: str,
                           batch_size: int = 32,
                           target_size: Tuple[int, int] = (224, 224)) -> Dict[str, tf.data.Dataset]:
    """
    Create train, validation, and test data generators.
    
    Args:
        data_dir: Root data directory
        batch_size: Batch size
        target_size: Target image size
        
    Returns:
        Dictionary with 'train', 'val', 'test' datasets
    """
    # Load full dataset
    full_dataset = BreedDataset(data_dir, target_size, augmentation=True)
    
    # Split
    train_dataset, val_dataset, test_dataset = full_dataset.split()
    
    # Create TensorFlow datasets
    generators = {
        'train': train_dataset.create_tf_dataset(batch_size, shuffle=True),
        'val': val_dataset.create_tf_dataset(batch_size, shuffle=False),
        'test': test_dataset.create_tf_dataset(batch_size, shuffle=False)
    }
    
    return generators
