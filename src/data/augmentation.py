"""
Data Augmentation Module
========================
Handles image augmentation for training data expansion.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import random


class DataAugmenter:
    """
    Augments images for training data expansion.
    
    Supports:
    - Rotation
    - Flipping
    - Brightness adjustment
    - Contrast adjustment
    - Zoom
    - Noise addition
    """
    
    def __init__(self, 
                 rotation_range: int = 15,
                 brightness_range: Tuple[float, float] = (0.8, 1.2),
                 contrast_range: Tuple[float, float] = (0.85, 1.15),
                 zoom_range: Tuple[float, float] = (0.9, 1.1),
                 horizontal_flip: bool = True,
                 noise_std: float = 0.02):
        """
        Initialize augmenter with parameters.
        
        Args:
            rotation_range: Maximum rotation angle in degrees
            brightness_range: Range for brightness adjustment
            contrast_range: Range for contrast adjustment
            zoom_range: Range for zoom factor
            horizontal_flip: Whether to apply horizontal flip
            noise_std: Standard deviation for Gaussian noise
        """
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip
        self.noise_std = noise_std
    
    def rotate(self, image: np.ndarray, angle: Optional[float] = None) -> np.ndarray:
        """
        Rotate image by given angle.
        
        Args:
            image: Input image
            angle: Rotation angle (random if None)
            
        Returns:
            Rotated image
        """
        if angle is None:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
        
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height),
                                  borderMode=cv2.BORDER_REFLECT)
        
        return rotated
    
    def flip_horizontal(self, image: np.ndarray) -> np.ndarray:
        """
        Flip image horizontally.
        
        Args:
            image: Input image
            
        Returns:
            Flipped image
        """
        return cv2.flip(image, 1)
    
    def adjust_brightness(self, image: np.ndarray, 
                          factor: Optional[float] = None) -> np.ndarray:
        """
        Adjust image brightness.
        
        Args:
            image: Input image
            factor: Brightness factor (random if None)
            
        Returns:
            Brightness-adjusted image
        """
        if factor is None:
            factor = random.uniform(*self.brightness_range)
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = hsv[:, :, 2] * factor
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def adjust_contrast(self, image: np.ndarray,
                        factor: Optional[float] = None) -> np.ndarray:
        """
        Adjust image contrast.
        
        Args:
            image: Input image
            factor: Contrast factor (random if None)
            
        Returns:
            Contrast-adjusted image
        """
        if factor is None:
            factor = random.uniform(*self.contrast_range)
        
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[:, :, 0] = ((lab[:, :, 0] - 128) * factor) + 128
        lab[:, :, 0] = np.clip(lab[:, :, 0], 0, 255)
        
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    
    def zoom(self, image: np.ndarray, 
             factor: Optional[float] = None) -> np.ndarray:
        """
        Zoom in/out on image.
        
        Args:
            image: Input image
            factor: Zoom factor (random if None)
            
        Returns:
            Zoomed image
        """
        if factor is None:
            factor = random.uniform(*self.zoom_range)
        
        height, width = image.shape[:2]
        
        # Calculate crop dimensions
        new_height = int(height / factor)
        new_width = int(width / factor)
        
        # Calculate crop position (center)
        y1 = (height - new_height) // 2
        x1 = (width - new_width) // 2
        
        # Crop and resize
        if factor > 1:  # Zoom in
            cropped = image[y1:y1+new_height, x1:x1+new_width]
            zoomed = cv2.resize(cropped, (width, height))
        else:  # Zoom out
            zoomed = cv2.resize(image, (new_width, new_height))
            padded = np.zeros_like(image)
            y_offset = (height - new_height) // 2
            x_offset = (width - new_width) // 2
            padded[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = zoomed
            zoomed = padded
        
        return zoomed
    
    def add_noise(self, image: np.ndarray,
                   std: Optional[float] = None) -> np.ndarray:
        """
        Add Gaussian noise to image.
        
        Args:
            image: Input image
            std: Noise standard deviation (random if None)
            
        Returns:
            Noisy image
        """
        if std is None:
            std = self.noise_std
        
        noise = np.random.normal(0, std * 255, image.shape).astype(np.float32)
        noisy = image.astype(np.float32) + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        
        return noisy
    
    def augment(self, image: np.ndarray) -> np.ndarray:
        """
        Apply random augmentation to image.
        
        Args:
            image: Input image
            
        Returns:
            Augmented image
        """
        augmented = image.copy()
        
        # Random rotation
        if random.random() > 0.5:
            augmented = self.rotate(augmented)
        
        # Random horizontal flip
        if self.horizontal_flip and random.random() > 0.5:
            augmented = self.flip_horizontal(augmented)
        
        # Random brightness
        if random.random() > 0.5:
            augmented = self.adjust_brightness(augmented)
        
        # Random contrast
        if random.random() > 0.5:
            augmented = self.adjust_contrast(augmented)
        
        # Random zoom
        if random.random() > 0.5:
            augmented = self.zoom(augmented)
        
        # Random noise
        if random.random() > 0.7:  # Less frequent
            augmented = self.add_noise(augmented)
        
        return augmented
    
    def generate_augmented_batch(self, 
                                  image: np.ndarray, 
                                  num_augmentations: int = 5) -> List[np.ndarray]:
        """
        Generate multiple augmented versions of an image.
        
        Args:
            image: Input image
            num_augmentations: Number of augmented images to generate
            
        Returns:
            List of augmented images
        """
        augmented_images = [image]  # Include original
        
        for _ in range(num_augmentations):
            augmented = self.augment(image)
            augmented_images.append(augmented)
        
        return augmented_images


def create_augmentation_pipeline():
    """
    Create a standard augmentation pipeline for training.
    
    Returns:
        Configured DataAugmenter instance
    """
    return DataAugmenter(
        rotation_range=15,
        brightness_range=(0.8, 1.2),
        contrast_range=(0.85, 1.15),
        zoom_range=(0.9, 1.1),
        horizontal_flip=True,
        noise_std=0.02
    )
