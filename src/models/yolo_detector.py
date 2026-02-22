"""
YOLO Detector Module
====================
YOLO-Nano based animal detection for cattle breed recognition.
"""

import os
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
import tensorflow as tf
from tensorflow.keras import Model, layers


class YOLODetector:
    """
    YOLO-Nano based animal detector.
    
    Detects and localizes cattle/buffalo in images.
    Used as first stage before breed classification.
    """
    
    def __init__(self, 
                 input_size: Tuple[int, int] = (416, 416),
                 num_classes: int = 1,
                 confidence_threshold: float = 0.5,
                 nms_threshold: float = 0.4):
        """
        Initialize YOLO detector.
        
        Args:
            input_size: Input image size
            num_classes: Number of detection classes (1 for cattle/buffalo)
            confidence_threshold: Minimum confidence for detection
            nms_threshold: Non-maximum suppression threshold
        """
        self.input_size = input_size
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.model = None
    
    def build_model(self) -> Model:
        """
        Build YOLO-Nano model architecture.
        
        Returns:
            Keras Model instance
        """
        inputs = layers.Input(shape=(*self.input_size, 3))
        
        # Backbone - Lightweight CNN
        x = self._conv_block(inputs, 16, 3, strides=1)
        x = self._conv_block(x, 32, 3, strides=2)
        x = self._conv_block(x, 64, 3, strides=2)
        x = self._conv_block(x, 128, 3, strides=2)
        x = self._conv_block(x, 256, 3, strides=2)
        x = self._conv_block(x, 512, 3, strides=1)
        
        # Detection head
        x = self._conv_block(x, 256, 1, strides=1)
        x = self._conv_block(x, 512, 3, strides=1)
        
        # Output layer
        # 5 values: x, y, w, h, confidence + num_classes
        output_channels = 5 + self.num_classes
        outputs = layers.Conv2D(output_channels, 1, padding='same')(x)
        
        self.model = Model(inputs, outputs, name='yolo_nano')
        
        return self.model
    
    def _conv_block(self, x, filters: int, kernel_size: int, 
                    strides: int = 1) -> layers.Layer:
        """
        Create a convolutional block with BatchNorm and LeakyReLU.
        
        Args:
            x: Input tensor
            filters: Number of filters
            kernel_size: Kernel size
            strides: Stride
            
        Returns:
            Output tensor
        """
        x = layers.Conv2D(filters, kernel_size, strides=strides, 
                          padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        return x
    
    def load_pretrained(self, model_path: str):
        """
        Load pretrained weights.
        
        Args:
            model_path: Path to weights file
        """
        if self.model is None:
            self.build_model()
        
        self.model.load_weights(model_path)
        print(f"Loaded weights from {model_path}")
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for YOLO.
        
        Args:
            image: Input image (BGR)
            
        Returns:
            Preprocessed image tensor
        """
        # Resize
        image_resized = cv2.resize(image, self.input_size)
        
        # Normalize to [0, 1]
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        image_batch = np.expand_dims(image_normalized, axis=0)
        
        return image_batch
    
    def predict(self, image: np.ndarray) -> List[Dict]:
        """
        Detect animals in image.
        
        Args:
            image: Input image (BGR)
            
        Returns:
            List of detections with bounding boxes
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Preprocess
        input_tensor = self.preprocess(image)
        
        # Predict
        predictions = self.model.predict(input_tensor, verbose=0)
        
        # Post-process
        detections = self._postprocess(predictions[0], image.shape[:2])
        
        return detections
    
    def _postprocess(self, predictions: np.ndarray, 
                     original_size: Tuple[int, int]) -> List[Dict]:
        """
        Post-process predictions to get bounding boxes.
        
        Args:
            predictions: Raw model predictions
            original_size: Original image size (height, width)
            
        Returns:
            List of detection dictionaries
        """
        detections = []
        
        # Get grid dimensions
        grid_h, grid_w = predictions.shape[:2]
        
        # Reshape predictions
        predictions = predictions.reshape(-1, 5 + self.num_classes)
        
        # Extract boxes
        for i, pred in enumerate(predictions):
            # Get confidence
            confidence = pred[4]
            
            if confidence < self.confidence_threshold:
                continue
            
            # Get grid position
            grid_x = i % grid_w
            grid_y = i // grid_w
            
            # Decode bounding box
            x = (pred[0] + grid_x) / grid_w
            y = (pred[1] + grid_y) / grid_h
            w = pred[2]
            h = pred[3]
            
            # Convert to original image coordinates
            orig_h, orig_w = original_size
            x1 = int((x - w/2) * orig_w)
            y1 = int((y - h/2) * orig_h)
            x2 = int((x + w/2) * orig_w)
            y2 = int((y + h/2) * orig_h)
            
            # Clip to image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(orig_w, x2)
            y2 = min(orig_h, y2)
            
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(confidence),
                'class': 'cattle'
            })
        
        # Apply NMS
        detections = self._nms(detections)
        
        return detections
    
    def _nms(self, detections: List[Dict]) -> List[Dict]:
        """
        Apply Non-Maximum Suppression.
        
        Args:
            detections: List of detections
            
        Returns:
            Filtered detections
        """
        if len(detections) == 0:
            return detections
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        
        while len(detections) > 0:
            # Keep highest confidence detection
            best = detections[0]
            keep.append(best)
            
            # Remove overlapping detections
            remaining = []
            for det in detections[1:]:
                iou = self._compute_iou(best['bbox'], det['bbox'])
                if iou < self.nms_threshold:
                    remaining.append(det)
            
            detections = remaining
        
        return keep
    
    def _compute_iou(self, box1: List[int], box2: List[int]) -> float:
        """
        Compute Intersection over Union.
        
        Args:
            box1: First box [x1, y1, x2, y2]
            box2: Second box [x1, y1, x2, y2]
            
        Returns:
            IoU value
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def crop_animal(self, image: np.ndarray, 
                    detection: Dict) -> np.ndarray:
        """
        Crop detected animal from image.
        
        Args:
            image: Original image
            detection: Detection dictionary with bbox
            
        Returns:
            Cropped animal image
        """
        x1, y1, x2, y2 = detection['bbox']
        cropped = image[y1:y2, x1:x2]
        
        return cropped
    
    def export_to_tflite(self, output_path: str):
        """
        Export model to TFLite format.
        
        Args:
            output_path: Output file path
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # Optimization for size
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"Exported TFLite model to {output_path}")
        print(f"Model size: {len(tflite_model) / 1024 / 1024:.2f} MB")


def create_yolo_detector(weights_path: Optional[str] = None) -> YOLODetector:
    """
    Create and optionally load YOLO detector.
    
    Args:
        weights_path: Path to pretrained weights
        
    Returns:
        YOLODetector instance
    """
    detector = YOLODetector()
    detector.build_model()
    
    if weights_path and os.path.exists(weights_path):
        detector.load_pretrained(weights_path)
    
    return detector
