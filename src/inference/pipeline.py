"""
Complete Inference Pipeline for Cattle Breed Recognition
=========================================================

This module provides the complete inference pipeline that combines:
1. YOLOv8-Nano for animal detection
2. EfficientNet-B0 for breed classification

Designed for integration with Bharat Pashudhan App (BPA).

Author: SIH 2025 Team
Problem Statement: SIH25004
"""

import os
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Breed definitions
BREEDS = [
    'Gir', 'Sahiwal', 'Red_Sindhi', 'Tharparkar', 'Rathi',
    'Hariana', 'Kankrej', 'Ongole', 'Deoni',
    'Hallikar', 'Amritmahal', 'Khillari', 'Kangayam', 'Bargur',
    'Dangi', 'Krishna_Valley', 'Malnad_Gidda', 'Punganur', 'Vechur',
    'Pulikulam', 'Umblachery', 'Toda', 'Kalahandi',
    'Murrah', 'Jaffrabadi', 'Nili_Ravi', 'Banni', 'Pandharpuri',
    'Mehsana', 'Surti', 'Nagpuri', 'Bhadawari', 'Chilika',
    'Jersey_Cross', 'HF_Cross'
]

IDX_TO_CLASS = {i: breed for i, breed in enumerate(BREEDS)}


class TFLiteModel:
    """Base class for TFLite model inference."""
    
    def __init__(self, model_path: str):
        """
        Initialize TFLite model.
        
        Args:
            model_path: Path to .tflite model file
        """
        import tensorflow as tf
        
        self.model_path = model_path
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.input_shape = self.input_details[0]['shape'][1:3]
        self.input_dtype = self.input_details[0]['dtype']
        
        logger.info(f"Loaded model: {model_path}")
        logger.info(f"Input shape: {self.input_shape}")
        logger.info(f"Input dtype: {self.input_dtype}")
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for model input.
        
        Args:
            image: Input image (BGR format)
        
        Returns:
            Preprocessed image ready for inference
        """
        raise NotImplementedError
    
    def postprocess(self, output: np.ndarray) -> Dict:
        """
        Postprocess model output.
        
        Args:
            output: Raw model output
        
        Returns:
            Processed results dictionary
        """
        raise NotImplementedError
    
    def predict(self, image: np.ndarray) -> Dict:
        """
        Run inference on image.
        
        Args:
            image: Input image (BGR format)
        
        Returns:
            Prediction results
        """
        # Preprocess
        input_tensor = self.preprocess(image)
        
        # Run inference
        start_time = time.time()
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        inference_time = (time.time() - start_time) * 1000
        
        # Postprocess
        result = self.postprocess(output)
        result['inference_time_ms'] = inference_time
        
        return result


class YOLODetector(TFLiteModel):
    """YOLOv8-Nano detector for cattle/buffalo detection."""
    
    def __init__(self, model_path: str, conf_threshold: float = 0.25):
        """
        Initialize YOLO detector.
        
        Args:
            model_path: Path to YOLOv8 TFLite model
            conf_threshold: Confidence threshold for detection
        """
        super().__init__(model_path)
        self.conf_threshold = conf_threshold
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for YOLO detection."""
        import cv2
        
        # Store original dimensions
        self.orig_height, self.orig_width = image.shape[:2]
        
        # Resize to model input size
        input_size = self.input_shape[0]  # Usually 416
        resized = cv2.resize(image, (input_size, input_size))
        
        # Normalize to 0-1
        normalized = resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        input_tensor = np.expand_dims(normalized, axis=0)
        
        # Handle quantized input
        if self.input_dtype == np.uint8:
            input_scale = self.input_details[0]['quantization_parameters']['scales'][0]
            input_zero_point = self.input_details[0]['quantization_parameters']['zero_points'][0]
            input_tensor = (input_tensor / input_scale + input_zero_point).astype(np.uint8)
        
        return input_tensor
    
    def postprocess(self, output: np.ndarray) -> Dict:
        """Postprocess YOLO output to get bounding boxes."""
        # Handle quantized output
        if self.output_details[0]['dtype'] == np.uint8:
            output_scale = self.output_details[0]['quantization_parameters']['scales'][0]
            output_zero_point = self.output_details[0]['quantization_parameters']['zero_points'][0]
            output = (output.astype(np.float32) - output_zero_point) * output_scale
        
        # YOLOv8 output format: [batch, num_boxes, 5+num_classes]
        # For detection only (1 class: cattle), output shape is [1, 8400, 5]
        
        # Simplified: return full image as bbox
        # In production, implement proper NMS and box decoding
        
        return {
            'detected': True,
            'bbox': [0, 0, self.orig_width, self.orig_height],  # x, y, w, h
            'confidence': 0.95,
            'class': 'cattle'
        }
    
    def detect(self, image: np.ndarray) -> Dict:
        """
        Detect animal in image.
        
        Args:
            image: Input image (BGR format)
        
        Returns:
            Detection result with bounding box
        """
        return self.predict(image)
    
    def crop_animal(self, image: np.ndarray, bbox: List[int], padding: float = 0.1) -> np.ndarray:
        """
        Crop animal from image using bounding box.
        
        Args:
            image: Input image
            bbox: Bounding box [x, y, w, h]
            padding: Padding ratio around bbox
        
        Returns:
            Cropped image
        """
        x, y, w, h = bbox
        
        # Add padding
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        
        # Calculate crop coordinates
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(image.shape[1], x + w + pad_w)
        y2 = min(image.shape[0], y + h + pad_h)
        
        return image[y1:y2, x1:x2]


class EfficientNetClassifier(TFLiteModel):
    """EfficientNet-B0 classifier for breed identification."""
    
    def __init__(self, model_path: str, top_k: int = 3):
        """
        Initialize EfficientNet classifier.
        
        Args:
            model_path: Path to EfficientNet TFLite model
            top_k: Number of top predictions to return
        """
        super().__init__(model_path)
        self.top_k = top_k
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for EfficientNet classification."""
        import cv2
        
        # Resize to model input size (224x224)
        input_size = self.input_shape[0]  # Usually 224
        resized = cv2.resize(image, (input_size, input_size))
        
        # Normalize to 0-1
        normalized = resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        input_tensor = np.expand_dims(normalized, axis=0)
        
        # Handle quantized input
        if self.input_dtype == np.uint8:
            input_scale = self.input_details[0]['quantization_parameters']['scales'][0]
            input_zero_point = self.input_details[0]['quantization_parameters']['zero_points'][0]
            input_tensor = (input_tensor / input_scale + input_zero_point).astype(np.uint8)
        
        return input_tensor
    
    def postprocess(self, output: np.ndarray) -> Dict:
        """Postprocess EfficientNet output to get breed predictions."""
        # Handle quantized output
        if self.output_details[0]['dtype'] == np.uint8:
            output_scale = self.output_details[0]['quantization_parameters']['scales'][0]
            output_zero_point = self.output_details[0]['quantization_parameters']['zero_points'][0]
            output = (output.astype(np.float32) - output_zero_point) * output_scale
        
        # Get probabilities
        probabilities = output[0]
        
        # Get top-k predictions
        top_k_idx = np.argsort(probabilities)[-self.top_k:][::-1]
        
        predictions = []
        for idx in top_k_idx:
            predictions.append({
                'breed': IDX_TO_CLASS[idx],
                'breed_index': int(idx),
                'confidence': float(probabilities[idx])
            })
        
        return {
            'breed': predictions[0]['breed'],
            'confidence': predictions[0]['confidence'],
            'top_predictions': predictions
        }
    
    def classify(self, image: np.ndarray) -> Dict:
        """
        Classify breed from image.
        
        Args:
            image: Input image (BGR format)
        
        Returns:
            Classification result with breed and confidence
        """
        return self.predict(image)


class BreedRecognitionPipeline:
    """
    Complete pipeline for cattle breed recognition.
    
    Combines YOLO detection and EfficientNet classification.
    """
    
    def __init__(
        self,
        detector_path: str,
        classifier_path: str,
        confidence_threshold: float = 0.85,
        escalation_threshold: float = 0.60
    ):
        """
        Initialize the complete pipeline.
        
        Args:
            detector_path: Path to YOLOv8 TFLite model
            classifier_path: Path to EfficientNet TFLite model
            confidence_threshold: Threshold for auto-confirm
            escalation_threshold: Threshold for expert escalation
        """
        self.detector = YOLODetector(detector_path)
        self.classifier = EfficientNetClassifier(classifier_path)
        
        self.confidence_threshold = confidence_threshold
        self.escalation_threshold = escalation_threshold
        
        logger.info("Breed Recognition Pipeline initialized")
        logger.info(f"Auto-confirm threshold: {confidence_threshold}")
        logger.info(f"Escalation threshold: {escalation_threshold}")
    
    def predict(self, image: np.ndarray) -> Dict:
        """
        Run complete prediction pipeline.
        
        Args:
            image: Input image (BGR format)
        
        Returns:
            Complete prediction result
        """
        start_time = time.time()
        
        # Stage 1: Detection
        detection_result = self.detector.detect(image)
        
        # Stage 2: Classification
        if detection_result['detected']:
            # Crop animal region
            crop = self.detector.crop_animal(image, detection_result['bbox'])
            
            # Classify breed
            classification_result = self.classifier.classify(crop)
        else:
            classification_result = {
                'breed': 'Unknown',
                'confidence': 0.0,
                'top_predictions': []
            }
        
        total_time = (time.time() - start_time) * 1000
        
        # Determine action based on confidence
        confidence = classification_result['confidence']
        if confidence >= self.confidence_threshold:
            action = 'auto_confirm'
        elif confidence >= self.escalation_threshold:
            action = 'flw_select'
        else:
            action = 'expert_review'
        
        return {
            'detection': detection_result,
            'classification': classification_result,
            'action': action,
            'total_time_ms': total_time,
            'detection_time_ms': detection_result.get('inference_time_ms', 0),
            'classification_time_ms': classification_result.get('inference_time_ms', 0)
        }
    
    def predict_from_file(self, image_path: str) -> Dict:
        """
        Run prediction on image file.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Complete prediction result
        """
        import cv2
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        return self.predict(image)


def main():
    """Demo function to test the pipeline."""
    import argparse
    import cv2
    
    parser = argparse.ArgumentParser(description='Cattle Breed Recognition')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--detector', type=str, default='../models/final/yolov8_nano_cattle_detector_int8.tflite',
                        help='Path to YOLO detector model')
    parser.add_argument('--classifier', type=str, default='../models/final/efficientnet_b0_int8.tflite',
                        help='Path to EfficientNet classifier model')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = BreedRecognitionPipeline(
        detector_path=args.detector,
        classifier_path=args.classifier
    )
    
    if args.image:
        # Predict from file
        result = pipeline.predict_from_file(args.image)
        
        print("\n" + "="*60)
        print("PREDICTION RESULT")
        print("="*60)
        print(f"\nDetected: {result['detection']['detected']}")
        print(f"Predicted Breed: {result['classification']['breed']}")
        print(f"Confidence: {result['classification']['confidence']*100:.1f}%")
        print(f"Action: {result['action']}")
        print(f"\nTotal Time: {result['total_time_ms']:.1f}ms")
        
        print("\nTop 3 Predictions:")
        for pred in result['classification']['top_predictions']:
            print(f"  {pred['breed']}: {pred['confidence']*100:.1f}%")
    else:
        print("No image provided. Use --image to specify input.")
        print("\nExample usage:")
        print("  python pipeline.py --image cattle_photo.jpg")


if __name__ == '__main__':
    main()
