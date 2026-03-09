"""
EfficientNet Classifier Module
==============================
EfficientNet-B0 based breed classification for cattle and buffaloes.
"""

import os
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

from ..data.preprocessing import IDX_TO_BREED, NUM_CLASSES


class EfficientNetClassifier:
    """
    EfficientNet-B0 based breed classifier.
    
    Classifies detected cattle/buffalo into specific breeds.
    Second stage of the breed recognition pipeline.
    """
    
    def __init__(self,
                 input_size: Tuple[int, int] = (224, 224),
                 num_classes: int = NUM_CLASSES,
                 confidence_threshold: float = 0.85):
        """
        Initialize EfficientNet classifier.
        
        Args:
            input_size: Input image size
            num_classes: Number of breed classes
            confidence_threshold: Threshold for high confidence predictions
        """
        self.input_size = input_size
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.model = None
    
    def build_model(self, 
                    pretrained: bool = True,
                    freeze_backbone: bool = True) -> Model:
        """
        Build EfficientNet-B0 based classifier.
        
        Args:
            pretrained: Use ImageNet pretrained weights
            freeze_backbone: Freeze backbone layers initially
            
        Returns:
            Keras Model instance
        """
        # Load EfficientNet-B0 backbone
        if pretrained:
            backbone = EfficientNetB0(
                include_top=False,
                weights='imagenet',
                input_shape=(*self.input_size, 3)
            )
        else:
            backbone = EfficientNetB0(
                include_top=False,
                weights=None,
                input_shape=(*self.input_size, 3)
            )
        
        # Freeze backbone if specified
        if freeze_backbone:
            for layer in backbone.layers:
                layer.trainable = False
        
        # Build classification head
        inputs = backbone.input
        
        x = backbone.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = Model(inputs, outputs, name='efficientnet_breed_classifier')
        
        return self.model
    
    def unfreeze_backbone(self, num_layers: int = 20):
        """
        Unfreeze last N layers of backbone for fine-tuning.
        
        Args:
            num_layers: Number of layers to unfreeze
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Unfreeze last N layers
        for layer in self.model.layers[-num_layers:]:
            layer.trainable = True
        
        print(f"Unfrozen last {num_layers} layers for fine-tuning")
    
    def compile_model(self, 
                      learning_rate: float = 0.001):
        """
        Compile model with optimizer and loss.
        
        Args:
            learning_rate: Learning rate for optimizer
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        print("Model compiled successfully")
    
    def load_weights(self, weights_path: str):
        """
        Load model weights.
        
        Args:
            weights_path: Path to weights file
        """
        if self.model is None:
            self.build_model(pretrained=False)
        
        self.model.load_weights(weights_path)
        print(f"Loaded weights from {weights_path}")
    
    def save_weights(self, weights_path: str):
        """
        Save model weights.
        
        Args:
            weights_path: Path to save weights
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        self.model.save_weights(weights_path)
        print(f"Saved weights to {weights_path}")
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for EfficientNet.
        
        Args:
            image: Input image (BGR or RGB)
            
        Returns:
            Preprocessed image tensor
        """
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assume BGR, convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Resize
        image_resized = cv2.resize(image_rgb, self.input_size)
        
        # Add batch dimension
        image_batch = np.expand_dims(image_resized, axis=0)
        
        # Use EfficientNet preprocessing
        image_preprocessed = preprocess_input(image_batch)
        
        return image_preprocessed
    
    def predict(self, image: np.ndarray, 
                return_top_k: int = 3) -> Dict:
        """
        Predict breed from image.
        
        Args:
            image: Input image
            return_top_k: Number of top predictions to return
            
        Returns:
            Dictionary with predictions and confidence
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Preprocess
        input_tensor = self.preprocess(image)
        
        # Predict
        predictions = self.model.predict(input_tensor, verbose=0)[0]
        
        # Get top K predictions
        top_k_indices = np.argsort(predictions)[::-1][:return_top_k]
        
        top_predictions = []
        for idx in top_k_indices:
            breed_name = IDX_TO_BREED.get(idx, f"Unknown_{idx}")
            confidence = float(predictions[idx])
            top_predictions.append({
                'breed': breed_name,
                'breed_id': int(idx),
                'confidence': confidence
            })
        
        # Determine action based on confidence
        top_confidence = top_predictions[0]['confidence']
        
        if top_confidence >= self.confidence_threshold:
            action = 'auto_confirm'
        elif top_confidence >= 0.60:
            action = 'flw_select'
        else:
            action = 'expert_review'
        
        return {
            'top_prediction': top_predictions[0],
            'top_k_predictions': top_predictions,
            'action_required': action,
            'confidence_level': self._get_confidence_level(top_confidence)
        }
    
    def _get_confidence_level(self, confidence: float) -> str:
        """
        Get confidence level description.
        
        Args:
            confidence: Confidence score
            
        Returns:
            Confidence level string
        """
        if confidence >= 0.90:
            return 'Very High'
        elif confidence >= 0.85:
            return 'High'
        elif confidence >= 0.70:
            return 'Medium'
        elif confidence >= 0.60:
            return 'Low'
        else:
            return 'Very Low'
    
    def train(self,
              train_data: tf.data.Dataset,
              val_data: tf.data.Dataset,
              epochs: int = 20,
              callbacks: Optional[List] = None):
        """
        Train the model.
        
        Args:
            train_data: Training dataset
            val_data: Validation dataset
            epochs: Number of training epochs
            callbacks: Optional list of callbacks
            
        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Default callbacks
        if callbacks is None:
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    min_lr=1e-7
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    'best_model.h5',
                    monitor='val_accuracy',
                    save_best_only=True,
                    mode='max'
                )
            ]
        
        history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=callbacks
        )
        
        return history
    
    def evaluate(self, test_data: tf.data.Dataset) -> Dict:
        """
        Evaluate model on test data.
        
        Args:
            test_data: Test dataset
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        results = self.model.evaluate(test_data, verbose=1)
        
        return {
            'loss': results[0],
            'accuracy': results[1],
            'top_k_accuracy': results[2] if len(results) > 2 else None
        }
    
    def export_to_tflite(self, output_path: str, quantize: bool = True):
        """
        Export model to TFLite format.
        
        Args:
            output_path: Output file path
            quantize: Apply quantization for smaller size
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        if quantize:
            # Apply full integer quantization
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                tf.lite.OpsSet.TFLITE_BUILTINS
            ]
        
        tflite_model = converter.convert()
        
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"Exported TFLite model to {output_path}")
        print(f"Model size: {len(tflite_model) / 1024 / 1024:.2f} MB")
    
    def export_to_tflite_int8(self, 
                               output_path: str,
                               representative_dataset: tf.data.Dataset = None,
                               enable_quantization: bool = True) -> str:
        """
        Export model to TFLite format with INT8 quantization.
        
        This creates an optimized edge-deployment-ready model with:
        - INT8 quantization for ~4x size reduction
        - DEFAULT optimization for latency improvements
        - Compatible with edge devices (mobile, TPU, NPU)
        
        Args:
            output_path: Output .tflite file path
            representative_dataset: Dataset for quantization calibration
                                    (required for INT8). Use tf.data.Dataset
                                    with batches of preprocessed images.
            enable_quantization: Whether to apply quantization
        
        Returns:
            Path to exported model
        
        Example:
            >>> # Create representative dataset from validation data
            >>> val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            ...     'data/val',
            ...     labels='inferred',
            ...     label_mode='categorical',
            ...     batch_size=32,
            ...     image_size=(224, 224)
            ... )
            >>> 
            >>> # Export with INT8 quantization
            >>> classifier.export_to_tflite_int8(
            ...     output_path='models/tflite/efficientnet_b0_int8.tflite',
            ...     representative_dataset=val_ds
            ... )
        """
        import tensorflow as tf
        
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        print(f"\n{'='*60}")
        print("Exporting EfficientNet-B0 to INT8 TFLite")
        print(f"{'='*60}")
        
        # Create converter
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # Apply optimizations
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        if enable_quantization:
            # Set quantization parameters for INT8
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                tf.lite.OpsSet.TFLITE_BUILTINS
            ]
            
            # Use full integer quantization
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            
            # Set representative dataset for calibration
            if representative_dataset is not None:
                def representative_data_gen():
                    count = 0
                    for images, _ in representative_dataset:
                        # Yield images for calibration
                        yield [images]
                        count += 1
                        if count >= 100:  # Calibrate with 100 batches
                            break
                
                converter.representative_dataset = representative_data_gen
                print("✓ Representative dataset configured for INT8 calibration")
            else:
                print("⚠ Warning: No representative dataset provided.")
                print("  Using float fallback quantization.")
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS,
                    tf.lite.OpsSet.TFLITE_BUILTINS_NET
                ]
        
        # Convert model
        tflite_model = converter.convert()
        
        # Save model
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        # Calculate sizes
        # Estimate original size (params * 4 bytes per float32)
        param_count = self.model.count_params()
        original_size_mb = param_count * 4 / (1024 * 1024)
        quantized_size_mb = len(tflite_model) / (1024 * 1024)
        compression_ratio = original_size_mb / quantized_size_mb if quantized_size_mb > 0 else 0
        
        print(f"\n{'='*60}")
        print("TFLite Export Summary")
        print(f"{'='*60}")
        print(f"Output path: {output_path}")
        print(f"Parameter count: {param_count:,}")
        print(f"Original model size: ~{original_size_mb:.1f} MB (estimated)")
        print(f"Quantized model size: {quantized_size_mb:.2f} MB")
        print(f"Compression ratio: {compression_ratio:.1f}x")
        print(f"Quantization: INT8" if enable_quantization else "Float16")
        print(f"{'='*60}\n")
        
        return output_path
    
    def get_model_summary(self):
        """Print model summary."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        return self.model.summary()


def create_classifier(weights_path: Optional[str] = None) -> EfficientNetClassifier:
    """
    Create and optionally load EfficientNet classifier.
    
    Args:
        weights_path: Path to pretrained weights
        
    Returns:
        EfficientNetClassifier instance
    """
    classifier = EfficientNetClassifier()
    classifier.build_model(pretrained=True)
    classifier.compile_model()
    
    if weights_path and os.path.exists(weights_path):
        classifier.load_weights(weights_path)
    
    return classifier
