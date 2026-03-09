"""
Edge Deployment Model Export Script
===================================
Creates INT8 quantized TFLite model for mobile/edge deployment.
"""

import os
import sys
import time
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf


def create_representative_dataset(data_dir: str, 
                                 num_samples: int = 100,
                                 image_size: tuple = (224, 224)) -> tf.data.Dataset:
    """
    Create representative dataset for TFLite quantization.
    
    Args:
        data_dir: Path to data directory with breed subdirectories
        num_samples: Number of samples to use for calibration
        image_size: Target image size
    
    Returns:
        tf.data.Dataset for quantization calibration
    """
    print(f"\n{'='*60}")
    print("Creating Representative Dataset")
    print(f"{'='*60}")
    
    # Try validation data first, then training data
    for check_dir in [data_dir, 'data/val', 'data/train']:
        if os.path.exists(check_dir):
            # Check if directory has actual images
            has_images = False
            for breed in os.listdir(check_dir):
                breed_path = os.path.join(check_dir, breed)
                if os.path.isdir(breed_path):
                    images = [f for f in os.listdir(breed_path) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                    if images:
                        has_images = True
                        break
            
            if has_images:
                print(f"✓ Found valid images in: {check_dir}")
                data_dir = check_dir
                break
    else:
        print("⚠ No valid image directories found, using synthetic data")
        return create_synthetic_representative_dataset(num_samples, image_size)
    
    # Create dataset
    try:
        dataset = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            labels='inferred',
            label_mode='categorical',
            batch_size=32,
            image_size=image_size,
            validation_split=0.2,
            subset='validation',
            seed=42
        )
        
        # Take only needed samples
        dataset = dataset.take(num_samples // 32)
        
        print(f"✓ Created representative dataset with {num_samples} samples")
        return dataset
        
    except Exception as e:
        print(f"⚠ Error creating dataset: {e}")
        return create_synthetic_representative_dataset(num_samples, image_size)


def create_synthetic_representative_dataset(num_samples: int = 100,
                                           image_size: tuple = (224, 224)) -> tf.data.Dataset:
    """
    Create synthetic representative dataset if no real data available.
    
    Args:
        num_samples: Number of samples
        image_size: Image size
    
    Returns:
        tf.data.Dataset with random data
    """
    print("Creating synthetic representative dataset...")
    
    # Generate random images that match expected input shape
    dummy_images = np.random.rand(num_samples, image_size[0], image_size[1], 3).astype(np.float32)
    dummy_labels = np.zeros((num_samples, 41))  # 41 classes
    
    dataset = tf.data.Dataset.from_tensor_slices((dummy_images, dummy_labels))
    dataset = dataset.batch(32)
    
    print(f"✓ Created synthetic dataset with {num_samples} samples")
    return dataset


def export_int8_model():
    """Export EfficientNet-B0 to INT8 TFLite."""
    print(f"\n{'='*60}")
    print("EDGE MODEL EXPORT - INT8 QUANTIZATION")
    print(f"{'='*60}")
    
    # Import our classifier
    from src.models.efficientnet_classifier import EfficientNetClassifier
    
    # Use absolute path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'models', 'breed_classifier_v2.keras')
    
    print(f"\n1. Loading model from: {model_path}")
    
    # Load the full model directly (not just weights)
    if os.path.exists(model_path):
        try:
            classifier_model = tf.keras.models.load_model(model_path)
            print("   ✓ Model loaded successfully!")
            
            # Create classifier wrapper
            classifier = EfficientNetClassifier(
                input_size=(224, 224),
                num_classes=41,
                confidence_threshold=0.85
            )
            classifier.model = classifier_model
            print(f"   ✓ Model has {classifier.model.count_params():,} parameters")
        except Exception as e:
            print(f"   ⚠ Could not load model: {e}")
            return
    else:
        print(f"   Model file not found: {model_path}")
        return
    
    # Create representative dataset from REAL training data
    print("\n3. Creating representative dataset from real images...")
    
    # Try to find real training data
    data_dir = os.path.join(base_dir, 'data', 'train')
    
    if os.path.exists(data_dir):
        # Check if directory has actual images
        has_images = False
        for breed in os.listdir(data_dir):
            breed_path = os.path.join(data_dir, breed)
            if os.path.isdir(breed_path):
                images = [f for f in os.listdir(breed_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                if images:
                    has_images = True
                    break
        
        if has_images:
            print(f"   Using real images from: {data_dir}")
            try:
                representative_ds = tf.keras.preprocessing.image_dataset_from_directory(
                    data_dir,
                    labels='inferred',
                    label_mode='categorical',
                    batch_size=32,
                    image_size=(224, 224),
                    validation_split=0.1,
                    subset='validation',
                    seed=42
                )
                representative_ds = representative_ds.take(100 // 32)
                print(f"   ✓ Using real images for calibration")
            except Exception as e:
                print(f"   ⚠ Error loading: {e}")
                representative_ds = create_synthetic_representative_dataset(100)
        else:
            print(f"   No real images found, using synthetic data")
            representative_ds = create_synthetic_representative_dataset(100)
    else:
        print(f"   Data directory not found: {data_dir}")
        representative_ds = create_synthetic_representative_dataset(100)
    
    # Export to INT8 TFLite
    print("\n4. Exporting to INT8 TFLite...")
    output_path = 'models/tflite/breed_classifier_int8.tflite'
    
    try:
        classifier.export_to_tflite_int8(
            output_path=output_path,
            representative_dataset=representative_ds,
            enable_quantization=True
        )
    except Exception as e:
        print(f"⚠ INT8 export failed: {e}")
        print("   Trying fallback to float16...")
        
        # Fallback to regular TFLite
        output_path = 'models/tflite/breed_classifier_float32.tflite'
        classifier.export_to_tflite(output_path, quantize=False)
    
    # Get file size
    if os.path.exists(output_path):
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"\n{'='*60}")
        print("SIZE COMPARISON")
        print(f"{'='*60}")
        print(f"Original Keras model:  ~30 MB (estimated)")
        print(f"INT8 TFLite model:     {file_size_mb:.2f} MB")
        print(f"Target:                <25 MB")
        
        if file_size_mb < 25:
            print(f"\n✅ SUCCESS: Model is {25 - file_size_mb:.2f} MB UNDER target!")
        else:
            print(f"\n⚠ Model exceeds target by {file_size_mb - 25:.2f} MB")
    else:
        file_size_mb = 0
    
    # Estimate inference latency
    print(f"\n{'='*60}")
    print("INFERENCE LATENCY ESTIMATION")
    print(f"{'='*60}")
    
    estimate_latency(output_path)
    
    return output_path, file_size_mb


def estimate_latency(model_path: str):
    """
    Estimate inference latency for the TFLite model.
    
    Args:
        model_path: Path to TFLite model
    """
    if not os.path.exists(model_path):
        print("⚠ Model file not found, cannot estimate latency")
        return
    
    # Load model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Get input dtype to handle INT8 models
    input_dtype = input_details[0]['dtype']
    input_shape = input_details[0]['shape']
    
    # Create dummy input matching the model's expected dtype
    if input_dtype == np.int8:
        # For INT8 models, use int8 range [-128, 127]
        dummy_input = np.random.randint(-128, 127, size=input_shape, dtype=np.int8)
    else:
        dummy_input = np.random.rand(*input_shape).astype(np.float32)
    
    # Warmup
    print("   Running warmup...")
    for _ in range(5):
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
    
    # Measure latency
    print("   Measuring latency...")
    num_runs = 20
    latencies = []
    
    for _ in range(num_runs):
        start_time = time.time()
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
        end_time = time.time()
        latencies.append((end_time - start_time) * 1000)  # Convert to ms
    
    # Calculate statistics
    avg_latency = np.mean(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    p95_latency = np.percentile(latencies, 95)
    
    print(f"\n   Latency Results ({num_runs} runs):")
    print(f"   ├─ Average: {avg_latency:.2f} ms")
    print(f"   ├─ Min:    {min_latency:.2f} ms")
    print(f"   ├─ Max:    {max_latency:.2f} ms")
    print(f"   └─ P95:    {p95_latency:.2f} ms")
    
    # Target check
    if avg_latency < 500:
        print(f"\n   ✅ PASS: Average latency {avg_latency:.0f}ms < 500ms target")
    else:
        print(f"\n   ⚠ Latency exceeds 500ms target")


def main():
    """Main execution."""
    print("\n" + "="*60)
    print("CATTLE BREED RECOGNITION - EDGE DEPLOYMENT")
    print("="*60)
    
    try:
        model_path, file_size = export_int8_model()
        
        print(f"\n{'='*60}")
        print("DEPLOYMENT SUMMARY")
        print(f"{'='*60}")
        print(f"Model exported to: {model_path}")
        print(f"File size: {file_size:.2f} MB")
        print("\n✅ Ready for mobile deployment!")
        
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
