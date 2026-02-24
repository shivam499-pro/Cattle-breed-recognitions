"""
Two-Stage Cattle Breed Recognition Pipeline Test
================================================
Stage 1: YOLOv8-Nano - Detect cattle in image
Stage 2: EfficientNet-B0 - Classify breed

Author: SIH 2025 Team
"""

import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# TensorFlow for EfficientNet
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# Ultralytics for YOLO
from ultralytics import YOLO


class CattleBreedPipeline:
    """Two-stage pipeline for cattle breed recognition."""
    
    def __init__(self, yolo_path, classifier_path, mapping_path):
        """
        Initialize the pipeline with both models.
        
        Args:
            yolo_path: Path to YOLOv8-Nano model (.pt file)
            classifier_path: Path to EfficientNet-B0 model (.keras file)
            mapping_path: Path to breed mapping JSON file
        """
        print("=" * 60)
        print("Initializing Two-Stage Pipeline...")
        print("=" * 60)
        
        # Load YOLO detector
        print("\n📦 Loading YOLOv8-Nano (Stage 1: Detection)...")
        self.yolo_model = YOLO(yolo_path)
        print("   ✅ YOLOv8-Nano loaded successfully!")
        
        # Load breed classifier
        print("\n📦 Loading EfficientNet-B0 (Stage 2: Classification)...")
        self.classifier = load_model(classifier_path)
        print("   ✅ EfficientNet-B0 loaded successfully!")
        
        # Load breed mapping
        print("\n📋 Loading breed mapping...")
        with open(mapping_path, 'r') as f:
            self.breed_mapping = json.load(f)
        self.idx_to_breed = {int(k): v for k, v in self.breed_mapping.items()}
        print(f"   ✅ Loaded {len(self.idx_to_breed)} breed classes:")
        for idx, breed in sorted(self.idx_to_breed.items()):
            print(f"      {idx}: {breed}")
        
        print("\n" + "=" * 60)
        print("✅ Pipeline Ready!")
        print("=" * 60)
    
    def detect_cattle(self, image_path, conf_threshold=0.5):
        """
        Stage 1: Detect cattle in image using YOLOv8-Nano.
        
        Args:
            image_path: Path to input image
            conf_threshold: Minimum confidence for detection
            
        Returns:
            List of bounding boxes [(x1, y1, x2, y2, confidence), ...]
        """
        # Run YOLO detection
        results = self.yolo_model.predict(image_path, conf=conf_threshold, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                detections.append((x1, y1, x2, y2, conf))
        
        return detections
    
    def classify_breed(self, image_crop):
        """
        Stage 2: Classify breed using EfficientNet-B0.
        
        Args:
            image_crop: PIL Image or numpy array of cattle crop
            
        Returns:
            Tuple of (breed_name, confidence, top_3_predictions)
        """
        # Convert to PIL Image if numpy array
        if isinstance(image_crop, np.ndarray):
            image_crop = Image.fromarray(image_crop)
        
        # Resize to EfficientNet input size
        image_resized = image_crop.resize((224, 224))
        
        # Convert to array and preprocess
        img_array = np.array(image_resized)
        if img_array.shape[-1] == 4:  # RGBA to RGB
            img_array = img_array[:, :, :3]
        img_array = preprocess_input(img_array)
        img_batch = np.expand_dims(img_array, axis=0)
        
        # Predict
        predictions = self.classifier.predict(img_batch, verbose=0)[0]
        
        # Get top 3 predictions
        top_3_indices = np.argsort(predictions)[-3:][::-1]
        top_3 = [(self.idx_to_breed[idx], predictions[idx]) for idx in top_3_indices]
        
        # Get best prediction
        best_idx = np.argmax(predictions)
        best_breed = self.idx_to_breed[best_idx]
        best_conf = predictions[best_idx]
        
        return best_breed, best_conf, top_3
    
    def process_image(self, image_path, conf_threshold=0.5, save_result=True):
        """
        Process an image through the complete two-stage pipeline.
        
        Args:
            image_path: Path to input image
            conf_threshold: Minimum confidence for YOLO detection
            save_result: Whether to save the result image
            
        Returns:
            Dictionary with detection and classification results
        """
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(image_path)}")
        print(f"{'='*60}")
        
        # Load image
        image = Image.open(image_path)
        image_np = np.array(image)
        
        # Stage 1: Detect cattle
        print("\n🔍 Stage 1: Detecting cattle...")
        detections = self.detect_cattle(image_path, conf_threshold)
        
        if not detections:
            print("   ❌ No cattle detected!")
            return {"success": False, "error": "No cattle detected"}
        
        print(f"   ✅ Detected {len(detections)} cattle!")
        
        # Process each detection
        results = []
        for i, (x1, y1, x2, y2, det_conf) in enumerate(detections):
            print(f"\n   🐄 Processing cattle {i+1}/{len(detections)}...")
            
            # Crop the detected region
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            crop = image_np[y1:y2, x1:x2]
            
            # Stage 2: Classify breed
            print(f"      🔬 Stage 2: Classifying breed...")
            breed, conf, top_3 = self.classify_breed(crop)
            
            print(f"      ✅ Predicted: {breed} ({conf*100:.1f}% confidence)")
            print(f"      📊 Top 3 predictions:")
            for b, c in top_3:
                print(f"         • {b}: {c*100:.1f}%")
            
            # Determine confidence level
            if conf >= 0.85:
                level = "HIGH (Auto-confirm)"
            elif conf >= 0.60:
                level = "MEDIUM (FLW select from top 3)"
            else:
                level = "LOW (Expert review required)"
            
            results.append({
                "bbox": (x1, y1, x2, y2),
                "detection_confidence": float(det_conf),
                "breed": breed,
                "breed_confidence": float(conf),
                "top_3": [(b, float(c)) for b, c in top_3],
                "confidence_level": level
            })
        
        # Visualize and save result
        if save_result:
            self._visualize_result(image_path, results)
        
        return {"success": True, "results": results}
    
    def _visualize_result(self, image_path, results, output_dir="pipeline_results"):
        """Create visualization of detection and classification results."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Load image
        image = Image.open(image_path)
        
        # Create figure
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image)
        
        # Draw each detection
        colors = ['#00FF00', '#FF6600', '#0066FF', '#FF00FF']
        for i, result in enumerate(results):
            x1, y1, x2, y2 = result["bbox"]
            color = colors[i % len(colors)]
            
            # Draw bounding box
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=3, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label
            label = f"{result['breed']} ({result['breed_confidence']*100:.1f}%)"
            ax.text(
                x1, y1-10, label,
                fontsize=12, fontweight='bold',
                color='white', backgroundcolor=color,
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.8)
            )
        
        # Add title
        ax.set_title(
            f"Two-Stage Pipeline Result\n"
            f"Detected: {len(results)} cattle",
            fontsize=14, fontweight='bold'
        )
        ax.axis('off')
        
        # Save
        output_path = os.path.join(output_dir, f"result_{os.path.basename(image_path)}")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n   📸 Result saved to: {output_path}")


def main():
    """Test the two-stage pipeline with sample images."""
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    
    YOLO_PATH = os.path.join(MODELS_DIR, "cattle_detector.pt")
    CLASSIFIER_PATH = os.path.join(MODELS_DIR, "breed_classifier.keras")
    MAPPING_PATH = os.path.join(MODELS_DIR, "breed_mapping.json")
    
    # Check if all model files exist
    for path, name in [(YOLO_PATH, "YOLO model"), 
                       (CLASSIFIER_PATH, "Classifier model"),
                       (MAPPING_PATH, "Breed mapping")]:
        if not os.path.exists(path):
            print(f"❌ Error: {name} not found at {path}")
            return
    
    # Initialize pipeline
    pipeline = CattleBreedPipeline(YOLO_PATH, CLASSIFIER_PATH, MAPPING_PATH)
    
    # Test images
    test_images = [
        ("test_dataset.jpg", "From Dataset"),
        ("test_google.jpg", "From Google"),
        ("test_ai.png", "AI Generated")
    ]
    
    print("\n" + "=" * 60)
    print("🧪 TESTING TWO-STAGE PIPELINE")
    print("=" * 60)
    
    for img_file, source in test_images:
        img_path = os.path.join(BASE_DIR, img_file)
        if os.path.exists(img_path):
            print(f"\n📸 Test Image: {source}")
            result = pipeline.process_image(img_path)
            
            if result["success"]:
                print(f"\n✅ SUCCESS!")
                for r in result["results"]:
                    print(f"   🐄 Breed: {r['breed']}")
                    print(f"   📊 Confidence: {r['breed_confidence']*100:.1f}%")
                    print(f"   📋 Level: {r['confidence_level']}")
        else:
            print(f"\n⚠️ Test image not found: {img_file}")
    
    print("\n" + "=" * 60)
    print("✅ Pipeline Testing Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
