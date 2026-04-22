"""
Model Testing Script
Tests all trained classification models and reports their accuracy.
"""
import os
import sys
import numpy as np
from pathlib import Path
import json

# Setup paths
PROJECT_DIR = Path("c:/Project Design thinking/cattle-breed-recognition")
sys.path.insert(0, str(PROJECT_DIR))

# TensorFlow setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

# Breed mappings
BREED_MAPPING_24 = {
    0: 'Gir', 1: 'Sahiwal', 2: 'Red Sindhi', 3: 'Tharparkar', 4: 'Rathi',
    5: 'Hariana', 6: 'Kankrej', 7: 'Ongole', 8: 'Deoni',
    9: 'Hallikar', 10: 'Amritmahal', 11: 'Khillari', 12: 'Kangayam', 13: 'Bargur',
    14: 'Dangi', 15: 'Krishna Valley', 16: 'Malnad Gidda', 17: 'Punganur', 18: 'Vechur',
    19: 'Pulikulam', 20: 'Umblachery', 21: 'Toda', 22: 'Kalahandi',
    23: 'Murrah'
}

with open(PROJECT_DIR / "models/breed_mapping_v2.json", "r") as f:
    BREED_MAPPING_41 = json.load(f)

API_BREEDS = sorted([
    'Alambadi', 'Amritmahal', 'Ayrshire', 'Banni', 'Bargur', 'Bhadawari',
    'Brahman', 'Brahman Cross', 'Brown_Swiss', 'Chhattisgarhi', 'Chilika', 'Cholistani',
    'Cholistani Cross', 'Dangi', 'Deoni', 'Dhani', 'Fresian Cross', 'Gir',
    'Gojri', 'Guernsey', 'Hallikar', 'Hariana', 'Holstein_Friesian', 'Jaffarabadi',
    'Jersey', 'Kalahandi', 'Kangayam', 'Kankrej', 'Kasargod', 'Kenkatha',
    'Kherigarh', 'Khillari', 'Krishna_Valley', 'Luit', 'Malnad_gidda', 'Marathwada',
    'Mehsana', 'Murrah', 'Nagori', 'Nagpuri', 'Nili-ravi', 'Nili_Ravi', 'Nimari',
    'Ongole', 'Pandharpuri', 'Pulikulam', 'Rathi', 'Red_Dane', 'Red_Sindhi',
    'Sahiwal', 'Sahiwal Cross', 'Sibbi', 'Surti', 'Tharparkar', 'Toda',
    'Umblachery', 'Vechur', 'luit_(swamp)', 'marathwadi', 'unidentified (mixed)'
])
BREED_MAPPING_60 = {i: b for i, b in enumerate(API_BREEDS)}


def get_train_folders():
    """Get list of training folders."""
    train_dir = PROJECT_DIR / "data" / "train"
    folders = [f for f in train_dir.iterdir() if f.is_dir()]
    return sorted(folders)


def load_test_images(num_per_class=1):
    """Load a small sample of test images from training data."""
    train_folders = get_train_folders()
    images = []
    labels = []
    
    for folder in train_folders[:10]:  # Limit to first 10 classes
        breed_name = folder.name
        image_files = list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")) + list(folder.glob("*.png"))
        if image_files:
            # Get first image
            img_path = image_files[0]
            try:
                img = keras_image.load_img(img_path, target_size=(224, 224))
                img_array = keras_image.img_to_array(img)
                img_array = img_array / 255.0
                images.append(img_array)
                labels.append(breed_name)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    
    return np.array(images), labels


def test_model(model_path, model_name):
    """Test a single model."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"Path: {model_path}")
    print(f"{'='*60}")
    
    try:
        # Load model
        model = load_model(model_path, compile=False)
        
        # Get model config
        print(f"\nModel loaded successfully!")
        print(f"Input shape: {model.input_shape}")
        
        # Try prediction
        test_images, test_labels = load_test_images()
        
        if len(test_images) == 0:
            print("No test images found!")
            return None
        
        # Predict
        predictions = model.predict(test_images, verbose=0)
        num_classes = predictions.shape[1]
        
        print(f"Output classes: {num_classes}")
        print(f"Test samples: {len(test_images)}")
        
        # Show sample predictions
        print(f"\nSample predictions:")
        for i in range(min(3, len(test_images))):
            pred_idx = np.argmax(predictions[i])
            confidence = predictions[i][pred_idx]
            
            # Try to map to known breeds
            if num_classes == 24:
                breed = BREED_MAPPING_24.get(pred_idx, f"Unknown_{pred_idx}")
            elif num_classes == 41:
                breed = BREED_MAPPING_41.get(str(pred_idx), f"Unknown_{pred_idx}")
            elif num_classes == 60:
                breed = BREED_MAPPING_60.get(pred_idx, f"Unknown_{pred_idx}")
            else:
                breed = f"Unknown_{pred_idx}"
            
            print(f"  {test_labels[i]} → {breed} ({confidence*100:.1f}%)")
        
        return {
            "name": model_name,
            "path": str(model_path),
            "num_classes": num_classes,
            "input_shape": str(model.input_shape),
            "sample_predictions": list(zip(test_labels[:3], [np.argmax(p) for p in predictions[:3]]))
        }
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Test all models."""
    print("="*60)
    print("CATTLE BREED MODEL TESTING")
    print("="*60)
    
    models_to_test = [
        ("models/breed_classifier.keras", "breed_classifier.keras"),
        ("models/breed_classifier_v2.keras", "breed_classifier_v2.keras"),
        ("models/cattle_breed_pro_v1.h5", "cattle_breed_pro_v1.h5"),
    ]
    
    results = []
    
    for model_path, model_name in models_to_test:
        full_path = PROJECT_DIR / model_path
        if full_path.exists():
            result = test_model(full_path, model_name)
            if result:
                results.append(result)
        else:
            print(f"Model not found: {full_path}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    for r in results:
        print(f"\n{r['name']}:")
        print(f"  Classes: {r['num_classes']}")
        print(f"  Input: {r['input_shape']}")
        
        # Match with known mappings
        if r['num_classes'] == 24:
            print(f"  Mapping: preprocessing.py (24 classes)")
        elif r['num_classes'] == 41:
            print(f"  Mapping: breed_mapping_v2.json (41 classes)")
        elif r['num_classes'] == 60:
            print(f"  Mapping: api/app.py (60+ classes)")
        else:
            print(f"  Mapping: UNKNOWN")


if __name__ == "__main__":
    main()