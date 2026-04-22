"""Comprehensive model testing script"""
import os
import sys
import json
import numpy as np
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress warnings

print("=" * 70)
print("CATTLE BREED CLASSIFICATION MODEL TESTING")
print("=" * 70)

# Imports
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

PROJECT_DIR = Path("c:/Project Design thinking/cattle-breed-recognition")
MODELS_DIR = PROJECT_DIR / "models"
TRAIN_DIR = PROJECT_DIR / "data" / "train"

# Load breed mappings
BREED_MAPPING_12 = {
    0: 'brahman', 1: 'brahman_cross', 2: 'cholistani', 3: 'cholistani_cross',
    4: 'dhani', 5: 'fresian', 6: 'fresian_cross', 7: 'kankarej',
    8: 'sahiwal', 9: 'sahiwal_cross', 10: 'sibbi', 11: 'unidentified_(mixed)'
}

with open(MODELS_DIR / "breed_mapping_v2.json", "r") as f:
    BREED_MAPPING_41 = json.load(f)

API_BREEDS = sorted([
    'Alambadi', 'Amritmahal', 'Ayrshire', 'Banni', 'Bargur', 'Bhadawari',
    'Brahman', 'Brahman_Cross', 'Brown_Swiss', 'Chhattisgarhi', 'Chilika', 'Cholistani',
    'Cholistani_Cross', 'Dangi', 'Deoni', 'Dhani', 'Fresian_Cross', 'Gir',
    'Gojri', 'Guernsey', 'Hallikar', 'Hariana', 'Holstein_Friesian', 'Jaffarabadi',
    'Jersey', 'Kalahandi', 'Kangayam', 'Kankrej', 'Kasargod', 'Kenkatha',
    'Kherigarh', 'Khillari', 'Krishna_Valley', 'Luit', 'Malnad_gidda', 'Marathwada',
    'Mehsana', 'Murrah', 'Nagori', 'Nagpuri', 'Nili-ravi', 'Nili_Ravi', 'Nimari',
    'Ongole', 'Pandharpuri', 'Pulikulam', 'Rathi', 'Red_Dane', 'Red_Sindhi',
    'Sahiwal', 'Sahiwal_Cross', 'Sibbi', 'Surti', 'Tharparkar', 'Toda',
    'Umblachery', 'Vechur', 'luit_(swamp)', 'marathwadi', 'unidentified_(mixed)'
])
BREED_MAPPING_60 = {i: b for i, b in enumerate(API_BREEDS)}


def load_test_images(max_per_class=3, max_classes=10):
    """Load test images from training data."""
    images = []
    labels = []
    filenames = []
    
    # Get all breed folders
    breed_folders = sorted([f for f in TRAIN_DIR.iterdir() if f.is_dir()])
    
    print(f"Found {len(breed_folders)} breed folders in train")
    
    count = 0
    for folder in breed_folders[:max_classes]:
        breed_name = folder.name
        
        # Look for jpg files
        image_files = list(folder.glob("*.jpg")) + list(folder.glob("*.JPG"))
        
        # Take up to max_per_class images
        for img_path in image_files[:max_per_class]:
            try:
                img = keras_image.load_img(str(img_path), target_size=(224, 224))
                img_array = keras_image.img_to_array(img)
                img_array = img_array / 255.0
                images.append(img_array)
                labels.append(breed_name)
                filenames.append(img_path.name)
                count += 1
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        
        if count >= max_classes * max_per_class:
            break
    
    print(f"Loaded {len(images)} images from {len(set(labels))} classes")
    return np.array(images), labels, filenames


def test_model(model, images, labels, filenames, num_classes):
    """Test a model and calculate accuracy."""
    # Process images one by one to avoid Keras 3 issues
    predictions = []
    for i in range(len(images)):
        img = np.expand_dims(images[i], axis=0)
        pred = model.predict(img, verbose=0)
        predictions.append(pred[0])
    
    predictions = np.array(predictions)
    
    correct = 0
    top3_correct = 0
    total = len(images)
    
    sample_preds = []
    
    for i in range(total):
        pred_probs = predictions[i]
        pred_idx = np.argmax(pred_probs)
        top3_idx = np.argsort(pred_probs)[-3:][::-1]
        
        true_label = labels[i]
        
        # Map prediction to breed name
        if num_classes == 12:
            pred_breed = BREED_MAPPING_12.get(pred_idx, f"Unknown_{pred_idx}")
            top3_breeds = [BREED_MAPPING_12.get(idx, f"Unknown_{idx}") for idx in top3_idx]
        elif num_classes == 41:
            pred_breed = BREED_MAPPING_41.get(str(pred_idx), f"Unknown_{pred_idx}")
            top3_breeds = [BREED_MAPPING_41.get(str(idx), f"Unknown_{idx}") for idx in top3_idx]
        else:  # 60
            pred_breed = BREED_MAPPING_60.get(pred_idx, f"Unknown_{pred_idx}")
            top3_breeds = [BREED_MAPPING_60.get(idx, f"Unknown_{idx}") for idx in top3_idx]
        
        # Check if correct (with normalization)
        normalized_true = true_label.replace(' ', '_').replace('-', '_').lower()
        normalized_pred = pred_breed.replace(' ', '_').replace('-', '_').lower()
        
        is_correct = False
        if normalized_true == normalized_pred:
            correct += 1
            is_correct = True
        
        # Top-3 check
        for tb in top3_breeds:
            normalized_tb = tb.replace(' ', '_').replace('-', '_').lower()
            if normalized_true == normalized_tb:
                top3_correct += 1
                break
        
        # Save sample predictions
        if i < 5:
            sample_preds.append({
                'file': filenames[i],
                'true': true_label,
                'pred': pred_breed,
                'conf': float(pred_probs[pred_idx]),
            })
    
    accuracy = (correct / total) * 100 if total > 0 else 0
    top3_accuracy = (top3_correct / total) * 100 if total > 0 else 0
    
    return {
        'accuracy': accuracy,
        'top3_accuracy': top3_accuracy,
        'total': total,
        'correct': correct,
        'top3_correct': top3_correct,
        'samples': sample_preds
    }


# Load test images
print("\nLoading test images from training data...")
images, labels, filenames = load_test_images(max_per_class=3, max_classes=10)

# Models to test
models_config = [
    ("breed_classifier.keras", "breed_classifier.keras", 12),
    ("breed_classifier_v2.keras", "breed_classifier_v2.keras", 41),
    ("cattle_breed_pro_v1.h5", "cattle_breed_pro_v1.h5", 60),
]

results = []

for model_file, model_name, expected_classes in models_config:
    print(f"\n{'='*70}")
    print(f"Testing: {model_name}")
    print(f"{'='*70}")
    
    model_path = MODELS_DIR / model_file
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        continue
    
    try:
        model = load_model(model_path, compile=False)
        print(f"[OK] Model loaded successfully!")
        
        input_shape = model.input_shape
        output_shape = model.output_shape
        num_classes = output_shape[-1]
        
        print(f"  Input shape: {input_shape}")
        print(f"  Output shape: {output_shape}")
        print(f"  Number of classes: {num_classes}")
        
        if len(images) > 0:
            # Test on images
            result = test_model(model, images, labels, filenames, num_classes)
            
            print(f"\n  Validation Metrics:")
            print(f"    Total test samples: {result['total']}")
            print(f"    Top-1 Accuracy: {result['accuracy']:.1f}% ({result['correct']}/{result['total']})")
            print(f"    Top-3 Accuracy: {result['top3_accuracy']:.1f}% ({result['top3_correct']}/{result['total']})")
            
            print(f"\n  Sample Predictions:")
            for s in result['samples'][:3]:
                status = "CORRECT" if s['true'].lower().replace('_', '') == s['pred'].lower().replace('_', '') else "WRONG"
                print(f"    {s['file']} -> {s['pred']} ({s['conf']*100:.1f}%) [True: {s['true']}] [{status}]")
            
            results.append({
                'name': model_name,
                'path': str(model_path),
                'num_classes': num_classes,
                'input_shape': str(input_shape),
                'accuracy': result['accuracy'],
                'top3_accuracy': result['top3_accuracy'],
                'samples': result['samples']
            })
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

# Match with known mappings
print(f"\n{'='*70}")
print("FINAL SUMMARY")
print(f"{'='*70}")

for r in results:
    num_classes = r['num_classes']
    mapping_match = "Unknown"
    if num_classes == 12:
        mapping_match = "breed_mapping.json (12 classes)"
    elif num_classes == 41:
        mapping_match = "breed_mapping_v2.json (41 classes)"
    elif num_classes == 60:
        mapping_match = "api/app.py (60 classes)"
    
    print(f"\n{r['name']}:")
    print(f"  Number of classes: {r['num_classes']}")
    print(f"  Model input size: {r['input_shape']}")
    print(f"  Validation accuracy: {r['accuracy']:.1f}%")
    print(f"  Top-3 accuracy: {r['top3_accuracy']:.1f}%")
    print(f"  Breed mapping: {mapping_match}")

print("\n" + "=" * 70)
print("Done!")
print("=" * 70)