# test_model.py - Test YOLOv8-Nano on 3 different image sources
from ultralytics import YOLO
import os

# Load trained model
print("Loading model...")
model = YOLO('models/cattle_detector.pt')
print("✅ Model loaded!\n")

# Test images
test_images = [
    ('test_dataset.jpg', 'From Dataset'),
    ('test_google.jpg', 'From Google'),
    ('test_ai.png', 'AI Generated')
]

print("=" * 60)
print("TESTING MODEL ON 3 DIFFERENT IMAGE SOURCES")
print("=" * 60)

for img_path, source in test_images:
    print(f"\n📸 Test Image: {source}")
    print("-" * 40)
    
    if os.path.exists(img_path):
        # Run prediction
        results = model.predict(img_path, save=True, conf=0.5, verbose=False)
        
        # Print results
        for result in results:
            boxes = result.boxes
            num_cattle = len(boxes)
            
            if num_cattle > 0:
                print(f"   ✅ Detected: {num_cattle} cattle")
                for i, box in enumerate(boxes):
                    conf = box.conf[0]
                    print(f"      Cattle {i+1}: {conf:.1%} confidence")
            else:
                print(f"   ❌ No cattle detected")
    else:
        print(f"   ⚠️ Image not found: {img_path}")

print("\n" + "=" * 60)
print("✅ Testing complete!")
print("Check the 'runs/detect' folder for saved results")
print("=" * 60)
