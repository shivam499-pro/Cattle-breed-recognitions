"""
Create v3 dataset by removing worst performing classes
"""
import os
import shutil
import json

# Classes to remove (worst performing from previous model)
REMOVE_CLASSES = ['Krishna_Valley', 'Red_Cattle', 'Nagpuri']

# Ensure all directories exist
os.makedirs('data/train_final_v3', exist_ok=True)
os.makedirs('data/val_final_v3', exist_ok=True)
os.makedirs('data/test_final_v3', exist_ok=True)

# Pairs: (source, destination)
pairs = [
    ('data/train_final_v2', 'data/train_final_v3'),
    ('data/val_final_v2', 'data/val_final_v3'),
    ('data/test_final_v2', 'data/test_final_v3')
]

# Process each split
for src_dir, dst_dir in pairs:
    print(f"\nProcessing {src_dir} -> {dst_dir}")
    
    # Get all classes from source
    if not os.path.exists(src_dir):
        print(f"  ERROR: Source {src_dir} does not exist!")
        continue
    
    classes = [c for c in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, c))]
    print(f"  Found {len(classes)} classes")
    
    for cls in classes:
        # Skip removed classes
        if cls in REMOVE_CLASSES:
            print(f"  REMOVING: {cls}")
            continue
        
        src_path = os.path.join(src_dir, cls)
        dst_path = os.path.join(dst_dir, cls)
        
        # Copy directory
        if os.path.exists(dst_path):
            print(f"  Already exists: {cls}")
        else:
            shutil.copytree(src_path, dst_path)
            # Count files
            num_files = len([f for f in os.listdir(dst_path) if os.path.isfile(os.path.join(dst_path, f))])
            print(f"  Copied: {cls} ({num_files} files)")

# Verify result
print("\n" + "="*50)
print("VERIFICATION")
print("="*50)

for split in ['train_final_v3', 'val_final_v3', 'test_final_v3']:
    path = f'data/{split}'
    classes = sorted([c for c in os.listdir(path) if os.path.isdir(os.path.join(path, c))])
    print(f"\n{split}: {len(classes)} classes")
    for cls in classes:
        num = len([f for f in os.listdir(os.path.join(path, cls)) if os.path.isfile(os.path.join(path, cls, f))])
        print(f"  {cls}: {num}")

# Create new mapping file
print("\n" + "="*50)
print("CREATING MAPPING")
print("="*50)

# Read old mapping
with open('models/breed_mapping_final_v2.json', 'r') as f:
    old_mapping = json.load(f)

# Create new mapping
new_classes = {}
new_stats = {}

# Original 25 classes in order
orig_classes = [old_mapping['classes'][str(i)] for i in range(25)]

for i, cls in enumerate(orig_classes):
    if cls not in REMOVE_CLASSES:
        new_classes[str(len(new_classes))] = cls
        if cls in old_mapping['stats']:
            new_stats[cls] = old_mapping['stats'][cls]

new_mapping = {
    'classes': new_classes,
    'stats': new_stats
}

with open('models/breed_mapping_final_v3.json', 'w') as f:
    json.dump(new_mapping, f, indent=2)

print(f"Created mapping with {len(new_classes)} classes")
print("Saved to models/breed_mapping_final_v3.json")

print("\nDone!")
