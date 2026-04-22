#!/usr/bin/env python
"""
Split the 41-class cattle breed dataset into train/val/test sets.
- 70% train, 15% val, 15% test
- Shuffle images randomly for each breed
- Move (not copy) files to maintain data integrity
"""

import os
import random
import shutil
from pathlib import Path

# Set seed for reproducibility
random.seed(42)

# Paths
SOURCE_DIR = Path('data/train_41')
VAL_DIR = Path('data/val_41')
TEST_DIR = Path('data/test_41')

# Create destination directories
VAL_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)

# Get all breed folders
breeds = sorted([d for d in SOURCE_DIR.iterdir() if d.is_dir()])
print(f"Found {len(breeds)} breeds in {SOURCE_DIR}")
print("=" * 70)

# Results storage
results = []
total_source = 0
total_train = 0
total_val = 0
total_test = 0

# Process each breed
for breed_folder in breeds:
    breed_name = breed_folder.name
    
    # Get all images (files only)
    images = [f for f in breed_folder.iterdir() if f.is_file() and not f.name.startswith('.')]
    images.sort()  # Sort for consistency
    
    count = len(images)
    total_source += count
    
    if count == 0:
        print(f"WARNING: {breed_name} has NO images!")
        results.append({
            'breed': breed_name,
            'total': 0,
            'train': 0,
            'val': 0,
            'test': 0,
            'warning': 'No images'
        })
        continue
    
    # Shuffle images
    random.shuffle(images)
    
    # Calculate split indices
    train_end = int(count * 0.70)
    val_end = train_end + int(count * 0.15)
    
    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]
    
    # Handle edge case: ensure at least 1 in each set if count >= 3
    if count >= 3:
        # Ensure no empty sets
        if len(train_images) == 0:
            train_images = [val_images.pop(0)]
        if len(val_images) == 0:
            val_images = [test_images.pop(0)]
        if len(test_images) == 0:
            test_images = [val_images.pop()]
    elif count == 2:
        # 2 images: 1 train, 1 val (0 test)
        train_images = [images[0]]
        val_images = [images[1]]
        test_images = []
    elif count == 1:
        # 1 image: all to train
        train_images = images
        val_images = []
        test_images = []
    else:
        # 0 images
        train_images = []
        val_images = []
        test_images = []
    
    train_count = len(train_images)
    val_count = len(val_images)
    test_count = len(test_images)
    
    total_train += train_count
    total_val += val_count
    total_test += test_count
    
    # Create breed directories in val and test
    val_breed_dir = VAL_DIR / breed_name
    test_breed_dir = TEST_DIR / breed_name
    val_breed_dir.mkdir(parents=True, exist_ok=True)
    test_breed_dir.mkdir(parents=True, exist_ok=True)
    
    # Move files
    for img in train_images:
        # Keep in source (train_41)
        pass
    
    for img in val_images:
        shutil.move(str(img), str(val_breed_dir / img.name))
    
    for img in test_images:
        shutil.move(str(img), str(test_breed_dir / img.name))
    
    # Record result
    warning = None
    if count < 10:
        warning = f"Low count ({count})"
    if val_count == 0 or test_count == 0:
        warning = "Empty set" if warning is None else warning + ", empty set"
    
    results.append({
        'breed': breed_name,
        'total': count,
        'train': train_count,
        'val': val_count,
        'test': test_count,
        'warning': warning
    })

# Print results
print()
print("PER-CLASS DISTRIBUTION:")
print("-" * 70)
print(f"{'Breed':<25} {'Total':>6} {'Train':>6} {'Val':>6} {'Test':>6}  Notes")
print("-" * 70)

low_count_breeds = []
for r in results:
    notes = f" [{r['warning']}]" if r['warning'] else ""
    print(f"{r['breed']:<25} {r['total']:>6} {r['train']:>6} {r['val']:>6} {r['test']:>6}{notes}")
    if r['total'] < 10:
        low_count_breeds.append(r['breed'])

print("=" * 70)
print()
print("OVERALL SUMMARY:")
print(f"  Total source images: {total_source}")
print(f"  Train images (70%):  {total_train} ({total_train*100/total_source:.1f}%)")
print(f"  Val images (15%):   {total_val} ({total_val*100/total_source:.1f}%)")
print(f"  Test images (15%):  {total_test} ({total_test*100/total_source:.1f}%)")
print(f"  Sum:               {total_train + total_val + total_test}")
print()

# Verify counts
train_41_count = sum(len([f for f in (SOURCE_DIR / r['breed']).iterdir() if f.is_file()]) for r in results)
val_41_count = sum(len([f for f in (VAL_DIR / r['breed']).iterdir() if f.is_file()]) for r in results)
test_41_count = sum(len([f for f in (TEST_DIR / r['breed']).iterdir() if f.is_file()]) for r in results)

print("VERIFICATION:")
print(f"  Remaining in train_41: {train_41_count}")
print(f"  Moved to val_41:       {val_41_count}")
print(f"  Moved to test_41:     {test_41_count}")
print(f"  Total:                {train_41_count + val_41_count + test_41_count}")
print(f"  Data loss:            {total_source - (train_41_count + val_41_count + test_41_count)}")
print()

# Check all breeds exist in each set
print("CHECKING BREEDS EXIST IN ALL SETS:")
all_good = True
for r in results:
    breed = r['breed']
    train_exists = (SOURCE_DIR / breed).exists()
    val_exists = (VAL_DIR / breed).exists()
    test_exists = (TEST_DIR / breed).exists()
    
    if not (train_exists and val_exists and test_exists):
        print(f"  WARNING: {breed} missing in some sets! train:{train_exists} val:{val_exists} test:{test_exists}")
        all_good = False

if all_good:
    print("  All 41 breeds exist in train_41, val_41, and test_41")

if low_count_breeds:
    print()
    print("WARNINGS - Classes with low image count (<10):")
    for b in low_count_breeds:
        print(f"  - {b}")

print()
print("Script completed successfully!")