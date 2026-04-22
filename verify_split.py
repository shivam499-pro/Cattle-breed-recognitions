#!/usr/bin/env python
"""Verify the split dataset structure and counts"""
from pathlib import Path

train_dir = Path('data/train_41')
val_dir = Path('data/val_41')
test_dir = Path('data/test_41')

# Get breeds in each directory
train_breeds = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
val_breeds = sorted([d.name for d in val_dir.iterdir() if d.is_dir()])
test_breeds = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])

print(f"Train_41 breeds: {len(train_breeds)}")
print(f"Val_41 breeds: {len(val_breeds)}")
print(f"Test_41 breeds: {len(test_breeds)}")
print()

# Check all breeds exist in all sets
all_match = train_breeds == val_breeds == test_breeds
print(f"All 41 breeds in all sets: {all_match}")
print()

# Get common breeds
common = sorted(set(train_breeds) & set(val_breeds) & set(test_breeds))
print(f"Common breeds ({len(common)}): {common}")
print()

# Count total images in each set
train_total = sum(len([f for f in (train_dir / b).iterdir() if f.is_file()]) for b in train_breeds)
val_total = sum(len([f for f in (val_dir / b).iterdir() if f.is_file()]) for b in val_breeds)
test_total = sum(len([f for f in (test_dir / b).iterdir() if f.is_file()]) for b in test_breeds)

print(f"Train images: {train_total}")
print(f"Val images: {val_total}")
print(f"Test images: {test_total}")
print(f"Total: {train_total + val_total + test_total}")