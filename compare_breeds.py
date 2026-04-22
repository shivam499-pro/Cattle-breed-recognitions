#!/usr/bin/env python3
"""Compare breed_mapping_v2.json with data/train folders."""

import json
import os
import shutil

def normalize(name):
    """Normalize breed name: lowercase, replace spaces/hyphens with underscores."""
    return name.lower().replace(' ', '_').replace('-', '_')

# Load mapping breeds
with open('models/breed_mapping_v2.json') as f:
    mapping = json.load(f)
mapping_breeds = {normalize(v): v for k, v in mapping.items()}  # normalized -> original

# Get train folders
train_dir = 'data/train'
train_folders = [f for f in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, f))]
train_folders_norm = {normalize(f): f for f in train_folders}

# Compare
matched = []
extra = []
missing = []

for norm_name, orig_name in mapping_breeds.items():
    if norm_name in train_folders_norm:
        matched.append((orig_name, train_folders_norm[norm_name]))
    elif norm_name == 'jaffrabadi' and 'jaffarabadi' in train_folders_norm:
        matched.append(('Jaffrabadi', 'Jaffarabadi'))
    else:
        missing.append((orig_name, norm_name))

for norm_name, orig_name in train_folders_norm.items():
    if norm_name not in mapping_breeds:
        if norm_name != 'jaffarabadi':  # already matched as jaffrabadi
            extra.append((orig_name, norm_name))

print("=" * 60)
print("BREED MATCHING REPORT")
print("=" * 60)
print(f"\n1. Source of Truth (breed_mapping_v2.json): {len(mapping_breeds)} breeds")
print(f"2. Dataset Folders (data/train): {len(train_folders)} folders")
print(f"\n3. Matching Results:")
print(f"   - Matched: {len(matched)}")
print(f"   - Extra (not in mapping): {len(extra)}")
print(f"   - Missing (in mapping but not in dataset): {len(missing)}")

print(f"\n   MATCHED ({len(matched)}):")
for m in sorted(matched, key=lambda x: x[0]):
    print(f"     {m[0]} <- {m[1]}")

if missing:
    print(f"\n   MISSING ({len(missing)}):")
    for m in sorted(missing, key=lambda x: x[0]):
        print(f"     {m[0]} (normalized: {m[1]})")

if extra:
    print(f"\n   EXTRA ({len(extra)}):")
    for e in sorted(extra, key=lambda x: x[0]):
        print(f"     {e[0]}")

# Decision
print("\n" + "=" * 60)
VALIDATION = len(matched) == 41
print(f"VALIDATION: {'PASS' if VALIDATION else 'FAIL'}")
print(f"   Expected: 41 matched")
print(f"   Actual: {len(matched)} matched")
if not VALIDATION:
    print("   >> CANNOT PROCEED - Mismatch detected")
print("=" * 60)