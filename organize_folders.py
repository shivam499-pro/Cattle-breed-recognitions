#!/usr/bin/env python3
"""Move matched folders to train_41, extra folders to extra_19."""

import json
import os
import shutil

def normalize(name):
    """Normalize breed name."""
    return name.lower().replace(' ', '_').replace('-', '_')

# Load mapping
with open('models/breed_mapping_v2.json') as f:
    mapping = json.load(f)
mapping_breeds = {normalize(v): v for k, v in mapping.items()}

# Get train folders
train_dir = 'data/train'
train_folders = [f for f in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, f))]
train_folders_norm = {normalize(f): f for f in train_folders}

# Identify matched and extra
matched_src = []
matched_dst = []
extra_src = []
extra_dst = []

# Handle matches
src_dir = 'data/train'
dst_train = 'data/train_41'
dst_extra = 'data/extra_19'

for norm_name, orig_name in mapping_breeds.items():
    if norm_name in train_folders_norm:
        folder_name = train_folders_norm[norm_name]
        matched_src.append(os.path.join(src_dir, folder_name))
        matched_dst.append(os.path.join(dst_train, folder_name))
    elif norm_name == 'jaffrabadi' and 'jaffarabadi' in train_folders_norm:
        # Special case: Jaffrabadi maps from Jaffarabadi (spelling variation - no rename)
        folder_name = 'Jaffarabadi'
        matched_src.append(os.path.join(src_dir, folder_name))
        matched_dst.append(os.path.join(dst_train, folder_name))
        print(f"WARNING: {folder_name} used for Jaffrabadi (spelling variation)")

for norm_name, orig_name in train_folders_norm.items():
    if norm_name not in mapping_breeds and norm_name != 'jaffarabadi':
        folder_name = orig_name
        extra_src.append(os.path.join(src_dir, folder_name))
        extra_dst.append(os.path.join(dst_extra, folder_name))

# Move matched folders
print("\n=== MOVING TO train_41 ===")
for src in matched_src:
    folder = os.path.basename(src)
    # Get destination with renamed folder for Jaffrabadi
    if 'jaffrabadi' in matched_dst[matched_src.index(src)]:
        dst = os.path.join(dst_train, 'Jaffrabadi')
    else:
        dst = os.path.join(dst_train, folder)
    shutil.move(src, dst)
    print(f"  Moved: {folder} -> {os.path.basename(dst)}")

print(f"\n=== MOVING TO extra_19 ===")
for src in extra_src:
    folder = os.path.basename(src)
    dst = os.path.join(dst_extra, folder)
    shutil.move(src, dst)
    print(f"  Moved: {folder}")

# Verify
print("\n=== VERIFICATION ===")
train_41_count = len(os.listdir(dst_train))
extra_19_count = len(os.listdir(dst_extra))
print(f"data/train_41: {train_41_count} folders")
print(f"data/extra_19: {extra_19_count} folders")
print(f"\nExpected: train_41=41, extra_19=18")
print(f"Actual: train_41={train_41_count}, extra_19={extra_19_count}")
if train_41_count == 41 and extra_19_count == 18:
    print("\n✓ VALIDATION PASSED")
else:
    print("\n✗ VALIDATION FAILED")