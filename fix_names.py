#!/usr/bin/env python3
"""Final cleanup: rename Jaffarabadi and merge Nili-ravi."""

import os
import shutil

train_dir = 'data/train_41'

# List current folders
folders = [f for f in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, f))]
print(f"Current folders: {len(folders)}")
print("Folders:", sorted(folders))

# 1. Rename Jaffarabadi -> Jaffrabadi
jaffarabadi = os.path.join(train_dir, 'Jaffarabadi')
jaffrabadi = os.path.join(train_dir, 'Jaffrabadi')

if os.path.exists(jaffarabadi):
    os.rename(jaffarabadi, jaffrabadi)
    print("\n1. Renamed: Jaffarabadi -> Jaffrabadi")
else:
    print(f"\n1. Jaffarabadi not found (exists: {os.path.exists(jaffarabadi)})")

# 2. Check for Nili-ravi in train (not train_41)
nili_ravi_train = os.path.join('data/train', 'Nili-ravi')
nili_ravi_41 = os.path.join(train_dir, 'Nili-ravi')

for src_path in [nili_ravi_train, nili_ravi_41]:
    if os.path.exists(src_path):
        nili_ravi = os.path.join(train_dir, 'Nili_Ravi')
        if os.path.exists(nili_ravi):
            # Count images
            alt_images = os.listdir(src_path)
            img_count = len([f for f in alt_images if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))])
            print(f"\n2. Found {img_count} images in {src_path}")
            
            # Move images
            for img in alt_images:
                src = os.path.join(src_path, img)
                dst = os.path.join(nili_ravi, img)
                if os.path.isfile(src):
                    shutil.move(src, dst)
            
            # Delete empty folder
            os.rmdir(src_path)
            print(f"   Merged images to Nili_Ravi, deleted empty folder")

# 3. Final verification
folders = [f for f in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, f))]
print(f"\n3. Final folder count: {len(folders)}")
print(f"   Expected: 41")
print(f"   Status: {'PASS' if len(folders) == 41 else 'FAIL'}")

# 4. Check for mismatches
mapping_names = set()
with open('models/breed_mapping_v2.json') as f:
    import json
    data = json.load(f)
    mapping_names = set(v for v in data.values())

folder_names = set(os.listdir(train_dir))
missing = mapping_names - folder_names
extra = folder_names - mapping_names

if missing:
    print(f"\n4. Missing from train_41: {missing}")
if extra:
    print(f"\n4. Extra in train_41: {extra}")
if not missing and not extra:
    print("\n4. All folder names match mapping: YES")