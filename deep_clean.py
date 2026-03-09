"""
Deep Clean Dataset
=================
Cleans the cattle breed dataset by:
1. Verifying Integrity - deletes corrupted or 0 KB images
2. Deduplicating - uses MD5 hashing to remove exact duplicates
3. Filtering Size - removes images smaller than 10KB
"""

import os
import hashlib
from pathlib import Path

# Settings
TRAIN_DIR = 'data/train'
MIN_SIZE_KB = 10


def get_file_hash(filepath):
    """Calculate MD5 hash of a file."""
    md5 = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                md5.update(chunk)
        return md5.hexdigest()
    except Exception:
        return None


def is_valid_image(filepath):
    """Check if image is valid (can be opened)."""
    try:
        from PIL import Image
        with Image.open(filepath) as img:
            img.verify()
        return True
    except Exception:
        return False


def get_file_size_kb(filepath):
    """Get file size in KB."""
    return os.path.getsize(filepath) / 1024


def clean_breed_folder(breed_folder):
    """Clean a single breed folder."""
    breed_name = os.path.basename(breed_folder)
    images = [f for f in os.listdir(breed_folder) 
              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not images:
        return 0, 0, 0
    
    deleted_corrupt = 0
    deleted_small = 0
    deleted_duplicate = 0
    
    # Phase 1: Check integrity and size
    valid_images = []
    for img_name in images:
        img_path = os.path.join(breed_folder, img_name)
        
        # Check file size
        size_kb = get_file_size_kb(img_path)
        if size_kb < MIN_SIZE_KB:
            os.remove(img_path)
            deleted_small += 1
            continue
        
        # Check if valid image
        if not is_valid_image(img_path):
            os.remove(img_path)
            deleted_corrupt += 1
            continue
        
        valid_images.append(img_path)
    
    # Phase 2: Deduplicate using MD5
    hashes = {}
    for img_path in valid_images:
        file_hash = get_file_hash(img_path)
        if file_hash is None:
            continue
            
        if file_hash in hashes:
            # Duplicate found
            os.remove(img_path)
            deleted_duplicate += 1
        else:
            hashes[file_hash] = img_path
    
    total_deleted = deleted_corrupt + deleted_small + deleted_duplicate
    
    if total_deleted > 0:
        print(f"  {breed_name}:")
        if deleted_corrupt > 0:
            print(f"    - Deleted {deleted_corrupt} corrupt images")
        if deleted_small > 0:
            print(f"    - Deleted {deleted_small} small images (<{MIN_SIZE_KB}KB)")
        if deleted_duplicate > 0:
            print(f"    - Deleted {deleted_duplicate} duplicates")
    
    return len(valid_images) - deleted_duplicate, total_deleted


def main():
    """Clean all breed folders."""
    print("=" * 60)
    print("DEEP CLEAN DATASET")
    print("=" * 60)
    print(f"Directory: {TRAIN_DIR}")
    print(f"Min size: {MIN_SIZE_KB} KB")
    print("=" * 60)
    
    if not os.path.exists(TRAIN_DIR):
        print(f"ERROR: {TRAIN_DIR} not found!")
        return
    
    breeds = [d for d in os.listdir(TRAIN_DIR) 
              if os.path.isdir(os.path.join(TRAIN_DIR, d))]
    
    print(f"Breeds to clean: {len(breeds)}")
    print()
    
    total_before = 0
    total_after = 0
    total_deleted = 0
    
    for breed in sorted(breeds):
        breed_folder = os.path.join(TRAIN_DIR, breed)
        images_before = len([f for f in os.listdir(breed_folder) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        
        images_after, deleted = clean_breed_folder(breed_folder)
        
        total_before += images_before
        total_after += images_after
        total_deleted += deleted
        
        if deleted > 0:
            print(f"  [{images_before} → {images_after}] (-{deleted})")
    
    print()
    print("=" * 60)
    print("CLEAN COMPLETE!")
    print("=" * 60)
    print(f"Images before: {total_before}")
    print(f"Images deleted: {total_deleted}")
    print(f"Images after: {total_after}")
    print(f"Space saved: {total_deleted} images")


if __name__ == '__main__':
    main()
