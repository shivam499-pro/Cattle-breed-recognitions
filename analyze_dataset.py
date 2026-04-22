#!/usr/bin/env python3
"""Analyze dataset for simplification planning."""

import os
import json
from pathlib import Path

# Image extensions to count
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}

def count_images(folder_path):
    """Count images in a folder."""
    if not folder_path.exists():
        return 0
    
    count = 0
    for item in folder_path.iterdir():
        if item.is_file() and item.suffix.lower() in IMAGE_EXTS:
            count += 1
    return count

def main():
    base = Path('data')
    
    # Breeds to remove
    to_remove = ['Kherigarh', 'Umblachery', 'Kenkatha', 'Nimari', 'Nagori']
    
    # Merge pairs
    merges = [
        ('Red_Cattle', ['Red_Dane', 'Red_Sindhi']),
        ('South_Draft', ['Hallikar', 'Amritmahal']),
        ('Draft_Breed', ['Kankrej', 'Kangayam'])
    ]
    
    splits = ['train_41', 'val_41', 'test_41']
    
    # Collect counts per breed per split
    breed_counts = {}
    
    for split in splits:
        split_dir = base / split
        if not split_dir.exists():
            print(f"Warning: {split} does not exist")
            continue
            
        for breed_dir in sorted(split_dir.iterdir()):
            if breed_dir.is_dir():
                breed_name = breed_dir.name
                img_count = count_images(breed_dir)
                
                if breed_name not in breed_counts:
                    breed_counts[breed_name] = {'train': 0, 'val': 0, 'test': 0}
                
                # Map split name
                split_key = split.replace('_41', '')
                if split_key == 'train':
                    breed_counts[breed_name]['train'] = img_count
                elif split_key == 'val':
                    breed_counts[breed_name]['val'] = img_count
                else:
                    breed_counts[breed_name]['test'] = img_count
    
    # Print analysis
    print("=" * 80)
    print("CURRENT DATASET ANALYSIS")
    print("=" * 80)
    
    total_breeds = len(breed_counts)
    print(f"\nTotal breeds: {total_breeds}")
    
    # Print header
    print(f"\n{'Breed':<20} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    print("-" * 60)
    
    total_train = 0
    total_val = 0
    total_test = 0
    
    for breed in sorted(breed_counts.keys()):
        counts = breed_counts[breed]
        total = counts['train'] + counts['val'] + counts['test']
        print(f"{breed:<20} {counts['train']:>8} {counts['val']:>8} {counts['test']:>8} {total:>8}")
        
        total_train += counts['train']
        total_val += counts['val']
        total_test += counts['test']
    
    print("-" * 60)
    total_all = total_train + total_val + total_test
    print(f"{'TOTAL':<20} {total_train:>8} {total_val:>8} {total_test:>8} {total_all:>8}")
    
    # ========== ANALYSIS OF REMOVAL ==========
    print("\n" + "=" * 80)
    print("BREEDS PROPOSED FOR REMOVAL")
    print("=" * 80)
    
    remove_train = 0
    remove_val = 0
    remove_test = 0
    
    for breed in to_remove:
        if breed in breed_counts:
            counts = breed_counts[breed]
            removed = counts['train'] + counts['val'] + counts['test']
            print(f"{breed:<20} Train={counts['train']}, Val={counts['val']}, Test={counts['test']}, Total={removed}")
            remove_train += counts['train']
            remove_val += counts['val']
            remove_test += counts['test']
    
    print(f"\n{'TOTAL TO REMOVE:':<20} {remove_train:>8} {remove_val:>8} {remove_test:>8} {remove_train+remove_val+remove_test:>8}")
    
    # ========== ANALYSIS OF MERGES ==========
    print("\n" + "=" * 80)
    print("BREEDS PROPOSED FOR MERGE")
    print("=" * 80)
    
    for new_name, merge_from in merges:
        print(f"\nMerge: {new_name} <-- {merge_from}")
        
        m_train = 0
        m_val = 0
        m_test = 0
        
        for breed in merge_from:
            if breed in breed_counts:
                counts = breed_counts[breed]
                m_train += counts['train']
                m_val += counts['val']
                m_test += counts['test']
                print(f"  {breed:<18} Train={counts['train']}, Val={counts['val']}, Test={counts['test']}")
        
        print(f"  {'Combined total':<18} Train={m_train}, Val={m_val}, Test={m_test}, Total={m_train+m_val+m_test}")
    
    # ========== FINAL STATS ==========
    print("\n" + "=" * 80)
    print("FINAL DATASET ESTIMATE")
    print("=" * 80)
    
    final_breeds = total_breeds - len(to_remove) + len(merges)
    final_images = total_all - (remove_train + remove_val + remove_test)
    
    print(f"Original breeds: {total_breeds}")
    print(f"Breeds removed: {len(to_remove)}")
    print(f"Merge groups: {len(merges)}")
    print(f"Final breeds: {final_breeds}")
    print()
    print(f"Original images: {total_all}")
    print(f"Images removed: {remove_train + remove_val + remove_test}")
    print(f"Final images: {final_images}")
    
    # Check for low-image classes
    print("\n" + "=" * 80)
    print("CLASSES BELOW 100 IMAGES (after proposed changes)")
    print("=" * 80)
    
    low_image_classes = []
    for breed in sorted(breed_counts.keys()):
        if breed in to_remove:
            continue
            
        # Check if this breed is part of a merge
        merged = False
        for new_name, merge_from in merges:
            if breed in merge_from:
                merged = True
                break
        
        if merged:
            continue
            
        counts = breed_counts[breed]
        total = counts['train'] + counts['val'] + counts['test']
        if total < 100:
            low_image_classes.append((breed, total))
    
    if low_image_classes:
        for breed, total in low_image_classes:
            print(f"  {breed}: {total}")
    else:
        print("  No classes below 100 images")
    
    # Save the analysis
    analysis = {
        'original_breeds': total_breeds,
        'original_images': total_all,
        'breed_counts': breed_counts,
        'breeds_to_remove': to_remove,
        'merges': merges
    }
    
    with open('dataset_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print("\nAnalysis saved to dataset_analysis.json")

if __name__ == '__main__':
    main()