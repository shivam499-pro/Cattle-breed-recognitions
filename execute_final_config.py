#!/usr/bin/env python3
"""
Execute FINAL dataset transformation with full traceability.

STEPS:
1. Remove 5 breeds (Kherigarh, Umblachery, Kenkatha, Nimari, Nagori)
2. Merge Red_Dane + Red_Sindhi → Red_Cattle
3. Copy remaining to final directories
4. Create mapping file
5. Validate and report
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime

# Configuration
BASE_DIR = Path('data')
SPLITS = ['train_41', 'val_41', 'test_41']

# Classes to REMOVE completely
REMOVE_BREEDS = ['Kherigarh', 'Umblachery', 'Kenkatha', 'Nimari', 'Nagori']

# Classes to MERGE
MERGE_PAIRS = [
    ('Red_Cattle', ['Red_Dane', 'Red_Sindhi'])
]

# Image extensions
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}


def count_images(folder):
    """Count images in a folder."""
    if not folder.exists():
        return 0
    count = 0
    for f in folder.iterdir():
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
            count += 1
    return count


def copy_folder(src, dst):
    """Copy folder with all images."""
    if not src.exists():
        return 0
    
    dst.mkdir(parents=True, exist_ok=True)
    
    count = 0
    for f in src.iterdir():
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
            shutil.copy2(f, dst / f.name)
            count += 1
    return count


def analyze_current():
    """Analyze current dataset."""
    print("=" * 80)
    print("STEP 1: ANALYZING CURRENT DATASET")
    print("=" * 80)
    
    stats = {}
    
    for split in SPLITS:
        base = BASE_DIR / split
        if not base.exists():
            continue
            
        stats[split] = {}
        
        for breed_dir in sorted(base.iterdir()):
            if breed_dir.is_dir():
                breed = breed_dir.name
                if breed in REMOVE_BREEDS:
                    continue
                    
                count = count_images(breed_dir)
                if count > 0:
                    stats[split][breed] = count
    
    # Calculate totals
    totals = {}
    for split, breeds in stats.items():
        for breed, count in breeds.items():
            if breed not in totals:
                totals[breed] = 0
            totals[breed] += count
    
    # Print analysis
    print("\nCurrent class distribution (excluding removed):")
    print(f"{'Breed':<22} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    print("-" * 60)
    
    for breed in sorted(totals.keys()):
        train = stats.get('train_41', {}).get(breed, 0)
        val = stats.get('val_41', {}).get(breed, 0)
        test = stats.get('test_41', {}).get(breed, 0)
        print(f"{breed:<22} {train:>8} {val:>8} {test:>8} {totals[breed]:>8}")
    
    print(f"\nTotal (before transformation): {sum(totals.values())} images")
    
    return stats, totals


def execute_removal(stats):
    """Track removal of breeds."""
    print("\n" + "=" * 80)
    print("STEP 2: REMOVED BREEDS (for audit)")
    print("=" * 80)
    
    removed_stats = {}
    
    for split in SPLITS:
        base = BASE_DIR / split
        if not base.exists():
            continue
            
        removed_stats[split] = {}
        
        for breed in REMOVE_BREEDS:
            breed_dir = base / breed
            if breed_dir.exists():
                count = count_images(breed_dir)
                removed_stats[split][breed] = count
    
    print("\nRemoved breeds with image counts:")
    print(f"{'Breed':<22} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    print("-" * 60)
    
    totals = {}
    for breed in REMOVE_BREEDS:
        train = removed_stats.get('train_41', {}).get(breed, 0)
        val = removed_stats.get('val_41', {}).get(breed, 0)
        test = removed_stats.get('test_41', {}).get(breed, 0)
        print(f"{breed:<22} {train:>8} {val:>8} {test:>8} {train+val+test:>8}")
        totals[breed] = train + val + test
    
    return removed_stats


def execute_merge_and_copy():
    """Execute merge and copy to final directories."""
    print("\n" + "=" * 80)
    print("STEP 3: EXECUTING TRANSFORMATION")
    print("=" * 80)
    
    final_stats = {}
    merge_details = {}
    
    for split in SPLITS:
        source = BASE_DIR / split
        target = BASE_DIR / f'{split.replace("_41", "_final")}'
        
        print(f"\nProcessing: {split} → {split.replace('_41', '_final')}")
        
        target.mkdir(parents=True, exist_ok=True)
        final_stats[split] = {}
        
        # Get all breed directories from source
        for breed_dir in sorted(source.iterdir()):
            if not breed_dir.is_dir():
                continue
            
            breed_name = breed_dir.name
            
            # Skip removed breeds
            if breed_name in REMOVE_BREEDS:
                print(f"  SKIP (removed): {breed_name}")
                continue
            
            # Handle merge: Red_Dane + Red_Sindhi -> Red_Cattle
            if breed_name in ['Red_Dane', 'Red_Sindhi']:
                if 'Red_Cattle' not in merge_details:
                    merge_details['Red_Cattle'] = {
                        'sources': {},
                        'total': {'train': 0, 'val': 0, 'test': 0}
                    }
                
                split_key = split.replace('_41', '')
                count = count_images(breed_dir)
                
                merge_details['Red_Cattle']['sources'][breed_name] = {
                    split_key: count
                }
                merge_details['Red_Cattle']['total'][split_key] = (
                    merge_details['Red_Cattle']['total'][split_key] + count
                )
                
                # Copy to Red_Cattle folder
                red_cattle_dir = target / 'Red_Cattle'
                copied = copy_folder(breed_dir, red_cattle_dir)
                
                if 'Red_Cattle' not in final_stats[split]:
                    final_stats[split]['Red_Cattle'] = 0
                final_stats[split]['Red_Cattle'] += copied
                
                print(f"  MERGE: {breed_name} → Red_Cattle ({copied} images)")
                continue
            
            # Regular copy
            target_breed = target / breed_name
            count = copy_folder(breed_dir, target_breed)
            final_stats[split][breed_name] = count
            
            print(f"  COPY: {breed_name} ({count} images)")
    
    return final_stats, merge_details


def create_mapping(final_stats):
    """Create breed_mapping_final.json."""
    print("\n" + "=" * 80)
    print("STEP 4: CREATING breed_mapping_final.json")
    print("=" * 80)
    
    # Get all unique breeds from final stats
    all_breeds = set()
    for split in final_stats.values():
        all_breeds.update(split.keys())
    
    # Sort and assign indices
    sorted_breeds = sorted(all_breeds)
    
    mapping = {}
    for i, breed in enumerate(sorted_breeds):
        mapping[str(i)] = breed
    
    # Save
    mapping_path = Path('models/breed_mapping_final.json')
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(mapping_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"\nSaved to: {mapping_path}")
    print(f"Total classes: {len(mapping)}")
    
    return mapping, sorted_breeds


def validate(final_stats, sorted_breeds, merge_details):
    """Validate all constraints."""
    print("\n" + "=" * 80)
    print("STEP 5: VALIDATION")
    print("=" * 80)
    
    errors = []
    
    # Calculate totals
    totals = {}
    for split, breeds in final_stats.items():
        for breed, count in breeds.items():
            if breed not in totals:
                totals[breed] = 0
            totals[breed] += count
    
    # Rule 1: Class count 25-30
    class_count = len(totals)
    print(f"\n1. Class count check: {class_count}")
    if 25 <= class_count <= 30:
        print("   ✓ PASS")
    else:
        errors.append(f"Class count {class_count} not in 25-30 range")
    
    # Rule 2: Each class in train, val, test
    print("\n2. All classes in train/val/test:")
    valid_classes = []
    for breed in sorted_breeds:
        in_all = all(breed in final_stats[s] for s in final_stats)
        if in_all:
            valid_classes.append(breed)
        else:
            errors.append(f"{breed} missing in some splits")
    
    print(f"   {len(valid_classes)}/{len(sorted_breeds)} classes in all splits")
    if len(valid_classes) == len(sorted_breeds):
        print("   ✓ PASS")
    
    # Rule 3: No class < 100 images
    print("\n3. Minimum class size check:")
    small_classes = [(b, c) for b, c in totals.items() if c < 100]
    if small_classes:
        for b, c in small_classes:
            print(f"   ✗ {b}: {c}")
        errors.append(f"Classes below 100: {small_classes}")
    else:
        print(f"   ✓ PASS (min: {min(totals.values())})")
    
    # Rule 4: Verify Red_Cattle merge
    print("\n4. Merge verification:")
    red_cattle_total = totals.get('Red_Cattle', 0)
    expected = merge_details['Red_Cattle']['total']
    print(f"   Red_Cattle: train={expected['train']}, val={expected['val']}, test={expected['test']}, total={red_cattle_total}")
    
    if red_cattle_total == expected['train'] + expected['val'] + expected['test']:
        print("   ✓ PASS")
    else:
        errors.append("Red_Cattle merge count mismatch")
    
    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    if errors:
        print("\n✗ ERRORS FOUND:")
        for e in errors:
            print(f"  - {e}")
        return False
    else:
        print("\n✓ ALL VALIDATIONS PASSED")
        return True


def generate_report(final_stats, totals, removed_stats, merge_details, sorted_breeds, mapping):
    """Generate final audit report."""
    print("\n" + "=" * 80)
    print("FINAL AUDIT REPORT")
    print("=" * 80)
    
    # 1. Final class list
    print("\n1. FINAL CLASS LIST:")
    for i, breed in enumerate(sorted_breeds):
        print(f"   {i}: {breed}")
    
    # 2. Removed classes
    print("\n2. REMOVED CLASSES:")
    for breed in REMOVE_BREEDS:
        total = sum(removed_stats[s].get(breed, 0) for s in removed_stats)
        print(f"   {breed}: {total} images removed")
    
    # 3. Merge details
    print("\n3. MERGE DETAILS:")
    print(f"   Red_Cattle <- Red_Dane + Red_Sindhi")
    for src in ['Red_Dane', 'Red_Sindhi']:
        for split in ['train', 'val', 'test']:
            count = merge_details['Red_Cattle']['sources'][src].get(split, 0)
            print(f"      {src} ({split}): {count}")
    print(f"   Total in Red_Cattle: {totals['Red_Cattle']}")
    
    # 4. Per-class distribution
    print("\n4. PER-CLASS DISTRIBUTION:")
    print(f"{'Breed':<22} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    print("-" * 60)
    for breed in sorted(totals.keys()):
        train = final_stats.get('train_final', {}).get(breed, 0)
        val = final_stats.get('val_final', {}).get(breed, 0)
        test = final_stats.get('test_final', {}).get(breed, 0)
        print(f"{breed:<22} {train:>8} {val:>8} {test:>8} {totals[breed]:>8}")
    
    # 5. Dataset summary
    print("\n5. DATASET SUMMARY:")
    total_images = sum(totals.values())
    print(f"   Total classes: {len(totals)}")
    print(f"   Total images: {total_images}")
    
    # 6. Validation result
    print("\n6. VALIDATION:")
    print("   ✓ All constraints satisfied")
    
    # 7. Change log
    print("\n7. CHANGE LOG:")
    print(f"   [{datetime.now().isoformat()}]")
    print("   - Removed 5 breeds: Kherigarh, Umblachery, Kenkatha, Nimari, Nagori")
    print("   - Merged Red_Dane + Red_Sindhi -> Red_Cattle")
    print("   - Created data/train_final, data/val_final, data/test_final")
    print("   - Created models/breed_mapping_final.json")
    
    # Save report to file
    report = {
        'timestamp': datetime.now().isoformat(),
        'final_classes': sorted_breeds,
        'removed_classes': {b: sum(removed_stats[s].get(b, 0) for s in removed_stats) for b in REMOVE_BREEDS},
        'merge_details': merge_details,
        'per_class_distribution': totals,
        'summary': {
            'total_classes': len(totals),
            'total_images': total_images
        },
        'validation': 'PASSED'
    }
    
    with open('final_dataset_audit.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n   Report saved to: final_dataset_audit.json")


def main():
    """Execute main transformation."""
    print("=" * 80)
    print("FINAL DATASET TRANSFORMATION")
    print("Controlled execution with full traceability")
    print("=" * 80)
    
    # Step 1: Analyze
    stats, totals = analyze_current()
    
    # Step 2: Track removal
    removed_stats = execute_removal(stats)
    
    # Step 3: Execute transformation
    final_stats, merge_details = execute_merge_and_copy()
    
    # Step 4: Create mapping
    mapping, sorted_breeds = create_mapping(final_stats)
    
    # Step 5: Validate
    valid = validate(final_stats, sorted_breeds, merge_details)
    
    if valid:
        # Step 6: Generate report
        final_totals = {}
        for split, breeds in final_stats.items():
            for breed, count in breeds.items():
                if breed not in final_totals:
                    final_totals[breed] = 0
                final_totals[breed] += count
        
        generate_report(
            final_stats, final_totals, 
            removed_stats, merge_details,
            sorted_breeds, mapping
        )
        
        print("\n" + "=" * 80)
        print("✓ TRANSFORMATION COMPLETE")
        print("=" * 80)
    else:
        print("\n✗ VALIDATION FAILED - STOPPING")


if __name__ == '__main__':
    main()