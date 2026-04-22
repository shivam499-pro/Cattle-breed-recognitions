"""
Dataset Quality Evaluation Script for 41-Class Cattle Breed Model
Analyzes data/train_41/ folder for quality issues
"""

import os
import random
import hashlib
import json
from pathlib import Path
from collections import defaultdict

# Configuration
TRAIN_41_PATH = Path("data/train_41")
OUTPUT_REPORT = "dataset_quality_report.json"

# Image extensions to consider
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tif', '.tiff'}

# Irrelevant files to flag
IRRELEVANT_EXTS = {'.ini', '.txt', '.csv', '.json', '.xml', '.desktop'}

def get_image_files(folder_path):
    """Get all image files from a folder, excluding irrelevant files."""
    images = []
    irrelevant = []
    
    for file in os.listdir(folder_path):
        file_path = folder_path / file
        if file_path.is_file():
            ext = Path(file).suffix.lower()
            if ext in IMAGE_EXTS:
                images.append(file)
            elif ext in IRRELEVANT_EXTS or file.startswith('.'):
                irrelevant.append(file)
    
    return sorted(images), irrelevant

def calculate_file_hash(file_path):
    """Calculate MD5 hash of a file."""
    try:
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return None

def analyze_breed(breed_name, breed_path):
    """Analyze a single breed folder."""
    images, irrelevant = get_image_files(breed_path)
    
    # Get file sizes
    file_sizes = []
    hashes = defaultdict(list)
    
    for img in images:
        file_path = breed_path / img
        size = file_path.stat().st_size
        file_sizes.append((img, size))
        
        # Calculate hash for duplicate detection
        file_hash = calculate_file_hash(file_path)
        if file_hash:
            hashes[file_hash].append(img)
    
    # Find duplicates (same hash)
    duplicates = {h: files for h, files in hashes.items() if len(files) > 1}
    
    # Check for very small or large files (potential issues)
    small_files = [f for f, s in file_sizes if s < 5000]  # Less than 5KB
    large_files = [f for f, s in file_sizes if s > 50_000_000]  # Larger than 50MB
    
    # Randomly select 10 images for preview
    sample_images = []
    if len(images) >= 10:
        sample_images = random.sample(images, 10)
    else:
        sample_images = images
    
    # File type distribution
    ext_counts = defaultdict(int)
    for img in images:
        ext = Path(img).suffix.lower()
        ext_counts[ext] += 1
    
    return {
        'breed_name': breed_name,
        'total_images': len(images),
        'irrelevant_files': irrelevant,
        'file_type_distribution': dict(ext_counts),
        'sample_images': sample_images,
        'duplicates': {h: files for h, files in duplicates.items()},
        'small_files': small_files,
        'large_files': large_files,
        'min_size': min(file_sizes, key=lambda x: x[1])[1] if file_sizes else 0,
        'max_size': max(file_sizes, key=lambda x: x[1])[1] if file_sizes else 0,
        'avg_size': sum(s for _, s in file_sizes) // len(file_sizes) if file_sizes else 0
    }

def main():
    """Main function to analyze all breeds."""
    print("=" * 60)
    print("DATASET QUALITY EVALUATION - 41 CLASS CATTLE BREED MODEL")
    print("=" * 60)
    
    # Get all breed folders
    breeds = sorted([d for d in os.listdir(TRAIN_41_PATH) 
                   if (TRAIN_41_PATH / d).is_dir()])
    
    print(f"\nFound {len(breeds)} breeds:")
    for i, breed in enumerate(breeds, 1):
        print(f"  {i:2}. {breed}")
    
    # Analyze each breed
    results = []
    for breed in breeds:
        breed_path = TRAIN_41_PATH / breed
        print(f"\nAnalyzing: {breed}...", end=" ")
        result = analyze_breed(breed, breed_path)
        results.append(result)
        print(f"{result['total_images']} images")
    
    # Sort by number of images
    results_by_count = sorted(results, key=lambda x: x['total_images'], reverse=True)
    
    # Generate report
    print("\n" + "=" * 60)
    print("QUALITY ANALYSIS REPORT")
    print("=" * 60)
    
    # Per-breed statistics
    print("\n### PER-BREED STATISTICS (sorted by image count) ###\n")
    print(f"{'Breed':<25} {'Images':>8} {'Irrelevant':>10} {'Duplicates':>10} {'Small':>8} {'Type Dist':>30}")
    print("-" * 95)
    
    for r in results_by_count:
        dup_count = sum(len(files) for files in r['duplicates'].values())
        type_dist = ', '.join([f"{k}:{v}" for k, v in r['file_type_distribution'].items()])
        print(f"{r['breed_name']:<25} {r['total_images']:>8} {len(r['irrelevant_files']):>10} {dup_count:>10} {len(r['small_files']):>8} {type_dist[:28]:>30}")
    
    # Identify quality issues
    print("\n" + "=" * 60)
    print("IDENTIFIED ISSUES")
    print("=" * 60)
    
    # Breeds with no images
    no_images = [r for r in results if r['total_images'] == 0]
    if no_images:
        print("\n### BREEDS WITH NO IMAGES ###")
        for r in no_images:
            print(f"  - {r['breed_name']}")
    
    # Breeds with few images (< 50)
    few_images = [r for r in results if 0 < r['total_images'] < 50]
    if few_images:
        print("\n### BREEDS WITH FEW IMAGES (< 50) ###")
        for r in sorted(few_images, key=lambda x: x['total_images']):
            print(f"  - {r['breed_name']}: {r['total_images']} images")
    
    # Breeds with many irrelevant files
    irrelevant_issues = [r for r in results if len(r['irrelevant_files']) > 0]
    if irrelevant_issues:
        print("\n### BREEDS WITH IRRELEVANT FILES ###")
        for r in irrelevant_issues:
            print(f"  - {r['breed_name']}: {r['irrelevant_files']}")
    
    # Breeds with duplicates
    duplicate_issues = [r for r in results if r['duplicates']]
    if duplicate_issues:
        print("\n### BREEDS WITH DUPLICATE IMAGES ###")
        for r in duplicate_issues:
            dup_count = sum(len(files) for files in r['duplicates'].values())
            print(f"  - {r['breed_name']}: {dup_count} duplicates")
            for h, files in r['duplicates'].items():
                print(f"      {files}")
    
    # Breeds with small files
    small_file_issues = [r for r in results if r['small_files']]
    if small_file_issues:
        print("\n### BREEDS WITH SMALL FILES (< 5KB) ###")
        for r in small_file_issues:
            print(f"  - {r['breed_name']}: {r['small_files']}")
    
    # Ranking: most inconsistent (issues)
    print("\n" + "=" * 60)
    print("TOP 10 MOST INCONSISTENT BREEDS (by quality issues)")
    print("=" * 60)
    
    # Calculate inconsistency score
    for r in results:
        r['issues_score'] = (
            (1 if r['total_images'] < 50 else 0) * 2 +
            len(r['irrelevant_files']) +
            sum(len(files) for files in r['duplicates'].values()) +
            len(r['small_files'])
        )
    
    inconsistent_breeds = sorted(results, key=lambda x: x['issues_score'], reverse=True)[:10]
    
    print(f"\n{'Rank':<6} {'Breed':<25} {'Issues Score':>12} {'Issues':<40}")
    print("-" * 85)
    for i, r in enumerate(inconsistent_breeds, 1):
        issues = []
        if r['total_images'] < 50:
            issues.append(f"few imgs({r['total_images']})")
        if r['irrelevant_files']:
            issues.append(f"{len(r['irrelevant_files'])} irr")
        if r['duplicates']:
            issues.append(f"{sum(len(f) for f in r['duplicates'].values())} dup")
        if r['small_files']:
            issues.append(f"{len(r['small_files'])} small")
        
        issues_str = ', '.join(issues) if issues else 'OK'
        print(f"{i:<6} {r['breed_name']:<25} {r['issues_score']:>12} {issues_str:<40}")
    
    # Ranking: most clean breeds
    print("\n" + "=" * 60)
    print("TOP 10 MOST CLEAN BREEDS (fewest quality issues)")
    print("=" * 60)
    
    clean_breeds = sorted(results, key=lambda x: x['issues_score'])[:10]
    
    print(f"\n{'Rank':<6} {'Breed':<25} {'Issues Score':>12} {'Total Images':>15}")
    print("-" * 60)
    for i, r in enumerate(clean_breeds, 1):
        print(f"{i:<6} {r['breed_name']:<25} {r['issues_score']:>12} {r['total_images']:>15}")
    
    # Global statistics
    print("\n" + "=" * 60)
    print("GLOBAL STATISTICS")
    print("=" * 60)
    
    total_images = sum(r['total_images'] for r in results)
    total_irrelevant = sum(len(r['irrelevant_files']) for r in results)
    total_duplicates = sum(sum(len(files) for files in r['duplicates'].values()) for r in results)
    total_small = sum(len(r['small_files']) for r in results)
    
    avg_images = total_images // len(results) if results else 0
    
    print(f"\nTotal breeds: {len(results)}")
    print(f"Total images: {total_images}")
    print(f"Average images per breed: {avg_images}")
    print(f"Total irrelevant files: {total_irrelevant}")
    print(f"Total duplicate images: {total_duplicates}")
    print(f"Total small files: {total_small}")
    
    # Save detailed report to JSON
    report_data = {
        'summary': {
            'total_breeds': len(results),
            'total_images': total_images,
            'average_images_per_breed': avg_images,
            'total_irrelevant_files': total_irrelevant,
            'total_duplicates': total_duplicates,
            'total_small_files': total_small
        },
        'breeds': results,
        'top_10_inconsistent': [r['breed_name'] for r in inconsistent_breeds],
        'top_10_clean': [r['breed_name'] for r in clean_breeds]
    }
    
    with open(OUTPUT_REPORT, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\nDetailed report saved to: {OUTPUT_REPORT}")
    
    return results

if __name__ == "__main__":
    main()