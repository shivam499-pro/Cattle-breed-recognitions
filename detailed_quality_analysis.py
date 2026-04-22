"""
Detailed Dataset Quality Evaluation - Extended Analysis
Checks filename patterns, format diversity, and generates visual inspection lists
"""

import os
import random
import json
from pathlib import Path
from collections import Counter, defaultdict

TRAIN_41_PATH = Path("data/train_41")
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tif', '.tiff'}

def get_images_only(folder_path):
    """Get only image files."""
    images = []
    for file in os.listdir(folder_path):
        if (folder_path / file).is_file():
            ext = Path(file).suffix.lower()
            if ext in IMAGE_EXTS:
                images.append(file)
    return sorted(images)

def analyze_breed_detailed(breed_path, breed_name):
    """Detailed analysis of a breed folder."""
    images = get_images_only(breed_path)
    
    # Check filename patterns
    base_names = [Path(f).stem.lower() for f in images]
    
    # Check naming patterns - are images named after breed?
    name_pattern_matches = sum(1 for n in base_names if breed_name.lower().replace('_', '') in n.replace('_', ''))
    
    # Check for numbered files vs named files
    numbered_files = [f for f in base_names if any(c.isdigit() for c in f)]
    named_files = [f for f in base_names if not any(c.isdigit() for c in f)]
    
    # Format diversity
    formats = Counter([Path(f).suffix.lower() for f in images])
    unique_formats = len(formats)
    
    # Check for GIF files (often lower quality/animated)
    gif_count = formats.get('.gif', 0)
    
    # Check for consistent naming scheme
    name_counter = Counter()
    for f in images:
        # Extract prefix (e.g., "sahiwal" from "sahiwal_001.jpg")
        prefix = ''
        for i, c in enumerate(f):
            if c.isdigit():
                prefix = f[:i].lower()
                break
        else:
            prefix = Path(f).stem.lower()
        
        # Also check full stem
        stem = Path(f).stem.lower()
        # Try to extract breed name from filename
        for known_breed in ['sahiwal', 'nagpuri', 'gir', 'holstein', 'fresian', 'murrah', 
                          'jersey', 'kankrej', 'banni', 'jaffrabadi', 'alambadi',
                          'bargur', 'bhadawari', 'brown', 'swiss', 'dangi', 'deoni',
                          'hallikar', 'hariana', 'kangayam', 'kasargod', 'kenkatha',
                          'kherigarh', 'khillari', 'krishna', 'malnad', 'mehsana',
                          'nagori', 'nili', 'ravi', 'nimari', 'ongole', 'pulikulam',
                          'rathi', 'red', 'dane', 'sindhi', 'surti', 'tharparkar',
                          'toda', 'umblachery', 'vechur', 'amritmahal', 'ayrshire',
                          'guernsey']:
            if known_breed in stem:
                prefix = known_breed
                break
        name_counter[prefix] += 1
    
    # Check if dominant naming prefix matches breed name
    dominant_prefix = name_counter.most_common(1)[0][0] if name_counter else ''
    name_quality = 'OK' if breed_name.lower().replace('_', '').replace(' ', '').replace('-', '').startswith(dominant_prefix.replace('_', '')) else 'CHECK'
    
    # Random samples for visual inspection
    random.seed(42)  # For reproducibility
    sample_10 = random.sample(images, min(10, len(images))) if len(images) >= 10 else images
    
    return {
        'breed': breed_name,
        'total_images': len(images),
        'unique_formats': unique_formats,
        'format_breakdown': dict(formats),
        'gif_count': gif_count,
        'filename_quality': name_quality,
        'dominant_prefix': dominant_prefix,
        'name_pattern_matches': name_pattern_matches,
        'numbered_count': len(numbered_files),
        'named_count': len(named_files),
        'sample_10': sample_10,
        'all_image_files': images
    }

def main():
    # Get all breeds
    breeds = sorted([d for d in os.listdir(TRAIN_41_PATH) 
                     if (TRAIN_41_PATH / d).is_dir()])
    
    print("=" * 70)
    print("DETAILED DATASET QUALITY ANALYSIS - EXTENDED")
    print("=" * 70)
    print(f"\nAnalyzing {len(breeds)} breeds...\n")
    
    results = []
    for breed in breeds:
        result = analyze_breed_detailed(TRAIN_41_PATH / breed, breed)
        results.append(result)
        print(f"  {breed}: {result['total_images']} images, {result['unique_formats']} formats, {result['filename_quality']}")
    
    # Detailed issues report
    print("\n" + "=" * 70)
    print("DETAILED ISSUES ANALYSIS")
    print("=" * 70)
    
    # 1. Format Diversity Issues (multiple formats = different sources)
    multi_format = [r for r in results if r['unique_formats'] > 1]
    if multi_format:
        print("\n### BREEDS WITH MULTIPLE IMAGE FORMATS (mixed sources) ###")
        for r in sorted(multi_format, key=lambda x: x['unique_formats'], reverse=True):
            print(f"  {r['breed']}: {r['format_breakdown']}")
    
    # 2. GIF files (often lower quality)
    has_gif = [r for r in results if r['gif_count'] > 0]
    if has_gif:
        print("\n### BREEDS WITH GIF FILES (potential quality issues) ###")
        for r in sorted(has_gif, key=lambda x: x['gif_count'], reverse=True):
            print(f"  {r['breed']}: {r['gif_count']} GIF files")
    
    # 3. Filename quality issues
    filename_issues = [r for r in results if r['filename_quality'] == 'CHECK']
    if filename_issues:
        print("\n### BREEDS WITH FILENAME MISMATCHES ###")
        for r in filename_issues:
            print(f"  {r['breed']}: Expected prefix-like name, got '{r['dominant_prefix']}'")
    
    # 4. Low image counts
    low_count = [r for r in results if r['total_images'] < 60]
    if low_count:
        print("\n### BREEDS WITH LOW IMAGE COUNTS (< 60) ###")
        for r in sorted(low_count, key=lambda x: x['total_images']):
            print(f"  {r['breed']}: {r['total_images']} images")
    
    # Calculate consistency scores
    print("\n" + "=" * 70)
    print("CONSISTENCY SCORING")
    print("=" * 70)
    
    for r in results:
        score = 100  # Start with perfect score
        
        # Deduct for issues
        if r['total_images'] < 50:
            score -= 20
        elif r['total_images'] < 80:
            score -= 10
        elif r['total_images'] < 100:
            score -= 5
        
        if r['unique_formats'] > 1:
            score -= (r['unique_formats'] - 1) * 5
        
        if r['gif_count'] > 0:
            score -= r['gif_count'] * 2
        
        if r['filename_quality'] == 'CHECK':
            score -= 10
        
        r['consistency_score'] = max(0, score)
    
    # Sort by consistency
    results_sorted = sorted(results, key=lambda x: x['consistency_score'])
    
    # Top 10 most inconsistent
    print("\n### TOP 10 MOST INCONSISTENT BREEDS ###")
    print(f"{'Rank':<5} {'Breed':<22} {'Score':>7} {'Issues':<40}")
    print("-" * 75)
    for i, r in enumerate(results_sorted[:10], 1):
        issues = []
        if r['total_images'] < 50:
            issues.append(f"low count({r['total_images']})")
        if r['unique_formats'] > 1:
            issues.append(f"multi-format({r['unique_formats']})")
        if r['gif_count'] > 0:
            issues.append(f"{r['gif_count']} GIFs")
        if r['filename_quality'] == 'CHECK':
            issues.append("naming mismatch")
        
        issues_str = ', '.join(issues) if issues else 'OK'
        print(f"{i:<5} {r['breed']:<22} {r['consistency_score']:>7} {issues_str:<40}")
    
    # Top 10 most consistent/clean
    results_by_score = sorted(results, key=lambda x: x['consistency_score'], reverse=True)
    print("\n### TOP 10 MOST CONSISTENT/CLEAN BREEDS ###")
    print(f"{'Rank':<5} {'Breed':<22} {'Score':>7} {'Images':>8}")
    print("-" * 45)
    for i, r in enumerate(results_by_score[:10], 1):
        print(f"{i:<5} {r['breed']:<22} {r['consistency_score']:>7} {r['total_images']:>8}")
    
    # Global analysis
    print("\n" + "=" * 70)
    print("GLOBAL ANALYSIS")
    print("=" * 70)
    
    total = sum(r['total_images'] for r in results)
    avg = total // len(results)
    below_80 = len([r for r in results if r['total_images'] < 80])
    multi_fmt = len(multi_format)
    gif_total = sum(r['gif_count'] for r in results)
    
    print(f"\nTotal images: {total}")
    print(f"Average per breed: {avg}")
    print(f"Breeds below 80 images: {below_80}")
    print(f"Breeds with mixed formats: {multi_fmt}")
    print(f"Total GIF files: {gif_total}")
    
    # Similar breeds analysis
    print("\n### POTENTIALLY SIMILAR/CONFUSABLE BREEDS ###")
    print("(Breeds with similar naming or that might be visually similar)")
    
    similar_groups = [
        {'Sahiwal', 'Surti', 'Surti'},
        {'Red_Sindhi', 'Red_Dane', 'Red_Sindhi'},
        {'Jersey', 'Jersey_Cross', 'Holstein_Friesian'},
        {'Murrah', 'Nili_Ravi', 'Nili_Ravi'},
        {'Gir', 'Hariana', 'Hariana'},
        {'Hallikar', 'Amritmahal', 'Amritmahal'},
        {'Kankrej', 'Kangayam', 'Kangayam'},
        {'Khillari', 'Krishna_Valley', 'Krishna_Valley'},
    ]
    
    # Check which exist
    for group in similar_groups:
        existing = [g for g in group if g in [r['breed'] for r in results]]
        if len(existing) > 1:
            print(f"  Potential confusion: {existing}")
    
    # Save detailed report
    report = {
        'summary': {
            'total_breeds': len(results),
            'total_images': total,
            'avg_per_breed': avg,
            'breeds_below_80': below_80,
            'mixed_formats': multi_fmt,
            'total_gif_files': gif_total
        },
        'detailed_analysis': results,
        'top_10_inconsistent': [r['breed'] for r in results_sorted[:10]],
        'top_10_clean': [r['breed'] for r in results_by_score[:10]]
    }
    
    with open('detailed_quality_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nDetailed report saved to: detailed_quality_report.json")
    
    # Generate sample paths for manual visual inspection
    print("\n" + "=" * 70)
    print("SAMPLE IMAGE PATHS FOR VISUAL INSPECTION")
    print("=" * 70)
    print("(Run these commands or open files to visually inspect)")
    print()
    
    for r in results[:5]:  # Show first 5 as example
        print(f"\n### {r['breed']} ({r['total_images']} images) ###")
        for img in r['sample_10'][:5]:  # Show 5 samples
            print(f"  data/train_41/{r['breed']}/{img}")
    
    print("\n... (run detailed_quality_report.json for complete list)")

if __name__ == "__main__":
    main()