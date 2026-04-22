#!/usr/bin/env python3
"""
Create 3 candidate dataset configurations for optimal model performance.

Analysis based on:
- Image count (from dataset_analysis.json)
- Data quality score (from detailed_quality_report.json)
- Visual distinctiveness and breed clusters
"""

import json
from pathlib import Path

# Load data
with open('dataset_analysis.json', 'r') as f:
    analysis = json.load(f)

breed_counts = analysis['breed_counts']

# Calculate totals
breed_totals = {}
for breed, counts in breed_counts.items():
    total = counts['train'] + counts['val'] + counts['test']
    breed_totals[breed] = {
        'total': total,
        'train': counts['train'],
        'val': counts['val'],
        'test': counts['test']
    }

# ============================================================
# RANKING BY MULTIPLE FACTORS
# ============================================================

def calculate_composite_score(breed, total):
    """Calculate composite score based on multiple factors."""
    score = 0
    
    # Factor 1: Image count (0-40 points)
    if total >= 300:
        score += 40
    elif total >= 200:
        score += 30
    elif total >= 150:
        score += 20
    elif total >= 100:
        score += 10
    else:
        score += 5
    
    # Factor 2: Data Quality (based on known quality issues)
    issue_breeds = {
        'Bargur': -5,
        'Bhadawari': -3,
        'Holstein_Friesian': -2,
        'Kankrej': -2,
    }
    score += issue_breeds.get(breed, 0)
    
    return score

# Score all breeds
breed_scores = {}
for breed, data in breed_totals.items():
    score = calculate_composite_score(breed, data['total'])
    breed_scores[breed] = {
        'score': score,
        'total': data['total'],
        'train': data['train'],
        'val': data['val'],
        'test': data['test'],
        'tier': None
    }

# Assign tiers based on score
sorted_breeds = sorted(breed_scores.items(), key=lambda x: x[1]['score'], reverse=True)
for i, (breed, data) in enumerate(sorted_breeds):
    if i < 10:
        data['tier'] = 'TIER_1'
    elif i < 25:
        data['tier'] = 'TIER_2'
    elif i < 35:
        data['tier'] = 'TIER_3'
    else:
        data['tier'] = 'TIER_4'

# ============================================================
# IDENTIFY BREED CLUSTERS
# ============================================================

BREED_CLUSTERS = {
    'red_cattle': ['Red_Dane', 'Red_Sindhi'],
    'south_draft': ['Hallikar', 'Amritmahal'],
    'gujarat_draft': ['Kankrej', 'Kangayam'],
    'buffalo_indian': ['Murrah', 'Mehsana', 'Nagpuri'],
    'bos_taurus': ['Holstein_Friesian', 'Guernsey', 'Jersey', 'Ayrshire', 'Brown_Swiss'],
    'zebu': ['Sahiwal', 'Hariana', 'Ongole', 'Gir'],
}

# ============================================================
# CREATE CONFIGURATIONS
# ============================================================

def create_config_a():
    """Config A: High Accuracy - ~22 classes (aggressive)"""
    classes = {}
    
    # Keep all TIER 1 (10 breeds)
    for breed, d in breed_scores.items():
        if d['tier'] == 'TIER_1':
            classes[breed] = breed_totals[breed]['total']
    
    # Keep top 5 from TIER 2
    tier2 = [b for b, d in breed_scores.items() if d['tier'] == 'TIER_2']
    tier2_sorted = sorted(tier2, key=lambda b: breed_totals[b]['total'], reverse=True)
    for breed in tier2_sorted[:5]:
        classes[breed] = breed_totals[breed]['total']
    
    # Merge strategies (all remaining into clusters)
    # Red cattle -> one class
    classes['Red_Cattle'] = (
        breed_totals.get('Red_Dane', {}).get('total', 0) +
        breed_totals.get('Red_Sindhi', {}).get('total', 0)
    )
    
    # Draft breeds -> one class (Hallikar + Amritmahal + Kankrej + Kangayam)
    classes['Draft_Breeds'] = (
        breed_totals.get('Hallikar', {}).get('total', 0) +
        breed_totals.get('Amritmahal', {}).get('total', 0) +
        breed_totals.get('Kankrej', {}).get('total', 0) +
        breed_totals.get('Kangayam', {}).get('total', 0)
    )
    
    # Buffalo -> one class (Nagpuri + Mehsana)
    classes['Buffalo'] = (
        breed_totals.get('Nagpuri', {}).get('total', 0) +
        breed_totals.get('Mehsana', {}).get('total', 0)
    )
    
    # Zebu group -> one class (Hariana + Ongole)
    classes['Zebu'] = (
        breed_totals.get('Hariana', {}).get('total', 0) +
        breed_totals.get('Ongole', {}).get('total', 0)
    )
    
    return classes


def create_config_b():
    """Config B: Balanced - ~27 classes (moderate)"""
    classes = {}
    
    # Keep TIER 1
    for breed, d in breed_scores.items():
        if d['tier'] == 'TIER_1':
            classes[breed] = breed_totals[breed]['total']
    
    # Keep all TIER 2 
    for breed, d in breed_scores.items():
        if d['tier'] == 'TIER_2':
            classes[breed] = breed_totals[breed]['total']
    
    # Keep TIER 3 above 100 images
    tier3 = [b for b, d in breed_scores.items() if d['tier'] == 'TIER_3']
    for breed in tier3:
        if breed_totals[breed]['total'] >= 100:
            classes[breed] = breed_totals[breed]['total']
    
    # Merge only TIER 4 (5 weakest breeds)
    weak = ['Kherigarh', 'Umblachery', 'Kenkatha', 'Nimari', 'Nagori']
    classes['Rare_Breeds'] = sum(breed_totals[b]['total'] for b in weak if b in breed_totals)
    
    # One strategic merge - red cattle
    classes['Red_Cattle'] = (
        breed_totals.get('Red_Dane', {}).get('total', 0) +
        breed_totals.get('Red_Sindhi', {}).get('total', 0)
    )
    
    # Draft merge
    classes['Draft_Breeds'] = (
        breed_totals.get('Hallikar', {}).get('total', 0) +
        breed_totals.get('Amritmahal', {}).get('total', 0)
    )
    
    return classes


def create_config_c():
    """Config C: High Coverage - ~34 classes (minimal)"""
    classes = {}
    
    # Keep TIER 1, 2, 3
    for breed, d in breed_scores.items():
        if d['tier'] in ['TIER_1', 'TIER_2', 'TIER_3']:
            classes[breed] = breed_totals[breed]['total']
    
    # Only remove 4 weakest breeds
    weak = ['Kherigarh', 'Kenkatha', 'Nagori', 'Nimari']
    removed_total = sum(breed_totals[b]['total'] for b in weak if b in breed_totals)
    
    # Keep these small breeds for coverage
    if 'Umblachery' in breed_totals:
        classes['Umblachery'] = breed_totals['Umblachery']['total']
    if 'Vechur' in breed_totals:
        classes['Vechur'] = breed_totals['Vechur']['total']
    
    return classes


# ============================================================
# OUTPUT
# ============================================================

print("=" * 80)
print("BREED RANKING BY COMPOSITE SCORE")
print("=" * 80)
print()
print(f"{'Rank':<5} {'Breed':<20} {'Score':<7} {'Tier':<10} {'Total':<8}")
print("-" * 55)

for rank, (breed, data) in enumerate(sorted_breeds, 1):
    print(f"{rank:<5} {breed:<20} {data['score']:<7} {data['tier']:<10} {data['total']:<8}")

# Generate configs
config_a = create_config_a()
config_b = create_config_b()
config_c = create_config_c()

print()
print("=" * 80)
print("CONFIGURATION A: HIGH ACCURACY")
print("=" * 80)
print(f"Total classes: {len(config_a)}")
print(f"Total images: {sum(config_a.values())}")
print(f"Expected accuracy: HIGH")
print(f"Training difficulty: LOW")
print()
print("Classes:")
for c in sorted(config_a.keys()):
    print(f"  {c}: {config_a[c]}")

print()
print("=" * 80)
print("CONFIGURATION B: BALANCED")
print("=" * 80)
print(f"Total classes: {len(config_b)}")
print(f"Total images: {sum(config_b.values())}")
print(f"Expected accuracy: MEDIUM-HIGH")
print(f"Training difficulty: MEDIUM")
print()
print("Classes:")
for c in sorted(config_b.keys()):
    print(f"  {c}: {config_b[c]}")

print()
print("=" * 80)
print("CONFIGURATION C: HIGH COVERAGE")
print("=" * 80)
print(f"Total classes: {len(config_c)}")
print(f"Total images: {sum(config_c.values())}")
print(f"Expected accuracy: MEDIUM")
print(f"Training difficulty: HIGH")
print()
print("Classes:")
for c in sorted(config_c.keys()):
    print(f"  {c}: {config_c[c]}")

# Save configs
configs = {
    'config_a': {'classes': config_a, 'count': len(config_a), 'total_images': sum(config_a.values())},
    'config_b': {'classes': config_b, 'count': len(config_b), 'total_images': sum(config_b.values())},
    'config_c': {'classes': config_c, 'count': len(config_c), 'total_images': sum(config_c.values())},
    'breed_scores': {k: {'score': v['score'], 'tier': v['tier'], 'total': v['total']} for k, v in breed_scores.items()},
}

with open('candidate_configurations.json', 'w') as f:
    json.dump(configs, f, indent=2)

print()
print("Data saved to candidate_configurations.json")