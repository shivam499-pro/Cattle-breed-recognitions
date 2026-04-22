# Dataset Quality Report - 41-Class Cattle Breed Model

**Dataset**: `data/train_41/`  
**Date**: 2026-04-18  
**Total Breeds**: 41  
**Total Images**: 6,125  
**Average Images per Breed**: 149

---

## 1. Per-Class Quality Summary

### Summary Table

| Breed | Images | Formats | GIFs | Issues Score | Notes |
|-------|-------|---------|------|-------------|-------|
| Sahiwal | 416 | 4 | 0 | 85 | Largest class, mixed sources |
| Nagpuri | 320 | 4 | 2 | 81 | Mixed formats, GIF files |
| Holstein_Friesian | 275 | 2 | 0 | 99 | Good quality |
| Murrah | 271 | 3 | 0 | 90 | Mixed formats |
| Gir | 256 | 1 | 0 | 100 | Excellent - single format |
| Alambadi | 221 | 2 | 0 | 95 | Mixed JPG/PNG |
| Kankrej | 214 | 2 | 0 | 99 | Good quality |
| Jaffrabadi | 211 | 3 | 1 | 78 | Naming mismatch (jaffarabadi) |
| Banni | 201 | 4 | 1 | 83 | Mixed formats |
| Nili_Ravi | 196 | 4 | 1 | 83 | Mixed formats, duplicates found |
| Bargur | 187 | 1 | 0 | 99 | Desktop.ini file present |
| Ayrshire | 173 | 1 | 0 | 100 | Clean |
| Bhadawari | 173 | 1 | 0 | 100 | Clean |
| Brown_Swiss | 170 | 1 | 0 | 100 | Clean |
| Toda | 170 | 2 | 0 | 100 | Clean |
| Surti | 166 | 4 | 4 | 77 | Issues: 4 GIFs, 4 formats |
| Mehsana | 165 | 3 | 0 | 90 | Mixed formats |
| Hallikar | 156 | 1 | 0 | 100 | Clean |
| Jersey | 156 | 1 | 0 | 100 | Clean |
| Tharparkar | 150 | 1 | 0 | 100 | Clean |
| Ongole | 127 | 1 | 0 | 100 | Clean |
| Krishna_Valley | 116 | 1 | 0 | 100 | Clean |
| Red_Dane | 116 | 1 | 0 | 100 | Clean |
| Red_Sindhi | 115 | 1 | 0 | 100 | Clean |
| Khillari | 111 | 1 | 0 | 100 | Clean |
| Malnad_gidda | 110 | 1 | 0 | 100 | Clean |
| Hariana | 109 | 1 | 0 | 100 | Clean |
| Rathi | 103 | 1 | 0 | 100 | Clean |
| Vechur | 98 | 1 | 0 | 100 | Clean |
| Kangayam | 91 | 1 | 0 | 100 | Clean |
| Deoni | 90 | 1 | 0 | 100 | Clean |
| Guernsey | 88 | 1 | 0 | 100 | Clean |
| Amritmahal | 88 | 1 | 0 | 100 | Clean |
| Pulikulam | 87 | 1 | 0 | 100 | Clean |
| Kasargod | 81 | 1 | 0 | 100 | Clean |
| Dangi | 75 | 1 | 0 | 90 | Below 80 images |
| Nagori | 60 | 1 | 0 | 100 | Clean |
| Kenkatha | 58 | 1 | 0 | 90 | Below 80 images |
| Nimari | 58 | 1 | 0 | 90 | Below 80 images |
| Umblachery | 53 | 1 | 0 | 95 | Below 60 images |
| Kherigarh | 44 | 1 | 0 | 80 | Fewest images |

---

## 2. Top 10 Most Inconsistent Breeds

| Rank | Breed | Score | Issues |
|------|-------|-------|-------|
| 1 | Surti | 77 | 4 formats, 4 GIFs |
| 2 | Jaffrabadi | 78 | 3 formats, 1 GIF, naming mismatch |
| 3 | Kherigarh | 80 | Only 44 images (need data) |
| 4 | Nagpuri | 81 | 4 formats, 2 GIFs |
| 5 | Banni | 83 | 4 formats, 1 GIF |
| 6 | Nili_Ravi | 83 | 4 formats, 1 GIF, duplicates |
| 7 | Sahiwal | 85 | 4 formats (largest but mixed) |
| 8 | Dangi | 90 | Only 75 images |
| 9 | Kenkatha | 90 | Only 58 images |
| 10 | Mehsana | 90 | 3 formats |

---

## 3. Top 10 Most Clean Breeds

| Rank | Breed | Score | Images |
|------|-------|-------|--------|
| 1 | Ayrshire | 100 | 173 |
| 2 | Bargur | 100 | 187 |
| 3 | Bhadawari | 100 | 173 |
| 4 | Brown_Swiss | 100 | 170 |
| 5 | Gir | 100 | 256 |
| 6 | Hallikar | 100 | 156 |
| 7 | Hariana | 100 | 109 |
| 8 | Jersey | 100 | 156 |
| 9 | Khillari | 100 | 111 |
| 10 | Krishna_Valley | 100 | 116 |

---

## 4. Identified Issues

### 4.1 Low Image Counts (< 60 images)
- **Kherigarh**: 44 images - CRITICAL
- **Umblachery**: 53 images - Need more data
- **Kenkatha**: 58 images - Need more data
- **Nimari**: 58 images - Need more data

### 4.2 Mixed Image Formats (Multiple Sources)
12 breeds have multiple image formats, indicating data from different sources:
- **Banni**: jpg:181, jpeg:15, png:4, gif:1
- **Nagpuri**: jpg:293, png:23, gif:2, jpeg:2
- **Nili_Ravi**: jpg:190, png:4, jpeg:1, gif:1
- **Sahiwal**: jpg:383, png:28, webp:4, jpeg:1
- **Surti**: jpg:149, gif:4, png:12, jpeg:1
- **Jaffrabadi**: jpg:175, png:35, gif:1
- **Mehsana**: jpg:128, png:34, jpeg:3
- **Murrah**: jpg:252, jpeg:11, png:8

### 4.3 GIF Files (Quality Issues)
9 GIF files across 5 breeds (GIF format indicates potential quality issues):
- **Surti**: 4 GIFs
- **Nagpuri**: 2 GIFs
- **Banni**: 1 GIF
- **Jaffrabadi**: 1 GIF
- **Nili_Ravi**: 1 GIF

### 4.4 Duplicates Found
- **Nili_Ravi**: 10 duplicate image pairs (same content with different names)
- **Surti**: 2 duplicate GIFs

### 4.5 Irrelevant Files
- **Bargur**: desktop.ini
- **Holstein_Friesian**: fresian_70.ini
- **Kankrej**: kankarej_80.ini

### 4.6 Filename Mismatch
- **Jaffrabadi**: Images named "jaffarabadi_*" but folder is "Jaffrabadi"

---

## 5. Global Analysis

### 5.1 Dataset Statistics
- **Total Images**: 6,125
- **Average per Breed**: 149
- **Breeds Below 80 Images**: 6
- **Breeds with Mixed Formats**: 12
- **Total GIF Files**: 9
- **Duplicate Images**: 12

### 5.2 Potentially Confusable Breeds
These breed pairs may be visually similar and could cause classification confusion:
- **Surti** vs **Sahiwal** - Similar coloring patterns
- **Red_Sindhi** vs **Red_Dane** - Both red cattle breeds
- **Holstein_Friesian** vs **Jersey** - Different color patterns (black-white vs brown)
- **Nili_Ravi** vs **Murrah** - Both buffalo breeds, very similar
- **Gir** vs **Hariana** - Both Indian dairy breeds
- **Hallikar** vs **Amritmahal** - Similar southern Indian breeds
- **Kankrej** vs **Kangayam** - Both draft breeds
- **Krishna_Valley** vs **Khillari** - Both draft breeds

### 5.3 Visual Consistency Assessment
Based on filename and format analysis (visual inspection would be needed for definitive assessment):
- **Consistent**: ~35 breeds (single format, properly named)
- **Needs Review**: ~6 breeds (multiple formats, potential issues)

---

## 6. Recommended Actions

### 6.1 High Priority - Data Collection Needed
| Breed | Current Images | Recommended | Action |
|-------|---------------|--------------|--------|
| Kherigarh | 44 | 100+ | Collect more data |
| Umblachery | 53 | 100+ | Collect more data |
| Kenkatha | 58 | 100+ | Collect more data |
| Nimari | 58 | 100+ | Collect more data |
| Nagori | 60 | 100+ | Collect more data |

### 6.2 Medium Priority - Clean Up
| Breed | Issue | Action |
|-------|-------|--------|
| Nili_Ravi | 10 duplicates | Remove duplicate files |
| Surti | 4 GIFs | Convert to JPG or remove |
| Jaffrabadi | Naming mismatch | Rename files (jaffarabadi → jaffrabadi) |
| Bargur | desktop.ini | Remove irrelevant file |
| Holstein_Friesian | fresian_70.ini | Remove irrelevant file |
| Kankrej | kankarej_80.ini | Remove irrelevant file |

### 6.3 Consider Merging Similar Classes
If model performance is poor, consider merging:
- **Red_Sindhi** + **Red_Dane** → "Red_Cattle" (if visually indistinguishable)
- **Nili_Ravi** + **Murrah** → Note: These are different species (buffalo vs cattle), NOT recommended to merge

### 6.4 No Action Needed
These breeds are clean and ready for training:
- Gir, Hallikar, Jersey, Khillari, Krishna_Valley, Tharparkar, Ongole, etc.

---

## 7. Sample Image Paths for Visual Inspection

For manual visual verification, here are sample paths:
```
data/train_41/Gir/gir_001.jpg
data/train_41/Sahiwal/sahiwal_001.jpg
data/train_41/Kherigarh/kherigarh_001.jpg
data/train_41/Jaffrabadi/jaffarabadi_001.jpg
data/train_41/Surti/surti_001.gif
```

---

## 8. Report Files Generated
- `dataset_quality_report.json` - Basic quality metrics
- `detailed_quality_report.json` - Full analysis with all filenames
- `DATASET_QUALITY_REPORT.md` - This report

---

*Report generated by dataset_quality_analysis.py*