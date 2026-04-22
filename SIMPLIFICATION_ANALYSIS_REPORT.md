# Dataset Simplification Analysis Report

## Executive Summary
This report analyzes the impact of proposed dataset simplification changes including removing low-data breeds and optional merges. Analysis is based on the current 41-class dataset (train_41, val_41, test_41 splits).

---

## 1. Current Dataset State

| Metric | Value |
|--------|-------|
| **Total Classes** | 41 |
| **Total Images** | 8,780 |
| **Train Split** | 6,125 (69.8%) |
| **Validation Split** | 1,299 (14.8%) |
| **Test Split** | 1,356 (15.4%) |
| **Average per Class** | 214 images |
| **Min Class Size** | Dangi: 108 images |
| **Max Class Size** | Sahiwal: 594 images |
| **Imbalance Ratio** | 5.5:1 (max/min) |

---

## 2. Impact of Proposed Removals

### 2.1 Breeds to Remove (5 breeds, 394 images)

| Breed | Train | Val | Test | Total | % of Dataset |
|-------|-------|-----|------|-------|-------------|
| **Kherigarh** | 44 | 9 | 11 | 64 | 0.73% |
| **Umblachery** | 53 | 11 | 12 | 76 | 0.87% |
| **Kenkatha** | 58 | 12 | 14 | 84 | 0.96% |
| **Nimari** | 58 | 12 | 14 | 84 | 0.96% |
| **Nagori** | 60 | 12 | 14 | 86 | 0.98% |
| **TOTAL** | 273 | 56 | 65 | 394 | 4.5% |

**Key Observations:**
- All 5 breeds have < 100 images each (significantly below average of 214)
- Kherigarh has only 64 images - the smallest class in the dataset
- These 5 breeds represent only 4.5% of total images
- Removing them eliminates extreme minority classes that may cause poor classification

---

## 3. Impact of Proposed Merges

### 3.1 Merge Option 1: Red_Cattle

| Source Breed | Train | Val | Test | Total |
|-------------|-------|-----|------|-------|
| Red_Dane | 116 | 24 | 26 | 166 |
| Red_Sindhi | 115 | 24 | 26 | 165 |
| **Combined Total** | 231 | 24 | 26 | **331** |

**Rationale:** Both are red-colored breeds from different origins (Danish crossbreeding vs. Indian). Merging creates a more robust "Red Cattle" category.

### 3.2 Merge Option 2: South_Draft

| Source Breed | Train | Val | Test | Total |
|-------------|-------|-----|------|-------|
| Hallikar | 156 | 33 | 34 | 223 |
| Amritmahal | 88 | 18 | 20 | 126 |
| **Combined Total** | 244 | 51 | 54 | **349** |

**Rationale:** Both are South Indian draft breeds with similar characteristics. Hallikar is more common, Amritmahal benefits from its larger sample size.

### 3.3 Merge Option 3: Draft_Breed

| Source Breed | Train | Val | Test | Total |
|-------------|-------|-----|------|-------|
| Kankrej | 214 | 46 | 47 | 307 |
| Kangayam | 91 | 19 | 20 | 130 |
| **Combined Total** | 305 | 65 | 67 | **437** |

**Rationale:** Both are Indian draft breeds from Gujarat/Tamil Nadu regions. Kankrej has decent data, Kangayam benefits from merged sample.

---

## 4. Final Class List (Proposed)

After applying removals and merges, the final dataset would have **39 classes**.

### Complete Per-Class Distribution:

| Class | Train | Val | Test | Total | Status |
|-------|-------|-----|------|-------|--------|
| Alambadi | 221 | 47 | 49 | 317 | Keep |
| Amritmahal | 88 | 18 | 20 | 126 | → Merged to South_Draft |
| Ayrshire | 173 | 37 | 38 | 248 | Keep |
| Banni | 201 | 43 | 44 | 288 | Keep |
| Bargur | 187 | 40 | 41 | 268 | Keep |
| Bhadawari | 173 | 37 | 38 | 248 | Keep |
| Brown_Swiss | 170 | 36 | 38 | 244 | Keep |
| Dangi | 75 | 16 | 17 | 108 | Keep |
| Deoni | 90 | 19 | 20 | 129 | Keep |
| **Draft_Breed** | 305 | 65 | 67 | **437** | NEW MERGE |
| Gir | 256 | 55 | 56 | 367 | Keep |
| Guernsey | 88 | 18 | 20 | 126 | Keep |
| Hallikar | 156 | 33 | 34 | 223 | → Merged to South_Draft |
| Hariana | 109 | 23 | 25 | 157 | Keep |
| Holstein_Friesian | 275 | 59 | 60 | 394 | Keep |
| Jaffrabadi | 211 | 45 | 46 | 302 | Keep |
| Jersey | 156 | 33 | 34 | 223 | Keep |
| Kangayam | 91 | 19 | 20 | 130 | → Merged to Draft_Breed |
| Kankrej | 214 | 46 | 47 | 307 | → Merged to Draft_Breed |
| Kasargod | 81 | 17 | 19 | 117 | Keep |
| Kherigarh | 44 | 9 | 11 | 64 | REMOVED |
| Khenkatha | 58 | 12 | 14 | 84 | REMOVED |
| Khillari | 111 | 23 | 25 | 159 | Keep |
| Krishna_Valley | 116 | 25 | 26 | 167 | Keep |
| Malnad_gidda | 110 | 23 | 25 | 158 | Keep |
| Mehsana | 165 | 35 | 37 | 237 | Keep |
| Murrah | 271 | 58 | 59 | 388 | Keep |
| Nagori | 60 | 12 | 14 | 86 | REMOVED |
| Nagpuri | 320 | 68 | 70 | 458 | Keep |
| Nili_Ravi | 196 | 42 | 42 | 280 | Keep |
| Nimari | 58 | 12 | 14 | 84 | REMOVED |
| Ongole | 127 | 27 | 28 | 182 | Keep |
| Pulikulam | 87 | 18 | 20 | 125 | Keep |
| Rathi | 103 | 22 | 23 | 148 | Keep |
| **Red_Cattle** | 231 | 48 | 52 | **331** | NEW MERGE |
| Red_Dane | 116 | 24 | 26 | 166 | → Merged to Red_Cattle |
| Red_Sindhi | 115 | 24 | 26 | 165 | → Merged to Red_Cattle |
| Sahiwal | 416 | 89 | 89 | 594 | Keep |
| **South_Draft** | 244 | 51 | 54 | **349** | NEW MERGE |
| Surti | 166 | 35 | 37 | 238 | Keep |
| Tharparkar | 150 | 32 | 33 | 215 | Keep |
| Toda | 170 | 36 | 38 | 244 | Keep |
| Umblachery | 53 | 11 | 12 | 76 | REMOVED |
| Vechur | 98 | 21 | 21 | 140 | Keep |

---

## 5. Before vs After Comparison

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| **Total Classes** | 41 | 39 | -2 (-4.9%) |
| **Total Images** | 8,780 | 8,386 | -394 (-4.5%) |
| **Min Class Size** | 64 (Kherigarh) | 108 (Dangi) | +44 |
| **Max Class Size** | 594 (Sahiwal) | 594 (Sahiwal) | 0 |
| **Classes < 100 images** | 6 | 0 | -6 ✓ |
| **Average per Class** | 214 | 215 | +1 |

### Split Distribution:
- Train: 5,852 images (69.8%)
- Val: 1,243 images (14.8%)
- Test: 1,291 images (15.4%)

*(Approximately preserved)*

---

## 6. Class Balance Analysis (After Changes)

| Category | Count |
|----------|-------|
| **Largest Class** | Sahiwal: 594 (7.1%) |
| **Smallest Class** | Dangi: 108 (1.3%) |
| **Imbalance Ratio** | 5.5:1 |
| **Standard Deviation** | ~98 images |

**Distribution Quality:**
- < 150 images: 4 classes (Dangi, Deoni, Guernsey, Vechur)
- 150-200 images: 8 classes
- 200-300 images: 17 classes
- > 300 images: 10 classes

---

## 7. Benefits Analysis

### Expected Improvements:

1. **Better Model Convergence**
   - No classes below 100 images (minimum is now 108)
   - Reduced risk of overfitting to minority classes
   - More balanced gradient updates during training

2. **Reduced Risk of Overfitting**
   - Smaller, problematic classes removed
   - Fewer but more robust categories

3. **Improved Generalization**
   - Merged classes have 300+ images each
   - Better feature learning for draft/red breed variants

4. **Faster Training**
   - 4.5% fewer images to process
   - Fewer fine-grained distinctions to learn

5. **Simplified Model Output**
   - 39 classes vs 41 (simpler debugging and evaluation)

---

## 8. Risks Analysis

### Critical Risks:

1. **Loss of Rare Breed Distinctions**
   - **Kherigarh**: Only 64 images - may be valuable for conservation
   - **Umblachery**: Endangered breed from Tamil Nadu
   - **Nimari**: Historical breed from Maharashtra
   - These breeds may have unique phenotypic markers being lost

2. **Merge Validity Concerns**
   - Red_Dane vs Red_Sindhi: Not truly equivalent breeds
   - Hallikar vs Amritmahal: Different breed characteristics
   - Kankrej vs Kangayam: Distinct regional breeds

3. **Research Impact**
   - Loss of ability to classify rare Indian breeds
   - May affect conservation efforts
   - Reduced classification granularity for specific use cases

4. **Potential Performance Issues**
   - Draft breeds have very different uses (transport vs meat)
   - Merged classes may confuse the model
   - Loss of fine-grained features from rare breeds

---

## 9. Recommendations

### Primary Recommendation: **PROCEED WITH MODIFICATIONS**

The proposed changes are beneficial for model training performance:

1. ✅ Remove the 5 low-data breeds (Kherigarh, Umblachery, Kenkatha, Nimari, Nagori)
   
2. ⚠️ **Consider alternatives to merges:**
   - Option A: Apply all three merges (as proposed) - BEST for training
   - Option B: Only merge Hallikar + Amritmahal (most similar)
   - Option C: Skip all merges, keep all original breed distinctions

3. **Recommended Implementation Path:**
   - First, verify model performance on 36-class dataset (removals only)
   - If imbalance remains problematic, apply merges incrementally
   - Monitor per-class accuracy during training

### Alternative Plan (If Rare Breeds Critical):

- **Keep all 41 breeds**
- **Apply aggressive data augmentation** to minority classes
- **Consider collecting more images** for breeds below 100
- **Use class-weighted loss** to handle imbalance

---

## 10. Conclusion

The proposed simplification removes 394 images (4.5%) and reduces classes from 41 to 39 while ensuring all remaining classes have 100+ images. This should significantly improve model training stability and reduce overfitting to rare classes.

**However**, the removal of endangered/rare Indian breeds (Kherigarh, Umblachery, Nimari) may have implications for breed conservation research. If breed-specific classification is important, consider retaining these classes with augmentation.

**Final Decision**: Proceed with removals (5 breeds) + consider merge options based on use case priorities.

---

*Analysis Date: 2026-04-18*
*Data Source: data/train_41, data/val_41, data/test_41*