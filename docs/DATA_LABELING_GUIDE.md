# Data Labeling Guide

## Cattle Breed Recognition - Image Labeling Instructions

This guide explains how to label images for training the cattle breed recognition model.

---

## Table of Contents

1. [Overview](#overview)
2. [Labeling Tools](#labeling-tools)
3. [Labeling for Detection (YOLO)](#labeling-for-detection-yolo)
4. [Labeling for Classification](#labeling-for-classification)
5. [Labeling Standards](#labeling-standards)
6. [Quality Checklist](#quality-checklist)

---

## Overview

We need to label images for two tasks:

| Task | Purpose | Label Type |
|------|---------|------------|
| **Detection** | Locate animal in image | Bounding box (x, y, width, height) |
| **Classification** | Identify breed | Class label (breed name) |

---

## Labeling Tools

### Free Tools

| Tool | Best For | Platform |
|------|----------|----------|
| **LabelImg** | Bounding boxes | Desktop (Python) |
| **CVAT** | All annotation types | Web-based |
| **Roboflow** | All annotation types | Web-based (free tier) |
| **Label Studio** | All annotation types | Web-based |

### Installing LabelImg

```bash
pip install labelImg
labelImg
```

### Using CVAT (Free)

1. Go to https://cvat.ai
2. Create free account
3. Create project and upload images
4. Annotate using the interface

---

## Labeling for Detection (YOLO)

### Step-by-Step Process

1. **Open LabelImg**
   ```bash
   labelImg ./data/raw ./data/labels classes.txt
   ```

2. **Create Bounding Box**
   - Click "Create RectBox" (or press 'W')
   - Draw box around the animal
   - Select class: "cattle"

3. **Save Annotation**
   - Click "Save" (or press Ctrl+S)
   - Format: YOLO (.txt)

### YOLO Format

Each image gets a corresponding `.txt` file:

```
# Format: class_id x_center y_center width height
# All values normalized to [0, 1]

0 0.5 0.5 0.8 0.6
```

| Value | Description |
|-------|-------------|
| `class_id` | 0 for cattle/buffalo |
| `x_center` | Center X of box (normalized) |
| `y_center` | Center Y of box (normalized) |
| `width` | Box width (normalized) |
| `height` | Box height (normalized) |

### Bounding Box Guidelines

```
┌─────────────────────────────────────┐
│                                     │
│    ┌───────────────────────┐       │
│    │                       │       │
│    │      ANIMAL           │       │
│    │      (tight box)      │       │
│    │                       │       │
│    └───────────────────────┘       │
│                                     │
└─────────────────────────────────────┘

✓ CORRECT: Tight box around animal
✗ WRONG: Too much background
✗ WRONG: Cutting off animal parts
```

---

## Labeling for Classification

### Folder Structure

Organize images by breed in folders:

```
data/
├── train/
│   ├── Gir/
│   │   ├── gir_001.jpg
│   │   ├── gir_002.jpg
│   │   └── ...
│   ├── Sahiwal/
│   │   ├── sahiwal_001.jpg
│   │   └── ...
│   └── Murrah/
│       └── ...
├── val/
│   └── (same structure)
└── test/
    └── (same structure)
```

### Breed Identification Guide

#### Cattle Breeds

| Breed | Key Characteristics | Origin |
|-------|---------------------|--------|
| **Gir** | Red/white spotted, curved ears, prominent hump | Gujarat |
| **Sahiwal** | Reddish brown, loose skin, medium hump | Punjab |
| **Red Sindhi** | Deep red color, compact body | Sindh |
| **Tharparkar** | White/light gray, lyre-shaped horns | Rajasthan |
| **Kankrej** | Silver-gray, lyre horns, strong build | Gujarat |
| **Hariana** | White/gray, medium size, curved horns | Haryana |
| **Ongole** | White, muscular, long horns | Andhra Pradesh |
| **Hallikar** | Gray, long horns, draught breed | Karnataka |

#### Buffalo Breeds

| Breed | Key Characteristics | Origin |
|-------|---------------------|--------|
| **Murrah** | Black, tightly curved horns, heavy build | Haryana |
| **Jaffrabadi** | Black, heavy, drooping horns | Gujarat |
| **Nili-Ravi** | Black, white markings on face | Punjab |
| **Banni** | Black, long horns, desert breed | Gujarat |
| **Pandharpuri** | Black, long sword-like horns | Maharashtra |

### Visual Identification Tips

```
GIR CATTLE
┌─────────────────────────────────┐
│  • Distinctive curved ears      │
│  • Red/white spotted coat       │
│  • Prominent forehead           │
│  • Well-developed hump          │
└─────────────────────────────────┘

MURRAH BUFFALO
┌─────────────────────────────────┐
│  • Solid black color            │
│  • Tightly curled horns         │
│  • Heavy, massive build         │
│  • Short, thick neck            │
└─────────────────────────────────┘
```

---

## Labeling Standards

### Image Quality Requirements

| Criterion | Requirement |
|-----------|-------------|
| **Resolution** | Minimum 300×300 pixels |
| **Clarity** | Animal clearly visible |
| **Lighting** | Not too dark or bright |
| **Focus** | Sharp, not blurry |
| **Content** | Single animal preferred |

### What to Include

✓ **Include:**
- Clear side profile images
- Full body visible
- Natural lighting
- Various backgrounds
- Different ages (adult preferred)

✗ **Exclude:**
- Blurry images
- Multiple animals (unless for detection)
- Heavily edited images
- Cartoon/drawings
- Watermarked images

### Handling Difficult Cases

| Case | Action |
|------|--------|
| **Crossbreed** | Label as closest breed + "Cross" suffix |
| **Young animal** | Label breed if identifiable, else "Unknown" |
| **Partial view** | Label if breed identifiable |
| **Multiple animals** | Create separate crops if possible |

---

## Quality Checklist

Before submitting labeled data, verify:

### Detection Labels
- [ ] Bounding box tightly fits the animal
- [ ] No parts of animal cut off
- [ ] Box doesn't include excessive background
- [ ] All animals in image are labeled
- [ ] YOLO format is correct

### Classification Labels
- [ ] Correct breed folder
- [ ] Image quality meets standards
- [ ] No duplicate images
- [ ] Filename is descriptive

### Overall
- [ ] Consistent labeling across similar images
- [ ] No spelling errors in breed names
- [ ] Train/val/test split is balanced

---

## Annotation Statistics

Track your labeling progress:

| Breed | Target | Labeled | Progress |
|-------|--------|---------|----------|
| Gir | 500 | 0 | 0% |
| Sahiwal | 500 | 0 | 0% |
| Murrah | 500 | 0 | 0% |
| ... | ... | ... | ... |
| **Total** | **12,000** | **0** | **0%** |

---

## Tips for Efficient Labeling

1. **Batch Similar Images**: Group images by breed before labeling
2. **Use Keyboard Shortcuts**: Learn LabelImg shortcuts
3. **Take Breaks**: Avoid fatigue to maintain quality
4. **Review Work**: Periodically check previous labels
5. **Get Help**: Have others verify difficult cases

---

## Contact

For questions about labeling:
- Create an issue in the project repository
- Refer to NBAGR website for breed references: https://nbagr.icar.gov.in
