"""
Create Dataset Folder Structure
================================
Run this script to create all breed folders for data organization.

Usage:
    python create_folder_structure.py
"""

import os

# Define base data directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Indian Cattle Breeds
CATTLE_BREEDS = [
    # Milch Breeds (High Priority)
    "Gir",
    "Sahiwal",
    "Red_Sindhi",
    "Tharparkar",
    "Rathi",
    
    # Draught Breeds (Medium Priority)
    "Hallikar",
    "Amritmahal",
    "Khillari",
    "Kangayam",
    "Bargur",
    "Pulikulam",
    "Umblachery",
    
    # Dual Purpose Breeds (Medium Priority)
    "Hariana",
    "Kankrej",
    "Ongole",
    "Deoni",
    "Krishna_Valley",
    "Dangi",
    
    # Hill Cattle (Low Priority)
    "Punganur",
    "Vechur",
    "Malnad_Gidda",
    
    # Crossbreeds
    "Jersey_Cross",
    "HF_Cross",
]

# Indian Buffalo Breeds
BUFFALO_BREEDS = [
    # High Priority
    "Murrah",
    "Jaffrabadi",
    "Nili_Ravi",
    
    # Medium Priority
    "Banni",
    "Pandharpuri",
    "Mehsana",
    "Surti",
    "Nagpuri",
    
    # Low Priority
    "Toda",
    "Bhadawari",
    "Kalahandi",
    "Chilika",
]

# All breeds combined
ALL_BREEDS = CATTLE_BREEDS + BUFFALO_BREEDS

# Data splits
SPLITS = ['train', 'val', 'test']

def create_folder_structure():
    """Create complete folder structure for dataset."""
    
    print("=" * 60)
    print("Creating Dataset Folder Structure")
    print("=" * 60)
    
    total_folders = 0
    
    for split in SPLITS:
        split_dir = os.path.join(DATA_DIR, split)
        
        for breed in ALL_BREEDS:
            breed_dir = os.path.join(split_dir, breed)
            os.makedirs(breed_dir, exist_ok=True)
            total_folders += 1
            print(f"Created: {split}/{breed}/")
    
    # Create additional directories
    additional_dirs = [
        os.path.join(DATA_DIR, 'raw'),
        os.path.join(DATA_DIR, 'processed'),
        os.path.join(DATA_DIR, 'augmented'),
        os.path.join(BASE_DIR, 'models', 'checkpoints'),
        os.path.join(BASE_DIR, 'models', 'logs'),
        os.path.join(BASE_DIR, 'models', 'tflite'),
    ]
    
    for dir_path in additional_dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created: {os.path.relpath(dir_path, BASE_DIR)}/")
    
    print("\n" + "=" * 60)
    print("Folder Structure Created Successfully!")
    print("=" * 60)
    print(f"\nTotal breed folders: {total_folders}")
    print(f"Total breeds: {len(ALL_BREEDS)}")
    print(f"  - Cattle breeds: {len(CATTLE_BREEDS)}")
    print(f"  - Buffalo breeds: {len(BUFFALO_BREEDS)}")
    print(f"\nData splits: {SPLITS}")
    print("\nNext steps:")
    print("1. Add images to each breed folder in 'train/'")
    print("2. Keep some images aside for 'val/' and 'test/'")
    print("3. Recommended: 70% train, 15% val, 15% test")
    
    return True


def print_structure():
    """Print the folder structure."""
    print("\n" + "=" * 60)
    print("FOLDER STRUCTURE")
    print("=" * 60)
    print("""
data/
├── train/
│   ├── Gir/              ← Add Gir cattle images here
│   ├── Sahiwal/          ← Add Sahiwal images here
│   ├── Red_Sindhi/
│   ├── Tharparkar/
│   ├── Murrah/           ← Add Murrah buffalo images here
│   └── ... (all other breeds)
│
├── val/
│   ├── Gir/
│   ├── Sahiwal/
│   └── ... (all other breeds)
│
├── test/
│   ├── Gir/
│   ├── Sahiwal/
│   └── ... (all other breeds)
│
├── raw/                  ← Original downloaded images
├── processed/            ← Preprocessed images
└── augmented/            ← Augmented images
""")


if __name__ == '__main__':
    create_folder_structure()
    print_structure()
