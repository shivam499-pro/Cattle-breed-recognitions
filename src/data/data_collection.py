"""
Data Collection Script
======================
Script to collect cattle breed images from various sources.

Usage:
    python data_collection.py --source kaggle --output ./data/raw
    python data_collection.py --source nbagr --output ./data/raw
    python data_collection.py --source google --breed "Gir" --output ./data/raw
"""

import os
import argparse
import requests
from bs4 import BeautifulSoup
from typing import List, Dict
import time
import json
from pathlib import Path


# Indian Cattle Breeds
CATTLE_BREEDS = [
    "Gir", "Sahiwal", "Red Sindhi", "Tharparkar", "Rathi",
    "Hallikar", "Amritmahal", "Khillari", "Kangayam", "Bargur",
    "Hariana", "Kankrej", "Ongole", "Deoni", "Krishna Valley",
    "Punganur", "Vechur", "Malnad Gidda",
    "Jersey Cross", "HF Cross"
]

# Indian Buffalo Breeds
BUFFALO_BREEDS = [
    "Murrah", "Jaffrabadi", "Nili-Ravi", "Banni",
    "Pandharpuri", "Mehsana", "Surti", "Nagpuri",
    "Toda", "Bhadawari"
]

ALL_BREEDS = CATTLE_BREEDS + BUFFALO_BREEDS


def download_kaggle_dataset(dataset_name: str, output_dir: str):
    """
    Download dataset from Kaggle.
    
    Args:
        dataset_name: Kaggle dataset name (e.g., 'username/dataset-name')
        output_dir: Output directory
    """
    try:
        import kaggle
        
        print(f"Downloading Kaggle dataset: {dataset_name}")
        kaggle.api.dataset_download_files(dataset_name, path=output_dir, unzip=True)
        print(f"Downloaded to {output_dir}")
        
    except ImportError:
        print("Error: Kaggle API not installed.")
        print("Install with: pip install kaggle")
        print("Then setup your API key from https://www.kaggle.com/settings")


def scrape_nbagr_images(output_dir: str, breeds: List[str] = None):
    """
    Scrape breed images from NBAGR website.
    
    Args:
        output_dir: Output directory
        breeds: List of breeds to scrape (all if None)
    """
    base_url = "https://nbagr.icar.gov.in"
    
    if breeds is None:
        breeds = ALL_BREEDS
    
    print(f"Scraping NBAGR for {len(breeds)} breeds...")
    
    for breed in breeds:
        breed_dir = os.path.join(output_dir, breed)
        os.makedirs(breed_dir, exist_ok=True)
        
        print(f"\nProcessing: {breed}")
        
        # Note: This is a placeholder. Actual scraping requires
        # inspecting the NBAGR website structure
        # You may need to adjust URLs and selectors
        
        try:
            # Search for breed page
            search_url = f"{base_url}/breed/{breed.lower().replace(' ', '-')}"
            
            response = requests.get(search_url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find images (adjust selector based on actual website)
                images = soup.find_all('img')
                
                for i, img in enumerate(images[:10]):  # Limit to 10 images
                    src = img.get('src')
                    
                    if src and ('cattle' in src.lower() or 'buffalo' in src.lower() or breed.lower() in src.lower()):
                        if not src.startswith('http'):
                            src = base_url + src
                        
                        # Download image
                        try:
                            img_response = requests.get(src, timeout=10)
                            
                            if img_response.status_code == 200:
                                ext = src.split('.')[-1].lower()
                                if ext not in ['jpg', 'jpeg', 'png']:
                                    ext = 'jpg'
                                
                                img_path = os.path.join(breed_dir, f"{breed}_{i}.{ext}")
                                
                                with open(img_path, 'wb') as f:
                                    f.write(img_response.content)
                                
                                print(f"  Downloaded: {img_path}")
                                
                                time.sleep(1)  # Be respectful
                                
                        except Exception as e:
                            print(f"  Error downloading image: {e}")
            
        except Exception as e:
            print(f"  Error scraping {breed}: {e}")
    
    print("\nScraping complete!")


def download_google_images(breed: str, output_dir: str, limit: int = 100):
    """
    Download images from Google Images (manual approach).
    
    Note: Automated Google Images scraping is against ToS.
    This provides instructions for manual collection.
    
    Args:
        breed: Breed name
        output_dir: Output directory
        limit: Number of images to download
    """
    breed_dir = os.path.join(output_dir, breed)
    os.makedirs(breed_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Manual Image Collection for: {breed}")
    print(f"{'='*60}")
    print(f"\nOutput Directory: {breed_dir}")
    print(f"Target: {limit} images")
    print(f"\nInstructions:")
    print(f"1. Go to: https://www.google.com/search?q={breed}+cattle+breed&tbm=isch")
    print(f"2. Search for '{breed} cattle breed' or '{breed} buffalo'")
    print(f"3. Download images manually to: {breed_dir}")
    print(f"4. Ensure images show clear side profile of the animal")
    print(f"5. Avoid duplicate or low-quality images")
    print(f"\nAlternative: Use browser extensions like:")
    print(f"  - 'Fatkun Batch Download Image' (Chrome)")
    print(f"  - 'DownThemAll' (Firefox)")
    print(f"{'='*60}\n")


def organize_dataset(raw_dir: str, output_dir: str):
    """
    Organize raw images into breed folders.
    
    Args:
        raw_dir: Raw images directory
        output_dir: Organized output directory
    """
    from pathlib import Path
    import shutil
    
    print(f"Organizing dataset from {raw_dir}...")
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(raw_dir).rglob(f'*{ext}'))
        image_files.extend(Path(raw_dir).rglob(f'*{ext.upper()}'))
    
    print(f"Found {len(image_files)} images")
    
    # Organize by breed name in filename
    for img_path in image_files:
        filename = img_path.stem.lower()
        
        # Try to match breed name
        matched_breed = None
        for breed in ALL_BREEDS:
            if breed.lower() in filename:
                matched_breed = breed
                break
        
        if matched_breed:
            dest_dir = os.path.join(output_dir, matched_breed)
            os.makedirs(dest_dir, exist_ok=True)
            
            dest_path = os.path.join(dest_dir, img_path.name)
            
            if not os.path.exists(dest_path):
                shutil.copy2(str(img_path), dest_path)
                print(f"  {img_path.name} -> {matched_breed}/")
    
    print("\nOrganization complete!")


def create_dataset_manifest(data_dir: str, output_file: str = None):
    """
    Create a manifest file listing all images and their labels.
    
    Args:
        data_dir: Dataset directory
        output_file: Output manifest file path
    """
    if output_file is None:
        output_file = os.path.join(data_dir, 'manifest.json')
    
    manifest = {
        'data_dir': data_dir,
        'breeds': {},
        'total_images': 0
    }
    
    for breed in os.listdir(data_dir):
        breed_dir = os.path.join(data_dir, breed)
        
        if os.path.isdir(breed_dir):
            images = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                images.extend(Path(breed_dir).glob(ext))
            
            manifest['breeds'][breed] = {
                'count': len(images),
                'images': [str(img.relative_to(data_dir)) for img in images]
            }
            manifest['total_images'] += len(images)
    
    with open(output_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Created manifest: {output_file}")
    print(f"Total images: {manifest['total_images']}")
    print(f"Total breeds: {len(manifest['breeds'])}")
    
    return manifest


def main():
    parser = argparse.ArgumentParser(description='Collect cattle breed images')
    parser.add_argument('--source', type=str, choices=['kaggle', 'nbagr', 'google', 'organize', 'manifest'],
                        required=True, help='Data source')
    parser.add_argument('--output', type=str, default='./data/raw', help='Output directory')
    parser.add_argument('--breed', type=str, help='Specific breed (for google source)')
    parser.add_argument('--dataset', type=str, help='Kaggle dataset name')
    parser.add_argument('--limit', type=int, default=100, help='Number of images to download')
    parser.add_argument('--input', type=str, help='Input directory (for organize)')
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    if args.source == 'kaggle':
        if args.dataset:
            download_kaggle_dataset(args.dataset, args.output)
        else:
            print("Popular Kaggle datasets for cattle breeds:")
            print("  - saurabhsharma/cattle-breed-classification")
            print("  - datasets/cow-images")
            print("\nUse --dataset to specify a dataset")
    
    elif args.source == 'nbagr':
        breeds = [args.breed] if args.breed else None
        scrape_nbagr_images(args.output, breeds)
    
    elif args.source == 'google':
        breeds = [args.breed] if args.breed else ALL_BREEDS
        for breed in breeds:
            download_google_images(breed, args.output, args.limit)
    
    elif args.source == 'organize':
        if args.input:
            organize_dataset(args.input, args.output)
        else:
            print("Please specify --input directory")
    
    elif args.source == 'manifest':
        create_dataset_manifest(args.output)


if __name__ == '__main__':
    main()
