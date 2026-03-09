"""
Breed Image Downloader
=====================
Downloads cattle breed images from Bing using icrawler.
"""

import os
import sys
from icrawler.builtin import BingImageCrawler

# List of breeds to download
breeds = [
    "Alambadi", "Amritmahal", "Ayrshire", "Banni", "Bargur", 
    "Bhadawari", "Brahman", "Brahman Cross", "Brown_Swiss", 
    "Chhattisgarhi", "Chilika", "Cholistani", "Dangi", "Deoni", 
    "Dhani", "Fresian Cross", "Gir", "Gojri", "Guernsey", "Hallikar", 
    "Hariana", "Holstein_Friesian", "Jaffarabadi", "Jersey", "Kalahandi", 
    "Kangayam", "Kankrej", "Kasargod", "Kenkatha", "Kherigarh", 
    "Khillari", "Krishna_Valley", "Luit", "Malnad_gidda", "Marathwada", 
    "Mehsana", "Murrah", "Nagpuri", "Nili-ravi", "Pandharpuri", 
    "Sahiwal", "Sahiwal Cross", "Sibbi", "Surti", "Toda"
]

# Settings
MAX_IMAGES = 400  # Images per breed
TRAIN_DIR = 'data/train'


def get_existing_count(breed_folder):
    """Get existing image count in breed folder."""
    if not os.path.exists(breed_folder):
        return 0
    return len([f for f in os.listdir(breed_folder) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])


def download_breed_images(breed, max_images=400):
    """Download images for a single breed."""
    # Create folder
    breed_folder = os.path.join(TRAIN_DIR, breed)
    os.makedirs(breed_folder, exist_ok=True)
    
    # Get existing count
    existing = get_existing_count(breed_folder)
    print(f"\n{breed}:")
    print(f"  Existing images: {existing}")
    
    # Calculate how many more to download
    to_download = max_images - existing
    if to_download <= 0:
        print(f"  ✓ Already has {existing} images, skipping")
        return
    
    # Use file_idx_offset='auto' to avoid overwriting
    # But since we want to add NEW images, let's use offset based on existing count
    
    # Search keyword: breed name + "cattle photo"
    keyword = f"{breed} cattle photo"
    
    print(f"  Downloading {to_download} more images...")
    print(f"  Keyword: '{keyword}'")
    
    # Create crawler
    storage = {'root_dir': breed_folder}
    crawler = BingImageCrawler(storage=storage)
    
    # Download
    crawler.crawl(keyword=keyword, max_num=to_download, file_idx_offset='auto')
    
    # Final count
    final = get_existing_count(breed_folder)
    print(f"  ✓ Total now: {final} images")


def main():
    """Download images for all breeds."""
    print("=" * 60)
    print("CATTLE BREED IMAGE DOWNLOADER")
    print("=" * 60)
    print(f"Breeds: {len(breeds)}")
    print(f"Images per breed: {MAX_IMAGES}")
    print(f"Total to download: {len(breeds) * MAX_IMAGES}")
    print("=" * 60)
    
    # Check if icrawler is installed
    try:
        from icrawler import __version__
        print(f"icrawler version: {__version__}")
    except ImportError:
        print("ERROR: icrawler not installed!")
        print("Install with: pip install icrawler")
        return
    
    # Download each breed
    for i, breed in enumerate(breeds, 1):
        print(f"\n[{i}/{len(breeds)}] ", end="")
        download_breed_images(breed, MAX_IMAGES)
    
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE!")
    print("=" * 60)
    
    # Summary
    print("\nFinal dataset summary:")
    total = 0
    for breed in sorted(breeds):
        breed_folder = os.path.join(TRAIN_DIR, breed)
        count = get_existing_count(breed_folder)
        total += count
        print(f"  {breed}: {count}")
    print(f"\nTotal images: {total}")


if __name__ == '__main__':
    main()
