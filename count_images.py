"""Count images in each subfolder of data/train"""
import os

train_dir = 'data/train'

print("=" * 50)
print("Image Count in data/train")
print("=" * 50)

total = 0
counts = []

for breed in sorted(os.listdir(train_dir)):
    breed_path = os.path.join(train_dir, breed)
    if os.path.isdir(breed_path):
        images = [f for f in os.listdir(breed_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        count = len(images)
        counts.append((breed, count))
        total += count
        print(f"{breed:25s}: {count:4d}")

print("=" * 50)
print(f"{'TOTAL':25s}: {total:4d}")
print(f"Breeds: {len(counts)}")
