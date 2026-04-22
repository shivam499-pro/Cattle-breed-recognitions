import os, shutil, glob

def merge_breeds(src_list, target_folder, base_path):
    """Merge multiple breed folders into one, handling duplicate filenames."""
    os.makedirs(target_folder, exist_ok=True)
    
    for src in src_list:
        src_path = os.path.join(base_path, src)
        if not os.path.exists(src_path):
            print(f'Skipping {src} - not found')
            continue
            
        files = glob.glob(os.path.join(src_path, '*'))
        for f in files:
            filename = os.path.basename(f)
            target_path = os.path.join(target_folder, filename)
            
            # Handle duplicate filenames by adding prefix
            if os.path.exists(target_path):
                name, ext = os.path.splitext(filename)
                counter = 1
                new_filename = f"{name}_{counter}{ext}"
                target_path = os.path.join(target_folder, new_filename)
                while os.path.exists(target_path):
                    counter += 1
                    new_filename = f"{name}_{counter}{ext}"
                    target_path = os.path.join(target_folder, new_filename)
            
            shutil.move(f, target_path)
            print(f'Moved {filename} -> {os.path.basename(target_folder)}')
        
        # Remove empty source directory
        if os.path.exists(src_path) and not os.listdir(src_path):
            os.rmdir(src_path)
            print(f'Removed empty folder: {src}')

# Process each split
for split in ['train_final_v2', 'val_final_v2', 'test_final_v2']:
    base = f'data/{split}'
    print(f'\nProcessing {split}...')
    
    # Merge Hallikar + Amritmahal -> Draft_South
    merge_breeds(['Hallikar', 'Amritmahal'], f'{base}/Draft_South', base)
    
    # Merge Kankrej + Kangayam -> K_Draft
    merge_breeds(['Kankrej', 'Kangayam'], f'{base}/K_Draft', base)

print('\n=== All merges complete! ===')