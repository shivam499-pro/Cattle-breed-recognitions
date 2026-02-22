"""
File Utilities
==============
Common file operations for the project.
"""

import os
import glob
from typing import List, Optional
import shutil


def ensure_dir(directory: str) -> str:
    """
    Ensure directory exists, create if not.
    
    Args:
        directory: Directory path
        
    Returns:
        Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    
    return directory


def get_image_files(directory: str,
                    extensions: List[str] = None) -> List[str]:
    """
    Get all image files in directory.
    
    Args:
        directory: Directory to search
        extensions: Allowed file extensions
        
    Returns:
        List of image file paths
    """
    if extensions is None:
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
    
    image_files = []
    
    for ext in extensions:
        pattern = os.path.join(directory, '**', ext)
        image_files.extend(glob.glob(pattern, recursive=True))
        
        # Also check uppercase
        pattern = os.path.join(directory, '**', ext.upper())
        image_files.extend(glob.glob(pattern, recursive=True))
    
    return sorted(set(image_files))


def copy_files(file_list: List[str],
               destination: str,
               preserve_structure: bool = False):
    """
    Copy files to destination.
    
    Args:
        file_list: List of file paths to copy
        destination: Destination directory
        preserve_structure: Preserve directory structure
    """
    ensure_dir(destination)
    
    for file_path in file_list:
        if preserve_structure:
            # Preserve relative path
            rel_path = os.path.relpath(file_path)
            dest_path = os.path.join(destination, rel_path)
            ensure_dir(os.path.dirname(dest_path))
        else:
            # Flat copy
            filename = os.path.basename(file_path)
            dest_path = os.path.join(destination, filename)
        
        shutil.copy2(file_path, dest_path)
    
    print(f"Copied {len(file_list)} files to {destination}")


def count_files_by_class(data_dir: str) -> dict:
    """
    Count files in each class subdirectory.
    
    Args:
        data_dir: Root data directory with class subdirectories
        
    Returns:
        Dictionary with class names and file counts
    """
    counts = {}
    
    if not os.path.exists(data_dir):
        print(f"Directory not found: {data_dir}")
        return counts
    
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        
        if os.path.isdir(class_dir):
            files = get_image_files(class_dir)
            counts[class_name] = len(files)
    
    return counts


def split_dataset(source_dir: str,
                  train_dir: str,
                  val_dir: str,
                  test_dir: str,
                  train_ratio: float = 0.7,
                  val_ratio: float = 0.15,
                  test_ratio: float = 0.15,
                  seed: int = 42):
    """
    Split dataset into train/val/test directories.
    
    Args:
        source_dir: Source directory with class subdirectories
        train_dir: Training output directory
        val_dir: Validation output directory
        test_dir: Test output directory
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed
    """
    import random
    
    random.seed(seed)
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1"
    
    for class_name in os.listdir(source_dir):
        class_dir = os.path.join(source_dir, class_name)
        
        if not os.path.isdir(class_dir):
            continue
        
        # Get all files
        files = get_image_files(class_dir)
        random.shuffle(files)
        
        # Split
        n_total = len(files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]
        
        # Copy to destinations
        for split_name, split_dir, split_files in [
            ('train', train_dir, train_files),
            ('val', val_dir, val_files),
            ('test', test_dir, test_files)
        ]:
            dest_class_dir = os.path.join(split_dir, class_name)
            ensure_dir(dest_class_dir)
            
            for file_path in split_files:
                filename = os.path.basename(file_path)
                dest_path = os.path.join(dest_class_dir, filename)
                shutil.copy2(file_path, dest_path)
        
        print(f"{class_name}: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")


def create_dataset_structure(base_dir: str,
                             classes: List[str],
                             splits: List[str] = ['train', 'val', 'test']):
    """
    Create dataset directory structure.
    
    Args:
        base_dir: Base directory
        classes: List of class names
        splits: List of split names
    """
    for split in splits:
        split_dir = os.path.join(base_dir, split)
        ensure_dir(split_dir)
        
        for class_name in classes:
            class_dir = os.path.join(split_dir, class_name)
            ensure_dir(class_dir)
    
    print(f"Created dataset structure at {base_dir}")


def get_dataset_stats(data_dir: str) -> dict:
    """
    Get dataset statistics.
    
    Args:
        data_dir: Root data directory
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        'total_images': 0,
        'num_classes': 0,
        'class_distribution': {},
        'splits': {}
    }
    
    # Check for split directories
    splits = ['train', 'val', 'test']
    has_splits = all(os.path.exists(os.path.join(data_dir, s)) for s in splits)
    
    if has_splits:
        for split in splits:
            split_dir = os.path.join(data_dir, split)
            split_counts = count_files_by_class(split_dir)
            
            stats['splits'][split] = {
                'total': sum(split_counts.values()),
                'distribution': split_counts
            }
            
            stats['total_images'] += sum(split_counts.values())
    else:
        # Single directory structure
        class_counts = count_files_by_class(data_dir)
        stats['class_distribution'] = class_counts
        stats['total_images'] = sum(class_counts.values())
        stats['num_classes'] = len(class_counts)
    
    return stats


def print_dataset_stats(data_dir: str):
    """
    Print dataset statistics.
    
    Args:
        data_dir: Root data directory
    """
    stats = get_dataset_stats(data_dir)
    
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    print(f"Total Images: {stats['total_images']}")
    
    if stats['splits']:
        for split, split_stats in stats['splits'].items():
            print(f"\n{split.upper()}:")
            print(f"  Total: {split_stats['total']}")
            for class_name, count in split_stats['distribution'].items():
                print(f"    {class_name}: {count}")
    else:
        print(f"Number of Classes: {stats['num_classes']}")
        print("\nClass Distribution:")
        for class_name, count in stats['class_distribution'].items():
            print(f"  {class_name}: {count}")
    
    print("=" * 60)
