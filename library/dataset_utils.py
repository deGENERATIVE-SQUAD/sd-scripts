"""
Dataset utilities for teacher-student training.
Supports image-txt pairs and automatic caption extraction from folder names.
"""

import os
import re
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_dataset_folder_name(folder_name: str) -> Tuple[int, str]:
    """
    Parse dataset folder name in format "N_class" where N is the number of repetitions.
    
    Args:
        folder_name: Folder name like "1_woman", "5_cat", "10_landscape"
    
    Returns:
        Tuple of (repetitions, class_name)
    
    Examples:
        "1_woman" -> (1, "woman")
        "5_cat" -> (5, "cat")
        "10_landscape" -> (10, "landscape")
        "woman" -> (1, "woman")  # Default to 1 repetition
    """
    # Match pattern like "N_class" where N is a number
    match = re.match(r'^(\d+)_(.+)$', folder_name)
    
    if match:
        repetitions = int(match.group(1))
        class_name = match.group(2)
        return repetitions, class_name
    else:
        # If no number prefix, assume 1 repetition
        return 1, folder_name


def find_image_txt_pairs(dataset_dir: str) -> List[Tuple[str, str, str]]:
    """
    Find all image-txt pairs in a dataset directory.
    
    Args:
        dataset_dir: Path to dataset directory
    
    Returns:
        List of tuples: (image_path, txt_path, caption)
        If no txt file exists, caption will be extracted from folder name
    """
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists() or not dataset_path.is_dir():
        logger.warning(f"Dataset directory not found: {dataset_dir}")
        return []
    
    # Parse folder name for default caption
    folder_name = dataset_path.name
    repetitions, class_name = parse_dataset_folder_name(folder_name)
    default_caption = class_name
    
    logger.info(f"Dataset folder: {folder_name} -> {repetitions} repetitions of '{default_caption}'")
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
    
    # Supported text extensions
    text_extensions = {'.txt', '.caption'}
    
    pairs = []
    
    # Scan for image files
    for image_file in dataset_path.iterdir():
        if not image_file.is_file():
            continue
            
        # Check if it's an image
        if image_file.suffix.lower() not in image_extensions:
            continue
        
        image_path = str(image_file)
        base_name = image_file.stem
        
        # Look for corresponding txt file
        txt_path = None
        caption = default_caption
        
        # Check for txt file with same name
        for ext in text_extensions:
            potential_txt = image_file.with_suffix(ext)
            if potential_txt.exists():
                txt_path = str(potential_txt)
                try:
                    with open(potential_txt, 'r', encoding='utf-8') as f:
                        caption = f.read().strip()
                    logger.debug(f"Found caption for {image_file.name}: '{caption}'")
                    break
                except Exception as e:
                    logger.warning(f"Error reading txt file {potential_txt}: {e}")
                    # Fall back to default caption
                    caption = default_caption
                break
        
        # If no txt file found, use default caption
        if txt_path is None:
            logger.debug(f"No txt file found for {image_file.name}, using default caption: '{default_caption}'")
        
        pairs.append((image_path, txt_path, caption))
    
    logger.info(f"Found {len(pairs)} image-txt pairs in {dataset_dir}")
    return pairs


def create_dataset_metadata(dataset_dir: str, output_file: str = None) -> Dict[str, Dict]:
    """
    Create metadata dictionary from dataset directory.
    
    Args:
        dataset_dir: Path to dataset directory
        output_file: Optional path to save metadata JSON file
    
    Returns:
        Dictionary mapping image paths to metadata
    """
    pairs = find_image_txt_pairs(dataset_dir)
    
    metadata = {}
    for image_path, txt_path, caption in pairs:
        metadata[image_path] = {
            "caption": caption,
            "txt_file": txt_path,
            "dataset_folder": os.path.basename(dataset_dir)
        }
    
    # Save metadata if output file specified
    if output_file:
        import json
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved metadata to: {output_file}")
    
    return metadata


def scan_datasets_recursively(base_dir: str) -> List[Tuple[str, str, Dict]]:
    """
    Scan base directory recursively for dataset folders.
    
    Args:
        base_dir: Base directory to scan
    
    Returns:
        List of tuples: (dataset_path, dataset_name, metadata)
    """
    base_path = Path(base_dir)
    if not base_path.exists() or not base_path.is_dir():
        logger.warning(f"Base directory not found: {base_dir}")
        return []
    
    datasets = []
    
    # Scan for dataset folders (folders that might contain images)
    for item in base_path.iterdir():
        if not item.is_dir():
            continue
        
        # Check if folder contains images
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
        has_images = any(
            f.suffix.lower() in image_extensions 
            for f in item.iterdir() 
            if f.is_file()
        )
        
        if has_images:
            dataset_path = str(item)
            dataset_name = item.name
            
            # Create metadata for this dataset
            metadata = create_dataset_metadata(dataset_path)
            
            if metadata:  # Only add if images were found
                datasets.append((dataset_path, dataset_name, metadata))
                logger.info(f"Found dataset: {dataset_name} with {len(metadata)} images")
    
    logger.info(f"Found {len(datasets)} datasets in {base_dir}")
    return datasets


def get_caption_for_image(image_path: str, dataset_dir: str, metadata: Dict = None) -> str:
    """
    Get caption for a specific image.
    
    Args:
        image_path: Path to image file
        dataset_dir: Dataset directory path
        metadata: Optional metadata dictionary
    
    Returns:
        Caption for the image
    """
    # Try to get caption from metadata first
    if metadata and image_path in metadata:
        return metadata[image_path].get("caption", "")
    
    # Fall back to folder name parsing
    folder_name = os.path.basename(dataset_dir)
    _, class_name = parse_dataset_folder_name(folder_name)
    return class_name


def validate_dataset_structure(dataset_dir: str) -> bool:
    """
    Validate that dataset directory has proper structure.
    
    Args:
        dataset_dir: Path to dataset directory
    
    Returns:
        True if valid, False otherwise
    """
    if not os.path.exists(dataset_dir):
        logger.error(f"Dataset directory does not exist: {dataset_dir}")
        return False
    
    if not os.path.isdir(dataset_dir):
        logger.error(f"Dataset path is not a directory: {dataset_dir}")
        return False
    
    # Check if folder name follows expected format
    folder_name = os.path.basename(dataset_dir)
    repetitions, class_name = parse_dataset_folder_name(folder_name)
    
    if not class_name:
        logger.error(f"Invalid dataset folder name: {folder_name}")
        return False
    
    # Check if directory contains images
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
    has_images = False
    
    for file in os.listdir(dataset_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            has_images = True
            break
    
    if not has_images:
        logger.error(f"No images found in dataset directory: {dataset_dir}")
        return False
    
    logger.info(f"Dataset validation passed: {folder_name} -> {repetitions}x '{class_name}'")
    return True


def create_dataset_summary(datasets: List[Tuple[str, str, Dict]]) -> Dict:
    """
    Create summary of all datasets.
    
    Args:
        datasets: List of dataset tuples from scan_datasets_recursively
    
    Returns:
        Summary dictionary
    """
    summary = {
        "total_datasets": len(datasets),
        "total_images": 0,
        "datasets": []
    }
    
    for dataset_path, dataset_name, metadata in datasets:
        repetitions, class_name = parse_dataset_folder_name(dataset_name)
        image_count = len(metadata)
        
        dataset_info = {
            "name": dataset_name,
            "path": dataset_path,
            "class": class_name,
            "repetitions": repetitions,
            "image_count": image_count,
            "total_weight": repetitions * image_count
        }
        
        summary["datasets"].append(dataset_info)
        summary["total_images"] += image_count
    
    return summary


# Example usage and testing
if __name__ == "__main__":
    # Test the functions
    test_dir = "./test_dataset"
    
    print("Testing dataset utilities...")
    
    # Test folder name parsing
    test_names = ["1_woman", "5_cat", "10_landscape", "woman", "123_very_long_class_name"]
    for name in test_names:
        reps, cls = parse_dataset_folder_name(name)
        print(f"'{name}' -> {reps}x '{cls}'")
    
    # Test dataset scanning (if test directory exists)
    if os.path.exists(test_dir):
        print(f"\nScanning test directory: {test_dir}")
        datasets = scan_datasets_recursively(test_dir)
        
        for dataset_path, dataset_name, metadata in datasets:
            print(f"\nDataset: {dataset_name}")
            print(f"Path: {dataset_path}")
            print(f"Images: {len(metadata)}")
            
            # Show first few captions
            for i, (img_path, meta) in enumerate(list(metadata.items())[:3]):
                print(f"  {os.path.basename(img_path)}: '{meta['caption']}'")
    else:
        print(f"\nTest directory {test_dir} not found, skipping scan test")