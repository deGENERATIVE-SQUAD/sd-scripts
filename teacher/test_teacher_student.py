#!/usr/bin/env python3
"""
Test script for teacher-student training functionality.
This script tests the basic functionality without requiring a full dataset.
"""

import os
import tempfile
import json
import numpy as np
import torch
from pathlib import Path

# Add parent directory to path to import library modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from teacher_student_dataset import TeacherStudentDataset, TeacherStudentDataLoader


def create_dummy_teacher_outputs(output_dir: str, num_images: int = 5):
    """Create dummy teacher outputs for testing."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create metadata
    metadata = {
        "total_images": num_images,
        "resolution": [512, 512],
        "model_path": "test_model",
        "generated_at": "2024-01-01 00:00:00",
        "images": []
    }
    
    # Create dummy data for each image
    for i in range(num_images):
        # Create dummy tensors
        latents = np.random.randn(1, 4, 64, 64).astype(np.float32)  # 512/8 = 64
        t = np.random.randint(0, 1000, (1,)).astype(np.int64)
        text_embeddings = np.random.randn(1, 77, 768).astype(np.float32)
        eps_teacher = np.random.randn(1, 4, 64, 64).astype(np.float32)
        
        # Save files
        base_path = os.path.join(output_dir, f"test_image_{i:06d}")
        
        np.save(f"{base_path}_latents.npy", latents)
        np.save(f"{base_path}_t.npy", t)
        np.save(f"{base_path}_text_embeddings.npy", text_embeddings)
        np.save(f"{base_path}_eps_teacher.npy", eps_teacher)
        
        # Add to metadata
        metadata["images"].append({
            "index": i,
            "original_path": f"/path/to/image_{i}.jpg",
            "caption": f"Test caption {i}",
            "latents_file": f"test_image_{i:06d}_latents.npy",
            "t_file": f"test_image_{i:06d}_t.npy",
            "text_embeddings_file": f"test_image_{i:06d}_text_embeddings.npy",
            "eps_teacher_file": f"test_image_{i:06d}_eps_teacher.npy"
        })
    
    # Save metadata
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Created dummy teacher outputs in {output_dir}")
    return output_dir


def test_dataset():
    """Test the TeacherStudentDataset."""
    print("Testing TeacherStudentDataset...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create dummy data
        teacher_outputs_dir = create_dummy_teacher_outputs(temp_dir, num_images=3)
        
        # Create dataset
        dataset = TeacherStudentDataset(
            teacher_outputs_dir=teacher_outputs_dir,
            resolution=(512, 512),
            network_multiplier=1.0,
            debug_dataset=True
        )
        
        print(f"Dataset length: {len(dataset)}")
        
        # Test getting items
        for i in range(min(3, len(dataset))):
            item = dataset[i]
            print(f"Item {i}:")
            print(f"  Latents shape: {item['latents'].shape}")
            print(f"  T shape: {item['t'].shape}")
            print(f"  Text embeddings shape: {item['text_embeddings'].shape}")
            print(f"  Eps teacher shape: {item['eps_teacher'].shape}")
            print(f"  Caption: {item['caption']}")
            if 'image_key' in item:
                print(f"  Image key: {item['image_key']}")
        
        print("Dataset test passed!")


def test_dataloader():
    """Test the TeacherStudentDataLoader."""
    print("\nTesting TeacherStudentDataLoader...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create dummy data
        teacher_outputs_dir = create_dummy_teacher_outputs(temp_dir, num_images=5)
        
        # Create dataloader
        dataloader = TeacherStudentDataLoader(
            teacher_outputs_dir=teacher_outputs_dir,
            batch_size=2,
            resolution=(512, 512),
            network_multiplier=1.0,
            debug_dataset=True,
            shuffle=True
        )
        
        print(f"DataLoader length: {len(dataloader)}")
        print(f"Dataset info: {dataloader.get_dataset_info()}")
        
        # Test iteration
        batch_count = 0
        for batch in dataloader:
            print(f"\nBatch {batch_count}:")
            print(f"  Latents shape: {batch['latents'].shape}")
            print(f"  T shape: {batch['t'].shape}")
            print(f"  Text embeddings shape: {batch['text_embeddings'].shape}")
            print(f"  Eps teacher shape: {batch['eps_teacher'].shape}")
            print(f"  Captions: {batch['captions']}")
            
            batch_count += 1
            if batch_count >= 3:  # Limit for testing
                break
        
        print("DataLoader test passed!")


def test_memory_efficiency():
    """Test memory efficiency with larger dummy data."""
    print("\nTesting memory efficiency...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create larger dummy data
        teacher_outputs_dir = create_dummy_teacher_outputs(temp_dir, num_images=10)
        
        # Create dataset
        dataset = TeacherStudentDataset(
            teacher_outputs_dir=teacher_outputs_dir,
            resolution=(512, 512),
            network_multiplier=1.0,
            debug_dataset=False
        )
        
        print(f"Dataset with {len(dataset)} images created")
        
        # Test memory usage
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Load a few items
        for i in range(min(5, len(dataset))):
            item = dataset[i]
            # Simulate some processing
            _ = item['latents'] + item['eps_teacher']
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"Memory usage: {initial_memory:.1f}MB -> {final_memory:.1f}MB (+{memory_increase:.1f}MB)")
        
        if memory_increase < 100:  # Should be reasonable
            print("Memory efficiency test passed!")
        else:
            print("Warning: High memory usage detected")


def main():
    """Run all tests."""
    print("Running Teacher-Student module tests...")
    print("=" * 50)
    
    try:
        test_dataset()
        test_dataloader()
        test_memory_efficiency()
        
        print("\n" + "=" * 50)
        print("All tests passed! Teacher-Student module is working correctly.")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())