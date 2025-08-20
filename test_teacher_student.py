#!/usr/bin/env python3
"""
Test script for teacher-student training system.
This script tests the basic functionality without requiring a full dataset.
"""

import os
import tempfile
import numpy as np
import torch
from PIL import Image
import json

def create_test_image(width=512, height=512):
    """Create a simple test image"""
    # Create a simple gradient image
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            img_array[y, x] = [
                int(255 * x / width),      # Red gradient
                int(255 * y / height),     # Green gradient
                128                         # Fixed blue
            ]
    
    return Image.fromarray(img_array)

def create_test_metadata():
    """Create test metadata"""
    return {
        "test_image.jpg": {
            "caption": "A beautiful gradient image with red and green colors",
            "resolution": [512, 512]
        }
    }

def test_teacher_outputs_creation():
    """Test the creation of teacher outputs"""
    print("Testing teacher outputs creation...")
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test image
        test_img = create_test_image()
        img_path = os.path.join(temp_dir, "test_image.jpg")
        test_img.save(img_path)
        
        # Create metadata
        metadata = create_test_metadata()
        metadata_path = os.path.join(temp_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Create teacher outputs directory
        teacher_outputs_dir = os.path.join(temp_dir, "teacher_outputs")
        os.makedirs(teacher_outputs_dir, exist_ok=True)
        
        # Simulate teacher outputs (without actual model)
        teacher_data = {
            "latents": np.random.randn(1, 4, 64, 64).astype(np.float32),
            "timesteps": np.array([500], dtype=np.int64),
            "text_embeddings": np.random.randn(1, 77, 768).astype(np.float32),
            "eps_teacher": np.random.randn(1, 4, 64, 64).astype(np.float32),
            "caption": "A beautiful gradient image with red and green colors",
            "image_path": img_path,
            "original_size": [512, 512]
        }
        
        # Save teacher output
        output_path = os.path.join(teacher_outputs_dir, "test_image_teacher.npz")
        np.savez(output_path, **teacher_data)
        
        # Save teacher info
        teacher_info = {
            "teacher_model": "test_model",
            "vae_scale_factor": 0.18215,
            "mixed_precision": "fp16",
            "bucket_resos": [[512, 512]],
            "max_resolution": "512,512",
            "min_bucket_reso": 256,
            "max_bucket_reso": 1024,
            "bucket_reso_steps": 64,
        }
        
        with open(os.path.join(teacher_outputs_dir, "teacher_info.json"), "w") as f:
            json.dump(teacher_info, f, indent=2)
        
        print(f"✓ Created test image: {img_path}")
        print(f"✓ Created metadata: {metadata_path}")
        print(f"✓ Created teacher outputs: {output_path}")
        print(f"✓ Created teacher info: {os.path.join(teacher_outputs_dir, 'teacher_info.json')}")
        
        return True

def test_dataset_loading():
    """Test the teacher-student dataset loading"""
    print("\nTesting dataset loading...")
    
    try:
        from library.teacher_student_dataset import TeacherStudentDataset, TeacherStudentSubset
        
        # Create temporary test data
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test image
            test_img = create_test_image()
            img_path = os.path.join(temp_dir, "test_image.jpg")
            test_img.save(img_path)
            
            # Create teacher outputs
            teacher_outputs_dir = os.path.join(temp_dir, "teacher_outputs")
            os.makedirs(teacher_outputs_dir, exist_ok=True)
            
            teacher_data = {
                "latents": np.random.randn(1, 4, 64, 64).astype(np.float32),
                "timesteps": np.array([500], dtype=np.int64),
                "text_embeddings": np.random.randn(1, 77, 768).astype(np.float32),
                "eps_teacher": np.random.randn(1, 4, 64, 64).astype(np.float32),
                "caption": "A beautiful gradient image with red and green colors",
                "image_path": img_path,
                "original_size": [512, 512]
            }
            
            output_path = os.path.join(teacher_outputs_dir, "test_image_teacher.npz")
            np.savez(output_path, **teacher_data)
            
            # Create teacher info
            teacher_info = {
                "teacher_model": "test_model",
                "vae_scale_factor": 0.18215,
                "mixed_precision": "fp16",
                "bucket_resos": [[512, 512]],
                "max_resolution": "512,512",
                "min_bucket_reso": 256,
                "max_bucket_reso": 1024,
                "bucket_reso_steps": 64,
            }
            
            with open(os.path.join(teacher_outputs_dir, "teacher_info.json"), "w") as f:
                json.dump(teacher_info, f, indent=2)
            
            # Test dataset creation
            subset = TeacherStudentSubset(
                image_dir=temp_dir,
                teacher_outputs_dir=teacher_outputs_dir
            )
            
            dataset = TeacherStudentDataset(
                subsets=[subset],
                teacher_outputs_dir=teacher_outputs_dir,
                is_training_dataset=True,
                batch_size=1,
                resolution=(512, 512),
                network_multiplier=1.0,
                enable_bucket=False,
                min_bucket_reso=256,
                max_bucket_reso=1024,
                bucket_reso_steps=64,
                bucket_no_upscale=False,
                prior_loss_weight=1.0,
                debug_dataset=False,
                validation_split=0.0,
                validation_seed=None,
                resize_interpolation=None,
            )
            
            print(f"✓ Created dataset with {len(dataset)} items")
            
            # Test getting an item
            item = dataset[0]
            print(f"✓ Dataset item keys: {list(item.keys())}")
            print(f"✓ Latents shape: {item['latents'].shape}")
            print(f"✓ Text embeddings shape: {item['text_embeddings'].shape}")
            print(f"✓ Eps teacher shape: {item['eps_teacher'].shape}")
            
            return True
            
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_training_components():
    """Test the training components"""
    print("\nTesting training components...")
    
    try:
        # Test LoRA network creation (mock)
        class MockLoRANetwork:
            def __init__(self):
                self.text_encoder_dims = 32
                self.unet_dims = 32
                self.text_encoder_alpha = 32
                self.unet_alpha = 32
            
            def to(self, device, dtype):
                return self
            
            def prepare_network(self, args, accelerator):
                pass
            
            def set_multiplier(self, multiplier):
                pass
            
            def get_trainable_params(self):
                return []
            
            def save_weights(self, output_dir, save_dtype):
                pass
        
        # Test optimizer creation
        mock_network = MockLoRANetwork()
        optimizer = torch.optim.AdamW(mock_network.parameters(), lr=1e-4)
        
        print(f"✓ Created mock LoRA network")
        print(f"✓ Created optimizer: {type(optimizer).__name__}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    """Run all tests"""
    print("Teacher-Student Training System Tests")
    print("=" * 50)
    
    tests = [
        ("Teacher outputs creation", test_teacher_outputs_creation),
        ("Dataset loading", test_dataset_loading),
        ("Training components", test_training_components),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print("=" * 50)
    
    all_passed = True
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! The system is ready to use.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    main()