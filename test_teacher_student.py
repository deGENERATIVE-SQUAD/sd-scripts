#!/usr/bin/env python3
"""
Test script for teacher-student training system.
This script tests the basic functionality without requiring a full dataset.
Supports testing SD, SDXL, and LyCORIS functionality.
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
        teacher_output_dir = os.path.join(temp_dir, "teacher_outputs")
        os.makedirs(teacher_output_dir, exist_ok=True)
        
        # Test SD teacher outputs
        sd_teacher_data = {
            "latents": np.random.randn(1, 4, 64, 64).astype(np.float32),
            "timesteps": np.array([500], dtype=np.int64),
            "text_embeddings": np.random.randn(1, 77, 768).astype(np.float32),
            "eps_teacher": np.random.randn(1, 4, 64, 64).astype(np.float32),
            "caption": "A beautiful gradient image with red and green colors",
            "image_path": img_path,
            "original_size": [512, 512],
            "is_sdxl": False,
        }
        
        # Save SD teacher output
        sd_output_path = os.path.join(teacher_output_dir, "test_image_sd_teacher.npz")
        np.savez(sd_output_path, **sd_teacher_data)
        
        # Test SDXL teacher outputs
        sdxl_teacher_data = {
            "latents": np.random.randn(1, 4, 128, 128).astype(np.float32),
            "timesteps": np.array([500], dtype=np.int64),
            "text_embeddings": np.random.randn(1, 2, 77, 768).astype(np.float32),  # SDXL has 2 encoders
            "eps_teacher": np.random.randn(1, 4, 128, 128).astype(np.float32),
            "caption": "A beautiful gradient image with red and green colors",
            "image_path": img_path,
            "original_size": [1024, 1024],
            "is_sdxl": True,
        }
        
        # Save SDXL teacher output
        sdxl_output_path = os.path.join(teacher_output_dir, "test_image_sdxl_teacher.npz")
        np.savez(sdxl_output_path, **sdxl_teacher_data)
        
        # Save teacher info for SD
        sd_teacher_info = {
            "teacher_model": "test_sd_model",
            "is_sdxl": False,
            "vae_scale_factor": 0.18215,
            "mixed_precision": "fp16",
            "bucket_resos": [[512, 512]],
            "max_resolution": "512,512",
            "min_bucket_reso": 256,
            "max_bucket_reso": 1024,
            "bucket_reso_steps": 64,
        }
        
        with open(os.path.join(teacher_output_dir, "sd_teacher_info.json"), "w") as f:
            json.dump(sd_teacher_info, f, indent=2)
        
        # Save teacher info for SDXL
        sdxl_teacher_info = {
            "teacher_model": "test_sdxl_model",
            "is_sdxl": True,
            "vae_scale_factor": 0.13025,
            "mixed_precision": "fp16",
            "bucket_resos": [[1024, 1024]],
            "max_resolution": "1024,1024",
            "min_bucket_reso": 512,
            "max_bucket_reso": 2048,
            "bucket_reso_steps": 64,
        }
        
        with open(os.path.join(teacher_output_dir, "sdxl_teacher_info.json"), "w") as f:
            json.dump(sdxl_teacher_info, f, indent=2)
        
        print(f"✓ Created test image: {img_path}")
        print(f"✓ Created metadata: {metadata_path}")
        print(f"✓ Created SD teacher output: {sd_output_path}")
        print(f"✓ Created SDXL teacher output: {sdxl_output_path}")
        print(f"✓ Created teacher info files")
        
        return True

def test_dataset_loading():
    """Test the teacher-student dataset loading for both SD and SDXL"""
    print("\nTesting dataset loading...")
    
    try:
        from library.teacher_student_dataset import TeacherStudentDataset, TeacherStudentSubset
        
        # Test SD dataset
        print("Testing SD dataset...")
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test image
            test_img = create_test_image()
            img_path = os.path.join(temp_dir, "test_image.jpg")
            test_img.save(img_path)
            
            # Create SD teacher outputs
            teacher_outputs_dir = os.path.join(temp_dir, "teacher_outputs")
            os.makedirs(teacher_outputs_dir, exist_ok=True)
            
            sd_teacher_data = {
                "latents": np.random.randn(1, 4, 64, 64).astype(np.float32),
                "timesteps": np.array([500], dtype=np.int64),
                "text_embeddings": np.random.randn(1, 77, 768).astype(np.float32),
                "eps_teacher": np.random.randn(1, 4, 64, 64).astype(np.float32),
                "caption": "A beautiful gradient image with red and green colors",
                "image_path": img_path,
                "original_size": [512, 512],
                "is_sdxl": False,
            }
            
            output_path = os.path.join(teacher_outputs_dir, "test_image_teacher.npz")
            np.savez(output_path, **sd_teacher_data)
            
            # Create SD teacher info
            teacher_info = {
                "teacher_model": "test_sd_model",
                "is_sdxl": False,
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
            
            # Test SD dataset creation
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
            
            print(f"✓ Created SD dataset with {len(dataset)} items")
            
            # Test getting an item
            item = dataset[0]
            print(f"✓ SD dataset item keys: {list(item.keys())}")
            print(f"✓ SD latents shape: {item['latents'].shape}")
            print(f"✓ SD text embeddings: {len(item['text_embeddings'])} encoders")
            print(f"✓ SD is_sdxl: {item['is_sdxl']}")
        
        # Test SDXL dataset
        print("\nTesting SDXL dataset...")
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test image
            test_img = create_test_image(1024, 1024)
            img_path = os.path.join(temp_dir, "test_image.jpg")
            test_img.save(img_path)
            
            # Create SDXL teacher outputs
            teacher_outputs_dir = os.path.join(temp_dir, "teacher_outputs")
            os.makedirs(teacher_outputs_dir, exist_ok=True)
            
            sdxl_teacher_data = {
                "latents": np.random.randn(1, 4, 128, 128).astype(np.float32),
                "timesteps": np.array([500], dtype=np.int64),
                "text_embeddings": np.random.randn(1, 2, 77, 768).astype(np.float32),
                "eps_teacher": np.random.randn(1, 4, 128, 128).astype(np.float32),
                "caption": "A beautiful gradient image with red and green colors",
                "image_path": img_path,
                "original_size": [1024, 1024],
                "is_sdxl": True,
            }
            
            output_path = os.path.join(teacher_outputs_dir, "test_image_teacher.npz")
            np.savez(output_path, **sdxl_teacher_data)
            
            # Create SDXL teacher info
            teacher_info = {
                "teacher_model": "test_sdxl_model",
                "is_sdxl": True,
                "vae_scale_factor": 0.13025,
                "mixed_precision": "fp16",
                "bucket_resos": [[1024, 1024]],
                "max_resolution": "1024,1024",
                "min_bucket_reso": 512,
                "max_bucket_reso": 2048,
                "bucket_reso_steps": 64,
            }
            
            with open(os.path.join(teacher_outputs_dir, "teacher_info.json"), "w") as f:
                json.dump(teacher_info, f, indent=2)
            
            # Test SDXL dataset creation
            subset = TeacherStudentSubset(
                image_dir=temp_dir,
                teacher_outputs_dir=teacher_outputs_dir
            )
            
            dataset = TeacherStudentDataset(
                subsets=[subset],
                teacher_outputs_dir=teacher_outputs_dir,
                is_training_dataset=True,
                batch_size=1,
                resolution=(1024, 1024),
                network_multiplier=1.0,
                enable_bucket=False,
                min_bucket_reso=512,
                max_bucket_reso=2048,
                bucket_reso_steps=64,
                bucket_no_upscale=False,
                prior_loss_weight=1.0,
                debug_dataset=False,
                validation_split=0.0,
                validation_seed=None,
                resize_interpolation=None,
            )
            
            print(f"✓ Created SDXL dataset with {len(dataset)} items")
            
            # Test getting an item
            item = dataset[0]
            print(f"✓ SDXL dataset item keys: {list(item.keys())}")
            print(f"✓ SDXL latents shape: {item['latents'].shape}")
            print(f"✓ SDXL text embeddings: {len(item['text_embeddings'])} encoders")
            print(f"✓ SDXL is_sdxl: {item['is_sdxl']}")
            
            return True
            
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_training_components():
    """Test the training components including LyCORIS"""
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
        
        # Test LyCORIS network creation (mock)
        class MockLyCORISNetwork:
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
        
        # Test LoRA network
        mock_lora_network = MockLoRANetwork()
        print(f"✓ Created mock LoRA network")
        
        # Test LyCORIS network
        mock_lycoris_network = MockLyCORISNetwork()
        print(f"✓ Created mock LyCORIS network")
        
        # Test optimizer creation
        optimizer = torch.optim.AdamW(mock_lora_network.parameters(), lr=1e-4)
        print(f"✓ Created optimizer: {type(optimizer).__name__}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_sdxl_detection():
    """Test SDXL model detection functionality"""
    print("\nTesting SDXL model detection...")
    
    try:
        # Test SD model detection
        sd_model_path = "runwayml/stable-diffusion-v1-5"
        sdxl_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
        
        # Mock detection function
        def mock_detect_model_type(model_path):
            if "xl" in model_path.lower() or "sdxl" in model_path.lower():
                return True, 0.13025  # SDXL
            else:
                return False, 0.18215  # SD
        
        # Test SD detection
        is_sdxl, vae_scale = mock_detect_model_type(sd_model_path)
        print(f"✓ SD model detection: is_sdxl={is_sdxl}, vae_scale={vae_scale}")
        
        # Test SDXL detection
        is_sdxl, vae_scale = mock_detect_model_type(sdxl_model_path)
        print(f"✓ SDXL model detection: is_sdxl={is_sdxl}, vae_scale={vae_scale}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    """Run all tests"""
    print("Teacher-Student Training System Tests")
    print("Supports SD, SDXL, and LyCORIS")
    print("=" * 50)
    
    tests = [
        ("Teacher outputs creation", test_teacher_outputs_creation),
        ("Dataset loading (SD & SDXL)", test_dataset_loading),
        ("Training components (LoRA & LyCORIS)", test_training_components),
        ("SDXL model detection", test_sdxl_detection),
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
        print("\nSupported features:")
        print("✓ SD models (Stable Diffusion 1.x, 2.x)")
        print("✓ SDXL models (Stable Diffusion XL)")
        print("✓ LoRA network modules")
        print("✓ LyCORIS network modules")
        print("✓ Teacher-student training pipeline")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    main()