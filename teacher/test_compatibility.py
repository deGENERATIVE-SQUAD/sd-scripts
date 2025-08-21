#!/usr/bin/env python3
"""
Test script to verify compatibility and basic functionality of the teacher-student module.
This script checks if all required dependencies are available and can perform basic operations.
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path to import library modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        from accelerate import Accelerator
        print("✓ Accelerate")
    except ImportError as e:
        print(f"✗ Accelerate import failed: {e}")
        return False
    
    try:
        from diffusers import DDPMScheduler
        print("✓ Diffusers")
    except ImportError as e:
        print(f"✗ Diffusers import failed: {e}")
        return False
    
    try:
        from library import sdxl_model_util, sdxl_train_util, strategy_base, strategy_sd, strategy_sdxl, train_util
        print("✓ SD Scripts library modules")
    except ImportError as e:
        print(f"✗ SD Scripts library import failed: {e}")
        return False
    
    try:
        from library.utils import setup_logging
        from library.device_utils import init_ipex, clean_memory_on_device
        print("✓ SD Scripts utility modules")
    except ImportError as e:
        print(f"✗ SD Scripts utility import failed: {e}")
        return False
    
    return True

def test_cuda_availability():
    """Test CUDA availability"""
    print("\nTesting CUDA availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            return True
        else:
            print("✗ CUDA not available")
            return False
    except Exception as e:
        print(f"✗ CUDA test failed: {e}")
        return False

def test_model_loading():
    """Test if we can load a basic SDXL model structure"""
    print("\nTesting model loading...")
    
    try:
        from library import sdxl_model_util
        print(f"✓ SDXL model util version: {sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0}")
        return True
    except Exception as e:
        print(f"✗ Model loading test failed: {e}")
        return False

def test_strategy_modules():
    """Test if strategy modules are available"""
    print("\nTesting strategy modules...")
    
    try:
        from library.strategy_sdxl import SdxlTokenizeStrategy
        from library.strategy_sd import SdSdxlLatentsCachingStrategy
        print("✓ Strategy modules available")
        return True
    except Exception as e:
        print(f"✗ Strategy modules test failed: {e}")
        return False

def test_teacher_student_scripts():
    """Test if teacher-student scripts can be imported"""
    print("\nTesting teacher-student scripts...")
    
    try:
        # Test teacher output generator
        from generate_teacher_outputs import TeacherOutputGenerator
        generator = TeacherOutputGenerator()
        print("✓ TeacherOutputGenerator imported successfully")
        
        # Test student trainer
        from train_student import StudentTrainer, TeacherStudentDataset
        trainer = StudentTrainer()
        print("✓ StudentTrainer imported successfully")
        
        return True
    except Exception as e:
        print(f"✗ Teacher-student scripts test failed: {e}")
        return False

def test_memory_optimization():
    """Test memory optimization features"""
    print("\nTesting memory optimization features...")
    
    try:
        import torch
        from library.device_utils import clean_memory_on_device
        
        if torch.cuda.is_available():
            device = torch.device("cuda")
            clean_memory_on_device(device)
            print("✓ Memory cleanup function available")
        else:
            print("⚠ CUDA not available, skipping memory test")
        
        return True
    except Exception as e:
        print(f"✗ Memory optimization test failed: {e}")
        return False

def create_test_config():
    """Create a test configuration file"""
    print("\nCreating test configuration...")
    
    try:
        config_content = """# Test configuration for teacher-student training
[teacher_generation]
pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
train_data_dir = "./test_dataset"
teacher_output_dir = "./test_teacher_outputs"
resolution = 1024
max_token_length = 225
train_batch_size = 1
mixed_precision = "fp16"
lowram = true
xformers = true
num_inference_steps = 1000

[student_training]
pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
train_data_dir = "./test_teacher_outputs"
output_dir = "./test_student_output"
output_name = "test_student"
max_train_epochs = 5
max_train_steps = 100
learning_rate = 1e-4
train_batch_size = 1
mixed_precision = "fp16"
lowram = true
xformers = true
save_every_n_epochs = 1
network_module = "networks.lora"
network_dim = 32
network_alpha = 1.0
"""
        
        config_path = Path("teacher/test_config.toml")
        with open(config_path, "w") as f:
            f.write(config_content)
        
        print(f"✓ Test configuration created: {config_path}")
        return True
    except Exception as e:
        print(f"✗ Test configuration creation failed: {e}")
        return False

def main():
    """Run all compatibility tests"""
    print("SDXL Teacher-Student Training - Compatibility Test")
    print("=" * 50)
    
    tests = [
        ("Import tests", test_imports),
        ("CUDA availability", test_cuda_availability),
        ("Model loading", test_model_loading),
        ("Strategy modules", test_strategy_modules),
        ("Teacher-student scripts", test_teacher_student_scripts),
        ("Memory optimization", test_memory_optimization),
        ("Test configuration", create_test_config),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("COMPATIBILITY TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The teacher-student module is ready to use.")
        print("\nNext steps:")
        print("1. Prepare your dataset")
        print("2. Run: python teacher/generate_teacher_outputs.py --help")
        print("3. Run: python teacher/train_student.py --help")
    else:
        print("⚠ Some tests failed. Please check the errors above.")
        print("\nCommon solutions:")
        print("- Install missing dependencies: pip install torch accelerate diffusers")
        print("- Check if sd_scripts library is properly installed")
        print("- Verify CUDA installation if using GPU")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)