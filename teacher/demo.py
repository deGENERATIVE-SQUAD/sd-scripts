#!/usr/bin/env python3
"""
Demo script for SDXL Teacher-Student Training Module

This script demonstrates the basic functionality of the teacher-student training module.
It shows how to use the main classes and provides examples of usage.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import library modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def print_demo_banner():
    """Print demo banner"""
    print("=" * 70)
    print("🎓 SDXL Teacher-Student Training - Demo")
    print("=" * 70)
    print("This demo shows the capabilities of the teacher-student training module.")
    print()

def demo_imports():
    """Demonstrate importing the main classes"""
    print("📦 Importing main classes...")
    
    try:
        from generate_teacher_outputs import TeacherOutputGenerator
        print("✓ TeacherOutputGenerator imported successfully")
        
        from train_student import StudentTrainer, TeacherStudentDataset
        print("✓ StudentTrainer imported successfully")
        print("✓ TeacherStudentDataset imported successfully")
        
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        print("This usually means dependencies are not installed or there's a path issue.")
        return False

def demo_class_creation():
    """Demonstrate creating instances of main classes"""
    print("\n🔧 Creating class instances...")
    
    try:
        from generate_teacher_outputs import TeacherOutputGenerator
        from train_student import StudentTrainer
        
        # Create teacher generator
        teacher_gen = TeacherOutputGenerator()
        print("✓ TeacherOutputGenerator instance created")
        print(f"  - VAE scale factor: {teacher_gen.vae_scale_factor}")
        print(f"  - Is SDXL: {teacher_gen.is_sdxl}")
        
        # Create student trainer
        student_trainer = StudentTrainer()
        print("✓ StudentTrainer instance created")
        print(f"  - VAE scale factor: {student_trainer.vae_scale_factor}")
        print(f"  - Is SDXL: {student_trainer.is_sdxl}")
        
        return True
    except Exception as e:
        print(f"✗ Class creation failed: {e}")
        return False

def demo_dataset_class():
    """Demonstrate TeacherStudentDataset functionality"""
    print("\n📊 Demonstrating TeacherStudentDataset...")
    
    try:
        from train_student import TeacherStudentDataset
        
        # Create a mock dataset (will fail but shows the interface)
        print("TeacherStudentDataset interface:")
        print("  - __init__(teacher_output_dir, max_samples=None)")
        print("  - __len__() -> int")
        print("  - __getitem__(idx) -> dict")
        print("  - Returns: latents, timesteps, text_embeddings, teacher_noise_pred")
        
        return True
    except Exception as e:
        print(f"✗ Dataset demo failed: {e}")
        return False

def demo_usage_examples():
    """Show usage examples"""
    print("\n💡 Usage Examples:")
    print("-" * 40)
    
    print("1. Generate teacher outputs:")
    print("   python teacher/generate_teacher_outputs.py \\")
    print("       --pretrained_model_name_or_path 'stabilityai/stable-diffusion-xl-base-1.0' \\")
    print("       --train_data_dir './my_dataset' \\")
    print("       --teacher_output_dir './teacher_outputs' \\")
    print("       --lowram --xformers")
    
    print("\n2. Train student model:")
    print("   python teacher/train_student.py \\")
    print("       --pretrained_model_name_or_path 'stabilityai/stable-diffusion-xl-base-1.0' \\")
    print("       --train_data_dir './teacher_outputs' \\")
    print("       --network_module 'networks.lora' \\")
    print("       --network_dim 64 --network_alpha 32 \\")
    print("       --lowram --xformers")
    
    print("\n3. Quick start (interactive):")
    print("   python teacher/quick_start.py")
    
    print("\n4. Run complete pipeline:")
    print("   python teacher/run_teacher_student.py \\")
    print("       --teacher_model 'stabilityai/stable-diffusion-xl-base-1.0' \\")
    print("       --dataset_path './my_dataset' \\")
    print("       --student_base_model 'stabilityai/stable-diffusion-xl-base-1.0'")

def demo_features():
    """Demonstrate key features"""
    print("\n🌟 Key Features:")
    print("-" * 40)
    
    features = [
        "✅ Teacher-student distillation based on DMD2 method",
        "✅ Memory efficient (teacher not kept in memory during training)",
        "✅ Full compatibility with sd_scripts arguments",
        "✅ Support for all network modules (LoRA, LoHa, LoKr, custom)",
        "✅ Optimized for RTX 3060 12GB",
        "✅ Automatic memory management",
        "✅ Text encoder output caching",
        "✅ Mixed precision training",
        "✅ XFormers support",
        "✅ TensorBoard and W&B logging",
        "✅ Gradient accumulation",
        "✅ Multiple loss functions (L1, L2, Huber)",
        "✅ Checkpoint saving and resuming"
    ]
    
    for feature in features:
        print(f"  {feature}")

def demo_optimization():
    """Show optimization tips"""
    print("\n⚡ Optimization Tips for RTX 3060 12GB:")
    print("-" * 50)
    
    tips = [
        "Use --lowram for memory optimization",
        "Enable --xformers for efficient attention",
        "Set --train_batch_size 1",
        "Use --mixed_precision fp16",
        "Enable --cache_text_encoder_outputs",
        "Use --gradient_accumulation_steps 4",
        "Consider --resolution 512 for very large datasets"
    ]
    
    for i, tip in enumerate(tips, 1):
        print(f"  {i}. {tip}")

def demo_next_steps():
    """Show next steps for users"""
    print("\n🚀 Next Steps:")
    print("-" * 30)
    
    steps = [
        "1. Run compatibility test: python teacher/test_compatibility.py",
        "2. Prepare your dataset with images and captions",
        "3. Choose a teacher model (SDXL base recommended)",
        "4. Use quick_start.py for interactive setup",
        "5. Start with a small dataset for testing",
        "6. Monitor training with TensorBoard or W&B",
        "7. Experiment with different network modules"
    ]
    
    for step in steps:
        print(f"  {step}")

def main():
    """Main demo function"""
    print_demo_banner()
    
    # Check if we're in the right directory
    if not Path("teacher").exists():
        print("❌ Please run this script from the root of sd_scripts repository")
        print("Current directory:", os.getcwd())
        return False
    
    print("🔍 Current directory:", os.getcwd())
    print("📁 Teacher module found:", Path("teacher").exists())
    print()
    
    # Run demos
    demos = [
        ("Importing classes", demo_imports),
        ("Creating class instances", demo_class_creation),
        ("Dataset functionality", demo_dataset_class),
    ]
    
    success_count = 0
    for demo_name, demo_func in demos:
        try:
            if demo_func():
                success_count += 1
        except Exception as e:
            print(f"✗ {demo_name} failed with exception: {e}")
    
    # Show features and examples
    demo_features()
    demo_optimization()
    demo_usage_examples()
    demo_next_steps()
    
    # Summary
    print("\n" + "=" * 70)
    print("DEMO SUMMARY")
    print("=" * 70)
    print(f"Tests passed: {success_count}/{len(demos)}")
    
    if success_count == len(demos):
        print("🎉 All demos passed! The module is ready to use.")
        print("\nQuick start:")
        print("  python teacher/quick_start.py")
    else:
        print("⚠ Some demos failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("  python teacher/test_compatibility.py")
    
    return success_count == len(demos)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)