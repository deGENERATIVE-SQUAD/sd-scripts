#!/usr/bin/env python3
"""
Quick start script for SDXL Teacher-Student training.
This script provides a simple interface to get started quickly.
"""

import os
import sys
import argparse
from pathlib import Path

def print_banner():
    """Print welcome banner"""
    print("=" * 60)
    print("🎓 SDXL Teacher-Student Training - Quick Start")
    print("=" * 60)
    print("This script will help you set up and run teacher-student training")
    print("for SDXL models using the DMD2 distillation method.")
    print()

def check_environment():
    """Check if the environment is ready"""
    print("🔍 Checking environment...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    # Check if we're in the right directory
    if not Path("teacher").exists():
        print("❌ Please run this script from the root of sd_scripts repository")
        return False
    print("✓ Running from sd_scripts repository")
    
    # Check if teacher scripts exist
    required_files = [
        "teacher/generate_teacher_outputs.py",
        "teacher/train_student.py",
        "teacher/run_teacher_student.py"
    ]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            print(f"❌ Required file not found: {file_path}")
            return False
    
    print("✓ All required files found")
    return True

def interactive_setup():
    """Interactive setup for quick start"""
    print("\n📝 Interactive Setup")
    print("-" * 30)
    
    config = {}
    
    # Teacher model
    print("\n1. Teacher Model Setup")
    print("   This is the model that will generate the training targets")
    teacher_model = input("   Enter path to teacher model (e.g., stabilityai/stable-diffusion-xl-base-1.0): ").strip()
    if not teacher_model:
        teacher_model = "stabilityai/stable-diffusion-xl-base-1.0"
    config['teacher_model'] = teacher_model
    
    # Dataset
    print("\n2. Dataset Setup")
    print("   This should contain images with caption files")
    dataset_path = input("   Enter path to your dataset directory: ").strip()
    if not dataset_path:
        print("   ❌ Dataset path is required!")
        return None
    config['dataset_path'] = dataset_path
    
    # Student base model
    print("\n3. Student Base Model Setup")
    print("   This is the base model that will be trained")
    student_base = input("   Enter path to student base model (default: same as teacher): ").strip()
    if not student_base:
        student_base = teacher_model
    config['student_base_model'] = student_base
    
    # Output directories
    print("\n4. Output Setup")
    teacher_output = input("   Enter directory for teacher outputs (default: ./teacher_outputs): ").strip()
    if not teacher_output:
        teacher_output = "./teacher_outputs"
    config['teacher_output_dir'] = teacher_output
    
    student_output = input("   Enter directory for student model (default: ./student_output): ").strip()
    if not student_output:
        student_output = "./student_output"
    config['student_output_dir'] = student_output
    
    # Training parameters
    print("\n5. Training Parameters")
    epochs = input("   Enter number of training epochs (default: 10): ").strip()
    config['max_epochs'] = int(epochs) if epochs.isdigit() else 10
    
    lr = input("   Enter learning rate (default: 1e-4): ").strip()
    config['learning_rate'] = float(lr) if lr.replace('e', '').replace('.', '').isdigit() else 1e-4
    
    # Network type
    print("\n6. Network Configuration")
    print("   Available options: lora, loha, lokr, or custom path")
    network_module = input("   Enter network module (default: networks.lora): ").strip()
    if not network_module:
        network_module = "networks.lora"
    config['network_module'] = network_module
    
    if network_module == "networks.lora":
        dim = input("   Enter LoRA dimension (default: 32): ").strip()
        config['network_dim'] = int(dim) if dim.isdigit() else 32
        
        alpha = input("   Enter LoRA alpha (default: 1.0): ").strip()
        config['network_alpha'] = float(alpha) if alpha.replace('.', '').isdigit() else 1.0
    
    # Memory optimization
    print("\n7. Memory Optimization")
    print("   For RTX 3060 12GB, these are recommended:")
    lowram = input("   Enable low RAM mode? (y/n, default: y): ").strip().lower()
    config['lowram'] = lowram != 'n'
    
    xformers = input("   Enable xformers? (y/n, default: y): ").strip().lower()
    config['xformers'] = xformers != 'n'
    
    return config

def generate_command(config):
    """Generate the command to run"""
    cmd = [
        "python", "teacher/run_teacher_student.py",
        "--teacher_model", config['teacher_model'],
        "--dataset_path", config['dataset_path'],
        "--student_base_model", config['student_base_model'],
        "--teacher_output_dir", config['teacher_output_dir'],
        "--student_output_dir", config['student_output_dir'],
        "--max_epochs", str(config['max_epochs']),
        "--learning_rate", str(config['learning_rate']),
        "--network_module", config['network_module'],
        "--network_dim", str(config['network_dim']),
        "--network_alpha", str(config['network_alpha']),
    ]
    
    if config['lowram']:
        cmd.append("--lowram")
    if config['xformers']:
        cmd.append("--xformers")
    
    return cmd

def save_config(config, filename="quick_start_config.toml"):
    """Save configuration to file"""
    config_content = f"""# Quick start configuration generated by quick_start.py

[teacher_generation]
pretrained_model_name_or_path = "{config['teacher_model']}"
train_data_dir = "{config['dataset_path']}"
teacher_output_dir = "{config['teacher_output_dir']}"
resolution = 1024
max_token_length = 225
train_batch_size = 1
mixed_precision = "fp16"
lowram = {str(config['lowram']).lower()}
xformers = {str(config['xformers']).lower()}
num_inference_steps = 1000

[student_training]
pretrained_model_name_or_path = "{config['student_base_model']}"
train_data_dir = "{config['teacher_output_dir']}"
output_dir = "{config['student_output_dir']}"
output_name = "student_model"
max_train_epochs = {config['max_epochs']}
max_train_steps = 1000
learning_rate = {config['learning_rate']}
train_batch_size = 1
mixed_precision = "fp16"
lowram = {str(config['lowram']).lower()}
xformers = {str(config['xformers']).lower()}
save_every_n_epochs = 1
network_module = "{config['network_module']}"
network_dim = {config['network_dim']}
network_alpha = {config['network_alpha']}
optimizer_type = "AdamW"
lr_scheduler_type = "cosine"
loss_type = "l2"
gradient_accumulation_steps = 4
"""
    
    config_path = Path("teacher") / filename
    with open(config_path, "w") as f:
        f.write(config_content)
    
    print(f"✓ Configuration saved to: {config_path}")
    return config_path

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Quick start for SDXL Teacher-Student training")
    parser.add_argument("--config", type=str, help="Path to existing config file")
    parser.add_argument("--run", action="store_true", help="Run training after setup")
    parser.add_argument("--test", action="store_true", help="Run compatibility test first")
    
    args = parser.parse_args()
    
    print_banner()
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment check failed. Please fix the issues above.")
        return False
    
    # Run compatibility test if requested
    if args.test:
        print("\n🧪 Running compatibility test...")
        test_cmd = ["python", "teacher/test_compatibility.py"]
        print(f"Command: {' '.join(test_cmd)}")
        print("Please run this command manually to test compatibility.")
        print()
    
    # Load existing config or create new one
    if args.config:
        print(f"📋 Loading configuration from: {args.config}")
        # TODO: Implement config loading
        print("Config loading not yet implemented. Please use interactive setup.")
        return False
    else:
        print("📋 No configuration provided. Starting interactive setup...")
        config = interactive_setup()
        if not config:
            print("❌ Setup cancelled.")
            return False
    
    # Save configuration
    config_path = save_config(config)
    
    # Generate command
    cmd = generate_command(config)
    
    print("\n🚀 Ready to start training!")
    print("=" * 60)
    print("Generated command:")
    print(" ".join(cmd))
    print()
    
    # Ask if user wants to run now
    if args.run:
        run_now = "y"
    else:
        run_now = input("Run training now? (y/n): ").strip().lower()
    
    if run_now == 'y':
        print("\n🎯 Starting training...")
        print("Note: This may take a long time depending on your dataset size.")
        print("You can monitor progress in the logs above.")
        print()
        
        # Import and run the main script
        try:
            sys.path.append("teacher")
            from run_teacher_student import main as run_main
            
            # Convert config to argparse namespace
            import argparse
            namespace = argparse.Namespace(**config)
            namespace.student_name = "student_model"
            namespace.resolution = 1024
            namespace.max_token_length = 225
            namespace.batch_size = 1
            namespace.mixed_precision = "fp16"
            namespace.num_inference_steps = 1000
            namespace.vae_batch_size = 1
            namespace.cache_text_encoder_outputs = False
            namespace.max_steps = 1000
            namespace.save_every_n_epochs = 1
            namespace.network_train_unet_only = False
            namespace.network_train_text_encoder_only = False
            namespace.gradient_accumulation_steps = 4
            namespace.optimizer_type = "AdamW"
            namespace.lr_scheduler_type = "cosine"
            namespace.loss_type = "l2"
            namespace.max_samples = None
            namespace.skip_teacher = False
            namespace.skip_student = False
            namespace.config_file = None
            namespace.network_args = []
            
            # Set global args for the main script
            import run_teacher_student
            run_teacher_student.args = namespace
            
            success = run_main()
            if success:
                print("\n🎉 Training completed successfully!")
            else:
                print("\n❌ Training failed. Check the logs above.")
            
            return success
            
        except Exception as e:
            print(f"❌ Error running training: {e}")
            print("Please run the command manually:")
            print(" ".join(cmd))
            return False
    else:
        print("\n📋 Training not started.")
        print("You can run it later with:")
        print(" ".join(cmd))
        print(f"\nOr use the saved configuration: {config_path}")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)