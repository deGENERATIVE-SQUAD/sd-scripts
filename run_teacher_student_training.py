#!/usr/bin/env python3
"""
Complete teacher-student training pipeline runner.
This script automates the entire process from teacher outputs preparation to student training.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(command)}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(command, check=True, capture_output=False)
        elapsed_time = time.time() - start_time
        print(f"\n✅ {description} completed successfully in {elapsed_time:.2f} seconds")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n❌ Command not found: {command[0]}")
        print("Please make sure the script is in the correct directory")
        return False

def check_requirements():
    """Check if required files and directories exist"""
    print("🔍 Checking requirements...")
    
    required_files = [
        "finetune/prepare_teacher_outputs.py",
        "library/teacher_student_dataset.py",
        "train_network_teacher_student.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        print("Please make sure you're in the correct directory")
        return False
    
    print("✅ All required files found")
    return True

def create_directories(output_dir, teacher_outputs_dir):
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(teacher_outputs_dir, exist_ok=True)
    
    print(f"✅ Created directories:")
    print(f"   - Output: {output_dir}")
    print(f"   - Teacher outputs: {teacher_outputs_dir}")

def main():
    parser = argparse.ArgumentParser(description="Complete teacher-student training pipeline")
    
    # Required arguments
    parser.add_argument("--train_data_dir", type=str, required=True, 
                       help="Directory containing training images")
    parser.add_argument("--in_json", type=str, required=True,
                       help="Metadata JSON file")
    parser.add_argument("--teacher_model", type=str, required=True,
                       help="Path to teacher model")
    parser.add_argument("--base_model", type=str, required=True,
                       help="Base model for LoRA training")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for trained LoRA")
    
    # Optional arguments
    parser.add_argument("--teacher_outputs_dir", type=str, default=None,
                       help="Directory for teacher outputs (auto-generated if not specified)")
    parser.add_argument("--skip_teacher_prep", action="store_true",
                       help="Skip teacher outputs preparation (use existing)")
    parser.add_argument("--skip_training", action="store_true",
                       help="Skip student training (only prepare teacher outputs)")
    parser.add_argument("--config_file", type=str, default=None,
                       help="Configuration file with training parameters")
    
    # Training parameters
    parser.add_argument("--train_batch_size", type=int, default=1,
                       help="Training batch size")
    parser.add_argument("--num_train_epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--network_dim", type=int, default=32,
                       help="LoRA network dimension")
    parser.add_argument("--network_alpha", type=float, default=32,
                       help="LoRA network alpha")
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                       choices=["no", "fp16", "bf16"])
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Gradient accumulation steps")
    
    args = parser.parse_args()
    
    # Auto-generate teacher outputs directory if not specified
    if args.teacher_outputs_dir is None:
        args.teacher_outputs_dir = os.path.join(args.output_dir, "teacher_outputs")
    
    print("🚀 Teacher-Student Training Pipeline")
    print("=" * 60)
    print(f"Train data directory: {args.train_data_dir}")
    print(f"Metadata file: {args.in_json}")
    print(f"Teacher model: {args.teacher_model}")
    print(f"Base model: {args.base_model}")
    print(f"Output directory: {args.output_dir}")
    print(f"Teacher outputs directory: {args.teacher_outputs_dir}")
    print("=" * 60)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Create directories
    create_directories(args.output_dir, args.teacher_outputs_dir)
    
    # Step 1: Prepare teacher outputs
    if not args.skip_teacher_prep:
        teacher_prep_cmd = [
            "python", "finetune/prepare_teacher_outputs.py",
            "--train_data_dir", args.train_data_dir,
            "--in_json", args.in_json,
            "--teacher_model_name_or_path", args.teacher_model,
            "--output_dir", args.teacher_outputs_dir,
            "--mixed_precision", args.mixed_precision,
            "--max_resolution", "512,512",
            "--min_bucket_reso", "256",
            "--max_bucket_reso", "1024",
            "--bucket_reso_steps", "64"
        ]
        
        if not run_command(teacher_prep_cmd, "Preparing teacher outputs"):
            print("❌ Teacher outputs preparation failed. Exiting.")
            sys.exit(1)
    else:
        print("⏭️  Skipping teacher outputs preparation")
    
    # Step 2: Train student model
    if not args.skip_training:
        training_cmd = [
            "python", "train_network_teacher_student.py",
            "--train_data_dir", args.train_data_dir,
            "--teacher_outputs_dir", args.teacher_outputs_dir,
            "--output_dir", args.output_dir,
            "--model_name_or_path", args.base_model,
            "--train_batch_size", str(args.train_batch_size),
            "--num_train_epochs", str(args.num_train_epochs),
            "--learning_rate", str(args.learning_rate),
            "--network_dim", str(args.network_dim),
            "--network_alpha", str(args.network_alpha),
            "--mixed_precision", args.mixed_precision,
            "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
            "--save_every_n_steps", "1000",
            "--logging_steps", "10"
        ]
        
        if not run_command(training_cmd, "Training student LoRA model"):
            print("❌ Student training failed. Exiting.")
            sys.exit(1)
    else:
        print("⏭️  Skipping student training")
    
    print("\n🎉 Pipeline completed successfully!")
    print(f"📁 Teacher outputs saved to: {args.teacher_outputs_dir}")
    if not args.skip_training:
        print(f"📁 Trained LoRA saved to: {args.output_dir}")
    
    print("\n📋 Next steps:")
    print("1. Check the output directories for generated files")
    print("2. Test the trained LoRA model")
    print("3. Adjust parameters if needed and re-run training")

if __name__ == "__main__":
    main()