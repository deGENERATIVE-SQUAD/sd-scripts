#!/usr/bin/env python3
"""
Quick start script for teacher-student training.
This script provides a simple way to get started with minimal configuration.
"""

import os
import argparse
import subprocess
import sys
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = ['torch', 'diffusers', 'transformers', 'numpy', 'tqdm']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True


def check_gpu():
    """Check GPU availability and memory."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU detected: {gpu_name}")
            print(f"GPU memory: {gpu_memory:.1f} GB")
            
            if gpu_memory < 8:
                print("Warning: GPU memory is less than 8GB. Training may be slow or fail.")
                return False
            return True
        else:
            print("No GPU detected. Training will be very slow on CPU.")
            return False
    except Exception as e:
        print(f"Error checking GPU: {e}")
        return False


def create_sample_dataset():
    """Create a sample dataset structure."""
    sample_dir = "./sample_dataset"
    os.makedirs(sample_dir, exist_ok=True)
    
    # Create sample images directory
    images_dir = os.path.join(sample_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Create sample captions
    captions = [
        "A beautiful landscape with mountains and trees",
        "A cute cat sitting on a windowsill",
        "A modern city skyline at sunset",
        "A flower garden in spring",
        "A cozy coffee shop interior"
    ]
    
    for i, caption in enumerate(captions):
        caption_file = os.path.join(images_dir, f"image_{i:03d}.txt")
        with open(caption_file, 'w') as f:
            f.write(caption)
    
    print(f"Sample dataset created in {sample_dir}")
    print("Please add your training images to the 'images' folder")
    print("Each image should have a corresponding .txt file with the same name")
    
    return sample_dir


def run_teacher_generation(model_path, data_dir, output_dir, use_fp16=True):
    """Run teacher output generation."""
    cmd = [
        sys.executable, "teacher/generate_teacher_outputs.py",
        "--model_path", model_path,
        "--train_data_dir", data_dir,
        "--output_dir", output_dir,
        "--resolution", "512", "512",
        "--use_fp16" if use_fp16 else ""
    ]
    
    # Remove empty arguments
    cmd = [arg for arg in cmd if arg]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Teacher output generation completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during teacher generation: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False


def run_student_training(model_path, teacher_outputs_dir, output_dir, max_steps=1000):
    """Run student training."""
    cmd = [
        sys.executable, "teacher/train_student.py",
        "--model_path", model_path,
        "--teacher_outputs_dir", teacher_outputs_dir,
        "--output_dir", output_dir,
        "--max_train_steps", str(max_steps),
        "--batch_size", "1",
        "--learning_rate", "1e-4",
        "--use_fp16"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Student training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during student training: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Quick start for teacher-student training")
    
    parser.add_argument("--model_path", type=str, 
                       default="runwayml/stable-diffusion-v1-5",
                       help="Path to the model (HuggingFace model ID or local path)")
    
    parser.add_argument("--data_dir", type=str, default=None,
                       help="Path to training data directory")
    
    parser.add_argument("--output_dir", type=str, default="./teacher_student_outputs",
                       help="Output directory for all outputs")
    
    parser.add_argument("--max_train_steps", type=int, default=1000,
                       help="Maximum training steps")
    
    parser.add_argument("--skip_teacher", action="store_true",
                       help="Skip teacher generation (use existing outputs)")
    
    parser.add_argument("--skip_student", action="store_true",
                       help="Skip student training")
    
    parser.add_argument("--create_sample", action="store_true",
                       help="Create a sample dataset structure")
    
    args = parser.parse_args()
    
    print("Teacher-Student Training Quick Start")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Check GPU
    if not check_gpu():
        print("Continuing anyway, but training may be slow...")
    
    # Create sample dataset if requested
    if args.create_sample:
        data_dir = create_sample_dataset()
        print(f"\nSample dataset created. Please add your images and run again.")
        print(f"Or use: --data_dir {data_dir}")
        return 0
    
    # Determine data directory
    if args.data_dir is None:
        print("No data directory specified. Creating sample dataset...")
        data_dir = create_sample_dataset()
        print(f"\nPlease add your training images to {data_dir}/images and run again.")
        return 0
    
    data_dir = args.data_dir
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Data directory does not exist: {data_dir}")
        return 1
    
    # Create output directories
    teacher_outputs_dir = os.path.join(args.output_dir, "teacher_outputs")
    student_outputs_dir = os.path.join(args.output_dir, "student_outputs")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Generate teacher outputs
    if not args.skip_teacher:
        print(f"\nStep 1: Generating teacher outputs...")
        print(f"Model: {args.model_path}")
        print(f"Data: {data_dir}")
        print(f"Output: {teacher_outputs_dir}")
        
        if not run_teacher_generation(args.model_path, data_dir, teacher_outputs_dir):
            print("Teacher generation failed. Exiting.")
            return 1
    else:
        print(f"\nSkipping teacher generation. Using existing outputs in: {teacher_outputs_dir}")
        if not os.path.exists(teacher_outputs_dir):
            print(f"Teacher outputs directory does not exist: {teacher_outputs_dir}")
            return 1
    
    # Step 2: Train student model
    if not args.skip_student:
        print(f"\nStep 2: Training student model...")
        print(f"Model: {args.model_path}")
        print(f"Teacher outputs: {teacher_outputs_dir}")
        print(f"Output: {student_outputs_dir}")
        
        if not run_student_training(args.model_path, teacher_outputs_dir, student_outputs_dir, args.max_train_steps):
            print("Student training failed. Exiting.")
            return 1
    else:
        print(f"\nSkipping student training.")
    
    print(f"\n" + "=" * 50)
    print("Quick start completed successfully!")
    print(f"Teacher outputs: {teacher_outputs_dir}")
    print(f"Student outputs: {student_outputs_dir}")
    
    if not args.skip_student:
        print(f"\nYour trained student model is ready!")
        print(f"Checkpoint files are in: {student_outputs_dir}/checkpoints/")
    
    return 0


if __name__ == "__main__":
    exit(main())