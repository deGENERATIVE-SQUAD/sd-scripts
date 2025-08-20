#!/usr/bin/env python3
"""
Main script to run the complete teacher-student training pipeline.
This script orchestrates both teacher output generation and student training.
"""

import argparse
import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import Optional

# Add parent directory to path to import library modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger(__name__)

def run_command(cmd: list, description: str, logger: logging.Logger) -> bool:
    """Run a command and return success status"""
    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"✓ {description} completed successfully")
        if result.stdout:
            logger.debug(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed with exit code {e.returncode}")
        if e.stdout:
            logger.error(f"stdout: {e.stdout}")
        if e.stderr:
            logger.error(f"stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error(f"✗ Command not found: {cmd[0]}")
        return False

def generate_teacher_outputs(args, logger: logging.Logger) -> bool:
    """Generate teacher outputs using the dataset"""
    cmd = [
        sys.executable, "teacher/generate_teacher_outputs.py",
        "--pretrained_model_name_or_path", args.teacher_model,
        "--train_data_dir", args.dataset_path,
        "--teacher_output_dir", args.teacher_output_dir,
        "--resolution", str(args.resolution),
        "--max_token_length", str(args.max_token_length),
        "--train_batch_size", str(args.batch_size),
        "--mixed_precision", args.mixed_precision,
        "--num_inference_steps", str(args.num_inference_steps),
    ]
    
    # Add optional flags
    if args.lowram:
        cmd.append("--lowram")
    if args.xformers:
        cmd.append("--xformers")
    if args.cache_text_encoder_outputs:
        cmd.append("--cache_text_encoder_outputs")
    if args.vae_batch_size:
        cmd.extend(["--vae_batch_size", str(args.vae_batch_size)])
    
    return run_command(cmd, "Teacher output generation", logger)

def train_student(args, logger: logging.Logger) -> bool:
    """Train the student model using teacher outputs"""
    cmd = [
        sys.executable, "teacher/train_student.py",
        "--pretrained_model_name_or_path", args.student_base_model,
        "--train_data_dir", args.teacher_output_dir,
        "--output_dir", args.student_output_dir,
        "--output_name", args.student_name,
        "--max_train_epochs", str(args.max_epochs),
        "--max_train_steps", str(args.max_steps),
        "--learning_rate", str(args.learning_rate),
        "--train_batch_size", str(args.batch_size),
        "--mixed_precision", args.mixed_precision,
        "--save_every_n_epochs", str(args.save_every_n_epochs),
        "--network_module", args.network_module,
        "--network_dim", str(args.network_dim),
        "--network_alpha", str(args.network_alpha),
    ]
    
    # Add optional flags
    if args.lowram:
        cmd.append("--lowram")
    if args.xformers:
        cmd.append("--xformers")
    if args.network_train_unet_only:
        cmd.append("--network_train_unet_only")
    if args.network_train_text_encoder_only:
        cmd.append("--network_train_text_encoder_only")
    if args.gradient_accumulation_steps:
        cmd.extend(["--gradient_accumulation_steps", str(args.gradient_accumulation_steps)])
    if args.optimizer_type:
        cmd.extend(["--optimizer_type", args.optimizer_type])
    if args.lr_scheduler_type:
        cmd.extend(["--lr_scheduler_type", args.lr_scheduler_type])
    if args.loss_type:
        cmd.extend(["--loss_type", args.loss_type])
    if args.max_samples:
        cmd.extend(["--max_samples", str(args.max_samples)])
    
    # Add network arguments if provided
    if args.network_args:
        cmd.extend(["--network_args"] + args.network_args)
    
    return run_command(cmd, "Student training", logger)

def check_prerequisites(logger: logging.Logger) -> bool:
    """Check if all required files and dependencies are available"""
    logger.info("Checking prerequisites...")
    
    # Check if teacher scripts exist
    required_files = [
        "teacher/generate_teacher_outputs.py",
        "teacher/train_student.py",
    ]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            logger.error(f"Required file not found: {file_path}")
            return False
    
    # Check if dataset directory exists
    if not Path(args.dataset_path).exists():
        logger.error(f"Dataset directory not found: {args.dataset_path}")
        return False
    
    logger.info("✓ All prerequisites met")
    return True

def main():
    """Main function to run the complete pipeline"""
    parser = argparse.ArgumentParser(description="Run complete teacher-student training pipeline")
    
    # Teacher generation arguments
    parser.add_argument("--teacher_model", type=str, required=True,
                       help="Path to teacher model")
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to dataset directory")
    parser.add_argument("--teacher_output_dir", type=str, default="./teacher_outputs",
                       help="Directory to save teacher outputs")
    
    # Student training arguments
    parser.add_argument("--student_base_model", type=str, required=True,
                       help="Path to base model for student")
    parser.add_argument("--student_output_dir", type=str, default="./student_output",
                       help="Directory to save trained student model")
    parser.add_argument("--student_name", type=str, default="student_model",
                       help="Name for the student model output files")
    
    # Common parameters
    parser.add_argument("--resolution", type=int, default=1024,
                       help="Image resolution")
    parser.add_argument("--max_token_length", type=int, default=225,
                       help="Maximum token length")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Training batch size")
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                       choices=["no", "fp16", "bf16"],
                       help="Mixed precision type")
    
    # Teacher generation specific
    parser.add_argument("--num_inference_steps", type=int, default=1000,
                       help="Number of inference steps for noise generation")
    parser.add_argument("--vae_batch_size", type=int, default=1,
                       help="VAE batch size for memory optimization")
    parser.add_argument("--cache_text_encoder_outputs", action="store_true",
                       help="Cache text encoder outputs")
    
    # Student training specific
    parser.add_argument("--max_epochs", type=int, default=10,
                       help="Maximum training epochs")
    parser.add_argument("--max_steps", type=int, default=1000,
                       help="Maximum training steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--save_every_n_epochs", type=int, default=1,
                       help="Save checkpoint every N epochs")
    
    # Network configuration
    parser.add_argument("--network_module", type=str, default="networks.lora",
                       help="Network module to use")
    parser.add_argument("--network_dim", type=int, default=32,
                       help="Network dimension")
    parser.add_argument("--network_alpha", type=float, default=1.0,
                       help="Network alpha parameter")
    parser.add_argument("--network_args", nargs="*", default=[],
                       help="Additional network arguments")
    
    # Training strategy
    parser.add_argument("--network_train_unet_only", action="store_true",
                       help="Train only UNet network")
    parser.add_argument("--network_train_text_encoder_only", action="store_true",
                       help="Train only text encoder networks")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="Gradient accumulation steps")
    
    # Optimizer and scheduler
    parser.add_argument("--optimizer_type", type=str, default="AdamW",
                       help="Optimizer type")
    parser.add_argument("--lr_scheduler_type", type=str, default="constant",
                       help="Learning rate scheduler type")
    parser.add_argument("--loss_type", type=str, default="l2",
                       help="Loss function type")
    
    # Memory optimization
    parser.add_argument("--lowram", action="store_true",
                       help="Enable low RAM mode")
    parser.add_argument("--xformers", action="store_true",
                       help="Enable xformers memory efficient attention")
    
    # Optional parameters
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to use")
    parser.add_argument("--skip_teacher", action="store_true",
                       help="Skip teacher output generation (use existing)")
    parser.add_argument("--skip_student", action="store_true",
                       help="Skip student training")
    parser.add_argument("--config_file", type=str, default=None,
                       help="Path to configuration file")
    
    global args
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    logger.info("Starting SDXL Teacher-Student Training Pipeline")
    logger.info("=" * 60)
    
    # Check prerequisites
    if not check_prerequisites(logger):
        logger.error("Prerequisites check failed. Exiting.")
        return False
    
    # Create output directories
    Path(args.teacher_output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.student_output_dir).mkdir(parents=True, exist_ok=True)
    
    success = True
    
    # Step 1: Generate teacher outputs
    if not args.skip_teacher:
        logger.info("Step 1: Generating teacher outputs")
        if not generate_teacher_outputs(args, logger):
            logger.error("Teacher output generation failed")
            success = False
        else:
            logger.info("✓ Teacher output generation completed")
    else:
        logger.info("Skipping teacher output generation (using existing outputs)")
    
    # Step 2: Train student model
    if success and not args.skip_student:
        logger.info("Step 2: Training student model")
        if not train_student(args, logger):
            logger.error("Student training failed")
            success = False
        else:
            logger.info("✓ Student training completed")
    elif args.skip_student:
        logger.info("Skipping student training")
    
    # Summary
    logger.info("=" * 60)
    if success:
        logger.info("🎉 Teacher-Student training pipeline completed successfully!")
        logger.info(f"Teacher outputs saved to: {args.teacher_output_dir}")
        logger.info(f"Student model saved to: {args.student_output_dir}")
    else:
        logger.error("❌ Teacher-Student training pipeline failed")
        logger.error("Check the logs above for error details")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)