#!/usr/bin/env python3
"""
Train a student model using pre-computed teacher outputs.
This script implements teacher-student training where the student learns
to mimic the teacher's predictions without needing the teacher model in memory.
"""

import argparse
import os
import json
import logging
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from diffusers import (
    StableDiffusionPipeline,
    DDPMScheduler,
    AutoencoderKL,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer

# Add parent directory to path to import library modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from library import model_util, train_util
from library.train_util import DreamBoothDataset, DreamBoothSubset, ImageInfo
from library.config_util import ConfigSanitizer, BlueprintGenerator
from teacher_student_dataset import TeacherStudentDataset, TeacherStudentDataLoader


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


class TeacherStudentTrainer:
    """Trainer for teacher-student learning."""
    
    def __init__(self, args):
        self.args = args
        self.logger = logging.getLogger(__name__)
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        # Load student model
        self.load_student_model()
        
        # Setup dataset
        self.setup_dataset()
        
        # Setup training components
        self.setup_training()
        
    def load_student_model(self):
        """Load the student model components."""
        self.logger.info("Loading student model...")
        
        # Load VAE
        if self.args.vae_path:
            self.vae = AutoencoderKL.from_pretrained(
                self.args.vae_path,
                torch_dtype=torch.float16 if self.args.use_fp16 else torch.float32
            )
        else:
            self.vae = AutoencoderKL.from_pretrained(
                self.args.model_path,
                subfolder="vae",
                torch_dtype=torch.float16 if self.args.use_fp16 else torch.float32
            )
        self.vae.to(self.device)
        self.vae.eval()  # VAE is not trained
        
        # Load UNet (this is what we'll train)
        self.unet = UNet2DConditionModel.from_pretrained(
            self.args.model_path,
            subfolder="unet",
            torch_dtype=torch.float16 if self.args.use_fp16 else torch.float32
        )
        self.unet.to(self.device)
        self.unet.train()
        
        # Load text encoder (not trained)
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.args.model_path,
            subfolder="text_encoder",
            torch_dtype=torch.float16 if self.args.use_fp16 else torch.float32
        )
        self.text_encoder.to(self.device)
        self.text_encoder.eval()
        
        # Load tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.args.model_path,
            subfolder="tokenizer"
        )
        
        # Load scheduler
        self.scheduler = DDPMScheduler.from_pretrained(
            self.args.model_path,
            subfolder="scheduler"
        )
        
        self.logger.info("Student model loaded successfully")
        
    def setup_dataset(self):
        """Setup the teacher-student dataset."""
        self.logger.info("Setting up teacher-student dataset...")
        
        self.dataset = TeacherStudentDataset(
            teacher_outputs_dir=self.args.teacher_outputs_dir,
            resolution=self.args.resolution,
            network_multiplier=1.0,
            debug_dataset=self.args.debug_dataset
        )
        
        self.dataloader = TeacherStudentDataLoader(
            teacher_outputs_dir=self.args.teacher_outputs_dir,
            batch_size=self.args.batch_size,
            resolution=self.args.resolution,
            network_multiplier=1.0,
            debug_dataset=self.args.debug_dataset,
            shuffle=True
        )
        
        self.logger.info(f"Dataset setup complete. Total images: {len(self.dataset)}")
        
    def setup_training(self):
        """Setup training components (optimizer, scheduler, etc.)."""
        self.logger.info("Setting up training components...")
        
        # Setup optimizer
        if self.args.optimizer_type == "AdamW":
            self.optimizer = torch.optim.AdamW(
                self.unet.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer_type == "AdamW8bit":
            try:
                import bitsandbytes as bnb
                self.optimizer = bnb.optim.AdamW8bit(
                    self.unet.parameters(),
                    lr=self.args.learning_rate,
                    weight_decay=self.args.weight_decay
                )
            except ImportError:
                self.logger.warning("bitsandbytes not available, falling back to AdamW")
                self.optimizer = torch.optim.AdamW(
                    self.unet.parameters(),
                    lr=self.args.learning_rate,
                    weight_decay=self.args.weight_decay
                )
        else:
            raise ValueError(f"Unknown optimizer type: {self.args.optimizer_type}")
        
        # Setup learning rate scheduler
        if self.args.lr_scheduler == "constant":
            self.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                self.optimizer,
                factor=1.0,
                total_iters=self.args.max_train_steps
            )
        elif self.args.lr_scheduler == "linear":
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=self.args.max_train_steps
            )
        elif self.args.lr_scheduler == "cosine":
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.args.max_train_steps
            )
        else:
            raise ValueError(f"Unknown LR scheduler: {self.args.lr_scheduler}")
        
        # Setup loss function
        self.loss_fn = F.mse_loss
        
        self.logger.info("Training components setup complete")
        
    def compute_loss(self, batch: Dict) -> torch.Tensor:
        """Compute the teacher-student loss."""
        # Get batch data
        latents = batch['latents'].to(self.device)
        t = batch['t'].to(self.device)
        text_embeddings = batch['text_embeddings'].to(self.device)
        eps_teacher = batch['eps_teacher'].to(self.device)
        
        # Get student prediction
        noise_pred = self.unet(
            latents,
            t,
            encoder_hidden_states=text_embeddings
        ).sample
        
        # Compute loss between student and teacher predictions
        loss = self.loss_fn(noise_pred, eps_teacher)
        
        return loss
    
    def train_step(self, batch: Dict) -> Dict:
        """Perform a single training step."""
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Compute loss
        loss = self.compute_loss(batch)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if self.args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.unet.parameters(), self.args.max_grad_norm)
        
        # Update weights
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def save_checkpoint(self, step: int, loss: float):
        """Save a training checkpoint."""
        checkpoint_dir = os.path.join(self.args.output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save UNet (the part we're training)
        checkpoint_path = os.path.join(checkpoint_dir, f"unet_step_{step:08d}.safetensors")
        
        # Convert to CPU for saving
        unet_state_dict = {k: v.cpu() for k, v in self.unet.state_dict().items()}
        
        # Save using safetensors
        try:
            import safetensors.torch
            safetensors.torch.save_file(unet_state_dict, checkpoint_path)
        except ImportError:
            # Fallback to torch.save
            torch.save(unet_state_dict, checkpoint_path.replace('.safetensors', '.pt'))
        
        # Save training state
        training_state = {
            'step': step,
            'loss': loss,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'args': vars(self.args)
        }
        
        state_path = os.path.join(checkpoint_dir, f"training_state_step_{step:08d}.json")
        with open(state_path, 'w') as f:
            json.dump(training_state, f, indent=2)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting teacher-student training...")
        
        # Training statistics
        total_loss = 0.0
        best_loss = float('inf')
        
        # Training loop
        for step in tqdm(range(self.args.max_train_steps), desc="Training"):
            try:
                # Get batch
                batch = next(self.dataloader)
                
                # Training step
                step_result = self.train_step(batch)
                loss = step_result['loss']
                total_loss += loss
                
                # Update learning rate
                self.lr_scheduler.step()
                
                # Logging
                if (step + 1) % self.args.logging_steps == 0:
                    avg_loss = total_loss / (step + 1)
                    self.logger.info(
                        f"Step {step + 1}/{self.args.max_train_steps}: "
                        f"Loss: {loss:.6f}, Avg Loss: {avg_loss:.6f}, "
                        f"LR: {step_result['learning_rate']:.6f}"
                    )
                
                # Save checkpoint
                if (step + 1) % self.args.save_steps == 0:
                    self.save_checkpoint(step + 1, loss)
                
                # Save best model
                if loss < best_loss:
                    best_loss = loss
                    self.save_checkpoint(step + 1, loss, is_best=True)
                
                # Clear GPU memory periodically
                if (step + 1) % 100 == 0:
                    torch.cuda.empty_cache()
                    
            except StopIteration:
                # Restart dataloader
                self.dataloader = iter(self.dataloader)
                continue
            except Exception as e:
                self.logger.error(f"Error in training step {step}: {e}")
                continue
        
        # Save final checkpoint
        self.save_checkpoint(self.args.max_train_steps, loss)
        
        self.logger.info("Training completed!")
        self.logger.info(f"Final loss: {loss:.6f}")
        self.logger.info(f"Best loss: {best_loss:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Train a student model using teacher outputs")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the student model")
    parser.add_argument("--vae_path", type=str, default=None,
                       help="Path to VAE (optional, will use model's VAE if not specified)")
    parser.add_argument("--use_fp16", action="store_true",
                       help="Use FP16 precision for model loading")
    
    # Teacher outputs
    parser.add_argument("--teacher_outputs_dir", type=str, required=True,
                       help="Directory containing teacher outputs")
    
    # Training arguments
    parser.add_argument("--max_train_steps", type=int, default=1000,
                       help="Maximum number of training steps")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-6,
                       help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Maximum gradient norm for clipping")
    
    # Optimizer arguments
    parser.add_argument("--optimizer_type", type=str, default="AdamW",
                       choices=["AdamW", "AdamW8bit"],
                       help="Optimizer type")
    parser.add_argument("--lr_scheduler", type=str, default="constant",
                       choices=["constant", "linear", "cosine"],
                       help="Learning rate scheduler")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./student_outputs",
                       help="Directory to save training outputs")
    parser.add_argument("--resolution", type=int, nargs=2, default=[512, 512],
                       help="Target resolution for training (width height)")
    
    # Other arguments
    parser.add_argument("--logging_steps", type=int, default=10,
                       help="Log every N steps")
    parser.add_argument("--save_steps", type=int, default=100,
                       help="Save checkpoint every N steps")
    parser.add_argument("--debug_dataset", action="store_true",
                       help="Enable dataset debug mode")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create trainer and run training
    trainer = TeacherStudentTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()