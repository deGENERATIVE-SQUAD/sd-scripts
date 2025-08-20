#!/usr/bin/env python3
"""
Generate teacher outputs for teacher-student training.
This script processes a dataset through a teacher model and saves the outputs
(latents, t, text_embeddings, eps_teacher) for later use in student training.
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
import numpy as np
from tqdm import tqdm
from PIL import Image

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


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


class TeacherOutputGenerator:
    """Generate teacher outputs for teacher-student training."""
    
    def __init__(self, args):
        self.args = args
        self.logger = logging.getLogger(__name__)
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        # Load teacher model
        self.load_teacher_model()
        
        # Setup dataset
        self.setup_dataset()
        
    def load_teacher_model(self):
        """Load the teacher model components."""
        self.logger.info("Loading teacher model...")
        
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
        self.vae.eval()
        
        # Load UNet
        self.unet = UNet2DConditionModel.from_pretrained(
            self.args.model_path,
            subfolder="unet",
            torch_dtype=torch.float16 if self.args.use_fp16 else torch.float32
        )
        self.unet.to(self.device)
        self.unet.eval()
        
        # Load text encoder
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
        
        self.logger.info("Teacher model loaded successfully")
        
    def setup_dataset(self):
        """Setup the dataset for processing."""
        self.logger.info("Setting up dataset...")
        
        # Create dataset subset
        subset = DreamBoothSubset(
            image_dir=self.args.train_data_dir,
            metadata_file=self.args.metadata_file,
            is_reg=False,
            flip_aug=False,
            color_aug=False,
            face_crop_aug_range=None,
            random_crop=False,
            caption_extension=self.args.caption_extension,
            enable_wildcard=False,
            cache_info=False,
            custom_attributes={}
        )
        
        # Create dataset
        self.dataset = DreamBoothDataset(
            subsets=[subset],
            is_training_dataset=True,
            batch_size=1,  # Process one image at a time
            resolution=self.args.resolution,
            network_multiplier=1.0,
            enable_bucket=False,
            min_bucket_reso=None,
            max_bucket_reso=None,
            bucket_reso_steps=None,
            bucket_no_upscale=False,
            prior_loss_weight=1.0,
            debug_dataset=False,
            validation_split=0.0,
            validation_seed=None,
            resize_interpolation=None
        )
        
        self.logger.info(f"Dataset setup complete. Total images: {len(self.dataset)}")
        
    def process_image(self, image_info: ImageInfo, caption: str) -> Dict:
        """Process a single image through the teacher model."""
        # Load and preprocess image
        image = Image.open(image_info.absolute_path).convert("RGB")
        
        # Resize image to target resolution
        if self.args.resolution:
            image = image.resize(self.args.resolution, Image.Resampling.LANCZOS)
        
        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        image_tensor = image_tensor.to(self.device)
        
        # Encode image to latents
        with torch.no_grad():
            latents = self.vae.encode(image_tensor).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        
        # Tokenize caption
        inputs = self.tokenizer(
            caption,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        input_ids = inputs.input_ids.to(self.device)
        
        # Get text embeddings
        with torch.no_grad():
            text_embeddings = self.text_encoder(input_ids)[0]
        
        # Generate noise and timestep
        batch_size = 1
        height, width = latents.shape[2], latents.shape[3]
        
        # Generate random timestep
        t = torch.randint(0, self.scheduler.num_train_timesteps, (batch_size,), device=self.device)
        
        # Generate noise
        noise = torch.randn_like(latents)
        
        # Get teacher prediction
        with torch.no_grad():
            noise_pred = self.unet(
                torch.cat([latents] * 2),
                t,
                encoder_hidden_states=torch.cat([text_embeddings] * 2)
            ).sample
            
            # Extract teacher prediction for the image (not unconditional)
            eps_teacher = noise_pred[0:1]
        
        return {
            "latents": latents.cpu().numpy(),
            "t": t.cpu().numpy(),
            "text_embeddings": text_embeddings.cpu().numpy(),
            "eps_teacher": eps_teacher.cpu().numpy(),
            "caption": caption,
            "image_path": image_info.absolute_path
        }
    
    def save_outputs(self, outputs: List[Dict], output_dir: str):
        """Save teacher outputs to disk."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metadata
        metadata = {
            "total_images": len(outputs),
            "resolution": self.args.resolution,
            "model_path": self.args.model_path,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "images": []
        }
        
        for i, output in enumerate(tqdm(outputs, desc="Saving outputs")):
            # Create filename based on image path
            image_name = os.path.splitext(os.path.basename(output["image_path"]))[0]
            safe_name = "".join(c for c in image_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            
            # Save individual files
            base_path = os.path.join(output_dir, f"{safe_name}_{i:06d}")
            
            # Save latents
            np.save(f"{base_path}_latents.npy", output["latents"])
            
            # Save timestep
            np.save(f"{base_path}_t.npy", output["t"])
            
            # Save text embeddings
            np.save(f"{base_path}_text_embeddings.npy", output["text_embeddings"])
            
            # Save teacher prediction
            np.save(f"{base_path}_eps_teacher.npy", output["eps_teacher"])
            
            # Add to metadata
            metadata["images"].append({
                "index": i,
                "original_path": output["image_path"],
                "caption": output["caption"],
                "latents_file": f"{safe_name}_{i:06d}_latents.npy",
                "t_file": f"{safe_name}_{i:06d}_t.npy",
                "text_embeddings_file": f"{safe_name}_{i:06d}_text_embeddings.npy",
                "eps_teacher_file": f"{safe_name}_{i:06d}_eps_teacher.npy"
            })
        
        # Save metadata
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Saved {len(outputs)} teacher outputs to {output_dir}")
    
    def generate_all_outputs(self):
        """Generate teacher outputs for all images in the dataset."""
        self.logger.info("Starting teacher output generation...")
        
        outputs = []
        
        for i in tqdm(range(len(self.dataset)), desc="Processing images"):
            try:
                # Get dataset item
                item = self.dataset[i]
                
                # Get image info
                image_key = item.get("image_keys", [f"image_{i}"])[0]
                image_info = self.dataset.image_data[image_key]
                caption = item["captions"][0] if item["captions"] else ""
                
                # Process image
                output = self.process_image(image_info, caption)
                outputs.append(output)
                
                # Clear GPU memory periodically
                if (i + 1) % 10 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                self.logger.error(f"Error processing image {i}: {e}")
                continue
        
        # Save outputs
        self.save_outputs(outputs, self.args.output_dir)
        
        self.logger.info("Teacher output generation completed!")


def main():
    parser = argparse.ArgumentParser(description="Generate teacher outputs for teacher-student training")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the teacher model")
    parser.add_argument("--vae_path", type=str, default=None,
                       help="Path to VAE (optional, will use model's VAE if not specified)")
    parser.add_argument("--use_fp16", action="store_true",
                       help="Use FP16 precision for model loading")
    
    # Dataset arguments
    parser.add_argument("--train_data_dir", type=str, required=True,
                       help="Directory containing training images")
    parser.add_argument("--metadata_file", type=str, default=None,
                       help="Metadata file for training data")
    parser.add_argument("--caption_extension", type=str, default=".txt",
                       help="Extension for caption files")
    parser.add_argument("--resolution", type=int, nargs=2, default=[512, 512],
                       help="Target resolution for images (width height)")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./teacher_outputs",
                       help="Directory to save teacher outputs")
    
    # Other arguments
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
    
    # Create output generator and run
    generator = TeacherOutputGenerator(args)
    generator.generate_all_outputs()


if __name__ == "__main__":
    main()