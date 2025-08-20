#!/usr/bin/env python3
"""
Prepare teacher outputs for teacher-student training.
This script runs a teacher model on the dataset and saves the outputs for student training.
"""

import argparse
import os
import json
import math
from pathlib import Path
from typing import List, Optional, Tuple
from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2

import torch
from library.device_utils import init_ipex, get_preferred_device
from diffusers import DDPMScheduler, AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel

init_ipex()

from torchvision import transforms
import library.model_util as model_util
import library.train_util as train_util
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

DEVICE = get_preferred_device()

IMAGE_TRANSFORMS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


def collate_fn_remove_corrupted(batch):
    """Collate function that allows to remove corrupted examples in the
    dataloader. It expects that the dataloader returns 'None' when that occurs.
    The 'None's in the batch are removed.
    """
    # Filter out all the Nones (corrupted examples)
    batch = list(filter(lambda x: x is not None, batch))
    return batch


def get_npz_filename(data_dir, image_key, is_full_path, recursive):
    if is_full_path:
        base_name = os.path.splitext(os.path.basename(image_key))[0]
        relative_path = os.path.relpath(os.path.dirname(image_key), data_dir)
    else:
        base_name = image_key
        relative_path = ""

    if recursive and relative_path:
        return os.path.join(data_dir, relative_path, base_name) + ".npz"
    else:
        return os.path.join(data_dir, base_name) + ".npz"


def main(args):
    if args.bucket_reso_steps % 8 > 0:
        logger.warning(f"resolution of buckets in training time is a multiple of 8 / 学習時の各bucketの解像度は8単位になります")
    if args.bucket_reso_steps % 32 > 0:
        logger.warning(
            f"WARNING: bucket_reso_steps is not divisible by 32. It is not working with SDXL / bucket_reso_stepsが32で割り切れません。SDXLでは動作しません"
        )

    train_data_dir_path = Path(args.train_data_dir)
    image_paths: List[str] = [str(p) for p in train_util.glob_images_pathlib(train_data_dir_path, args.recursive)]
    logger.info(f"found {len(image_paths)} images.")

    if os.path.exists(args.in_json):
        logger.info(f"loading existing metadata: {args.in_json}")
        with open(args.in_json, "rt", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        logger.error(f"no metadata / メタデータファイルがありません: {args.in_json}")
        return

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load teacher model components
    logger.info(f"Loading teacher model from: {args.teacher_model_name_or_path}")
    
    # Load VAE
    vae = model_util.load_vae(args.teacher_model_name_or_path, weight_dtype)
    vae.eval()
    vae.to(DEVICE, dtype=weight_dtype)
    
    # Load text encoder
    text_encoder = model_util.load_text_encoder(args.teacher_model_name_or_path, weight_dtype)
    text_encoder.eval()
    text_encoder.to(DEVICE, dtype=weight_dtype)
    
    # Load tokenizer
    tokenizer = model_util.load_tokenizer(args.teacher_model_name_or_path)
    
    # Load UNet
    unet = model_util.load_unet(args.teacher_model_name_or_path, weight_dtype)
    unet.eval()
    unet.to(DEVICE, dtype=weight_dtype)
    
    # Load noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(args.teacher_model_name_or_path, subfolder="scheduler")

    # Calculate bucket sizes
    max_reso = tuple([int(t) for t in args.max_resolution.split(",")])
    assert (
        len(max_reso) == 2
    ), f"illegal resolution (not 'width,height') / 画像サイズに誤りがあります。'幅,高さ'で指定してください: {args.max_resolution}"

    bucket_manager = train_util.BucketManager(
        args.bucket_no_upscale, max_reso, args.min_bucket_reso, args.max_bucket_reso, args.bucket_reso_steps
    )

    # Process images
    bucket_resos = bucket_manager.get_bucket_resos()
    logger.info(f"bucket resolutions: {bucket_resos}")

    # Create output directory for teacher outputs
    teacher_output_dir = args.output_dir
    os.makedirs(teacher_output_dir, exist_ok=True)
    
    # Save metadata about teacher model
    teacher_info = {
        "teacher_model": args.teacher_model_name_or_path,
        "vae_scale_factor": 0.18215,
        "mixed_precision": args.mixed_precision,
        "bucket_resos": bucket_resos,
        "max_resolution": args.max_resolution,
        "min_bucket_reso": args.min_bucket_reso,
        "max_bucket_reso": args.max_bucket_reso,
        "bucket_reso_steps": args.bucket_reso_steps,
    }
    
    with open(os.path.join(teacher_output_dir, "teacher_info.json"), "w", encoding="utf-8") as f:
        json.dump(teacher_info, f, indent=2, ensure_ascii=False)

    # Process each image
    for image_path in tqdm(image_paths, desc="Processing images"):
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            image_tensor = IMAGE_TRANSFORMS(image).unsqueeze(0).to(DEVICE, dtype=weight_dtype)
            
            # Get image info
            if image_path in metadata:
                image_metadata = metadata[image_path]
                caption = image_metadata.get("caption", "")
            else:
                caption = ""
            
            # Tokenize caption
            input_ids = tokenizer(
                caption,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            ).input_ids.to(DEVICE)
            
            # Get text embeddings
            with torch.no_grad():
                text_embeddings = text_encoder(input_ids)[0]
            
            # Encode image to latents
            with torch.no_grad():
                latents = vae.encode(image_tensor).latent_dist.sample() * vae.config.scaling_factor
            
            # Sample noise and timesteps
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
            
            # Add noise to latents
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Get teacher prediction
            with torch.no_grad():
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample
            
            # Calculate teacher epsilon (noise prediction)
            if args.v_parameterization:
                eps_teacher = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                eps_teacher = noise_pred
            
            # Save teacher outputs
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(teacher_output_dir, f"{base_name}_teacher.npz")
            
            np.savez(
                output_path,
                latents=latents.cpu().numpy(),
                timesteps=timesteps.cpu().numpy(),
                text_embeddings=text_embeddings.cpu().numpy(),
                eps_teacher=eps_teacher.cpu().numpy(),
                caption=caption,
                image_path=image_path,
                original_size=[image.width, image.height],
            )
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            continue
    
    logger.info(f"Teacher outputs saved to: {teacher_output_dir}")
    logger.info("You can now use these outputs for student training with --teacher_outputs_dir argument")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Required arguments
    parser.add_argument("--train_data_dir", type=str, required=True, help="Directory containing training images")
    parser.add_argument("--in_json", type=str, required=True, help="Metadata JSON file")
    parser.add_argument("--teacher_model_name_or_path", type=str, required=True, help="Path to teacher model")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for teacher outputs")
    
    # Optional arguments
    parser.add_argument("--recursive", action="store_true", help="Search for images recursively")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"], help="Mixed precision")
    parser.add_argument("--max_resolution", type=str, default="512,512", help="Maximum resolution")
    parser.add_argument("--min_bucket_reso", type=int, default=256, help="Minimum bucket resolution")
    parser.add_argument("--max_bucket_reso", type=int, default=1024, help="Maximum bucket resolution")
    parser.add_argument("--bucket_reso_steps", type=int, default=64, help="Bucket resolution steps")
    parser.add_argument("--bucket_no_upscale", action="store_true", help="Don't upscale images to bucket resolution")
    parser.add_argument("--v_parameterization", action="store_true", help="Use v-parameterization")
    
    args = parser.parse_args()
    main(args)