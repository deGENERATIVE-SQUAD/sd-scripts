#!/usr/bin/env python3
"""
Train a student SDXL model using pre-computed teacher outputs.
This script implements teacher-student training where the student learns to mimic
the teacher's noise predictions instead of learning from scratch.
"""

import argparse
import json
import os
import sys
import logging
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import DDPMScheduler
from tqdm import tqdm

# Add parent directory to path to import library modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from library import sdxl_model_util, sdxl_train_util, strategy_base, strategy_sd, strategy_sdxl, train_util
from library.utils import setup_logging
from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

setup_logging()
logger = logging.getLogger(__name__)


class TeacherStudentDataset(torch.utils.data.Dataset):
    """Dataset that loads pre-computed teacher outputs"""
    
    def __init__(self, teacher_output_dir: str, max_samples: Optional[int] = None):
        self.teacher_output_dir = Path(teacher_output_dir)
        self.sample_files = sorted([f for f in self.teacher_output_dir.glob("sample_*.pt")])
        
        if max_samples is not None:
            self.sample_files = self.sample_files[:max_samples]
        
        # Load dataset info
        info_file = self.teacher_output_dir / "dataset_info.json"
        if info_file.exists():
            with open(info_file, "r") as f:
                import json
                self.dataset_info = json.load(f)
        else:
            self.dataset_info = {}
        
        logger.info(f"Loaded {len(self.sample_files)} teacher output samples from {teacher_output_dir}")
    
    def __len__(self):
        return len(self.sample_files)
    
    def __getitem__(self, idx):
        sample_file = self.sample_files[idx]
        sample_data = torch.load(sample_file, map_location="cpu")
        
        return {
            "latents": sample_data["latents"],
            "noisy_latents": sample_data["noisy_latents"],
            "timesteps": sample_data["timesteps"],
            "text_embeddings1": sample_data["text_embeddings1"],
            "text_embeddings2": sample_data["text_embeddings2"],
            "pool2": sample_data["pool2"],
            "teacher_noise_pred": sample_data["teacher_noise_pred"],
            "original_size": sample_data["original_size"],
            "crop_top_left": sample_data["crop_top_left"],
            "target_size": sample_data["target_size"],
        }


class StudentTrainer:
    def __init__(self):
        self.vae_scale_factor = sdxl_model_util.VAE_SCALE_FACTOR
        self.is_sdxl = True

    def load_student_model(self, args, weight_dtype, accelerator):
        """Load the student model (same as original sdxl_train_network.py)"""
        (
            load_stable_diffusion_format,
            text_encoder1,
            text_encoder2,
            vae,
            unet,
            logit_scale,
            ckpt_info,
        ) = sdxl_train_util.load_target_model(args, accelerator, sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, weight_dtype)

        # Apply memory efficient attention
        train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers, args.sdpa)
        if torch.__version__ >= "2.0.0":
            vae.set_use_memory_efficient_attention_xformers(args.xformers)

        return text_encoder1, text_encoder2, vae, unet

    def get_tokenize_strategy(self, args):
        return strategy_sdxl.SdxlTokenizeStrategy(args.max_token_length, args.tokenizer_cache_dir)

    def get_tokenizers(self, tokenize_strategy: strategy_sdxl.SdxlTokenizeStrategy):
        return [tokenize_strategy.tokenizer1, tokenize_strategy.tokenizer2]

    def get_latents_caching_strategy(self, args):
        latents_caching_strategy = strategy_sd.SdSdxlLatentsCachingStrategy(
            False, args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check
        )
        return latents_caching_strategy

    def get_text_encoding_strategy(self, args):
        return strategy_sdxl.SdxlTextEncodingStrategy()

    def get_models_for_text_encoding(self, args, accelerator, text_encoders):
        return text_encoders + [accelerator.unwrap_model(text_encoders[-1])]

    def get_text_encoder_outputs_caching_strategy(self, args):
        if args.cache_text_encoder_outputs:
            return strategy_sdxl.SdxlTextEncoderOutputsCachingStrategy(
                args.cache_text_encoder_outputs_to_disk, None, args.skip_cache_check, is_weighted=args.weighted_captions
            )
        else:
            return None

    def call_student_unet(
        self,
        args,
        accelerator,
        unet,
        noisy_latents,
        timesteps,
        text_conds,
        batch,
        weight_dtype,
    ):
        """Call student UNet to get noise predictions"""
        noisy_latents = noisy_latents.to(weight_dtype)

        # get size embeddings
        orig_size = batch["original_size"]
        crop_size = batch["crop_top_left"]
        target_size = batch["target_size"]
        embs = sdxl_train_util.get_size_embeddings(orig_size, crop_size, target_size, accelerator.device).to(weight_dtype)

        # concat embeddings
        encoder_hidden_states1, encoder_hidden_states2, pool2 = text_conds
        vector_embedding = torch.cat([pool2, embs], dim=1).to(weight_dtype)
        text_embedding = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=2).to(weight_dtype)

        noise_pred = unet(noisy_latents, timesteps, text_embedding, vector_embedding)
        return noise_pred

    def compute_loss(self, student_pred, teacher_pred, loss_type="l2"):
        """Compute loss between student and teacher predictions"""
        if loss_type == "l2":
            loss = torch.nn.functional.mse_loss(student_pred, teacher_pred, reduction="mean")
        elif loss_type == "l1":
            loss = torch.nn.functional.l1_loss(student_pred, teacher_pred, reduction="mean")
        elif loss_type == "huber":
            loss = torch.nn.functional.huber_loss(student_pred, teacher_pred, reduction="mean")
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        return loss

    def train_student(self, args, accelerator, text_encoders, vae, unet, dataset, weight_dtype):
        """Train the student model using teacher outputs"""
        logger.info("Starting student training...")
        
        # Move models to appropriate devices
        if args.lowram:
            text_encoders[0].to(accelerator.device, dtype=weight_dtype)
            text_encoders[1].to(accelerator.device, dtype=weight_dtype)
            vae.to(accelerator.device, dtype=weight_dtype)
            unet.to(accelerator.device, dtype=weight_dtype)
        else:
            text_encoders[0].to(accelerator.device, dtype=weight_dtype)
            text_encoders[1].to(accelerator.device, dtype=weight_dtype)
            vae.to(accelerator.device, dtype=weight_dtype)
            unet.to(accelerator.device, dtype=weight_dtype)

        # Setup optimizer
        if args.network_train_unet_only:
            trainable_params = list(unet.parameters())
        elif args.network_train_text_encoder_only:
            trainable_params = list(text_encoders[0].parameters()) + list(text_encoders[1].parameters())
        else:
            trainable_params = list(unet.parameters()) + list(text_encoders[0].parameters()) + list(text_encoders[1].parameters())

        if args.optimizer_type.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=args.learning_rate,
                weight_decay=0.01,
                eps=1e-8,
                betas=(0.9, 0.999),
            )
        elif args.optimizer_type.lower() == "lion":
            try:
                from lion_pytorch import Lion
                optimizer = Lion(trainable_params, lr=args.learning_rate, weight_decay=0.01)
            except ImportError:
                logger.warning("Lion optimizer not available, falling back to AdamW")
                optimizer = torch.optim.AdamW(
                    trainable_params,
                    lr=args.learning_rate,
                    weight_decay=0.01,
                    eps=1e-8,
                    betas=(0.9, 0.999),
                )
        else:
            raise ValueError(f"Unknown optimizer type: {args.optimizer_type}")

        # Setup learning rate scheduler
        if args.lr_scheduler_type.lower() == "constant":
            lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
        elif args.lr_scheduler_type.lower() == "cosine":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_train_steps)
        elif args.lr_scheduler_type.lower() == "linear":
            lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=args.max_train_steps)
        else:
            lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

        # Prepare models and optimizer with accelerator
        unet, optimizer, lr_scheduler = accelerator.prepare(unet, optimizer, lr_scheduler)
        if not args.network_train_unet_only:
            text_encoders = [accelerator.prepare(te) for te in text_encoders]

        # Create data loader
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=args.max_data_loader_n_workers,
            persistent_workers=args.persistent_data_loader_workers,
        )

        # Training loop
        global_step = 0
        total_loss = 0.0
        
        for epoch in range(args.max_train_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{args.max_train_epochs}")
            
            for batch in progress_bar:
                # Move batch to device
                batch = {k: v.to(accelerator.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Get text conditioning
                text_conds = (
                    batch["text_embeddings1"].to(weight_dtype),
                    batch["text_embeddings2"].to(weight_dtype),
                    batch["pool2"].to(weight_dtype)
                )
                
                # Get teacher noise prediction
                teacher_noise_pred = batch["teacher_noise_pred"].to(weight_dtype)
                
                # Forward pass through student UNet
                student_noise_pred = self.call_student_unet(
                    args, accelerator, unet, batch["noisy_latents"], 
                    batch["timesteps"], text_conds, batch, weight_dtype
                )
                
                # Compute loss
                loss = self.compute_loss(student_noise_pred, teacher_noise_pred, args.loss_type)
                
                # Backward pass
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                
                # Update progress
                epoch_loss += loss.item()
                total_loss += loss.item()
                num_batches += 1
                global_step += 1
                
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.6f}",
                    "avg_loss": f"{epoch_loss/num_batches:.6f}",
                    "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}"
                })
                
                # Check if we should stop
                if args.max_train_steps > 0 and global_step >= args.max_train_steps:
                    break
            
            # Save checkpoint
            if (epoch + 1) % args.save_every_n_epochs == 0 or epoch == args.max_train_epochs - 1:
                self.save_checkpoint(args, accelerator, unet, text_encoders, vae, epoch, global_step)
            
            if args.max_train_steps > 0 and global_step >= args.max_train_steps:
                break
        
        logger.info(f"Training completed. Total steps: {global_step}, Average loss: {total_loss/global_step:.6f}")

    def save_checkpoint(self, args, accelerator, unet, text_encoders, vae, epoch, global_step):
        """Save training checkpoint"""
        if not accelerator.is_main_process:
            return
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save UNet
        if not args.network_train_text_encoder_only:
            unet_path = output_dir / f"{args.output_name}-epoch{epoch+1:03d}-step{global_step:08d}.safetensors"
            accelerator.unwrap_model(unet).save_pretrained(unet_path)
            logger.info(f"Saved UNet to {unet_path}")
        
        # Save text encoders
        if not args.network_train_unet_only:
            for i, text_encoder in enumerate(text_encoders):
                te_path = output_dir / f"{args.output_name}-te{i+1}-epoch{epoch+1:03d}-step{global_step:08d}.safetensors"
                accelerator.unwrap_model(text_encoder).save_pretrained(te_path)
                logger.info(f"Saved Text Encoder {i+1} to {te_path}")


def setup_parser() -> argparse.ArgumentParser:
    """Setup argument parser with all original arguments plus teacher-student specific ones"""
    parser = argparse.ArgumentParser(description="Train SDXL student model using teacher outputs")
    
    # Add all original arguments from sdxl_train_network.py
    parser.add_argument("--console_log_level", type=str, default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--console_log_file", type=str)
    parser.add_argument("--console_log_simple", action="store_true")
    parser.add_argument("--v2", action="store_true")
    parser.add_argument("--v_parameterization", action="store_true")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True,
                       help="Path to student model (base model)")
    parser.add_argument("--tokenizer_cache_dir", type=str)
    parser.add_argument("--train_data_dir", type=str, required=True,
                       help="Path to teacher outputs directory")
    parser.add_argument("--cache_info", action="store_true")
    parser.add_argument("--shuffle_caption", action="store_true")
    parser.add_argument("--caption_separator", type=str, default=",")
    parser.add_argument("--caption_extension", type=str, default=".caption")
    parser.add_argument("--keep_tokens", type=int, default=0)
    parser.add_argument("--keep_tokens_separator", type=str, default="|")
    parser.add_argument("--secondary_separator", type=str, default="|")
    parser.add_argument("--enable_wildcard", action="store_true")
    parser.add_argument("--caption_prefix", type=str, default="")
    parser.add_argument("--caption_suffix", type=str, default="")
    parser.add_argument("--color_aug", action="store_true")
    parser.add_argument("--flip_aug", action="store_true")
    parser.add_argument("--face_crop_aug_range", type=str, default="0,0")
    parser.add_argument("--random_crop", action="store_true")
    parser.add_argument("--debug_dataset", action="store_true")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--cache_latents", action="store_true")
    parser.add_argument("--vae_batch_size", type=int, default=1)
    parser.add_argument("--cache_latents_to_disk", action="store_true")
    parser.add_argument("--skip_cache_check", action="store_true")
    parser.add_argument("--enable_bucket", action="store_true")
    parser.add_argument("--min_bucket_reso", type=int, default=256)
    parser.add_argument("--max_bucket_reso", type=int, default=1024)
    parser.add_argument("--bucket_reso_steps", type=int, default=64)
    parser.add_argument("--bucket_no_upscale", action="store_true")
    parser.add_argument("--resize_interpolation", type=str, default="lanczos",
                       choices=["lanczos", "nearest", "bilinear", "linear", "bicubic", "cubic", "area"])
    parser.add_argument("--token_warmup_min", type=int, default=1)
    parser.add_argument("--token_warmup_step", type=int, default=0)
    parser.add_argument("--alpha_mask", action="store_true")
    parser.add_argument("--dataset_class", type=str)
    parser.add_argument("--caption_dropout_rate", type=float, default=0.0)
    parser.add_argument("--caption_dropout_every_n_epochs", type=int, default=0)
    parser.add_argument("--caption_tag_dropout_rate", type=float, default=0.0)
    parser.add_argument("--reg_data_dir", type=str)
    parser.add_argument("--in_json", type=str)
    parser.add_argument("--dataset_repeats", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--output_name", type=str, default="student_model")
    parser.add_argument("--save_precision", type=str, default="fp16", choices=["None", "float", "fp16", "bf16"])
    parser.add_argument("--save_every_n_epochs", type=int, default=1)
    parser.add_argument("--save_every_n_steps", type=int, default=0)
    parser.add_argument("--save_n_epoch_ratio", type=float, default=0.0)
    parser.add_argument("--save_last_n_epochs", type=int, default=0)
    parser.add_argument("--save_last_n_epochs_state", type=int, default=0)
    parser.add_argument("--save_last_n_steps", type=int, default=0)
    parser.add_argument("--save_last_n_steps_state", type=int, default=0)
    parser.add_argument("--save_state", action="store_true")
    parser.add_argument("--save_state_on_train_end", action="store_true")
    parser.add_argument("--resume", type=str)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--max_token_length", type=int, default=225)
    parser.add_argument("--mem_eff_attn", action="store_true")
    parser.add_argument("--fixed_noise", action="store_true")
    parser.add_argument("--torch_compile", action="store_true")
    parser.add_argument("--dynamo_backend", type=str, default="eager")
    parser.add_argument("--xformers", action="store_true")
    parser.add_argument("--sdpa", action="store_true")
    parser.add_argument("--vae", type=str)
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--max_train_epochs", type=int, default=10)
    parser.add_argument("--max_data_loader_n_workers", type=int, default=8)
    parser.add_argument("--persistent_data_loader_workers", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--full_fp16", action="store_true")
    parser.add_argument("--full_bf16", action="store_true")
    parser.add_argument("--fp8_base", action="store_true")
    parser.add_argument("--ddp_timeout", type=int, default=1800)
    parser.add_argument("--ddp_gradient_as_bucket_view", action="store_true")
    parser.add_argument("--ddp_static_graph", action="store_true")
    parser.add_argument("--clip_skip", type=int, default=1)
    parser.add_argument("--logging_dir", type=str, default="./logs")
    parser.add_argument("--log_with", type=str, default="tensorboard", choices=["tensorboard", "wandb", "all"])
    parser.add_argument("--log_prefix", type=str, default="")
    parser.add_argument("--log_tracker_name", type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument("--log_tracker_config", type=str, default="")
    parser.add_argument("--wandb_api_key", type=str, default="")
    parser.add_argument("--log_config", action="store_true")
    parser.add_argument("--noise_offset", type=float, default=0.0)
    parser.add_argument("--noise_offset_random_strength", action="store_true")
    parser.add_argument("--multires_noise_iterations", type=int, default=6)
    parser.add_argument("--ip_noise_gamma", type=float, default=0.0)
    parser.add_argument("--ip_noise_gamma_random_strength", action="store_true")
    parser.add_argument("--multires_noise_discount", type=float, default=0.0)
    parser.add_argument("--adaptive_noise_scale", type=float, default=0.0)
    parser.add_argument("--zero_terminal_snr", action="store_true")
    parser.add_argument("--min_timestep", type=int, default=0)
    parser.add_argument("--max_timestep", type=int, default=1000)
    parser.add_argument("--loss_type", type=str, default="l2", choices=["l1", "l2", "huber", "smooth_l1", "dwt", "fft"])
    parser.add_argument("--huber_schedule", type=str, default="constant", choices=["constant", "exponential", "snr"])
    parser.add_argument("--huber_c", type=float, default=0.1)
    parser.add_argument("--huber_scale", type=float, default=1.0)
    parser.add_argument("--lowram", action="store_true")
    parser.add_argument("--highvram", action="store_true")
    parser.add_argument("--sample_every_n_steps", type=int, default=0)
    parser.add_argument("--sample_at_first", action="store_true")
    parser.add_argument("--sample_every_n_epochs", type=int, default=0)
    parser.add_argument("--sample_prompts", type=str, default="")
    parser.add_argument("--sample_sampler", type=str, default="ddim")
    parser.add_argument("--config_file", type=str, default="")
    parser.add_argument("--output_config", action="store_true")
    parser.add_argument("--metadata_title", type=str, default="")
    parser.add_argument("--metadata_author", type=str, default="")
    parser.add_argument("--metadata_description", type=str, default="")
    parser.add_argument("--metadata_license", type=str, default="")
    parser.add_argument("--metadata_tags", type=str, default="")
    parser.add_argument("--prior_loss_weight", type=float, default=1.0)
    parser.add_argument("--conditioning_data_dir", type=str)
    parser.add_argument("--masked_loss", action="store_true")
    parser.add_argument("--deepspeed", action="store_true")
    parser.add_argument("--zero_stage", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("--offload_optimizer_device", type=str, default="None", choices=["None", "cpu", "nvme"])
    parser.add_argument("--offload_optimizer_nvme_path", type=str, default="")
    parser.add_argument("--offload_param_device", type=str, default="None", choices=["None", "cpu", "nvme"])
    parser.add_argument("--offload_param_nvme_path", type=str, default="")
    parser.add_argument("--zero3_init_flag", action="store_true")
    parser.add_argument("--zero3_save_16bit_model", action="store_true")
    parser.add_argument("--fp16_master_weights_and_gradients", action="store_true")
    parser.add_argument("--optimizer_type", type=str, default="AdamW")
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--use_lion_optimizer", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--optimizer_args", nargs="*", default=[])
    parser.add_argument("--lr_scheduler_type", type=str, default="constant")
    parser.add_argument("--lr_scheduler_args", nargs="*", default=[])
    parser.add_argument("--lr_scheduler", type=str, default="")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--lr_decay_steps", type=int, default=0)
    parser.add_argument("--lr_scheduler_num_cycles", type=float, default=1.0)
    parser.add_argument("--lr_scheduler_power", type=float, default=1.0)
    parser.add_argument("--fused_backward_pass", action="store_true")
    parser.add_argument("--lr_scheduler_timescale", type=int, default=1000)
    parser.add_argument("--lr_scheduler_min_lr_ratio", type=float, default=0.0)
    parser.add_argument("--dataset_config", type=str, default="")
    parser.add_argument("--min_snr_gamma", type=float, default=0.0)
    parser.add_argument("--scale_v_pred_loss_like_noise_pred", action="store_true")
    parser.add_argument("--v_pred_like_loss", action="store_true")
    parser.add_argument("--debiased_estimation_loss", action="store_true")
    parser.add_argument("--weighted_captions", action="store_true")
    parser.add_argument("--cpu_offload_checkpointing", action="store_true")
    parser.add_argument("--no_metadata", action="store_true")
    parser.add_argument("--save_model_as", type=str, default="safetensors", choices=["None", "ckpt", "pt", "safetensors"])
    parser.add_argument("--unet_lr", type=float, default=1e-4)
    parser.add_argument("--text_encoder_lr", nargs="*", type=float, default=[1e-5])
    parser.add_argument("--fp8_base_unet", action="store_true")
    parser.add_argument("--network_weights", type=str, default="")
    parser.add_argument("--network_module", type=str, default="")
    parser.add_argument("--network_dim", type=int, default=32)
    parser.add_argument("--network_alpha", type=float, default=1.0)
    parser.add_argument("--network_dropout", type=float, default=0.0)
    parser.add_argument("--network_args", nargs="*", default=[])
    parser.add_argument("--network_train_unet_only", action="store_true")
    parser.add_argument("--network_train_text_encoder_only", action="store_true")
    parser.add_argument("--training_comment", type=str, default="")
    parser.add_argument("--dim_from_weights", action="store_true")
    parser.add_argument("--scale_weight_norms", type=float, default=0.0)
    parser.add_argument("--base_weights", nargs="*", type=str, default=[])
    parser.add_argument("--base_weights_multiplier", nargs="*", type=float, default=[])
    parser.add_argument("--no_half_vae", action="store_true")
    parser.add_argument("--skip_until_initial_step", action="store_true")
    parser.add_argument("--initial_epoch", type=int, default=0)
    parser.add_argument("--initial_step", type=int, default=0)
    parser.add_argument("--validation_seed", type=int, default=42)
    parser.add_argument("--validation_split", type=float, default=0.0)
    parser.add_argument("--validate_every_n_steps", type=int, default=0)
    parser.add_argument("--validate_every_n_epochs", type=int, default=0)
    parser.add_argument("--max_validation_steps", type=int, default=0)
    parser.add_argument("--cache_text_encoder_outputs", action="store_true")
    parser.add_argument("--cache_text_encoder_outputs_to_disk", action="store_true")
    parser.add_argument("--disable_mmap_load_safetensors", action="store_true")
    
    # Teacher-student specific arguments
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to use from teacher outputs")
    
    return parser


def main():
    parser = setup_parser()
    args = parser.parse_args()
    
    # Verify arguments
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)
    
    # Set seed
    if hasattr(args, 'seed') and args.seed is not None:
        set_seed(args.seed)
    
    # Initialize accelerator
    ddp_kwargs = dict(
        find_unused_parameters=False,
        static_graph=True,
    )
    
    # Prepare logging configuration
    log_kwargs = {}
    if hasattr(args, 'log_with') and args.log_with:
        log_kwargs['log_with'] = args.log_with
    if hasattr(args, 'logging_dir') and args.logging_dir:
        log_kwargs['logging_dir'] = args.logging_dir
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        ddp_kwargs=ddp_kwargs,
        **log_kwargs
    )
    
    # Setup logging
    if accelerator.is_main_process:
        log_level = getattr(args, 'console_log_level', 'INFO')
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()],
        )
    
    # Initialize trainer
    trainer = StudentTrainer()
    
    # Load student model
    weight_dtype = train_util.prepare_dtype(args)
    text_encoder1, text_encoder2, vae, unet = trainer.load_student_model(args, weight_dtype, accelerator)
    
    # Load teacher outputs dataset
    dataset = TeacherStudentDataset(args.train_data_dir, args.max_samples)
    
    # Setup tokenizers (needed for size embeddings)
    tokenize_strategy = trainer.get_tokenize_strategy(args)
    tokenizers = trainer.get_tokenizers(tokenize_strategy)
    
    # Train student model
    trainer.train_student(
        args, accelerator, [text_encoder1, text_encoder2], vae, unet, 
        dataset, weight_dtype
    )


if __name__ == "__main__":
    main()