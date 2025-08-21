#!/usr/bin/env python3
"""
Generate teacher outputs for SDXL teacher-student training.
This script processes a dataset through a teacher model and saves latents, timesteps, 
text embeddings, and teacher noise predictions for later student training.
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


class TeacherOutputGenerator:
    def __init__(self):
        self.vae_scale_factor = sdxl_model_util.VAE_SCALE_FACTOR
        self.is_sdxl = True

    def load_teacher_model(self, args, weight_dtype, accelerator):
        """Load the teacher model (same as original sdxl_train_network.py)"""
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

    def cache_text_encoder_outputs_if_needed(
        self, args, accelerator: Accelerator, unet, vae, text_encoders, dataset: train_util.DatasetGroup, weight_dtype
    ):
        """Cache text encoder outputs if needed (same as original)"""
        if args.cache_text_encoder_outputs:
            if not args.lowram:
                logger.info("move vae and unet to cpu to save memory")
                org_vae_device = vae.device
                org_unet_device = unet.device
                vae.to("cpu")
                unet.to("cpu")
                clean_memory_on_device(accelerator.device)

            text_encoders[0].to(accelerator.device, dtype=weight_dtype)
            text_encoders[1].to(accelerator.device, dtype=weight_dtype)
            with accelerator.autocast():
                dataset.new_cache_text_encoder_outputs(text_encoders + [accelerator.unwrap_model(text_encoders[-1])], accelerator)
            accelerator.wait_for_everyone()

            text_encoders[0].to("cpu", dtype=torch.float32)
            text_encoders[1].to("cpu", dtype=torch.float32)
            clean_memory_on_device(accelerator.device)

            if not args.lowram:
                logger.info("move vae and unet back to original device")
                vae.to(org_vae_device)
                unet.to(org_unet_device)
        else:
            text_encoders[0].to(accelerator.device, dtype=weight_dtype)
            text_encoders[1].to(accelerator.device, dtype=weight_dtype)

    def get_text_cond(self, args, accelerator, batch, tokenizers, text_encoders, weight_dtype):
        """Get text conditioning (same as original)"""
        if "text_encoder_outputs1_list" not in batch or batch["text_encoder_outputs1_list"] is None:
            input_ids1 = batch["input_ids"]
            input_ids2 = batch["input_ids2"]
            with torch.enable_grad():
                input_ids1 = input_ids1.to(accelerator.device)
                input_ids2 = input_ids2.to(accelerator.device)
                encoder_hidden_states1, encoder_hidden_states2, pool2 = train_util.get_hidden_states_sdxl(
                    args.max_token_length,
                    input_ids1,
                    input_ids2,
                    tokenizers[0],
                    tokenizers[1],
                    text_encoders[0],
                    text_encoders[1],
                    None if not args.full_fp16 else weight_dtype,
                    accelerator=accelerator,
                )
        else:
            encoder_hidden_states1 = batch["text_encoder_outputs1_list"].to(accelerator.device).to(weight_dtype)
            encoder_hidden_states2 = batch["text_encoder_outputs2_list"].to(accelerator.device).to(weight_dtype)
            pool2 = batch["text_encoder_pool2_list"].to(accelerator.device).to(weight_dtype)

        return encoder_hidden_states1, encoder_hidden_states2, pool2

    def call_teacher_unet(
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
        """Call teacher UNet to get noise predictions"""
        noisy_latents = noisy_latents.to(weight_dtype)

        # get size embeddings
        orig_size = batch["original_sizes_hw"]
        crop_size = batch["crop_top_lefts"]
        target_size = batch["target_sizes_hw"]
        embs = sdxl_train_util.get_size_embeddings(orig_size, crop_size, target_size, accelerator.device).to(weight_dtype)

        # concat embeddings
        encoder_hidden_states1, encoder_hidden_states2, pool2 = text_conds
        vector_embedding = torch.cat([pool2, embs], dim=1).to(weight_dtype)
        text_embedding = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=2).to(weight_dtype)

        with torch.no_grad():
            noise_pred = unet(noisy_latents, timesteps, text_embedding, vector_embedding)
        
        return noise_pred

    def generate_teacher_outputs(self, args, accelerator, text_encoders, vae, unet, dataset, weight_dtype):
        """Generate teacher outputs for the entire dataset"""
        logger.info("Starting teacher output generation...")
        
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

        # Create scheduler for noise generation
        scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        scheduler.set_timesteps(args.num_inference_steps)

        # Process dataset
        dataset.set_max_length(args.max_token_length)
        dataset.set_has_cond(True)
        
        # Create output directory
        output_dir = Path(args.teacher_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save dataset info
        dataset_info = {
            "num_samples": len(dataset),
            "resolution": args.resolution,
            "max_token_length": args.max_token_length,
            "teacher_model": args.pretrained_model_name_or_path,
            "num_inference_steps": args.num_inference_steps,
        }
        
        with open(output_dir / "dataset_info.json", "w") as f:
            json.dump(dataset_info, f, indent=2)

        # Process each sample
        for i, batch in enumerate(tqdm(dataset, desc="Generating teacher outputs")):
            try:
                # Get text conditioning
                text_conds = self.get_text_cond(args, accelerator, batch, 
                                              [self.tokenizer1, self.tokenizer2], text_encoders, weight_dtype)
                
                # Get image and generate latents
                image = batch["image"].to(accelerator.device, dtype=weight_dtype)
                latents = vae.encode(image).latent_dist.sample() * self.vae_scale_factor
                
                # Generate noise and timesteps
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, scheduler.num_train_timesteps, (latents.shape[0],), device=latents.device)
                
                # Add noise to latents
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)
                
                # Get teacher noise prediction
                teacher_noise_pred = self.call_teacher_unet(args, accelerator, unet, noisy_latents, 
                                                          timesteps, text_conds, batch, weight_dtype)
                
                # Save outputs
                sample_output = {
                    "latents": latents.cpu(),
                    "noisy_latents": noisy_latents.cpu(),
                    "timesteps": timesteps.cpu(),
                    "text_embeddings1": text_conds[0].cpu(),
                    "text_embeddings2": text_conds[1].cpu(),
                    "pool2": text_conds[2].cpu(),
                    "teacher_noise_pred": teacher_noise_pred.cpu(),
                    "original_size": batch["original_sizes_hw"].cpu(),
                    "crop_top_left": batch["crop_top_lefts"].cpu(),
                    "target_size": batch["target_sizes_hw"].cpu(),
                }
                
                # Save individual sample
                torch.save(sample_output, output_dir / f"sample_{i:06d}.pt")
                
                # Clean up memory
                del latents, noise, timesteps, noisy_latents, teacher_noise_pred, sample_output
                torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                continue

        logger.info(f"Teacher output generation completed. Outputs saved to {output_dir}")
        
        # Move models back to CPU to free GPU memory
        if not args.lowram:
            text_encoders[0].to("cpu")
            text_encoders[1].to("cpu")
            vae.to("cpu")
            unet.to("cpu")
            clean_memory_on_device(accelerator.device)


def setup_parser() -> argparse.ArgumentParser:
    """Setup argument parser with all original arguments plus teacher-specific ones"""
    parser = argparse.ArgumentParser(description="Generate teacher outputs for SDXL teacher-student training")
    
    # Add all original arguments from sdxl_train_network.py
    parser.add_argument("--console_log_level", type=str, default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--console_log_file", type=str)
    parser.add_argument("--console_log_simple", action="store_true")
    parser.add_argument("--v2", action="store_true")
    parser.add_argument("--v_parameterization", action="store_true")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True,
                       help="Path to teacher model")
    parser.add_argument("--tokenizer_cache_dir", type=str)
    parser.add_argument("--train_data_dir", type=str, required=True)
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
    parser.add_argument("--output_name", type=str, default="teacher_outputs")
    parser.add_argument("--save_precision", type=str, default="fp16", choices=["None", "float", "fp16", "bf16"])
    parser.add_argument("--save_every_n_epochs", type=int, default=0)
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
    parser.add_argument("--max_train_steps", type=int, default=0)
    parser.add_argument("--max_train_epochs", type=int, default=0)
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
    
    # Teacher-specific arguments
    parser.add_argument("--teacher_output_dir", type=str, required=True,
                       help="Directory to save teacher outputs")
    parser.add_argument("--num_inference_steps", type=int, default=1000,
                       help="Number of inference steps for noise generation")
    
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
    
    # Initialize generator
    generator = TeacherOutputGenerator()
    
    # Load teacher model
    weight_dtype = train_util.prepare_dtype(args)
    text_encoder1, text_encoder2, vae, unet = generator.load_teacher_model(args, weight_dtype, accelerator)
    
    # Load dataset
    train_dataset_group = train_util.load_target_dataset(args, accelerator, weight_dtype)
    
    # Setup tokenizers
    tokenize_strategy = generator.get_tokenize_strategy(args)
    tokenizers = generator.get_tokenizers(tokenize_strategy)
    generator.tokenizer1, generator.tokenizer2 = tokenizers
    
    # Setup latents caching strategy
    latents_caching_strategy = generator.get_latents_caching_strategy(args)
    
    # Setup text encoding strategy
    text_encoding_strategy = generator.get_text_encoding_strategy(args)
    
    # Setup text encoder outputs caching strategy
    text_encoder_outputs_caching_strategy = generator.get_text_encoder_outputs_caching_strategy(args)
    
    # Prepare dataset
    train_dataset_group.prepare_text_encoder_outputs(
        tokenizers,
        text_encoders=[text_encoder1, text_encoder2],
        text_encoding_strategy=text_encoding_strategy,
        text_encoder_outputs_caching_strategy=text_encoder_outputs_caching_strategy,
        accelerator=accelerator,
        weight_dtype=weight_dtype,
    )
    
    # Cache text encoder outputs if needed
    generator.cache_text_encoder_outputs_if_needed(
        args, accelerator, unet, vae, [text_encoder1, text_encoder2], train_dataset_group, weight_dtype
    )
    
    # Generate teacher outputs
    generator.generate_teacher_outputs(
        args, accelerator, [text_encoder1, text_encoder2], vae, unet, 
        train_dataset_group, weight_dtype
    )


if __name__ == "__main__":
    main()