#!/usr/bin/env python3
"""
Teacher-Student LoRA Training Script.
This script trains a student LoRA model to mimic the outputs of a teacher model
without keeping the teacher model in memory during training.
Supports both SD and SDXL models, and LyCORIS network modules.
"""

import importlib
import argparse
import math
import os
import typing
from typing import Any, List, Union, Optional
import sys
import random
import time
import json
from multiprocessing import Value
import numpy as np
import toml

from tqdm import tqdm

import torch
from torch.types import Number
from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from accelerate.utils import set_seed
from accelerate import Accelerator
from diffusers import DDPMScheduler
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from library import deepspeed_utils, model_util, sai_model_spec, strategy_base, strategy_sd, strategy_sdxl, sai_model_spec

import library.train_util as train_util
from library.train_util import DreamBoothDataset
from library.teacher_student_dataset import TeacherStudentDataset, TeacherStudentSubset
import library.config_util as config_util
from library.config_util import (
    ConfigSanitizer,
    BlueprintGenerator,
)
import library.huggingface_util as huggingface_util
import library.custom_train_functions as custom_train_functions
from library.custom_train_functions import (
    apply_snr_weight,
    get_weighted_text_embeddings,
    prepare_scheduler_for_custom_training,
    scale_v_prediction_loss_like_noise_prediction,
    add_v_prediction_like_loss,
    apply_debiased_estimation,
    apply_masked_loss,
)
from library.utils import setup_logging, add_logging_arguments

setup_logging()
import logging

logger = logging.getLogger(__name__)


class TeacherStudentNetworkTrainer:
    def __init__(self):
        self.vae_scale_factor = 0.18215
        self.is_sdxl = False

    def generate_step_logs(
        self,
        args: argparse.Namespace,
        current_loss,
        avr_loss,
        lr_scheduler,
        lr_descriptions,
        optimizer=None,
        keys_scaled=None,
        mean_norm=None,
        maximum_norm=None,
        mean_grad_norm=None,
        mean_combined_norm=None,
    ):
        logs = {"loss/current": current_loss, "loss/average": avr_loss}

        if keys_scaled is not None:
            logs["max_norm/keys_scaled"] = keys_scaled
            logs["max_norm/max_key_norm"] = maximum_norm
        if mean_norm is not None:
            logs["norm/avg_key_norm"] = mean_norm
        if mean_grad_norm is not None:
            logs["norm/avg_grad_norm"] = mean_grad_norm
        if mean_combined_norm is not None:
            logs["norm/avg_combined_norm"] = mean_combined_norm

        lrs = lr_scheduler.get_last_lr()
        for i, lr in enumerate(lrs):
            if lr_descriptions is not None:
                lr_desc = lr_descriptions[i]
            else:
                idx = i - (0 if args.network_train_unet_only else -1)
                if idx == -1:
                    lr_desc = "textencoder"
                else:
                    if len(lrs) > 2:
                        lr_desc = f"group{idx}"
                    else:
                        lr_desc = "unet"

            logs[f"lr/{lr_desc}"] = lr

        return logs

    def detect_model_type(self, args):
        """Detect if the model is SDXL based on the model path or configuration"""
        model_path = args.model_name_or_path
        
        # Check if it's a local path
        if os.path.exists(model_path):
            # Check for SDXL-specific files
            if os.path.exists(os.path.join(model_path, "text_encoder_2")):
                self.is_sdxl = True
                logger.info("Detected SDXL model (text_encoder_2 found)")
            elif os.path.exists(os.path.join(model_path, "text_encoder")):
                self.is_sdxl = False
                logger.info("Detected SD model (text_encoder found)")
            else:
                # Try to load and check
                try:
                    from diffusers import StableDiffusionPipeline
                    pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
                    if hasattr(pipe, 'text_encoder_2'):
                        self.is_sdxl = True
                        logger.info("Detected SDXL model from pipeline")
                    else:
                        self.is_sdxl = False
                        logger.info("Detected SD model from pipeline")
                    del pipe
                except:
                    logger.warning("Could not determine model type, assuming SD")
                    self.is_sdxl = False
        else:
            # Check HuggingFace model names
            if "xl" in model_path.lower() or "sdxl" in model_path.lower():
                self.is_sdxl = True
                logger.info("Detected SDXL model from name")
            else:
                self.is_sdxl = False
                logger.info("Detected SD model from name")
        
        # Update VAE scale factor for SDXL
        if self.is_sdxl:
            self.vae_scale_factor = 0.13025
            logger.info(f"Using SDXL VAE scale factor: {self.vae_scale_factor}")
        else:
            self.vae_scale_factor = 0.18215
            logger.info(f"Using SD VAE scale factor: {self.vae_scale_factor}")

    def get_tokenize_strategy(self, args):
        if self.is_sdxl:
            return strategy_sdxl.SDXLTokenizeStrategy()
        else:
            return strategy_sd.SDTokenizeStrategy()

    def get_text_encoding_strategy(self, args):
        if self.is_sdxl:
            return strategy_sdxl.SDXLTextEncodingStrategy()
        else:
            return strategy_sd.SDTextEncodingStrategy()

    def get_latents_caching_strategy(self, args):
        if self.is_sdxl:
            return strategy_sdxl.SDXLLatentsCachingStrategy()
        else:
            return strategy_sd.SDLatentsCachingStrategy()

    def get_tokenizers(self, tokenize_strategy):
        if self.is_sdxl:
            return tokenize_strategy.get_tokenizers()
        else:
            return [tokenize_strategy.get_tokenizer()]

    def get_text_encoders(self, args, accelerator, weight_dtype):
        if self.is_sdxl:
            # SDXL has two text encoders
            text_encoder = model_util.load_text_encoder(args.model_name_or_path, weight_dtype)
            text_encoder_2 = model_util.load_text_encoder_2(args.model_name_or_path, weight_dtype)
            
            text_encoder.to(accelerator.device, dtype=weight_dtype)
            text_encoder_2.to(accelerator.device, dtype=weight_dtype)
            
            return [text_encoder, text_encoder_2]
        else:
            text_encoder = model_util.load_text_encoder(args.model_name_or_path, weight_dtype)
            text_encoder.to(accelerator.device, dtype=weight_dtype)
            return [text_encoder]

    def get_unet(self, args, accelerator, weight_dtype):
        if self.is_sdxl:
            unet = model_util.load_unet(args.model_name_or_path, weight_dtype, is_sdxl=True)
        else:
            unet = model_util.load_unet(args.model_name_or_path, weight_dtype)
        
        unet.to(accelerator.device, dtype=weight_dtype)
        return unet

    def get_vae(self, args, accelerator, weight_dtype):
        if self.is_sdxl:
            vae = model_util.load_vae(args.model_name_or_path, weight_dtype, is_sdxl=True)
        else:
            vae = model_util.load_vae(args.model_name_or_path, weight_dtype)
        
        vae.to(accelerator.device, dtype=weight_dtype)
        return vae

    def get_noise_scheduler(self, args):
        return DDPMScheduler.from_pretrained(args.model_name_or_path, subfolder="scheduler")

    def get_network(self, args, accelerator, weight_dtype):
        """Get network module (LoRA, LyCORIS, etc.)"""
        network_module = getattr(args, 'network_module', 'networks.lora')
        
        if network_module == 'lycoris.kohya':
            # LyCORIS network
            try:
                from lycoris.kohya import create_lycoris, create_lycoris_from_weights
                
                if hasattr(args, 'network_weights') and args.network_weights:
                    network = create_lycoris_from_weights(
                        args.network_weights,
                        text_encoder_dims=args.network_dim,
                        unet_dims=args.network_dim,
                        text_encoder_alpha=args.network_alpha,
                        unet_alpha=args.network_alpha,
                        vae_alpha=args.vae_alpha,
                        text_encoder_dropout=args.network_dropout,
                        unet_dropout=args.network_dropout,
                        vae_dropout=args.vae_dropout,
                        text_encoder_scale=args.text_encoder_lr_scale,
                        unet_scale=args.unet_lr_scale,
                        vae_scale=args.vae_lr_scale,
                        network_args=args.network_args,
                        network_alpha_args=args.network_alpha_args,
                        vae_alpha_args=args.vae_alpha_args,
                        network_dropout_args=args.network_dropout_args,
                        vae_dropout_args=args.vae_dropout_args,
                        text_encoder_scale_args=args.text_encoder_lr_scale_args,
                        unet_scale_args=args.unet_lr_scale_args,
                        vae_scale_args=args.vae_lr_scale_args,
                    )
                else:
                    network = create_lycoris(
                        text_encoder_dims=args.network_dim,
                        unet_dims=args.network_dim,
                        text_encoder_alpha=args.network_alpha,
                        unet_alpha=args.network_alpha,
                        vae_alpha=args.vae_alpha,
                        text_encoder_dropout=args.network_dropout,
                        unet_dropout=args.network_dropout,
                        vae_dropout=args.vae_dropout,
                        text_encoder_scale=args.text_encoder_lr_scale,
                        unet_scale=args.unet_lr_scale,
                        vae_scale=args.vae_lr_scale,
                        network_args=args.network_args,
                        network_alpha_args=args.network_alpha_args,
                        vae_alpha_args=args.vae_alpha_args,
                        network_dropout_args=args.network_dropout_args,
                        vae_dropout_args=args.vae_dropout_args,
                        text_encoder_scale_args=args.text_encoder_lr_scale_args,
                        unet_scale_args=args.unet_lr_scale_args,
                        vae_scale_args=args.vae_lr_scale_args,
                    )
                
                logger.info("Created LyCORIS network")
                
            except ImportError:
                logger.error("LyCORIS not installed. Please install it with: pip install lycoris")
                raise
            except Exception as e:
                logger.error(f"Failed to create LyCORIS network: {e}")
                raise
        else:
            # Default LoRA network
            try:
                from library.networks.lora import LoRANetwork
                network = LoRANetwork(
                    text_encoder_dims=args.network_dim,
                    unet_dims=args.network_dim,
                    text_encoder_alpha=args.network_alpha,
                    unet_alpha=args.network_alpha,
                    vae_alpha=args.vae_alpha,
                    text_encoder_dropout=args.network_dropout,
                    unet_dropout=args.network_dropout,
                    vae_dropout=args.vae_dropout,
                    text_encoder_scale=args.text_encoder_lr_scale,
                    unet_scale=args.unet_lr_scale,
                    vae_scale=args.vae_lr_scale,
                    network_args=args.network_args,
                    network_alpha_args=args.network_alpha_args,
                    vae_alpha_args=args.vae_alpha_args,
                    network_dropout_args=args.network_dropout_args,
                    vae_dropout_args=args.vae_dropout_args,
                    text_encoder_scale_args=args.text_encoder_lr_scale_args,
                    unet_scale_args=args.unet_lr_scale_args,
                    vae_scale_args=args.vae_lr_scale_args,
                )
                logger.info("Created LoRA network")
            except ImportError:
                logger.error("LoRA network module not found")
                raise
        
        network.to(accelerator.device, dtype=weight_dtype)
        return network

    def process_batch(
        self,
        batch,
        text_encoders,
        unet,
        network,
        vae,
        noise_scheduler,
        vae_dtype,
        weight_dtype,
        accelerator,
        args,
        text_encoding_strategy: strategy_base.TextEncodingStrategy,
        tokenize_strategy: strategy_base.TokenizeStrategy,
        is_train=True,
        train_text_encoder=True,
        train_unet=True,
    ) -> torch.Tensor:
        """
        Process a batch for teacher-student training.
        Uses pre-computed teacher outputs instead of generating them on-the-fly.
        """
        # Teacher outputs are already provided in the batch
        latents = batch["latents"].to(accelerator.device, dtype=weight_dtype)
        timesteps = batch["timesteps"].to(accelerator.device, dtype=weight_dtype)
        text_embeddings = batch["text_embeddings"].to(accelerator.device, dtype=weight_dtype)
        eps_teacher = batch["eps_teacher"].to(accelerator.device, dtype=weight_dtype)
        
        # Handle SDXL text embeddings (two encoders)
        if self.is_sdxl and len(text_encoders) == 2:
            # For SDXL, we need to handle two text encoders
            if text_embeddings.dim() == 3:
                # Single text embedding, duplicate for SDXL
                text_embeddings = [text_embeddings, text_embeddings]
            elif isinstance(text_embeddings, torch.Tensor) and text_embeddings.dim() == 4:
                # Split the embeddings for SDXL
                text_embeddings = [text_embeddings[:, 0], text_embeddings[:, 1]]
            else:
                # Already a list
                text_embeddings = [text_embeddings, text_embeddings]
        else:
            # For SD, wrap in list for compatibility
            text_embeddings = [text_embeddings]
        
        # Add noise to latents for training
        noise = torch.randn_like(latents)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Get student prediction
        with torch.set_grad_enabled(is_train), accelerator.autocast():
            noise_pred = self.call_unet(
                args,
                accelerator,
                unet,
                noisy_latents.requires_grad_(train_unet),
                timesteps,
                text_embeddings,
                batch,
                weight_dtype,
            )
        
        # Calculate loss: student should predict the same as teacher
        if args.v_parameterization:
            # For v-parameterization, compare velocity predictions
            target = eps_teacher
        else:
            # For epsilon-parameterization, compare noise predictions
            target = eps_teacher
        
        # Use MSE loss between student and teacher predictions
        loss = torch.nn.functional.mse_loss(noise_pred, target, reduction="mean")
        
        return loss

    def call_unet(
        self,
        args,
        accelerator,
        unet,
        noisy_latents,
        timesteps,
        text_encoder_conds,
        batch,
        weight_dtype,
        indices=None,
    ):
        """Call UNet with the given inputs"""
        if indices is not None:
            # Handle differential output preservation
            text_encoder_conds = [cond[indices] if cond is not None else None for cond in text_encoder_conds]
            noisy_latents = noisy_latents[indices]
            timesteps = timesteps[indices]
        
        # Apply network (LoRA/LyCORIS)
        if hasattr(unet, "set_lora_layer"):
            unet.set_lora_layer(True)
        
        # Forward pass
        if self.is_sdxl and len(text_encoder_conds) == 2:
            # SDXL with two text encoders
            noise_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=text_encoder_conds[0],
                added_cond_kwargs={
                    "text_embeds": text_encoder_conds[1],
                    "time_ids": batch.get("time_ids", None)
                }
            ).sample
        else:
            # SD with single text encoder
            noise_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=text_encoder_conds[0] if text_encoder_conds else None,
            ).sample
        
        return noise_pred

    def train(self, args):
        session_id = random.randint(0, 2**32)
        training_started_at = time.time()
        train_util.verify_training_args(args)
        train_util.prepare_dataset_args(args, True)
        deepspeed_utils.prepare_deepspeed_args(args)
        setup_logging(args, reset=True)

        # Check if teacher outputs directory is provided
        if not hasattr(args, 'teacher_outputs_dir') or not args.teacher_outputs_dir:
            logger.error("--teacher_outputs_dir is required for teacher-student training")
            return

        if args.seed is None:
            args.seed = random.randint(0, 2**32)
        set_seed(args.seed)

        # Detect model type (SD vs SDXL)
        self.detect_model_type(args)

        tokenize_strategy = self.get_tokenize_strategy(args)
        strategy_base.TokenizeStrategy.set_strategy(tokenize_strategy)
        tokenizers = self.get_tokenizers(tokenize_strategy)

        # Prepare caching strategy
        latents_caching_strategy = self.get_latents_caching_strategy(args)
        strategy_base.LatentsCachingStrategy.set_strategy(latents_caching_strategy)

        # Prepare dataset using teacher outputs
        if args.dataset_class is None:
            blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, args.masked_loss, True))
            
            # Create teacher-student dataset configuration
            user_config = {
                "datasets": [
                    {
                        "subsets": [
                            {
                                "image_dir": args.train_data_dir,
                                "teacher_outputs_dir": args.teacher_outputs_dir,
                                "metadata_file": args.in_json if hasattr(args, 'in_json') and args.in_json else None,
                            }
                        ]
                    }
                ]
            }
            
            blueprint = blueprint_generator.generate(user_config, args)
            train_dataset_group, val_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
            
            # Replace with teacher-student dataset
            for i, dataset in enumerate(train_dataset_group.datasets):
                if isinstance(dataset, DreamBoothDataset):
                    # Convert to teacher-student dataset
                    subset = dataset.subsets[0] if dataset.subsets else None
                    if subset:
                        teacher_subset = TeacherStudentSubset(
                            image_dir=subset.image_dir,
                            teacher_outputs_dir=args.teacher_outputs_dir,
                            metadata_file=getattr(subset, 'metadata_file', None),
                        )
                        
                        train_dataset_group.datasets[i] = TeacherStudentDataset(
                            subsets=[teacher_subset],
                            teacher_outputs_dir=args.teacher_outputs_dir,
                            is_training_dataset=True,
                            batch_size=args.train_batch_size,
                            resolution=dataset.resolution,
                            network_multiplier=dataset.network_multiplier,
                            enable_bucket=dataset.enable_bucket,
                            min_bucket_reso=dataset.min_bucket_reso,
                            max_bucket_reso=dataset.max_bucket_reso,
                            bucket_reso_steps=dataset.bucket_reso_steps,
                            bucket_no_upscale=dataset.bucket_no_upscale,
                            prior_loss_weight=dataset.prior_loss_weight,
                            debug_dataset=args.debug_dataset,
                            validation_split=0.0,
                            validation_seed=None,
                            resize_interpolation=dataset.resize_interpolation,
                        )
        else:
            # Use arbitrary dataset class
            train_dataset_group = train_util.load_arbitrary_dataset(args)
            val_dataset_group = None

        current_epoch = Value("i", 0)
        current_step = Value("i", 0)
        ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
        collator = train_util.collator_class(current_epoch, current_step, ds_for_collator)

        if args.debug_dataset:
            train_dataset_group.set_current_strategies()
            train_util.debug_dataset(train_dataset_group)
            if val_dataset_group is not None:
                val_dataset_group.set_current_strategies()
                train_util.debug_dataset(val_dataset_group)
            return

        if len(train_dataset_group) == 0:
            logger.error("No data found. Please verify arguments.")
            return

        # Create data loader
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset_group,
            batch_size=args.train_batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=args.max_data_loader_n_workers,
            persistent_workers=args.max_data_loader_n_workers > 0,
        )

        # Initialize accelerator
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with=args.log_with,
            project_config=args.log_with,
            project_dir=args.logging_dir,
            kwargs_handlers=[deepspeed_utils.prepare_deepspeed_kwargs(args)],
        )

        # Load models
        weight_dtype = train_util.prepare_dtype(args)
        vae_dtype = train_util.prepare_vae_dtype(args, weight_dtype)

        text_encoders = self.get_text_encoders(args, accelerator, weight_dtype)
        unet = self.get_unet(args, accelerator, weight_dtype)
        vae = self.get_vae(args, accelerator, weight_dtype)
        network = self.get_network(args, accelerator, weight_dtype)
        noise_scheduler = self.get_noise_scheduler(args)

        # Prepare for training
        network.prepare_network(args, accelerator)
        network.set_multiplier(args.network_multiplier)

        # Create optimizer and scheduler
        optimizer = train_util.get_optimizer(args, network)
        lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)

        # Prepare models
        training_model = network
        if args.train_text_encoder:
            training_model = torch.nn.ModuleList([network] + text_encoders)
        else:
            training_model = network

        training_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            training_model, optimizer, train_dataloader, lr_scheduler
        )

        # Training loop
        num_train_epochs = args.num_train_epochs
        max_train_steps = args.max_train_steps
        global_step = 0
        progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)

        for epoch in range(num_train_epochs):
            accelerator.print(f"\nepoch {epoch+1}/{num_train_epochs}\n")
            current_epoch.value = epoch + 1

            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(training_model):
                    # Process batch
                    loss = self.process_batch(
                        batch,
                        text_encoders,
                        unet,
                        network,
                        vae,
                        noise_scheduler,
                        vae_dtype,
                        weight_dtype,
                        accelerator,
                        args,
                        None,  # text_encoding_strategy not needed for teacher-student
                        tokenize_strategy,
                        is_train=True,
                        train_text_encoder=args.train_text_encoder,
                        train_unet=args.train_unet,
                    )

                    accelerator.backward(loss)
                    
                    if accelerator.sync_gradients:
                        if args.max_grad_norm != 0.0:
                            params_to_clip = accelerator.unwrap_model(network).get_trainable_params()
                            accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    # Log progress
                    if global_step % args.logging_steps == 0:
                        logs = self.generate_step_logs(
                            args, loss.item(), loss.item(), lr_scheduler, None
                        )
                        accelerator.log(logs, step=global_step)

                    # Save model
                    if global_step % args.save_every_n_steps == 0:
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            network.save_weights(args.output_dir, save_dtype=weight_dtype)

                    if global_step >= max_train_steps:
                        break

            if global_step >= max_train_steps:
                break

        # Save final model
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            network.save_weights(args.output_dir, save_dtype=weight_dtype)

        accelerator.end_training()


def main():
    parser = argparse.ArgumentParser()
    
    # Add all the standard training arguments
    add_logging_arguments(parser)
    
    # Required arguments
    parser.add_argument("--train_data_dir", type=str, required=True, help="Directory containing training images")
    parser.add_argument("--teacher_outputs_dir", type=str, required=True, help="Directory containing teacher outputs")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for trained model")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Base model path")
    
    # Optional arguments
    parser.add_argument("--in_json", type=str, help="Metadata JSON file")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Training batch size")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--max_train_steps", type=int, help="Maximum training steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--network_dim", type=int, default=32, help="Network dimension")
    parser.add_argument("--network_alpha", type=float, default=32, help="Network alpha")
    parser.add_argument("--network_module", type=str, default="networks.lora", 
                       choices=["networks.lora", "lycoris.kohya"], help="Network module to use")
    parser.add_argument("--network_weights", type=str, help="Path to network weights for LyCORIS")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_every_n_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--train_text_encoder", action="store_true", help="Train text encoder")
    parser.add_argument("--train_unet", action="store_true", default=True, help="Train UNet")
    parser.add_argument("--v_parameterization", action="store_true", help="Use v-parameterization")
    parser.add_argument("--debug_dataset", action="store_true", help="Debug dataset")
    parser.add_argument("--max_data_loader_n_workers", type=int, default=8)
    
    # Network-specific arguments
    parser.add_argument("--vae_alpha", type=float, default=1.0, help="VAE alpha")
    parser.add_argument("--network_dropout", type=float, default=0.0, help="Network dropout")
    parser.add_argument("--text_encoder_lr_scale", type=float, default=1.0, help="Text encoder LR scale")
    parser.add_argument("--unet_lr_scale", type=float, default=1.0, help="UNet LR scale")
    parser.add_argument("--vae_lr_scale", type=float, default=1.0, help="VAE LR scale")
    parser.add_argument("--network_args", type=str, help="Network arguments")
    parser.add_argument("--network_alpha_args", type=str, help="Network alpha arguments")
    parser.add_argument("--vae_alpha_args", type=str, help="VAE alpha arguments")
    parser.add_argument("--network_dropout_args", type=str, help="Network dropout arguments")
    parser.add_argument("--vae_dropout_args", type=str, help="VAE dropout arguments")
    parser.add_argument("--text_encoder_lr_scale_args", type=str, help="Text encoder LR scale arguments")
    parser.add_argument("--unet_lr_scale_args", type=str, help="UNet LR scale arguments")
    parser.add_argument("--vae_lr_scale_args", type=str, help="VAE LR scale arguments")
    
    args = parser.parse_args()
    
    # Set defaults
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * 1000  # Default assumption
    
    trainer = TeacherStudentNetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()