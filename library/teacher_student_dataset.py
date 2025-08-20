"""
Teacher-Student Dataset for LoRA training.
This dataset reads pre-computed teacher outputs instead of processing images from scratch.
Supports both SD and SDXL models.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from PIL import Image

from .train_util import BaseDataset, ImageInfo, DreamBoothSubset

logger = logging.getLogger(__name__)


class TeacherStudentDataset(BaseDataset):
    """
    Dataset for teacher-student training that uses pre-computed teacher outputs.
    This eliminates the need to keep the teacher model in memory during training.
    Supports both SD and SDXL models.
    """
    
    def __init__(
        self,
        subsets: List[DreamBoothSubset],
        teacher_outputs_dir: str,
        is_training_dataset: bool,
        batch_size: int,
        resolution,
        network_multiplier: float,
        enable_bucket: bool,
        min_bucket_reso: int,
        max_bucket_reso: int,
        bucket_reso_steps: int,
        bucket_no_upscale: bool,
        prior_loss_weight: float,
        debug_dataset: bool,
        validation_split: float,
        validation_seed: Optional[int],
        resize_interpolation: Optional[str],
    ) -> None:
        super().__init__(resolution, network_multiplier, debug_dataset, resize_interpolation)
        
        self.teacher_outputs_dir = teacher_outputs_dir
        self.batch_size = batch_size
        self.size = min(self.width, self.height)
        self.prior_loss_weight = prior_loss_weight
        self.is_training_dataset = is_training_dataset
        self.validation_seed = validation_seed
        self.validation_split = validation_split
        
        self.enable_bucket = enable_bucket
        if self.enable_bucket:
            min_bucket_reso, max_bucket_reso = self.adjust_min_max_bucket_reso_by_steps(
                resolution, min_bucket_reso, max_bucket_reso, bucket_reso_steps
            )
            self.min_bucket_reso = min_bucket_reso
            self.max_bucket_reso = max_bucket_reso
            self.bucket_reso_steps = bucket_reso_steps
            self.bucket_no_upscale = bucket_no_upscale
        else:
            self.min_bucket_reso = None
            self.max_bucket_reso = None
            self.bucket_reso_steps = None
            self.bucket_no_upscale = False
        
        # Load teacher info
        teacher_info_path = os.path.join(teacher_outputs_dir, "teacher_info.json")
        if os.path.exists(teacher_info_path):
            with open(teacher_info_path, "r", encoding="utf-8") as f:
                self.teacher_info = json.load(f)
            logger.info(f"Loaded teacher info: {self.teacher_info['teacher_model']}")
            self.is_sdxl = self.teacher_info.get("is_sdxl", False)
            logger.info(f"Model type: {'SDXL' if self.is_sdxl else 'SD'}")
        else:
            raise FileNotFoundError(f"Teacher info file not found: {teacher_info_path}")
        
        # Load dataset
        self.image_data: List[ImageInfo] = []
        self.load_dataset(subsets)
        
        if len(self.image_data) == 0:
            logger.warning("No images found in teacher outputs directory")
        
        logger.info(f"Loaded {len(self.image_data)} teacher outputs")
    
    def load_dataset(self, subsets: List[DreamBoothSubset]):
        """Load dataset from teacher outputs directory"""
        for subset in subsets:
            if not os.path.isdir(subset.image_dir):
                logger.warning(f"Image directory not found: {subset.image_dir}")
                continue
            
            # Get image files from the subset
            image_files = []
            if hasattr(subset, 'image_files') and subset.image_files:
                image_files = subset.image_files
            else:
                # Scan directory for images
                for ext in ['.jpg', '.jpeg', '.png', '.webp']:
                    image_files.extend(
                        [f for f in os.listdir(subset.image_dir) if f.lower().endswith(ext)]
                    )
            
            # Load metadata if available
            metadata = {}
            if hasattr(subset, 'metadata_file') and subset.metadata_file and os.path.exists(subset.metadata_file):
                with open(subset.metadata_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            
            # Process each image
            for image_file in image_files:
                image_path = os.path.join(subset.image_dir, image_file)
                base_name = os.path.splitext(image_file)[0]
                
                # Look for teacher output file
                teacher_output_path = os.path.join(self.teacher_outputs_dir, f"{base_name}_teacher.npz")
                
                if not os.path.exists(teacher_output_path):
                    logger.warning(f"Teacher output not found for {image_file}: {teacher_output_path}")
                    continue
                
                try:
                    # Load teacher output
                    teacher_data = np.load(teacher_output_path)
                    
                    # Get caption from metadata or teacher output
                    caption = ""
                    if image_path in metadata:
                        caption = metadata[image_path].get("caption", "")
                    elif "caption" in teacher_data:
                        caption = str(teacher_data["caption"])
                    
                    # Create ImageInfo
                    image_info = ImageInfo(
                        image_path=image_path,
                        absolute_path=os.path.abspath(image_path),
                        caption=caption,
                        is_reg=False,
                        teacher_output_path=teacher_output_path,
                    )
                    
                    self.image_data.append(image_info)
                    
                except Exception as e:
                    logger.error(f"Error loading teacher output for {image_file}: {e}")
                    continue
    
    def __len__(self) -> int:
        return len(self.image_data)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get a single training example with teacher outputs"""
        image_info = self.image_data[index]
        
        try:
            # Load teacher outputs
            teacher_data = np.load(image_info.teacher_output_path)
            
            # Convert numpy arrays to tensors
            latents = torch.from_numpy(teacher_data["latents"]).float()
            timesteps = torch.from_numpy(teacher_data["timesteps"]).long()
            text_embeddings = torch.from_numpy(teacher_data["text_embeddings"]).float()
            eps_teacher = torch.from_numpy(teacher_data["eps_teacher"]).float()
            
            # Get original size
            original_size = teacher_data.get("original_size", [self.width, self.height])
            
            # Handle SDXL vs SD text embeddings
            if self.is_sdxl and text_embeddings.dim() == 3 and text_embeddings.shape[1] == 2:
                # SDXL: text_embeddings is [batch, 2, seq_len, hidden_dim]
                # Split into two separate embeddings
                text_embeddings_1 = text_embeddings[:, 0]  # First text encoder
                text_embeddings_2 = text_embeddings[:, 1]  # Second text encoder
                # Store as a list for compatibility with training code
                text_embeddings = [text_embeddings_1, text_embeddings_2]
            elif self.is_sdxl and text_embeddings.dim() == 4:
                # SDXL: text_embeddings is [batch, 2, seq_len, hidden_dim]
                text_embeddings_1 = text_embeddings[:, 0]
                text_embeddings_2 = text_embeddings[:, 1]
                text_embeddings = [text_embeddings_1, text_embeddings_2]
            else:
                # SD: text_embeddings is [batch, seq_len, hidden_dim]
                # Wrap in list for compatibility
                text_embeddings = [text_embeddings]
            
            # Create batch item
            batch_item = {
                "latents": latents,
                "timesteps": timesteps,
                "text_embeddings": text_embeddings,
                "eps_teacher": eps_teacher,
                "caption": image_info.caption,
                "image_path": image_info.image_path,
                "original_size": original_size,
                "teacher_output_path": image_info.teacher_output_path,
                "is_sdxl": self.is_sdxl,
            }
            
            return batch_item
            
        except Exception as e:
            logger.error(f"Error loading teacher output for {image_info.image_path}: {e}")
            # Return a dummy item to avoid breaking the training loop
            if self.is_sdxl:
                # SDXL dummy item
                batch_item = {
                    "latents": torch.zeros(1, 4, self.height // 8, self.width // 8),
                    "timesteps": torch.zeros(1, dtype=torch.long),
                    "text_embeddings": [
                        torch.zeros(1, 77, 768),  # First text encoder
                        torch.zeros(1, 77, 1280)  # Second text encoder
                    ],
                    "eps_teacher": torch.zeros(1, 4, self.height // 8, self.width // 8),
                    "caption": "",
                    "image_path": image_info.image_path,
                    "original_size": [self.width, self.height],
                    "teacher_output_path": image_info.teacher_output_path,
                    "is_sdxl": self.is_sdxl,
                }
            else:
                # SD dummy item
                batch_item = {
                    "latents": torch.zeros(1, 4, self.height // 8, self.width // 8),
                    "timesteps": torch.zeros(1, dtype=torch.long),
                    "text_embeddings": [torch.zeros(1, 77, 768)],
                    "eps_teacher": torch.zeros(1, 4, self.height // 8, self.width // 8),
                    "caption": "",
                    "image_path": image_info.image_path,
                    "original_size": [self.width, self.height],
                    "teacher_output_path": image_info.teacher_output_path,
                    "is_sdxl": self.is_sdxl,
                }
            
            return batch_item
    
    def is_latent_cacheable(self) -> bool:
        """Teacher outputs are already cached, so this is always True"""
        return True
    
    def get_image_info(self, index: int) -> ImageInfo:
        """Get image info for the given index"""
        return self.image_data[index]
    
    def get_bucket_info(self, index: int) -> Tuple[int, int]:
        """Get bucket resolution for the given index"""
        if not self.enable_bucket:
            return self.width, self.height
        
        # For teacher-student training, we can use the original resolution
        # since teacher outputs are already processed
        image_info = self.image_data[index]
        try:
            teacher_data = np.load(image_info.teacher_output_path)
            original_size = teacher_data.get("original_size", [self.width, self.height])
            return original_size[0], original_size[1]
        except:
            return self.width, self.height


class TeacherStudentSubset(DreamBoothSubset):
    """Subset configuration for teacher-student training"""
    
    def __init__(
        self,
        image_dir: str,
        teacher_outputs_dir: str,
        metadata_file: Optional[str] = None,
        **kwargs
    ):
        super().__init__(image_dir=image_dir, **kwargs)
        self.teacher_outputs_dir = teacher_outputs_dir
        self.metadata_file = metadata_file