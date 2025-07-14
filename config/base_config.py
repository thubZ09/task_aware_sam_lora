import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class ModelConfig:
    """model architecture config"""
    
    #sam
    sam_model_type: str = "vit_h"
    sam_checkpoint: str = "checkpoints/sam_vit_h_4b8939.pth"
    freeze_image_encoder: bool = True
    freeze_prompt_encoder: bool = True
    
    #LoRA 
    lora_rank: int = 4
    lora_alpha: float = 32.0
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "output_upscaling.0",
        "output_upscaling.1", 
        "output_hypernetworks_mlps.0",
        "output_hypernetworks_mlps.1",
        "output_hypernetworks_mlps.2",
        "output_hypernetworks_mlps.3",
        "iou_prediction_head.0",
        "iou_prediction_head.1",
        "iou_prediction_head.2"
    ])
    
    #hypernetwork
    text_encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    text_embedding_dim: int = 384
    hypernetwork_hidden_dim: int = 512
    hypernetwork_num_layers: int = 4
    hypernetwork_num_heads: int = 8
    hypernetwork_dropout: float = 0.1
    hypernetwork_max_lora_params: int = 1000000  # 1M params max

@dataclass
class DataConfig:
    """for data loading and processing"""
    
    #paths
    coco_root: str = "data/coco"
    coco_ann_file: str = "data/coco/annotations/panoptic_train2017.json"
    coco_img_dir: str = "data/coco/train2017"
    coco_panoptic_dir: str = "data/coco/panoptic_train2017"
    
    #processing
    image_size: int = 1024
    max_samples: int = 10000
    num_workers: int = 4
    pin_memory: bool = True
    
    #task descriptions
    task_templates: List[str] = field(default_factory=lambda: [
        "segment all {}",
        "find and segment {}",
        "isolate all {} objects",
        "segment every {} in the image",
        "locate and segment all {} instances"
    ])
    
    #augment
    use_augmentation: bool = True
    aug_probability: float = 0.5
    color_jitter_strength: float = 0.4
    gaussian_blur_prob: float = 0.1

@dataclass
class TrainingConfig:
    """for training process."""
    
    #training parameters
    batch_size: int = 8
    num_epochs: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    
    #optimize
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    gradient_clip_norm: float = 1.0
    accumulation_steps: int = 1
    
    #loss weights
    seg_loss_weight: float = 1.0
    iou_loss_weight: float = 1.0
    lora_reg_weight: float = 0.01
    
    val_interval: int = 100
    save_interval: int = 500
    max_checkpoints: int = 3
    
    #mixed precision
    use_amp: bool = True
    amp_dtype: str = "float16"

@dataclass
class SystemConfig:
    """Configuration for system and hardware."""
    
    # Device settings
    device: str = "cuda"
    seed: int = 42
    
    # Memory optimization
    max_memory_gb: int = 14  # T4 has 16GB, leave some buffer
    gradient_checkpointing: bool = True
    dataloader_workers: int = 2
    
    # Logging
    log_level: str = "INFO"
    log_dir: str = "logs"
    
    # Experiment tracking
    use_wandb: bool = True
    wandb_project: str = "task-aware-sam-lora"
    wandb_entity: Optional[str] = None
    
    # Paths
    checkpoint_dir: str = "checkpoints"
    output_dir: str = "outputs"
    cache_dir: str = "cache"

@dataclass
class BaseConfig:
    """Main configuration class combining all sub-configs."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    def __post_init__(self):
        """Post-initialization setup."""
        # Create directories
        os.makedirs(self.system.checkpoint_dir, exist_ok=True)
        os.makedirs(self.system.output_dir, exist_ok=True)
        os.makedirs(self.system.log_dir, exist_ok=True)
        os.makedirs(self.system.cache_dir, exist_ok=True)
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration values."""
        assert self.model.lora_rank > 0, "LoRA rank must be positive"
        assert self.model.lora_alpha > 0, "LoRA alpha must be positive"
        assert self.training.batch_size > 0, "Batch size must be positive"
        assert self.training.learning_rate > 0, "Learning rate must be positive"
        assert self.data.max_samples > 0, "Max samples must be positive"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging."""
        return {
            "model": self.model.__dict__,
            "data": self.data.__dict__,
            "training": self.training.__dict__,
            "system": self.system.__dict__
        }
    
    def save(self, path: str):
        """Save configuration to file."""
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'BaseConfig':
        """Load configuration from file."""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        
        return cls(
            model=ModelConfig(**data['model']),
            data=DataConfig(**data['data']),
            training=TrainingConfig(**data['training']),
            system=SystemConfig(**data['system'])
        )