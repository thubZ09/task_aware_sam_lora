from .base_config import BaseConfig, ModelConfig, DataConfig, TrainingConfig, SystemConfig

def get_t4_optimized_config() -> BaseConfig:
    """for T4 GPU"""
    
    model_config = ModelConfig(
        #SAM settings
        sam_model_type="vit_h",
        sam_checkpoint="checkpoints/sam_vit_h_4b8939.pth",
        
        #LoRA 
        lora_alpha=16.0,  
        lora_dropout=0.1,
        
        #hypernetwork
        hypernetwork_hidden_dim=256,  
        hypernetwork_num_layers=3,   
        hypernetwork_num_heads=4,    
        hypernetwork_dropout=0.1,
        hypernetwork_max_lora_params=500000,  
    )
    
    data_config = DataConfig(
        max_samples=10000,
        num_workers=4, 
        pin_memory=True,
        
        image_size=1024,  
        use_augmentation=True,
        aug_probability=0.3,  #reduced for faster training
    )
    
    training_config = TrainingConfig(
        batch_size=1,  
        num_epochs=2,
        learning_rate=5e-5, 
        weight_decay=0.01,
        warmup_ratio=0.1,
        
        #gradient accumulation to simulate larger batches
        accumulation_steps=2,
        
        use_amp=True,
        amp_dtype="float16",

        seg_loss_weight=1.0,
        iou_loss_weight=1.0,
        lora_reg_weight=0.005,  
        
        val_interval=50,
        save_interval=250,
    )
    
    system_config = SystemConfig(
        max_memory_gb=14,  
        gradient_checkpointing=True,
        dataloader_workers=2,
        
        #logging
        log_level="INFO",
        use_wandb=True,
        wandb_project="task-aware-sam-lora-t4",
        
        #paths
        checkpoint_dir="checkpoints",
        output_dir="outputs",
        cache_dir="cache",
    )
    
    return BaseConfig(
        model=model_config,
        data=data_config,
        training=training_config,
        system=system_config
    )

def get_demo_config() -> BaseConfig:
    base_config = get_t4_optimized_config()
    
    base_config.training.batch_size = 1
    base_config.data.num_workers = 1
    base_config.system.use_wandb = False
    
    return base_config

def get_debug_config() -> BaseConfig:
    
    base_config = get_t4_optimized_config()
    
    base_config.data.max_samples = 100
    base_config.training.batch_size = 1
    base_config.training.num_epochs = 2
    base_config.training.val_interval = 10
    base_config.training.save_interval = 20
    base_config.system.log_level = "DEBUG"
    
    return base_config