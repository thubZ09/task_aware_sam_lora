import os
import sys
import argparse
import logging
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import json

sys.path.append(str(Path(__file__).parent.parent))

from src.models.hypernetwork import TaskAwareHyperNet
from src.models.sam_wrapper import SAMWithLoRA
from src.data.dataset import TaskAwareDataset
from src.data.coco_loader import COCOPanopticDataset
from src.training.trainer import HyperNetTrainer
from src.utils.text_processing import TextEncoder, create_coco_task_descriptions
from src.utils.checkpoint import CheckpointManager, create_config_backup
from config.training_config import TrainingConfig

def setup_logging(log_level: str = "INFO"):
    """setup logging config"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('training.log')
        ]
    )

def parse_args():
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description='train Task-Aware SAM LoRA')
    
    parser.add_argument('--config', type=str, default='config/training_config.py',
                       help='Path to config file')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--data_dir', type=str, default='data/coco_panoptic',
                       help='Path to data directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--wandb', action='store_true',
                       help='Enable wandb logging')
    parser.add_argument('--wandb_project', type=str, default='task-aware-sam-lora',
                       help='Wandb project name')
    
    return parser.parse_args()

def load_config(config_path: str) -> TrainingConfig:
    """load training config"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    #import config module
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    return config_module.TrainingConfig()

def create_datasets(config: TrainingConfig, data_dir: str):
    """Create training and validation datasets."""
    #create task descriptions
    task_descriptions = create_coco_task_descriptions()
    
    #create datasets
    train_dataset = COCOPanopticDataset(
        data_dir=data_dir,
        split='train',
        task_descriptions=task_descriptions,
        transforms=config.train_transforms,
        max_samples=config.max_train_samples
    )
    
    val_dataset = COCOPanopticDataset(
        data_dir=data_dir,
        split='val',
        task_descriptions=task_descriptions,
        transforms=config.val_transforms,
        max_samples=config.max_val_samples
    )
    
    #create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=val_dataset.collate_fn
    )
    
    return train_loader, val_loader

def create_models(config: TrainingConfig, device: str):
    """Create and initialize models."""
    #text encoder
    text_encoder = TextEncoder(
        model_name=config.text_encoder_model,
        hidden_dim=config.text_embedding_dim
    ).to(device)
    
    #hypernetwork
    hypernetwork = TaskAwareHyperNet(
     text_dim=config.text_embedding_dim,
     hidden_dim=config.hypernetwork_hidden_dim,
     num_layers=config.hypernetwork_layers,
     num_heads=config.hypernetwork_heads,
     lora_rank=config.lora_rank,
     target_layers=config.target_layers
    ) .to(device)
    
    #SAM with LoRA
    sam_model = SAMWithLoRA(
        checkpoint_path=config.sam_checkpoint_path,
        lora_rank=config.lora_rank,
        target_layers=config.target_layers
    ).to(device)
    
    return text_encoder, hypernetwork, sam_model

def main():
    """main training function."""
    args = parse_args()
    
    #setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    #load config
    config = load_config(args.config)
    
    #override config with command line args
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.learning_rate = args.lr
    
    #create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    #setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    #initialize wandb
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            config=config.__dict__,
            name=f"task-aware-sam-lora-{config.experiment_name}"
        )
    
    #create datasets
    logger.info("Creating datasets...")
    train_loader, val_loader = create_datasets(config, args.data_dir)
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    
    #create models
    logger.info("Creating models...")
    text_encoder, hypernetwork, sam_model = create_models(config, device)
    
    #log model info
    total_params = sum(p.numel() for p in hypernetwork.parameters())
    trainable_params = sum(p.numel() for p in hypernetwork.parameters() if p.requires_grad)
    logger.info(f"Hypernetwork parameters: {total_params:,} (trainable: {trainable_params:,})")
    
    #create optimizer
    optimizer = torch.optim.AdamW(
        hypernetwork.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    #create scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=config.min_learning_rate
    )
    
    #create checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=args.checkpoint_dir,
        max_checkpoints=config.max_checkpoints
    )
    
    #create trainer
    trainer = HyperNetTrainer(
        hypernetwork=hypernetwork,
        text_encoder=text_encoder,
        sam_model=sam_model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config
    )
    
    #resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint_data = checkpoint_manager.load_checkpoint(
            args.resume, hypernetwork, optimizer, device
        )
        start_epoch = checkpoint_data['epoch'] + 1
        logger.info(f"Resuming from epoch {start_epoch}")
    
    #save config backup
    create_config_backup(config.__dict__, args.checkpoint_dir)
    
    #training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, config.epochs):
        logger.info(f"Epoch {epoch + 1}/{config.epochs}")
        
        #training phase
        train_metrics = trainer.train_epoch(train_loader, epoch)
        
        #validation phase
        val_metrics = trainer.validate_epoch(val_loader, epoch)
        
        #update scheduler
        scheduler.step()
        
        #log metrics
        logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
        logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
        logger.info(f"Val mIoU: {val_metrics['miou']:.4f}")
        
        #wandb logging
        if args.wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'val_miou': val_metrics['miou'],
                'learning_rate': optimizer.param_groups[0]['lr']
            })
        
        #save checkpoint
        is_best = val_metrics['loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['loss']
        
        checkpoint_manager.save_checkpoint(
            model=hypernetwork,
            optimizer=optimizer,
            epoch=epoch,
            train_loss=train_metrics['loss'],
            val_loss=val_metrics['loss'],
            metrics=val_metrics,
            is_best=is_best
        )
        
        #early stopping check
        if hasattr(config, 'early_stopping_patience'):
            if trainer.early_stopping_counter >= config.early_stopping_patience:
                logger.info("Early stopping triggered")
                break

    #final evaluation
    logger.info("Training completed!")
    
    #load best model and evaluate
    checkpoint_manager.load_best_model(hypernetwork, device)
    final_metrics = trainer.validate_epoch(val_loader, config.epochs)
    
    logger.info(f"Final validation metrics:")
    for key, value in final_metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    #save final model
    final_path = os.path.join(args.checkpoint_dir, "final_model.pth")
    torch.save({
        'hypernetwork_state_dict': hypernetwork.state_dict(),
        'text_encoder_state_dict': text_encoder.state_dict(),
        'config': config.__dict__,
        'final_metrics': final_metrics
    }, final_path)
    
    logger.info(f"Final model saved: {final_path}")
    
    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    import importlib.util
    main()