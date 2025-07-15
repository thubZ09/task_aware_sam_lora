import torch
import os
from typing import Dict, Any, Optional
import json
import shutil
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class CheckpointManager:
    """manages model checkpoints with automatic saving and loading"""
    
    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        
    def save_checkpoint(self, 
                       model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       epoch: int,
                       train_loss: float,
                       val_loss: float,
                       metrics: Dict[str, float],
                       is_best: bool = False,
                       filename: Optional[str] = None) -> str:
        """
        save a checkpoint
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            train_loss: Training loss
            val_loss: Validation loss
            metrics: Dictionary of metrics
            is_best: Whether this is the best model so far
            filename: Optional custom filename
            
        Returns:
            path to saved checkpoint
        """
        if filename is None:
            filename = f"checkpoint_epoch_{epoch:03d}.pth"
        
        checkpoint_path = self.checkpoint_dir / filename
        
        #prepare checkpoint data
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'metrics': metrics,
            'model_config': getattr(model, 'config', {}),
        }
        
        #save checkpoint
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        #save best model separately
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            shutil.copy2(checkpoint_path, best_path)
            logger.info(f"Best model saved: {best_path}")
            
        #clean up old checkpoints
        self._cleanup_old_checkpoints()
        
        return str(checkpoint_path)
    
    def load_checkpoint(self, 
                       checkpoint_path: str,
                       model: torch.nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       device: str = 'cpu') -> Dict[str, Any]:
        """
        load a checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            device: Device to load tensors to
            
        Returns:
            dictionary containing checkpoint metadata
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        #load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        #load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        #load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        
        return {
            'epoch': checkpoint.get('epoch', 0),
            'train_loss': checkpoint.get('train_loss', 0.0),
            'val_loss': checkpoint.get('val_loss', 0.0),
            'metrics': checkpoint.get('metrics', {}),
            'model_config': checkpoint.get('model_config', {})
        }
    
    def load_best_model(self, 
                       model: torch.nn.Module,
                       device: str = 'cpu') -> Dict[str, Any]:
        """load the best saved model."""
        best_path = self.checkpoint_dir / "best_model.pth"
        return self.load_checkpoint(str(best_path), model, device=device)
    
    def _cleanup_old_checkpoints(self):
        """remove old checkpoints, keeping only the most recent ones"""
        #get all checkpoint files
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pth"))
        
        if len(checkpoint_files) <= self.max_checkpoints:
            return
        
        #sort by modification time 
        checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        #remove old checkpoints
        for checkpoint_file in checkpoint_files[self.max_checkpoints:]:
            checkpoint_file.unlink()
            logger.info(f"Removed old checkpoint: {checkpoint_file}")
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """list all available checkpoints with metadata"""
        checkpoints = []
        
        for checkpoint_file in self.checkpoint_dir.glob("checkpoint_epoch_*.pth"):
            try:
                checkpoint = torch.load(checkpoint_file, map_location='cpu')
                checkpoints.append({
                    'filename': checkpoint_file.name,
                    'path': str(checkpoint_file),
                    'epoch': checkpoint.get('epoch', 0),
                    'train_loss': checkpoint.get('train_loss', 0.0),
                    'val_loss': checkpoint.get('val_loss', 0.0),
                    'metrics': checkpoint.get('metrics', {}),
                    'size_mb': checkpoint_file.stat().st_size / (1024 * 1024)
                })
            except Exception as e:
                logger.warning(f"Could not load checkpoint {checkpoint_file}: {e}")
        
        #sort by epoch
        checkpoints.sort(key=lambda x: x['epoch'])
        
        return checkpoints

def save_hypernetwork_checkpoint(hypernetwork: torch.nn.Module,
                                text_encoder: torch.nn.Module,
                                optimizer: torch.optim.Optimizer,
                                epoch: int,
                                train_loss: float,
                                val_loss: float,
                                metrics: Dict[str, float],
                                config: Dict[str, Any],
                                save_path: str):
    """
    save a complete hypernetwork checkpoint
    
    Args:
        hypernetwork: Hypernetwork model
        text_encoder: Text encoder model
        optimizer: Optimizer state
        epoch: Current epoch
        train_loss: Training loss
        val_loss: Validation loss
        metrics: Dictionary of metrics
        config: Configuration dictionary
        save_path: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'hypernetwork_state_dict': hypernetwork.state_dict(),
        'text_encoder_state_dict': text_encoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'metrics': metrics,
        'config': config,
        'model_info': {
            'hypernetwork_params': sum(p.numel() for p in hypernetwork.parameters()),
            'text_encoder_params': sum(p.numel() for p in text_encoder.parameters()),
            'total_params': sum(p.numel() for p in hypernetwork.parameters()) + 
                          sum(p.numel() for p in text_encoder.parameters())
        }
    }
    
    torch.save(checkpoint, save_path)
    logger.info(f"hypernetwork checkpoint saved: {save_path}")

def load_hypernetwork_checkpoint(checkpoint_path: str,
                                hypernetwork: torch.nn.Module,
                                text_encoder: torch.nn.Module,
                                optimizer: Optional[torch.optim.Optimizer] = None,
                                device: str = 'cpu') -> Dict[str, Any]:
    """
    load a complete hypernetwork checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        hypernetwork: Hypernetwork model
        text_encoder: Text encoder model
        optimizer: Optional optimizer
        device: Device to load tensors to
        
    Returns:
        dictionary containing checkpoint metadata
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    #load model states
    hypernetwork.load_state_dict(checkpoint['hypernetwork_state_dict'])
    text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])
    
    #load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    logger.info(f"Hypernetwork checkpoint loaded: {checkpoint_path}")
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'train_loss': checkpoint.get('train_loss', 0.0),
        'val_loss': checkpoint.get('val_loss', 0.0),
        'metrics': checkpoint.get('metrics', {}),
        'config': checkpoint.get('config', {}),
        'model_info': checkpoint.get('model_info', {})
    }

def save_lora_weights(lora_weights: Dict[str, torch.Tensor],
                     task_description: str,
                     save_path: str,
                     metadata: Optional[Dict[str, Any]] = None):
    """
    save LoRA weights for a specific task
    
    Args:
        lora_weights: Dictionary of LoRA weights
        task_description: Task description
        save_path: Path to save weights
        metadata: Optional metadata dictionary
    """
    checkpoint = {
        'lora_weights': lora_weights,
        'task_description': task_description,
        'metadata': metadata or {},
        'weight_info': {
            'total_params': sum(w.numel() for w in lora_weights.values()),
            'layers': list(lora_weights.keys()),
            'shapes': {k: list(v.shape) for k, v in lora_weights.items()}
        }
    }
    
    torch.save(checkpoint, save_path)
    logger.info(f"LoRA weights saved: {save_path}")

def load_lora_weights(checkpoint_path: str,
                     device: str = 'cpu') -> Dict[str, Any]:
    """
    load LoRA weights from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load tensors to
        
    Returns:
        Dictionary containing LoRA weights and metadata
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"LoRA checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    logger.info(f"LoRA weights loaded: {checkpoint_path}")
    
    return checkpoint

def create_config_backup(config: Dict[str, Any], save_dir: str):
    """
    create a backup of the config file
    
    Args:
        config: Configuration dictionary
        save_dir: Directory to save config backup
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = save_dir / "config_backup.json"
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Config backup saved: {config_path}")

def get_model_summary(model: torch.nn.Module) -> Dict[str, Any]:
    """
    get a summary of model parameters and structure
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary containing model summary
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'frozen_parameters': total_params - trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
        'architecture': str(model)
    }