import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import os
import json
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import wandb

from src.training.loss import TaskAwareLoss
from src.training.metrics import SegmentationMetrics
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.utils.visualization import visualize_predictions

class TaskAwareTrainer:
    """trainer with hypernetwork"""
    
    def __init__(
        self,
        model,
        hypernetwork,
        sam_model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: torch.device,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
        use_wandb: bool = False
    ):
        self.model = model
        self.hypernetwork = hypernetwork
        self.sam_model = sam_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.use_wandb = use_wandb
        
        #create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        self.criterion = TaskAwareLoss(config.get('loss', {}))#initialize training components
        self.metrics = SegmentationMetrics()
        
        self.optimizer = self._create_optimizer()  #initialize optimizer
        self.scheduler = self._create_scheduler()
        
        self.writer = SummaryWriter(log_dir)#initialize logging
        self.logger = self._setup_logger()
        
        #taining state
        self.epoch = 0
        self.step = 0
        self.best_miou = 0.0
        self.train_losses = []
        self.val_losses = []
        
        #initialize wandb if enabled
        if use_wandb:
            wandb.init(
                project="task-aware-sam",
                config=config,
                name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
    
    def _create_optimizer(self) -> optim.Optimizer:
        """create optimizer"""
        opt_config = self.config.get('optimizer', {})
        opt_type = opt_config.get('type', 'adamw')
        lr = opt_config.get('lr', 1e-4)
        weight_decay = opt_config.get('weight_decay', 0.01)
        
        #only train hypernetwork parameters
        trainable_params = list(self.hypernetwork.parameters())
        
        if opt_type.lower() == 'adamw':
            return optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        elif opt_type.lower() == 'adam':
            return optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {opt_type}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """create lr scheduler"""
        scheduler_config = self.config.get('scheduler', {})
        if not scheduler_config:
            return None
        
        scheduler_type = scheduler_config.get('type', 'cosine')
        
        if scheduler_type == 'cosine':
            T_max = scheduler_config.get('T_max', len(self.train_loader) * self.config['epochs'])
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max)
        elif scheduler_type == 'step':
            step_size = scheduler_config.get('step_size', 30)
            gamma = scheduler_config.get('gamma', 0.1)
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        
        return None
    
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('TaskAwareTrainer')
        logger.setLevel(logging.INFO)
        
        #create file handler
        handler = logging.FileHandler(os.path.join(self.log_dir, 'training.log'))
        handler.setLevel(logging.INFO)
        
        #create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        return logger
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.hypernetwork.train()
        self.sam_model.eval()  #SAM stays in eval mode
        
        running_loss = 0.0
        running_metrics = {}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            #move batch to device
            batch = self._move_to_device(batch)
            
            #zero gradients
            self.optimizer.zero_grad()
            
            #forward pass
            loss, metrics = self._forward_pass(batch)
            
            #backward pass
            loss.backward()
            
            #gradient clipping
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.hypernetwork.parameters(), 
                    self.config['grad_clip']
                )
            
            #update parameters
            self.optimizer.step()
            
            if self.scheduler:
                self.scheduler.step()
            
            #update running metrics
            running_loss += loss.item()
            for key, value in metrics.items():
                running_metrics[key] = running_metrics.get(key, 0) + value
            
            #log to tensorboard
            if self.step % self.config.get('log_interval', 10) == 0:
                self._log_metrics(loss.item(), metrics, 'train')
            
            #update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
            
            self.step += 1
        
        #calculate epoch metrics
        epoch_loss = running_loss / len(self.train_loader)
        epoch_metrics = {k: v / len(self.train_loader) for k, v in running_metrics.items()}
        
        return {'loss': epoch_loss, **epoch_metrics}
    
    def validate_epoch(self) -> Dict[str, float]:
        """validate for one epoch"""
        self.hypernetwork.eval()
        self.sam_model.eval()
        
        running_loss = 0.0
        running_metrics = {}
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            
            for batch_idx, batch in enumerate(pbar):
                #move batch to device
                batch = self._move_to_device(batch)
                
                #forward pass
                loss, metrics = self._forward_pass(batch)
                
                #update running metrics
                running_loss += loss.item()
                for key, value in metrics.items():
                    running_metrics[key] = running_metrics.get(key, 0) + value
                
                #update progress bar
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        #xalculate epoch metrics
        epoch_loss = running_loss / len(self.val_loader)
        epoch_metrics = {k: v / len(self.val_loader) for k, v in running_metrics.items()}
        
        return {'loss': epoch_loss, **epoch_metrics}
    
    def _forward_pass(self, batch: Dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        """forward pass through the model"""
        images = batch['image']
        task_descriptions = batch['task_description']
        target_masks = batch['mask']
        points = batch['points']
        point_labels = batch['point_labels']
        
        #generate LoRA weights from hypernetwork
        lora_weights = self.hypernetwork(task_descriptions)
        self.model.apply_lora(lora_weights) #apply LoRA to SAM
        
        #get SAM predictions
        predictions = self.model(
            images, 
            points=points, 
            point_labels=point_labels,
            multimask_output=False
        )
        
        predicted_masks = predictions['masks']
        
        #calculate loss
        loss = self.criterion(predicted_masks, target_masks, lora_weights)
        
        #calculate metrics
        metrics = self.metrics(predicted_masks, target_masks)
        
        return loss, metrics
    
    def _move_to_device(self, batch: Dict) -> Dict:
        """move batch to device"""
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)
        return batch
    
    def _log_metrics(self, loss: float, metrics: Dict[str, float], phase: str):
        """log metrics to tensorboard and wandb"""
        #tensorboard
        self.writer.add_scalar(f'{phase}/loss', loss, self.step)
        for key, value in metrics.items():
            self.writer.add_scalar(f'{phase}/{key}', value, self.step)
        
        #wandb
        if self.use_wandb:
            wandb.log({
                f'{phase}/loss': loss,
                **{f'{phase}/{key}': value for key, value in metrics.items()},
                'step': self.step
            })
    
    def train(self, num_epochs: int):
        """main training loop"""
        self.logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            #training phase
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics['loss'])
            
            #validation phase
            val_metrics = self.validate_epoch()
            self.val_losses.append(val_metrics['loss'])
            
            #log epoch metrics
            self.logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val mIoU: {val_metrics.get('miou', 0):.4f}"
            )
            
            is_best = val_metrics.get('miou', 0) > self.best_miou   #save checkpoint
            if is_best:
                self.best_miou = val_metrics.get('miou', 0)
            
            self._save_checkpoint(epoch, is_best)
            
            #visualize predictions
            if epoch % self.config.get('viz_interval', 5) == 0:
                self._visualize_predictions()
        
        self.logger.info("Training completed!")
        
        self.writer.close()
        if self.use_wandb:
            wandb.finish()
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'hypernetwork_state_dict': self.hypernetwork.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_miou': self.best_miou,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        #save latest checkpoint
        save_checkpoint(checkpoint, os.path.join(self.checkpoint_dir, 'latest.pth'))
        
        #save best checkpoint
        if is_best:
            save_checkpoint(checkpoint, os.path.join(self.checkpoint_dir, 'best.pth'))
        
        #save periodic checkpoint
        if epoch % self.config.get('save_interval', 10) == 0:
            save_checkpoint(
                checkpoint, 
                os.path.join(self.checkpoint_dir, f'epoch_{epoch}.pth')
            )
    
    def _visualize_predictions(self):
        """visualize predictions on validation set"""
        self.hypernetwork.eval()
        
        with torch.no_grad():
            #get a batch from validation set
            batch = next(iter(self.val_loader))
            batch = self._move_to_device(batch)
            
            #generate predictions
            images = batch['image']
            task_descriptions = batch['task_description']
            target_masks = batch['mask']
            points = batch['points']
            point_labels = batch['point_labels']
            
            #generate LoRA weights
            lora_weights = self.hypernetwork(task_descriptions)
            self.model.apply_lora(lora_weights)
            
            #get predictions
            predictions = self.model(
                images, 
                points=points, 
                point_labels=point_labels,
                multimask_output=False
            )
            
            #visualize
            viz_path = os.path.join(self.log_dir, f'predictions_epoch_{self.epoch}.png')
            visualize_predictions(
                images, target_masks, predictions['masks'], 
                task_descriptions, viz_path
            )
    
    def load_checkpoint(self, checkpoint_path: str):
        """load checkpoint"""
        checkpoint = load_checkpoint(checkpoint_path)
        
        self.hypernetwork.load_state_dict(checkpoint['hypernetwork_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.best_miou = checkpoint['best_miou']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        self.logger.info(f"Loaded checkpoint from epoch {self.epoch}")
    
    def get_training_stats(self) -> Dict:
        """get training statistics"""
        return {
            'epoch': self.epoch,
            'best_miou': self.best_miou,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'total_steps': self.step
        }