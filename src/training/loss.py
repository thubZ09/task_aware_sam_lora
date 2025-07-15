import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

class TaskAwareLoss(nn.Module):
    """combined loss function"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        #weights
        self.seg_weight = config.get('seg_weight', 1.0)
        self.reg_weight = config.get('reg_weight', 0.01)
        self.consistency_weight = config.get('consistency_weight', 0.1)
        
        #loss functions
        self.seg_loss = SegmentationLoss(config.get('seg_loss', {}))
        self.reg_loss = RegularizationLoss(config.get('reg_loss', {}))
        self.consistency_loss = ConsistencyLoss(config.get('consistency_loss', {}))
    
    def forward(
        self, 
        pred_masks: torch.Tensor, 
        target_masks: torch.Tensor, 
        lora_weights: Optional[Dict] = None
    ) -> torch.Tensor:
        """
        forward pass of the loss function
        
        Args:
            pred_masks: Predicted masks [B, 1, H, W]
            target_masks: Ground truth masks [B, H, W]
            lora_weights: LoRA weights dictionary
        """
        #gegmentation loss
        seg_loss = self.seg_loss(pred_masks, target_masks)
        
        #regularization loss on LoRA weights
        reg_loss = torch.tensor(0.0, device=pred_masks.device)
        if lora_weights is not None:
            reg_loss = self.reg_loss(lora_weights)
        
        consistency_loss = torch.tensor(0.0, device=pred_masks.device)
        if self.consistency_weight > 0:
            consistency_loss = self.consistency_loss(pred_masks, target_masks)
        
        #combined loss
        total_loss = (
            self.seg_weight * seg_loss +
            self.reg_weight * reg_loss +
            self.consistency_weight * consistency_loss
        )
        
        return total_loss


class SegmentationLoss(nn.Module):
    """segmentation loss combining multiple loss functions"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        self.use_bce = config.get('use_bce', True)
        self.use_dice = config.get('use_dice', True)
        self.use_focal = config.get('use_focal', False)
        
        self.bce_weight = config.get('bce_weight', 1.0)
        self.dice_weight = config.get('dice_weight', 1.0)
        self.focal_weight = config.get('focal_weight', 1.0)
        
        #loss functions
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.focal_loss = FocalLoss(config.get('focal_alpha', 0.25), config.get('focal_gamma', 2.0))
    
    def forward(self, pred_masks: torch.Tensor, target_masks: torch.Tensor) -> torch.Tensor:
        """
        forward pass of segmentation loss
        
        Args:
            pred_masks: Predicted masks [B, 1, H, W]
            target_masks: Ground truth masks [B, H, W]
        """
        #ensure target masks have the same shape as predictions
        if target_masks.dim() == 3:
            target_masks = target_masks.unsqueeze(1)
        
        #resize target masks to match prediction size if needed
        if pred_masks.shape[-2:] != target_masks.shape[-2:]:
            target_masks = F.interpolate(
                target_masks, 
                size=pred_masks.shape[-2:], 
                mode='nearest'
            )
        
        total_loss = 0.0
        
        #binary cross entropy loss
        if self.use_bce:
            bce_loss = self.bce_loss(pred_masks, target_masks.float())
            total_loss += self.bce_weight * bce_loss
        
        #dice loss
        if self.use_dice:
            dice_loss = self.dice_loss(pred_masks, target_masks)
            total_loss += self.dice_weight * dice_loss
        
        #focal loss
        if self.use_focal:
            focal_loss = self.focal_loss(pred_masks, target_masks)
            total_loss += self.focal_weight * focal_loss
        
        return total_loss
    
    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
        """dice loss for segmentation"""
        pred = torch.sigmoid(pred)
        
        #flatten tensors
        pred = pred.view(-1)
        target = target.view(-1)
        
        #calculate intersection and union
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        
        #calculate dice coefficient
        dice = (2.0 * intersection + smooth) / (union + smooth)
        
        return 1.0 - dice


class FocalLoss(nn.Module):
    """focal loss for addressing class imbalance"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        forward pass of focal loss.
        
        Args:
            pred: Predicted logits [B, 1, H, W]
            target: Ground truth masks [B, 1, H, W]
        """
        #calculate BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(pred, target.float(), reduction='none')
        
        #calculate probabilities
        pt = torch.exp(-bce_loss)
        
        #calculate focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        return focal_loss.mean()


class RegularizationLoss(nn.Module):
    """regularization loss for LoRA weights"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        #regularization types
        self.use_l1 = config.get('use_l1', True)
        self.use_l2 = config.get('use_l2', False)
        self.use_orthogonal = config.get('use_orthogonal', False)
        
        #regularization weights
        self.l1_weight = config.get('l1_weight', 0.01)
        self.l2_weight = config.get('l2_weight', 0.001)
        self.orthogonal_weight = config.get('orthogonal_weight', 0.01)
    
    def forward(self, lora_weights: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        forward pass of regularization loss
        
        Args:
            lora_weights: Dictionary of LoRA weights
        """
        total_loss = 0.0
        
        for name, weight in lora_weights.items():
            if weight is None:
                continue
            
            #l1 regularization
            if self.use_l1:
                l1_loss = torch.norm(weight, p=1)
                total_loss += self.l1_weight * l1_loss
            
            #l2 regularization
            if self.use_l2:
                l2_loss = torch.norm(weight, p=2)
                total_loss += self.l2_weight * l2_loss
            
            #orthogonal regularization
            if self.use_orthogonal and weight.dim() == 2:
                orthogonal_loss = self.orthogonal_regularization(weight)
                total_loss += self.orthogonal_weight * orthogonal_loss
        
        return total_loss
    
    def orthogonal_regularization(self, weight: torch.Tensor) -> torch.Tensor:
        """orthogonal regularization for matrix weights"""
        wtw = torch.matmul(weight.t(), weight)
        
        # Calculate identity matrix nd orthogonal loss
        identity = torch.eye(wtw.size(0), device=weight.device)
        orthogonal_loss = torch.norm(wtw - identity, p='fro')
        
        return orthogonal_loss


class ConsistencyLoss(nn.Module):
    """consistency loss for stable training"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.temperature = config.get('temperature', 1.0)
    
    def forward(self, pred_masks: torch.Tensor, target_masks: torch.Tensor) -> torch.Tensor:
        """
        forward pass of consistency loss.
        
        Args:
            pred_masks: Predicted masks [B, 1, H, W]
            target_masks: Ground truth masks [B, H, W]
        """
        #simple consistency loss based on prediction smoothness
        pred_grad_x = torch.abs(pred_masks[:, :, :, 1:] - pred_masks[:, :, :, :-1])
        pred_grad_y = torch.abs(pred_masks[:, :, 1:, :] - pred_masks[:, :, :-1, :])
        
        consistency_loss = pred_grad_x.mean() + pred_grad_y.mean()
        
        return consistency_loss


class IoULoss(nn.Module):
    """IoU loss for segmentation"""
    
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        forward pass of IoU loss.
        
        Args:
            pred: Predicted masks [B, 1, H, W]
            target: Ground truth masks [B, 1, H, W]
        """
        pred = torch.sigmoid(pred)
        
        #flatten tensors
        pred = pred.view(-1)
        target = target.view(-1)
        
        #calculate intersection and union
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        
        #calculate IoU
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        return 1.0 - iou


class TverskyLoss(nn.Module):
    """tversky loss for segmentation with class imbalance"""
    
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1e-5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        forward pass of Tversky loss.
        
        Args:
            pred: Predicted masks [B, 1, H, W]
            target: Ground truth masks [B, 1, H, W]
        """
        pred = torch.sigmoid(pred)
        
        #flatten tensors
        pred = pred.view(-1)
        target = target.view(-1)
        
        #calculate true positives, false positives, and false negatives
        tp = (pred * target).sum()
        fp = ((1 - target) * pred).sum()
        fn = (target * (1 - pred)).sum()
        
        #calculate tversky index
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        return 1.0 - tversky


def get_loss_function(config: Dict) -> nn.Module:
    """factory function to get loss function based on config"""
    loss_type = config.get('type', 'task_aware')
    
    if loss_type == 'task_aware':
        return TaskAwareLoss(config)
    elif loss_type == 'segmentation':
        return SegmentationLoss(config)
    elif loss_type == 'bce':
        return nn.BCEWithLogitsLoss()
    elif loss_type == 'focal':
        return FocalLoss(config.get('alpha', 0.25), config.get('gamma', 2.0))
    elif loss_type == 'iou':
        return IoULoss(config.get('smooth', 1e-5))
    elif loss_type == 'tversky':
        return TverskyLoss(
            config.get('alpha', 0.7), 
            config.get('beta', 0.3), 
            config.get('smooth', 1e-5)
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")