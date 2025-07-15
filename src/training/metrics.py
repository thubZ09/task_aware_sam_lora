import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class SegmentationMetrics:
    """calculate segmentation metrics with LoRA adapters"""
    
    def __init__(self, num_classes: int = 80, ignore_index: int = -1):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """reset all accumulated metrics"""
        self.intersection = defaultdict(int)
        self.union = defaultdict(int)
        self.total_pixels = defaultdict(int)
        self.true_positives = defaultdict(int)
        self.false_positives = defaultdict(int)
        self.false_negatives = defaultdict(int)
        self.num_samples = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, 
               class_ids: Optional[List[int]] = None):
        """
        update metrics with batch predictions and targets
        
        Args:
            predictions: (B, H, W) predicted masks
            targets: (B, H, W) ground truth masks
            class_ids: List of class IDs for each sample in batch
        """
        batch_size = predictions.shape[0]
        self.num_samples += batch_size
        
        #cnvert to numpy for easier processing
        preds = predictions.detach().cpu().numpy()
        targs = targets.detach().cpu().numpy()
        
        for i in range(batch_size):
            pred = preds[i]
            targ = targs[i]
            
            #handle class-specific metrics
            class_id = class_ids[i] if class_ids else 0
            
            #binarize predictions (assuming threshold of 0.5)
            pred_binary = (pred > 0.5).astype(np.uint8)
            targ_binary = (targ > 0.5).astype(np.uint8)
            
            #calculate intersection and union for IoU
            intersection = np.logical_and(pred_binary, targ_binary).sum()
            union = np.logical_or(pred_binary, targ_binary).sum()
            
            self.intersection[class_id] += intersection
            self.union[class_id] += union
            
            #calculate precision/recall components
            tp = intersection
            fp = (pred_binary & ~targ_binary).sum()
            fn = (targ_binary & ~pred_binary).sum()
            
            self.true_positives[class_id] += tp
            self.false_positives[class_id] += fp
            self.false_negatives[class_id] += fn
    
    def compute_iou(self, class_id: int = None) -> float:
        """compute IoU for specific class or mean IoU"""
        if class_id is not None:
            if self.union[class_id] == 0:
                return 0.0
            return self.intersection[class_id] / self.union[class_id]
        
        #mean IoU across all classes
        ious = []
        for cid in self.intersection.keys():
            if self.union[cid] > 0:
                ious.append(self.intersection[cid] / self.union[cid])
        
        return np.mean(ious) if ious else 0.0
    
    def compute_dice(self, class_id: int = None) -> float:
        """Compute Dice score for specific class or mean Dice"""
        if class_id is not None:
            tp = self.true_positives[class_id]
            fp = self.false_positives[class_id]
            fn = self.false_negatives[class_id]
            
            if tp + fp + fn == 0:
                return 0.0
            return 2 * tp / (2 * tp + fp + fn)
        
        #mean Dice across all classes
        dice_scores = []
        for cid in self.true_positives.keys():
            tp = self.true_positives[cid]
            fp = self.false_positives[cid]
            fn = self.false_negatives[cid]
            
            if tp + fp + fn > 0:
                dice_scores.append(2 * tp / (2 * tp + fp + fn))
        
        return np.mean(dice_scores) if dice_scores else 0.0
    
    def compute_precision(self, class_id: int = None) -> float:
        """compute precision for specific class or mean precision"""
        if class_id is not None:
            tp = self.true_positives[class_id]
            fp = self.false_positives[class_id]
            
            if tp + fp == 0:
                return 0.0
            return tp / (tp + fp)
        
        #mean precision across all classes
        precisions = []
        for cid in self.true_positives.keys():
            tp = self.true_positives[cid]
            fp = self.false_positives[cid]
            
            if tp + fp > 0:
                precisions.append(tp / (tp + fp))
        
        return np.mean(precisions) if precisions else 0.0
    
    def compute_recall(self, class_id: int = None) -> float:
        """compute recall for specific class or mean recall"""
        if class_id is not None:
            tp = self.true_positives[class_id]
            fn = self.false_negatives[class_id]
            
            if tp + fn == 0:
                return 0.0
            return tp / (tp + fn)
        
        #mean recall across all classes
        recalls = []
        for cid in self.true_positives.keys():
            tp = self.true_positives[cid]
            fn = self.false_negatives[cid]
            
            if tp + fn > 0:
                recalls.append(tp / (tp + fn))
        
        return np.mean(recalls) if recalls else 0.0
    
    def compute_f1(self, class_id: int = None) -> float:
        """compute F1 score for specific class or mean F1"""
        precision = self.compute_precision(class_id)
        recall = self.compute_recall(class_id)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def get_metrics_summary(self) -> Dict[str, float]:
        """get summary of all computed metrics"""
        return {
            'mIoU': self.compute_iou(),
            'mDice': self.compute_dice(),
            'mPrecision': self.compute_precision(),
            'mRecall': self.compute_recall(),
            'mF1': self.compute_f1(),
            'num_samples': self.num_samples
        }
    
    def get_class_metrics(self, class_id: int) -> Dict[str, float]:
        """get metrics for specific class"""
        return {
            'IoU': self.compute_iou(class_id),
            'Dice': self.compute_dice(class_id),
            'Precision': self.compute_precision(class_id),
            'Recall': self.compute_recall(class_id),
            'F1': self.compute_f1(class_id)
        }

def calculate_mask_metrics(pred_masks: torch.Tensor, 
                          gt_masks: torch.Tensor,
                          threshold: float = 0.5) -> Dict[str, float]:
    """
    calculate segmentation metrics for a batch of masks
    
    Args:
        pred_masks: (B, H, W) predicted masks
        gt_masks: (B, H, W) ground truth masks
        threshold: Threshold for binarizing predictions
    
    Returns:
        dictionary of metrics
    """
    pred_binary = (pred_masks > threshold).float()
    gt_binary = (gt_masks > threshold).float()
    
    #calculate intersection and union
    intersection = (pred_binary * gt_binary).sum(dim=(1, 2))
    union = (pred_binary + gt_binary - pred_binary * gt_binary).sum(dim=(1, 2))
    
    #calculate IoU nd Dice
    iou = intersection / (union + 1e-8)    
    dice = 2 * intersection / (pred_binary.sum(dim=(1, 2)) + gt_binary.sum(dim=(1, 2)) + 1e-8)
    
    #calculate precision and recall
    tp = intersection
    fp = (pred_binary * (1 - gt_binary)).sum(dim=(1, 2))
    fn = (gt_binary * (1 - pred_binary)).sum(dim=(1, 2))
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        'IoU': iou.mean().item(),
        'Dice': dice.mean().item(),
        'Precision': precision.mean().item(),
        'Recall': recall.mean().item(),
        'F1': f1.mean().item()
    }

class SegmentationLoss:
    """combined loss functions for segmentation training"""
    
    def __init__(self, focal_alpha: float = 0.25, focal_gamma: float = 2.0,
                 dice_weight: float = 1.0, focal_weight: float = 1.0):
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
    
    def focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """focal loss for handling class imbalance"""
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss
        return focal_loss.mean()
    
    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """dice loss for segmentation"""
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(1, 2))
        dice = (2 * intersection) / (pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) + 1e-8)
        return 1 - dice.mean()
    
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """combined loss function"""
        focal = self.focal_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.focal_weight * focal + self.dice_weight * dice