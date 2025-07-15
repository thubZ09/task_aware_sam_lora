import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import cv2
from typing import List, Dict, Optional, Tuple, Any
from PIL import Image
import seaborn as sns
from matplotlib.colors import ListedColormap

def show_mask(mask: np.ndarray, ax: plt.Axes, color: Tuple[float, float, float, float] = (30/255, 144/255, 255/255, 0.6)):
    """display a segmentation mask on matplotlib axes"""
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * np.array(color).reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords: np.ndarray, labels: np.ndarray, ax: plt.Axes, marker_size: int = 375):
    """display point prompts on matplotlib axes"""
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box: np.ndarray, ax: plt.Axes, color: str = 'green'):
    """display bounding box on matplotlib axes"""
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(patches.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0, 0, 0, 0), lw=2))

def visualize_segmentation_results(image: np.ndarray, masks: List[np.ndarray], 
                                 task_description: str, scores: Optional[List[float]] = None,
                                 save_path: Optional[str] = None) -> plt.Figure:
    """
    visualize segmentation results with multiple masks.
    
    Args:
        image: Input image (H, W, 3)
        masks: List of segmentation masks
        task_description: Task description text
        scores: Optional list of confidence scores
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, len(masks) + 1, figsize=(4 * (len(masks) + 1), 4))
    
    if len(masks) == 0:
        axes = [axes]
    
    #priginal image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    #segmentation results
    colors = plt.cm.tab10(np.linspace(0, 1, len(masks)))
    
    for i, mask in enumerate(masks):
        ax = axes[i + 1]
        ax.imshow(image)
        show_mask(mask, ax, color=(*colors[i][:3], 0.6))
        
        title = f"Mask {i+1}"
        if scores is not None:
            title += f" (Score: {scores[i]:.3f})"
        ax.set_title(title)
        ax.axis('off')
    
    #add task description as suptitle
    fig.suptitle(f"Task: {task_description}", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

def compare_segmentation_results(image: np.ndarray, 
                               baseline_masks: List[np.ndarray],
                               lora_masks: List[np.ndarray],
                               task_description: str,
                               baseline_scores: Optional[List[float]] = None,
                               lora_scores: Optional[List[float]] = None,
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    compare baseline SAM vs LoRA-adapted SAM results.
    
    Args:
        image: Input image (H, W, 3)
        baseline_masks: Baseline SAM masks
        lora_masks: LoRA-adapted SAM masks
        task_description: Task description
        baseline_scores: Optional baseline confidence scores
        lora_scores: Optional LoRA confidence scores
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, max(len(baseline_masks), len(lora_masks)) + 1, 
                            figsize=(4 * (max(len(baseline_masks), len(lora_masks)) + 1), 8))
    
    if len(baseline_masks) == 0 and len(lora_masks) == 0:
        axes = axes.reshape(2, 1)
    
    #original images
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(image)
    axes[1, 0].set_title("Original Image")
    axes[1, 0].axis('off')
    
    #baseline results
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(baseline_masks), len(lora_masks))))
    
    for i, mask in enumerate(baseline_masks):
        ax = axes[0, i + 1]
        ax.imshow(image)
        show_mask(mask, ax, color=(*colors[i][:3], 0.6))
        
        title = f"Baseline {i+1}"
        if baseline_scores is not None:
            title += f" ({baseline_scores[i]:.3f})"
        ax.set_title(title)
        ax.axis('off')
    
    #LoRA results
    for i, mask in enumerate(lora_masks):
        ax = axes[1, i + 1]
        ax.imshow(image)
        show_mask(mask, ax, color=(*colors[i][:3], 0.6))
        
        title = f"LoRA {i+1}"
        if lora_scores is not None:
            title += f" ({lora_scores[i]:.3f})"
        ax.set_title(title)
        ax.axis('off')
    
    #labels
    axes[0, 0].set_ylabel("Baseline SAM", fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel("LoRA SAM", fontsize=14, fontweight='bold')
    
    fig.suptitle(f"Task: {task_description}", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

def plot_training_curves(train_losses: List[float], val_losses: List[float],
                        train_metrics: Dict[str, List[float]], 
                        val_metrics: Dict[str, List[float]],
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    plot training and validation curves
    
    Args:
        train_losses: Training loss values
        val_losses: Validation loss values
        train_metrics: Dictionary of training metrics
        val_metrics: Dictionary of validation metrics
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib Figure object
    """
    num_metrics = len(train_metrics) + 1  # +1 for loss
    fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 4))
    
    if num_metrics == 1:
        axes = [axes]
    
    #plot loss
    axes[0].plot(train_losses, label='Train Loss', color='blue')
    axes[0].plot(val_losses, label='Val Loss', color='red')
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    #plot metrics
    for i, (metric_name, train_values) in enumerate(train_metrics.items()):
        ax = axes[i + 1]
        val_values = val_metrics.get(metric_name, [])
        
        ax.plot(train_values, label=f'Train {metric_name}', color='blue')
        if val_values:
            ax.plot(val_values, label=f'Val {metric_name}', color='red')
        
        ax.set_title(f'{metric_name}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

def plot_miou_improvement(baseline_miou: Dict[str, float], 
                         lora_miou: Dict[str, float],
                         task_descriptions: Dict[str, str],
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    plot mIoU improvement for different tasks.
    
    Args:
        baseline_miou: Baseline mIoU scores per class
        lora_miou: LoRA mIoU scores per class
        task_descriptions: Task descriptions per class
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib Figure object
    """
    classes = list(baseline_miou.keys())
    baseline_scores = [baseline_miou[c] for c in classes]
    lora_scores = [lora_miou[c] for c in classes]
    improvements = [lora_scores[i] - baseline_scores[i] for i in range(len(classes))]
    
    #sort by improvement
    sorted_indices = np.argsort(improvements)[::-1]
    classes = [classes[i] for i in sorted_indices]
    baseline_scores = [baseline_scores[i] for i in sorted_indices]
    lora_scores = [lora_scores[i] for i in sorted_indices]
    improvements = [improvements[i] for i in sorted_indices]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    #mIoU comparison
    x = np.arange(len(classes))
    width = 0.35
    
    ax1.bar(x - width/2, baseline_scores, width, label='Baseline SAM', color='skyblue')
    ax1.bar(x + width/2, lora_scores, width, label='LoRA SAM', color='lightcoral')
    
    ax1.set_xlabel('Classes')
    ax1.set_ylabel('mIoU')
    ax1.set_title('mIoU Comparison: Baseline vs LoRA')
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    #improvement bars
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    ax2.bar(x, improvements, color=colors, alpha=0.7)
    ax2.set_xlabel('Classes')
    ax2.set_ylabel('mIoU Improvement')
    ax2.set_title('mIoU Improvement (LoRA - Baseline)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes, rotation=45, ha='right')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

def create_attention_heatmap(attention_weights: torch.Tensor, 
                           text_tokens: List[str],
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    create attention heatmap for text tokens.
    
    Args:
        attention_weights: Attention weights tensor (seq_len, seq_len)
        text_tokens: List of text tokens
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    #convert to numpy if tensor
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    #create heatmap
    sns.heatmap(attention_weights, 
                xticklabels=text_tokens, 
                yticklabels=text_tokens,
                annot=True, 
                fmt='.3f',
                cmap='Blues',
                ax=ax)
    
    ax.set_title('Attention Weights Heatmap')
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Token Position')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

def visualize_lora_weights(lora_weights: Dict[str, torch.Tensor],
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    visualize LoRA weight distributions.
    
    Args:
        lora_weights: Dictionary of LoRA weights
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib Figure object
    """
    num_layers = len(lora_weights)
    fig, axes = plt.subplots(2, (num_layers + 1) // 2, figsize=(4 * ((num_layers + 1) // 2), 8))
    
    if num_layers == 1:
        axes = [axes]
    elif num_layers == 2:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i, (layer_name, weights) in enumerate(lora_weights.items()):
        ax = axes[i] if num_layers > 1 else axes
        
        #convert to numpy
        if isinstance(weights, torch.Tensor):
            weights = weights.detach().cpu().numpy()
        
        #flatten weights
        weights_flat = weights.flatten()
        
        #plot histogram
        ax.hist(weights_flat, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_title(f'{layer_name} Weight Distribution')
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        #add statistics
        mean_val = np.mean(weights_flat)
        std_val = np.std(weights_flat)
        ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.4f}')
        ax.axvline(mean_val + std_val, color='orange', linestyle='--', alpha=0.7, label=f'±1σ: {std_val:.4f}')
        ax.axvline(mean_val - std_val, color='orange', linestyle='--', alpha=0.7)
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig