import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from pathlib import Path
import logging

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hypernetwork import TaskAwareHypernetwork
from models.sam_lora import SAMLoRA
from utils.visualization import visualize_segmentation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Demo Task-Aware SAM LoRA')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained hypernetwork checkpoint')
    parser.add_argument('--sam_checkpoint', type=str, 
                        default='sam_vit_h_4b8939.pth',
                        help='Path to SAM checkpoint')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--task_prompt', type=str, required=True,
                        help='Task description (e.g., "segment all round red fruit")')
    parser.add_argument('--output_dir', type=str, default='demo_results',
                        help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--compare_baseline', action='store_true',
                        help='Compare with baseline SAM')
    parser.add_argument('--point_prompt', type=str, default=None,
                        help='Point prompt as "x,y" coordinates')
    parser.add_argument('--box_prompt', type=str, default=None,
                        help='Box prompt as "x1,y1,x2,y2" coordinates')
    parser.add_argument('--save_masks', action='store_true',
                        help='Save individual mask files')
    
    return parser.parse_args()

def load_model(checkpoint_path: str, sam_checkpoint: str, device: str) -> tuple:
    """load SAM LoRA model and hypernetwork"""
    logger.info(f"Loading SAM from {sam_checkpoint}")
    sam_lora = SAMLoRA(sam_checkpoint_path=sam_checkpoint, device=device)
    
    logger.info(f"Loading hypernetwork from {checkpoint_path}")
    hypernetwork = TaskAwareHypernetwork(
        text_dim=512,
        lora_rank=16,
        lora_target_modules=['q_proj', 'v_proj', 'out_proj']
    ).to(device)
    
    #load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    hypernetwork.load_state_dict(checkpoint['hypernetwork_state_dict'])
    
    return sam_lora, hypernetwork

def load_image(image_path: str) -> tuple:
    """load and preprocess image"""
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    
    #convert to tensor format expected by SAM
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
    
    return image, image_np, image_tensor

def parse_prompts(point_prompt: str = None, box_prompt: str = None) -> tuple:
    """parse point and box prompts from string format"""
    points = None
    boxes = None
    
    if point_prompt:
        try:
            coords = [float(x) for x in point_prompt.split(',')]
            if len(coords) == 2:
                points = np.array([[coords[0], coords[1]]])
            else:
                logger.warning("Point prompt should be in format 'x,y'")
        except ValueError:
            logger.warning("Invalid point prompt format")
    
    if box_prompt:
        try:
            coords = [float(x) for x in box_prompt.split(',')]
            if len(coords) == 4:
                boxes = np.array([[coords[0], coords[1], coords[2], coords[3]]])
            else:
                logger.warning("Box prompt should be in format 'x1,y1,x2,y2'")
        except ValueError:
            logger.warning("Invalid box prompt format")
    
    return points, boxes

def segment_with_lora(sam_lora: SAMLoRA, hypernetwork: TaskAwareHypernetwork,
                     image_tensor: torch.Tensor, task_prompt: str,
                     points: np.ndarray = None, boxes: np.ndarray = None) -> np.ndarray:
    """perform segmentation with LoRA adaptation"""
    
    #generate LoRA weights for the task
    logger.info(f"generating LoRA weights for task: '{task_prompt}'")
    lora_weights = hypernetwork(task_prompt)
    
    #apply LoRA to SAM
    sam_lora.apply_lora(lora_weights)
    
    #perform segmentation
    with torch.no_grad():
        if points is not None or boxes is not None:
            masks = sam_lora.predict_mask(image_tensor, point_prompts=points, box_prompts=boxes)
        else:
            #use automatic mask generation
            masks = sam_lora.predict_mask(image_tensor)
    
    return masks

def segment_baseline(sam_lora: SAMLoRA, image_tensor: torch.Tensor,
                    points: np.ndarray = None, boxes: np.ndarray = None) -> np.ndarray:
    """perform segmentation with baseline SAM"""
    
    #remove LoRA adaptation
    sam_lora.remove_lora()
    
    #perform segmentation
    with torch.no_grad():
        if points is not None or boxes is not None:
            masks = sam_lora.predict_mask(image_tensor, point_prompts=points, box_prompts=boxes)
        else:
            #ue automatic mask generation
            masks = sam_lora.predict_mask(image_tensor)
    
    return masks

def visualize_results(image_np: np.ndarray, masks_lora: np.ndarray, 
                     masks_baseline: np.ndarray = None, task_prompt: str = "",
                     output_dir: str = "demo_results", save_masks: bool = False):
    """visualize segmentation results"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    if masks_baseline is not None:
        #create comparison visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        #original image
        axes[0].imshow(image_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        #loRA result
        visualize_segmentation(image_np, masks_lora, ax=axes[1])
        axes[1].set_title(f'Task-Aware LoRA\n"{task_prompt}"')
        axes[1].axis('off')
        
        #baseline result
        visualize_segmentation(image_np, masks_baseline, ax=axes[2])
        axes[2].set_title('Baseline SAM')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'comparison.png'), dpi=150, bbox_inches='tight')
        plt.show()
        
    else:
        #single result visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        #original image
        axes[0].imshow(image_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        #LoRA result
        visualize_segmentation(image_np, masks_lora, ax=axes[1])
        axes[1].set_title(f'Task-Aware LoRA\n"{task_prompt}"')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'result.png'), dpi=150, bbox_inches='tight')
        plt.show()
    
    #save individual masks if requested
    if save_masks:
        mask_dir = os.path.join(output_dir, 'masks')
        os.makedirs(mask_dir, exist_ok=True)
        
        #save LoRA masks
        if masks_lora is not None:
            for i, mask in enumerate(masks_lora):
                mask_uint8 = (mask * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(mask_dir, f'lora_mask_{i}.png'), mask_uint8)
        
        #save baseline masks
        if masks_baseline is not None:
            for i, mask in enumerate(masks_baseline):
                mask_uint8 = (mask * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(mask_dir, f'baseline_mask_{i}.png'), mask_uint8)

def main():
    args = parse_args()
    
    #create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    #load models
    sam_lora, hypernetwork = load_model(args.checkpoint, args.sam_checkpoint, args.device)
    
    #load image
    logger.info(f"Loading image from {args.image}")
    image_pil, image_np, image_tensor = load_image(args.image)
    image_tensor = image_tensor.unsqueeze(0).to(args.device)  # Add batch dimension
    
    #parse prompts
    points, boxes = parse_prompts(args.point_prompt, args.box_prompt)
    
    #segment with LoRA
    logger.info("Performing segmentation with Task-Aware LoRA...")
    masks_lora = segment_with_lora(
        sam_lora, hypernetwork, image_tensor, args.task_prompt, points, boxes
    )
    
    #convert to numpy for visualization
    masks_lora_np = masks_lora.cpu().numpy().squeeze()
    
    #segment with baseline if requested
    masks_baseline_np = None
    if args.compare_baseline:
        logger.info("Performing segmentation with baseline SAM...")
        masks_baseline = segment_baseline(sam_lora, image_tensor, points, boxes)
        masks_baseline_np = masks_baseline.cpu().numpy().squeeze()
    
    #visualize results
    logger.info("Visualizing results...")
    visualize_results(
        image_np, masks_lora_np, masks_baseline_np, 
        args.task_prompt, args.output_dir, args.save_masks
    )
    
    #print summary
    logger.info("\n=== RESULTS SUMMARY ===")
    logger.info(f"Task: {args.task_prompt}")
    logger.info(f"Image: {args.image}")
    logger.info(f"Output directory: {args.output_dir}")
    
    if masks_lora_np.ndim == 3:
        logger.info(f"Generated {masks_lora_np.shape[0]} masks")
    else:
        logger.info("Generated 1 mask")
    
    logger.info("Demo completed successfully!")

if __name__ == '__main__':
    main()