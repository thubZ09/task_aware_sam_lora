import os
import argparse
import json
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import logging
from typing import Dict, List, Tuple

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hypernetwork import TaskAwareHypernetwork
from models.sam_lora import SAMLoRA
from data.dataset import TaskAwareDataset
from training.metrics import MetricsCalculator, calculate_mask_metrics
from utils.coco_utils import load_coco_categories

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Task-Aware SAM LoRA')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained hypernetwork checkpoint')
    parser.add_argument('--sam_checkpoint', type=str, 
                        default='sam_vit_h_4b8939.pth',
                        help='Path to SAM checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to COCO dataset directory')
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate on')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--output_dir', type=str, default='eval_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for evaluation')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to evaluate (for debugging)')
    parser.add_argument('--task_prompts', type=str, nargs='+',
                        default=['segment all round red fruit', 'segment all cars', 'segment all people'],
                        help='Task prompts to evaluate')
    parser.add_argument('--baseline', action='store_true',
                        help='Evaluate baseline SAM without LoRA')
    
    return parser.parse_args()

def load_model(checkpoint_path: str, sam_checkpoint: str, device: str) -> Tuple[SAMLoRA, TaskAwareHypernetwork]:
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

def evaluate_on_dataset(sam_lora: SAMLoRA, hypernetwork: TaskAwareHypernetwork, 
                       dataset: TaskAwareDataset, task_prompt: str, 
                       device: str, max_samples: int = None) -> Dict[str, float]:
    """evaluate model on dataset for a specific task prompt"""
    
    sam_lora.eval()
    hypernetwork.eval()
    
    metrics_calc = MetricsCalculator()
    all_metrics = []
    
    #apply LoRA for this task
    lora_weights = hypernetwork(task_prompt)
    sam_lora.apply_lora(lora_weights)
    
    logger.info(f"Evaluating on task: '{task_prompt}'")
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataset)):
            if max_samples and i >= max_samples:
                break
                
            try:
                #prepare inputs
                image = batch['image'].to(device)
                mask = batch['mask'].to(device)
                prompt_box = batch.get('prompt_box', None)
                
                #get SAM prediction
                if prompt_box is not None:
                    prompt_box = prompt_box.to(device)
                    predicted_mask = sam_lora.predict_mask(image, prompt_box)
                else:
                    #use point prompts or automatic mask generation
                    predicted_mask = sam_lora.predict_mask(image)
                
                #calculate metrics
                batch_metrics = calculate_mask_metrics(predicted_mask, mask)
                all_metrics.append(batch_metrics)
                
                #update accumulated metrics
                metrics_calc.update(predicted_mask, mask)
                
            except Exception as e:
                logger.warning(f"Error processing batch {i}: {str(e)}")
                continue
    
    #compute final metrics
    final_metrics = metrics_calc.get_metrics_summary()
    
    #add per-batch statistics
    if all_metrics:
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            final_metrics[f'{key}_std'] = np.std(values)
            final_metrics[f'{key}_min'] = np.min(values)
            final_metrics[f'{key}_max'] = np.max(values)
    
    return final_metrics

def evaluate_baseline(sam_lora: SAMLoRA, dataset: TaskAwareDataset, 
                     device: str, max_samples: int = None) -> Dict[str, float]:
    """evaluate baseline SAM without LoRA"""
    
    sam_lora.eval()
    sam_lora.remove_lora()  #remove any LoRA adapters
    
    metrics_calc = MetricsCalculator()
    all_metrics = []
    
    logger.info("evaluating baseline SAM (no LoRA)")
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataset)):
            if max_samples and i >= max_samples:
                break
                
            try:
                #prepare inputs
                image = batch['image'].to(device)
                mask = batch['mask'].to(device)
                prompt_box = batch.get('prompt_box', None)
                
                #get SAM prediction
                if prompt_box is not None:
                    prompt_box = prompt_box.to(device)
                    predicted_mask = sam_lora.predict_mask(image, prompt_box)
                else:
                    predicted_mask = sam_lora.predict_mask(image)
                
                #calculate metrics
                batch_metrics = calculate_mask_metrics(predicted_mask, mask)
                all_metrics.append(batch_metrics)
                
                #update accumulated metrics
                metrics_calc.update(predicted_mask, mask)
                
            except Exception as e:
                logger.warning(f"Error processing batch {i}: {str(e)}")
                continue
    
    #compute final metrics
    final_metrics = metrics_calc.get_metrics_summary()
    
    #add per-batch statistics
    if all_metrics:
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            final_metrics[f'{key}_std'] = np.std(values)
            final_metrics[f'{key}_min'] = np.min(values)
            final_metrics[f'{key}_max'] = np.max(values)
    
    return final_metrics

def main():
    args = parse_args()
    
    #create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    #load models
    if not args.baseline:
        sam_lora, hypernetwork = load_model(args.checkpoint, args.sam_checkpoint, args.device)
    else:
        sam_lora = SAMLoRA(sam_checkpoint_path=args.sam_checkpoint, device=args.device)
        hypernetwork = None
    
    #load dataset
    logger.info(f"Loading dataset from {args.data_dir}")
    dataset = TaskAwareDataset(
        data_dir=args.data_dir,
        split=args.split,
        transform=None 
    )
    
    #eval results
    results = {}
    
    if args.baseline:
        #evaluate baseline
        baseline_metrics = evaluate_baseline(sam_lora, dataset, args.device, args.max_samples)
        results['baseline'] = baseline_metrics
        
        logger.info("Baseline Results:")
        for key, value in baseline_metrics.items():
            logger.info(f"  {key}: {value:.4f}")
    
    else:
        #evaluate each task prompt
        for task_prompt in args.task_prompts:
            task_metrics = evaluate_on_dataset(
                sam_lora, hypernetwork, dataset, task_prompt, 
                args.device, args.max_samples
            )
            results[task_prompt] = task_metrics
            
            logger.info(f"Results for '{task_prompt}':")
            for key, value in task_metrics.items():
                logger.info(f"  {key}: {value:.4f}")
    
    results_file = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    
    if not args.baseline:
        logger.info("\n=== SUMMARY ===")
        for task_prompt in args.task_prompts:
            miou = results[task_prompt]['mIoU']
            dice = results[task_prompt]['mDice']
            logger.info(f"Task: '{task_prompt}' - mIoU: {miou:.4f}, mDice: {dice:.4f}")

if __name__ == '__main__':
    main()