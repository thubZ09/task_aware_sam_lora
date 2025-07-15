import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import json
import os
from typing import Dict, List, Tuple, Optional
import cv2
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
import random

class TaskAwareDataset(Dataset):
    """dataset with textual descriptions"""
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        max_samples: int = None,
        transform=None,
        task_templates: Optional[List[str]] = None
    ):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.max_samples = max_samples
        
        #load annotations
        ann_file = os.path.join(data_dir, f"annotations/panoptic_{split}2017.json")
        self.coco = COCO(ann_file)
        
        #get image IDs
        self.image_ids = list(self.coco.imgs.keys())
        if max_samples:
            self.image_ids = self.image_ids[:max_samples]
        
        #load category info
        self.categories = self.coco.loadCats(self.coco.getCatIds())
        self.cat_id_to_name = {cat['id']: cat['name'] for cat in self.categories}
        
        #task description templates
        self.task_templates = task_templates or [
            "segment all {}",
            "find all {} in the image",
            "identify and segment {}",
            "locate all {} objects",
            "segment every {}",
            "extract all {} from the image",
            "highlight all {} regions",
            "detect and segment {}"
        ]
        
        #color nd shape descriptors for more complex tasks
        self.color_descriptors = [
            "red", "blue", "green", "yellow", "orange", "purple", "pink", 
            "brown", "black", "white", "gray", "silver"
        ]
        
        self.shape_descriptors = [
            "round", "square", "rectangular", "oval", "triangular", "circular",
            "long", "thin", "wide", "tall", "small", "large"
        ]
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        #load image
        img_info = self.coco.imgs[image_id]
        img_path = os.path.join(self.data_dir, f"{self.split}2017", img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        
        #get annotations for this image
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(ann_ids)
        
        #select a random annotation to create task for
        if annotations:
            selected_ann = random.choice(annotations)
            category_name = self.cat_id_to_name[selected_ann['category_id']]
            
            #generate task description
            task_description = self._generate_task_description(category_name)
            
            #create mask for the selected category
            mask = self._create_category_mask(annotations, selected_ann['category_id'], img_info)
            points, point_labels = self._generate_point_prompts(mask)
            
        else:
            #fallback for images without annotations
            task_description = "segment any object"
            mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
            points = np.array([[img_info['width']//2, img_info['height']//2]])
            point_labels = np.array([1])
        
        #apply transforms
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'task_description': task_description,
            'mask': torch.from_numpy(mask).float(),
            'points': torch.from_numpy(points).float(),
            'point_labels': torch.from_numpy(point_labels).long(),
            'image_id': image_id,
            'category_name': category_name if annotations else "unknown"
        }
    
    def _generate_task_description(self, category_name: str) -> str:
        """generate a task description for the given category"""
        template = random.choice(self.task_templates)
        
        #sometimes add descriptors for more complex tasks
        if random.random() < 0.3:
            if random.random() < 0.5:
                color = random.choice(self.color_descriptors)
                enhanced_name = f"{color} {category_name}"
            else:
                shape = random.choice(self.shape_descriptors)
                enhanced_name = f"{shape} {category_name}"
        else:
            enhanced_name = category_name
        
        return template.format(enhanced_name)
    
    def _create_category_mask(self, annotations: List[Dict], category_id: int, img_info: Dict) -> np.ndarray:
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        
        for ann in annotations:
            if ann['category_id'] == category_id:
                if 'segmentation' in ann:
                    if isinstance(ann['segmentation'], list):
                        #polygon format
                        for seg in ann['segmentation']:
                            poly = np.array(seg).reshape(-1, 2)
                            cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
                    else:
                        #RLE format
                        rle = ann['segmentation']
                        m = coco_mask.decode(rle)
                        mask = np.logical_or(mask, m).astype(np.uint8)
        
        return mask
    
    def _generate_point_prompts(self, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """generate point prompts from mask for SAM input"""
        #find positive points (inside mask)
        positive_points = []
        negative_points = []
        
        #get positive points from mask
        y_coords, x_coords = np.where(mask > 0)
        if len(y_coords) > 0:
            #sample a few positive points
            num_positive = min(3, len(y_coords))
            indices = np.random.choice(len(y_coords), num_positive, replace=False)
            for i in indices:
                positive_points.append([x_coords[i], y_coords[i]])
        
        #fet negative points (outside mask)
        y_coords, x_coords = np.where(mask == 0)
        if len(y_coords) > 0:
            # Sample a few negative points
            num_negative = min(2, len(y_coords))
            indices = np.random.choice(len(y_coords), num_negative, replace=False)
            for i in indices:
                negative_points.append([x_coords[i], y_coords[i]])
        
        #combine points and labels
        all_points = positive_points + negative_points
        labels = [1] * len(positive_points) + [0] * len(negative_points)
        
        if not all_points:
            #fallback
            h, w = mask.shape
            all_points = [[w//2, h//2]]
            labels = [1]
        
        return np.array(all_points), np.array(labels)


class COCOSubset(Dataset):
    """simplified COCO subset for quick experimentation"""
    
    def __init__(self, data_dir: str, num_samples: int = 10000, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        #create a subset of COCO data
        self.samples = self._create_subset(num_samples)
    
    def _create_subset(self, num_samples: int) -> List[Dict]:
        """create a subset of COCO data with diverse categorie"""
        samples = []
        categories = [
            "person", "car", "chair", "dog", "cat", "bird", "bicycle", 
            "bottle", "cup", "bowl", "apple", "banana", "orange"
        ]
        
        for i in range(num_samples):
            category = random.choice(categories)
            samples.append({
                'image_id': f"sample_{i:06d}",
                'category': category,
                'task_description': f"segment all {category}"
            })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        dummy_image = torch.randn(3, 224, 224)
        dummy_mask = torch.zeros(224, 224)
        dummy_points = torch.tensor([[112, 112]])
        dummy_labels = torch.tensor([1])
        
        return {
            'image': dummy_image,
            'task_description': sample['task_description'],
            'mask': dummy_mask,
            'points': dummy_points,
            'point_labels': dummy_labels,
            'image_id': sample['image_id'],
            'category_name': sample['category']
        }