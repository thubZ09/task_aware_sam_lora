import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import json
import os
from typing import Dict, List, Tuple, Optional, Sequence
import cv2
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
import random
from pycocotools import mask as coco_mask

class TaskAwareDataset(Dataset):
    """dataset with textual descriptions"""

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        max_samples: int = None,
        transform=None,
        task_templates: Optional[List[str]] = None,
        mode: str = "instance",  
    ):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.max_samples = max_samples
        self.mode = mode.lower()

        ann_dir = os.path.join(data_dir, "annotations")
        ann_file = None

        if self.mode == "panoptic":
            panoptic_ann = os.path.join(ann_dir, f"panoptic_{split}2017.json")
            if os.path.exists(panoptic_ann):
                ann_file = panoptic_ann
            else:
                raise FileNotFoundError(f"Panoptic annotation file not found: {panoptic_ann}")
        else:
            subset_ann = os.path.join(ann_dir, f"instances_{split}2017_subset.json")
            full_ann = os.path.join(ann_dir, f"instances_{split}2017.json")
            if os.path.exists(subset_ann):
                ann_file = subset_ann
            elif os.path.exists(full_ann):
                ann_file = full_ann
            else:
                raise FileNotFoundError(f"Instance annotation file not found: {subset_ann} or {full_ann}")

        self.coco = COCO(ann_file)

        self.image_ids = list(self.coco.imgs.keys())
        if max_samples:
            self.image_ids = self.image_ids[:max_samples]

        self.categories = self.coco.loadCats(self.coco.getCatIds())
        self.cat_id_to_name = {cat['id']: cat['name'] for cat in self.categories}

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
        img_info = self.coco.imgs[image_id]
        split_folder = f"{self.split}2017"
        
        #fixed image path construction
        img_path = os.path.join(self.data_dir, split_folder, img_info['file_name'])
        
        #alternative paths
        if not os.path.exists(img_path):
            # Try with images subfolder (original COCO structure)
            img_path = os.path.join(self.data_dir, "images", split_folder, img_info['file_name'])
            
            if not os.path.exists(img_path):
                #kaggle structure with coco2017 folder
                img_path = os.path.join("/kaggle/input/coco-2017-dataset/coco2017", split_folder, img_info['file_name'])
                
                if not os.path.exists(img_path):
                    if self.split == "train":
                        img_path = os.path.join("/kaggle/input/coco-2017-dataset/coco2017/train2017", img_info['file_name'])
                    else:
                        img_path = os.path.join("/kaggle/input/coco-2017-dataset/coco2017/val2017", img_info['file_name'])
                    
                    if not os.path.exists(img_path):
                        raise FileNotFoundError(f"Image not found: {img_info['file_name']} in any of the expected locations")
        
        image = Image.open(img_path).convert('RGB')

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(ann_ids)

        if annotations:
            selected_ann = random.choice(annotations)
            category_name = self.cat_id_to_name[selected_ann['category_id']]
            task_description = self._generate_task_description(category_name)
            mask = self._create_category_mask(annotations, selected_ann['category_id'], img_info)
            points, point_labels = self._generate_point_prompts(mask)
        else:
            task_description = "segment any object"
            mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
            points = np.array([[img_info['width']//2, img_info['height']//2]])
            point_labels = np.array([1])
            category_name = "unknown"

        if self.transform:
            image = self.transform(image)
        else:
            from torchvision import transforms
            image = transforms.ToTensor()(image)

        return {
            'image': image,
            'task_description': task_description,
            'mask': torch.from_numpy(mask).float(),
            'points': torch.from_numpy(points).float(),
            'point_labels': torch.from_numpy(point_labels).long(),
            'image_id': image_id,
            'category_name': category_name
        }

    def _generate_task_description(self, category_name: str) -> str:
        template = random.choice(self.task_templates)
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

    def _create_category_mask(self, annotations, category_id: int, img_info) -> np.ndarray:
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        for ann in annotations:
            if ann['category_id'] == category_id:
                segm = ann.get('segmentation', None)
                if segm is None:
                    continue
                if isinstance(segm, list) and segm and isinstance(segm[0], list):
                    for seg in segm:
                        poly = np.array(seg).reshape(-1, 2)
                        cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
                elif isinstance(segm, dict) and 'counts' in segm:
                    if isinstance(segm['counts'], list):
                        rle = coco_mask.frPyObjects(segm, img_info['height'], img_info['width'])
                    else:
                        rle = segm

                    m = coco_mask.decode(rle)
                    if m.ndim == 3:
                        m = m[:, :, 0]
                    mask = np.logical_or(mask, m).astype(np.uint8)
                elif isinstance(segm, list) and segm and isinstance(segm[0], dict):
                    for rle in segm:
                        if isinstance(rle['counts'], list):
                            rle = rle.copy()
                            rle['counts'] = ''.join(rle['counts']).encode('utf-8')
                        m = coco_mask.decode(rle)
                        if m.ndim == 3:
                            m = m[:, :, 0]
                        mask = np.logical_or(mask, m).astype(np.uint8)
        return mask

    def _generate_point_prompts(self, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        positive_points = []
        negative_points = []
        y_coords, x_coords = np.where(mask > 0)
        if len(y_coords) > 0:
            num_positive = min(3, len(y_coords))
            indices = np.random.choice(len(y_coords), num_positive, replace=False)
            for i in indices:
                positive_points.append([x_coords[i], y_coords[i]])
        y_coords, x_coords = np.where(mask == 0)
        if len(y_coords) > 0:
            num_negative = min(2, len(y_coords))
            indices = np.random.choice(len(y_coords), num_negative, replace=False)
            for i in indices:
                negative_points.append([x_coords[i], y_coords[i]])
        all_points = positive_points + negative_points
        labels = [1] * len(positive_points) + [0] * len(negative_points)
        if not all_points:
            h, w = mask.shape
            all_points = [[w//2, h//2]]
            labels = [1]
        return np.array(all_points), np.array(labels)