import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import numpy as np
from PIL import Image
import cv2
import random
from typing import Tuple, Dict, Any

class SAMTransform:
    """transform for SAM input - resize and normalize"""
    
    def __init__(self, size: int = 1024):
        self.size = size
        self.pixel_mean = [123.675, 116.28, 103.53]
        self.pixel_std = [58.395, 57.12, 57.375]
    
    def __call__(self, image):
        #resize
        image = F.resize(image, (self.size, self.size))
        
        #convert to tensor
        image = F.to_tensor(image)
        
        #normalize
        image = F.normalize(image, mean=self.pixel_mean, std=self.pixel_std)
        
        return image


class TaskAwareTransform:    
    def __init__(
        self,
        size: int = 1024,
        augment: bool = True,
        normalize: bool = True
    ):
        self.size = size
        self.augment = augment
        self.normalize = normalize
        
        #SAM normalization parameters
        self.pixel_mean = [123.675, 116.28, 103.53]
        self.pixel_std = [58.395, 57.12, 57.375]
        
        #augment transforms
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        )
        
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        image = sample['image']
        mask = sample.get('mask', None)
        points = sample.get('points', None)
        
        #get original size
        orig_size = image.size
        
        #apply augmentations
        if self.augment:
            image = self._apply_augmentations(image)
        
        image = F.resize(image, (self.size, self.size)) #resize
        
        if mask is not None:
            mask = torch.from_numpy(mask) if isinstance(mask, np.ndarray) else mask
            mask = F.resize(mask.unsqueeze(0), (self.size, self.size)).squeeze(0)
        
        if points is not None:
            points = self._scale_points(points, orig_size, (self.size, self.size))
        
        image = F.to_tensor(image)#convert to tensor and normalize
        if self.normalize:
            image = F.normalize(image, mean=self.pixel_mean, std=self.pixel_std)
        
        #update sample
        sample['image'] = image
        if mask is not None:
            sample['mask'] = mask
        if points is not None:
            sample['points'] = points
        
        return sample
    
    def _apply_augmentations(self, image: Image.Image) -> Image.Image:
        #random horizontal flip
        if random.random() < 0.5:
            image = F.hflip(image)
        
        #color jitter
        if random.random() < 0.7:
            image = self.color_jitter(image)
        
        if random.random() < 0.3:#random rotation
            angle = random.uniform(-10, 10)
            image = F.rotate(image, angle)
        
        return image
    
    def _scale_points(self, points: np.ndarray, orig_size: Tuple[int, int], new_size: Tuple[int, int]) -> np.ndarray:
        scale_x = new_size[0] / orig_size[0]
        scale_y = new_size[1] / orig_size[1]
        
        points = points.copy()
        points[:, 0] *= scale_x
        points[:, 1] *= scale_y
        
        return points


class ResizeTransform:    
    def __init__(self, size: int = 1024):
        self.size = size
    
    def __call__(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return F.resize(image, (self.size, self.size))


class MaskTransform:    
    def __init__(self, size: int = 1024):
        self.size = size
    
    def __call__(self, mask):
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        
        mask = F.resize(mask, (self.size, self.size))
        return mask.squeeze(0)


class PointTransform:    
    def __init__(self, orig_size: Tuple[int, int], new_size: Tuple[int, int]):
        self.scale_x = new_size[0] / orig_size[0]
        self.scale_y = new_size[1] / orig_size[1]
    
    def __call__(self, points):
        if isinstance(points, np.ndarray):
            points = torch.from_numpy(points)
        
        points = points.clone()
        points[:, 0] *= self.scale_x
        points[:, 1] *= self.scale_y
        
        return points


class RandomCrop:    
    def __init__(self, size: int, padding: int = 32):
        self.size = size
        self.padding = padding
    
    def __call__(self, image):
        #aadd padding
        image = F.pad(image, self.padding)
        
        #random crop
        i, j, h, w = transforms.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, i, j, h, w)
        
        return image

class RandomHorizontalFlip:
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, sample):
        if random.random() < self.p:
            sample['image'] = F.hflip(sample['image'])
            if 'mask' in sample:
                sample['mask'] = F.hflip(sample['mask'])
            if 'points' in sample:
                # Flip x coordinates
                w = sample['image'].shape[-1]
                sample['points'][:, 0] = w - sample['points'][:, 0]
        
        return sample


class Compose:
    
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)
        return sample


def get_train_transforms(size: int = 1024) -> TaskAwareTransform:
    return TaskAwareTransform(size=size, augment=True, normalize=True)


def get_val_transforms(size: int = 1024) -> TaskAwareTransform:
    return TaskAwareTransform(size=size, augment=False, normalize=True)


def get_sam_transforms(size: int = 1024) -> SAMTransform:
    return SAMTransform(size=size)


#util functions
def denormalize_image(image: torch.Tensor) -> torch.Tensor:
    pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
    pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)
    
    image = image * pixel_std + pixel_mean
    image = image / 255.0
    
    return torch.clamp(image, 0, 1)

def resize_mask(mask: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(0)
    
    mask = F.interpolate(mask, size=size, mode='nearest')
    return mask.squeeze()


def augment_points(points: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
    noise = torch.randn_like(points) * 5  # 5 pixel noise
    points = points + noise

    points[:, 0] = torch.clamp(points[:, 0], 0, image_size[1] - 1) #clamp to image boundaries
    points[:, 1] = torch.clamp(points[:, 1], 0, image_size[0] - 1)
    
    return points