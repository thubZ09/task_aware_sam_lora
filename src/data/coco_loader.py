import os
import requests
import zipfile
from tqdm import tqdm
import json
from typing import List, Dict, Optional

class COCOLoader:
    """utility class for downloading and loading"""
    
    def __init__(self, data_dir: str = "data/coco"):
        self.data_dir = data_dir
        self.base_url = "http://images.cocodataset.org"
        
        # COCO dataset URLs
        self.urls = {
            "train_images": f"{self.base_url}/zips/train2017.zip",
            "val_images": f"{self.base_url}/zips/val2017.zip",
            "annotations": f"{self.base_url}/annotations/annotations_trainval2017.zip"
        }
    
    def download_subset(self, subset_size: int = 10000, split: str = "train"):
        print(f"downloading COCO subset ({subset_size} samples) for {split}...")
        
        #create directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(f"{self.data_dir}/images", exist_ok=True)
        os.makedirs(f"{self.data_dir}/annotations", exist_ok=True)
        
        #download annotations first 
        self._download_annotations()
        
        #download and extract subset of images
        self._download_image_subset(subset_size, split)
        
        #create subset annotation file
        self._create_subset_annotations(subset_size, split)
        
        print(f"COCO subset downloaded successfully to {self.data_dir}")
    
    def _download_file(self, url: str, filepath: str):
        """download a file with progress bar"""
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f, tqdm(
            desc=f"Downloading {os.path.basename(filepath)}",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    def _download_annotations(self):
        """download COCO annotations"""
        ann_path = f"{self.data_dir}/annotations_trainval2017.zip"
        
        if not os.path.exists(ann_path):
            print("downloading COCO annotations...")
            self._download_file(self.urls["annotations"], ann_path)
            
            #extract annotations
            with zipfile.ZipFile(ann_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            
            #clean up zip file
            os.remove(ann_path)
    
    def _download_image_subset(self, subset_size: int, split: str):
        """download a subset of images"""
        # Load annotations to get image info
        ann_file = f"{self.data_dir}/annotations/instances_{split}2017.json"
        
        if not os.path.exists(ann_file):
            raise FileNotFoundError(f"annotations not found: {ann_file}")
        
        with open(ann_file, 'r') as f:
            data = json.load(f)
        
        #get subset of images
        images = data['images'][:subset_size]
        
        #create images directory
        img_dir = f"{self.data_dir}/images/{split}2017"
        os.makedirs(img_dir, exist_ok=True)
        
        #download images
        for img_info in tqdm(images, desc=f"Downloading {split} images"):
            img_url = f"{self.base_url}/{split}2017/{img_info['file_name']}"
            img_path = f"{img_dir}/{img_info['file_name']}"
            
            if not os.path.exists(img_path):
                try:
                    self._download_file(img_url, img_path)
                except Exception as e:
                    print(f"Failed to download {img_info['file_name']}: {e}")
                    continue
    
    def _create_subset_annotations(self, subset_size: int, split: str):
        """create annotation file for the subset"""
        ann_file = f"{self.data_dir}/annotations/instances_{split}2017.json"
        subset_ann_file = f"{self.data_dir}/annotations/instances_{split}2017_subset.json"
        
        with open(ann_file, 'r') as f:
            data = json.load(f)
        
        #get subset of images and their annotations
        subset_images = data['images'][:subset_size]
        subset_image_ids = {img['id'] for img in subset_images}
        
        #filter annotations for subset images
        subset_annotations = [
            ann for ann in data['annotations'] 
            if ann['image_id'] in subset_image_ids
        ]
        
        #create subset data
        subset_data = {
            'info': data['info'],
            'licenses': data['licenses'],
            'categories': data['categories'],
            'images': subset_images,
            'annotations': subset_annotations
        }        
        with open(subset_ann_file, 'w') as f:
            json.dump(subset_data, f)
        
        print(f"Created subset annotations: {len(subset_images)} images, {len(subset_annotations)} annotations")
    
    def get_dataset_stats(self) -> Dict:
        stats = {}
        
        #count images
        for split in ['train', 'val']:
            img_dir = f"{self.data_dir}/images/{split}2017"
            if os.path.exists(img_dir):
                stats[f'{split}_images'] = len(os.listdir(img_dir))
        
        #count annotations
        for split in ['train', 'val']:
            ann_file = f"{self.data_dir}/annotations/instances_{split}2017_subset.json"
            if os.path.exists(ann_file):
                with open(ann_file, 'r') as f:
                    data = json.load(f)
                stats[f'{split}_annotations'] = len(data['annotations'])
                stats[f'{split}_categories'] = len(data['categories'])
        
        return stats
    
    def cleanup(self):
        """clean up temporary files"""
        for file in os.listdir(self.data_dir):
            if file.endswith('.zip'):
                os.remove(os.path.join(self.data_dir, file))


def download_coco_subset(data_dir: str = "data/coco", size: int = 10000):
    """onvenient function to download COCO subset"""
    loader = COCOLoader(data_dir)
    
    loader.download_subset(size, "train")    #download smaller validation subset
    loader.download_subset(size // 5, "val")
    
    stats = loader.get_dataset_stats()
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
        
    loader.cleanup()
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download COCO dataset subset")
    parser.add_argument("--data_dir", default="data/coco", help="Data directory")
    parser.add_argument("--size", type=int, default=10000, help="Subset size")
    
    args = parser.parse_args()
    
    download_coco_subset(args.data_dir, args.size)