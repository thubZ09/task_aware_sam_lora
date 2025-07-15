import re
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Optional, Tuple
import numpy as np

class TextEncoder(nn.Module):
    """text encoder for task descriptions using pre-trained language models"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 hidden_dim: int = 512):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.hidden_dim = hidden_dim
        
        #freeze pre-trained model
        for param in self.model.parameters():
            param.requires_grad = False
        
        #projection layer to match hypernetwork input
        self.projection = nn.Linear(self.model.config.hidden_size, hidden_dim)
        
    def forward(self, text: List[str]) -> torch.Tensor:
        """
        encode text descriptions to embeddings
        
        Args:
            text: List of text descriptions
            
        Returns:
            text embeddings of shape (batch_size, hidden_dim)
        """
        #tokenize
        tokens = self.tokenizer(
            text, 
            padding=True, 
            truncation=True, 
            max_length=128,
            return_tensors="pt"
        )
        
        #move to same device as model
        tokens = {k: v.to(next(self.parameters()).device) for k, v in tokens.items()}
        
        #get embeddings
        with torch.no_grad():
            outputs = self.model(**tokens)
            embeddings = outputs.last_hidden_state[:, 0, :]  #CLS token
            
        #project to target dimension
        embeddings = self.projection(embeddings)
        
        return embeddings

class TaskDescriptionProcessor:
    """processes and augments task descriptions for training"""
    
    def __init__(self):
        self.templates = [
            "segment all {objects}",
            "find and segment {objects}",
            "identify {objects} in the image",
            "extract {objects} from the scene",
            "highlight all {objects}",
            "mask {objects} in the image",
            "locate {objects}",
            "segment {objects} with {attribute}",
            "find {attribute} {objects}",
            "segment {objects} that are {attribute}",
        ]
        
        self.attributes = [
            "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown", "black", "white",
            "large", "small", "tiny", "huge", "medium", "big", "little",
            "round", "square", "rectangular", "circular", "oval", "triangular",
            "bright", "dark", "light", "shiny", "matte", "glossy",
            "transparent", "opaque", "clear", "solid",
            "soft", "hard", "rough", "smooth", "textured"
        ]
        
    def generate_task_description(self, class_name: str, category_info: Dict) -> str:
        """Generate a task description for a given class."""
        #clean class name
        clean_name = self._clean_class_name(class_name)        
        template = np.random.choice(self.templates)
        
        #add attributes randomly
        if "{attribute}" in template and np.random.random() < 0.3:
            attribute = np.random.choice(self.attributes)
            description = template.format(objects=clean_name, attribute=attribute)
        else:
            #use simpler template
            simple_templates = [t for t in self.templates if "{attribute}" not in t]
            template = np.random.choice(simple_templates)
            description = template.format(objects=clean_name)
            
        return description
    
    def _clean_class_name(self, class_name: str) -> str:
        """clean and normalize class names"""
        #remove special characters
        clean_name = re.sub(r'[^a-zA-Z0-9\s]', '', class_name)
        
        #convert to lowercase
        clean_name = clean_name.lower()
        
        #handle plurals
        if not clean_name.endswith('s') and not clean_name.endswith('es'):
            if clean_name.endswith('y'):
                clean_name = clean_name[:-1] + 'ies'
            elif clean_name.endswith(('s', 'sh', 'ch', 'x', 'z')):
                clean_name = clean_name + 'es'
            else:
                clean_name = clean_name + 's'
                
        return clean_name
    
    def augment_description(self, description: str) -> str:
        """apply simple augmentations to task descriptions"""
        #random synonym replacement
        synonyms = {
            "segment": ["find", "identify", "locate", "extract", "highlight"],
            "all": ["every", "each", "any"],
            "objects": ["items", "things", "elements"],
            "red": ["crimson", "scarlet", "cherry"],
            "blue": ["azure", "navy", "cobalt"],
            "large": ["big", "huge", "massive"],
            "small": ["tiny", "little", "mini"]
        }
        
        words = description.split()
        for i, word in enumerate(words):
            if word in synonyms and np.random.random() < 0.1:
                words[i] = np.random.choice(synonyms[word])
                
        return " ".join(words)
    
    def create_task_variants(self, base_description: str, num_variants: int = 3) -> List[str]:
        """create multiple variants of a task description"""
        variants = [base_description]
        
        for _ in range(num_variants - 1):
            variant = self.augment_description(base_description)
            variants.append(variant)
            
        return variants

def create_coco_task_descriptions() -> Dict[str, List[str]]:
    """create task descriptions for COCO classes"""
    
    # COCO class names
    coco_classes = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"
    ]
    
    processor = TaskDescriptionProcessor()
    task_descriptions = {}
    
    for class_name in coco_classes:
        #generate multiple descriptions per class
        descriptions = []
        for _ in range(5):
            desc = processor.generate_task_description(class_name, {})
            descriptions.append(desc)
        
        task_descriptions[class_name] = descriptions
    
    return task_descriptions

def batch_encode_texts(text_encoder: TextEncoder, texts: List[str], 
                      batch_size: int = 32) -> torch.Tensor:
    """encode texts in batches to avoid memory issues"""
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        embeddings = text_encoder(batch_texts)
        all_embeddings.append(embeddings)
    
    return torch.cat(all_embeddings, dim=0)