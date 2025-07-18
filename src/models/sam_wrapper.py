import torch
import torch.nn as nn
from segment_anything import sam_model_registry, SamPredictor
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .lora_adapter import LoRAAdapter, LoRAConfig
from .hypernetwork import TaskAwareHyperNet, create_sam_lora_config

class SAMWithLoRA(nn.Module):
    """SAM model with LoRA adaptation capability"""
    
    def __init__(
        self,
        sam_checkpoint: str,
        model_type: str = "vit_h",
        lora_config: Optional[LoRAConfig] = None,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.device = device
        self.model_type = model_type
        
        # load model
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device)
        
        # freeze image encoder and prompt encoder
        self._freeze_encoders()
        
        # create LoRA adapter
        if lora_config is None:
            lora_config = LoRAConfig()
        self.lora_adapter = lora_config.create_adapter()
        
        # add LoRA to mask decoder
        self.sam.mask_decoder = self.lora_adapter.add_lora_to_model(self.sam.mask_decoder)
        
        # create predictor for inference
        self.predictor = SamPredictor(self.sam)
        
        # store original state for reset
        self.original_lora_state = self.lora_adapter.get_lora_params()
        
        print(f"SAM with LoRA initialized:")
        print(f"  Model type: {model_type}")
        print(f"  LoRA parameters: {self.lora_adapter.get_param_count():,}")
        print(f"  Device: {device}")
    
    def _freeze_encoders(self):
        """freeze image and prompt encoders"""
        for param in self.sam.image_encoder.parameters():
            param.requires_grad = False
        for param in self.sam.prompt_encoder.parameters():
            param.requires_grad = False
        print("Frozen SAM encoders")
    
    def apply_lora(self, lora_params: Dict[str, torch.Tensor]):
        """apply LoRA parameters to the model"""
        self.lora_adapter.set_lora_params(lora_params)
    
    def reset_lora(self):
        """reset to original state"""
        self.lora_adapter.set_lora_params(self.original_lora_state)
    
    def enable_lora_training(self):
        """enable LoRA adaptation training"""
        self.lora_adapter.enable_lora()
    
    def disable_lora_training(self):
        """disable LoRA adaptation training"""
        self.lora_adapter.disable_lora()
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """get trainable parameters (LoRA only)"""
        return self.lora_adapter.get_trainable_params()
    
    def forward(
        self,
        image_embeddings: torch.Tensor,
        point_coords: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        has_mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through SAM with LoRA"""
        
        # Prepare prompts
        points = (point_coords, point_labels) if point_coords is not None and point_labels is not None else None
        boxes = None
        masks = mask_input
        
        # Prompt encoder
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=masks,
        )
        
        # Mask decoder (single mask output)
        masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )
        
        return masks, iou_predictions
    
    def predict(
        self,
        image: np.ndarray,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """predict masks using the predictor interface"""
        self.predictor.set_image(image)
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            mask_input=mask_input,
            multimask_output=multimask_output
        )
        return masks, scores, logits
    
    def get_image_embeddings(self, image: np.ndarray) -> torch.Tensor:
        """get image embeddings from SAM encoder"""
        self.predictor.set_image(image)
        return self.predictor.features
    
    def compute_loss(
        self,
        predicted_masks: torch.Tensor,
        target_masks: torch.Tensor,
        predicted_iou: torch.Tensor,
        target_iou: torch.Tensor,
        seg_loss_weight: float = 1.0,
        iou_loss_weight: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """compute training loss"""
        seg_loss = nn.functional.binary_cross_entropy_with_logits(
            predicted_masks.flatten(1),
            target_masks.flatten(1),
            reduction='mean'
        )
        iou_loss = nn.functional.mse_loss(predicted_iou, target_iou)
        total_loss = seg_loss_weight * seg_loss + iou_loss_weight * iou_loss
        return {'total_loss': total_loss, 'seg_loss': seg_loss, 'iou_loss': iou_loss}

class TaskAwareSAM(nn.Module):
    """complete system integrating SAMWithLoRA and a hypernetwork"""
    
    def __init__(
        self,
        sam_checkpoint: str,
        model_type: str = "vit_h",
        lora_config: Optional[LoRAConfig] = None,
        hypernet_config: Optional[Dict[str, Any]] = None,
        device: str = "cuda"
    ):
        super().__init__()
        self.device = device
        self.sam_lora = SAMWithLoRA(
            sam_checkpoint=sam_checkpoint,
            model_type=model_type,
            lora_config=lora_config,
            device=device
        )
        lora_layer_config = create_sam_lora_config(self.sam_lora.sam.mask_decoder)
        hypernet_config = hypernet_config or {}
        self.hypernetwork = TaskAwareHyperNet(
            lora_config=lora_layer_config,
            lora_rank=lora_config.rank if lora_config else 4,
            **hypernet_config
        )
        self.hypernetwork.to(device)
        self.current_task = None
        self.current_lora_params = None
        print(f"TaskAwareSAM initialized:")
        print(f"  SAM LoRA params: {self.sam_lora.lora_adapter.get_param_count():,}")
        print(f"  Hypernetwork params: {self.hypernetwork.get_param_count()}")
    
    def set_task(self, task_description: str):
        self.current_task = task_description
        self.current_lora_params = self.hypernetwork.generate_lora_for_task(task_description)
        self.sam_lora.apply_lora(self.current_lora_params)
        print(f"Task set to: {task_description}")
    
    def predict_with_task(
        self,
        image: np.ndarray,
        task_description: str,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        multimask_output: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if task_description != self.current_task:
            self.set_task(task_description)
        return self.sam_lora.predict(
            image=image,
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=multimask_output
        )
    
    def forward(
        self,
        images: torch.Tensor,
        task_descriptions: List[str],
        point_coords: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
        multimask_output: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = images.shape[0]
        image_embeddings = []
        for i in range(batch_size):
            img_np = images[i].permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            emb = self.sam_lora.get_image_embeddings(img_np)
            image_embeddings.append(emb)
        image_embeddings = torch.stack(image_embeddings, dim=0)
        lora_params_batch = self.hypernetwork(task_descriptions)
        all_masks, all_ious = [], []
        for i in range(batch_size):
            sample_params = {k: v[i] for k, v in lora_params_batch.items()}
            self.sam_lora.apply_lora(sample_params)
            masks, iou_pred = self.sam_lora(
                image_embeddings=image_embeddings[i:i+1],
                point_coords=point_coords[i:i+1] if point_coords is not None else None,
                point_labels=point_labels[i:i+1] if point_labels is not None else None,
                multimask_output=False
            )
            all_masks.append(masks)
            all_ious.append(iou_pred)
        all_masks = torch.cat(all_masks, dim=0)
        all_ious = torch.cat(all_ious, dim=0)
        return all_masks, all_ious
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """get all trainable parameters of the hypernetwork"""
        return self.hypernetwork.get_hypernetwork_params()
