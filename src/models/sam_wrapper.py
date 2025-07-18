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
        self.sam.mask_decoder = self.lora_adapter.add_lora_to_model(self.sam.mask_decoder).to(self.device)
        
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
    
    def _validate_and_fix_shapes(self, 
                                 image_embeddings: torch.Tensor,
                                 point_coords: Optional[torch.Tensor] = None,
                                 point_labels: Optional[torch.Tensor] = None,
                                 mask_input: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Validate and fix tensor shapes to ensure compatibility"""
        
        # Fix image_embeddings shape
        if len(image_embeddings.shape) == 3:
            image_embeddings = image_embeddings.unsqueeze(0)
        elif len(image_embeddings.shape) != 4:
            raise ValueError(f"Expected image_embeddings to have 3 or 4 dimensions, got {len(image_embeddings.shape)}")
        
        # Fix point coordinates and labels
        if point_coords is not None:
            if len(point_coords.shape) == 2:
                point_coords = point_coords.unsqueeze(0)
            elif len(point_coords.shape) == 1:
                point_coords = point_coords.unsqueeze(0).unsqueeze(0)
        
        if point_labels is not None:
            if len(point_labels.shape) == 1:
                point_labels = point_labels.unsqueeze(0)
            elif len(point_labels.shape) == 0:
                point_labels = point_labels.unsqueeze(0).unsqueeze(0)
        
        # Fix mask input
        if mask_input is not None:
            if len(mask_input.shape) == 2:
                mask_input = mask_input.unsqueeze(0).unsqueeze(0)
            elif len(mask_input.shape) == 3:
                mask_input = mask_input.unsqueeze(0)
        
        return image_embeddings, point_coords, point_labels, mask_input
    
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
        
        # Validate and fix shapes
        image_embeddings, point_coords, point_labels, mask_input = self._validate_and_fix_shapes(
            image_embeddings, point_coords, point_labels, mask_input
        )
        
        # Debug shapes
        print(f"[DEBUG] image_embeddings shape: {image_embeddings.shape}")
        if point_coords is not None:
            print(f"[DEBUG] point_coords shape: {point_coords.shape}")
        if point_labels is not None:
            print(f"[DEBUG] point_labels shape: {point_labels.shape}")
        if mask_input is not None:
            print(f"[DEBUG] mask_input shape: {mask_input.shape}")
        
        # Prepare prompts
        points = (point_coords, point_labels) if point_coords is not None and point_labels is not None else None
        boxes = None
        masks = mask_input
        
        try:
            # Prompt encoder
            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=points,
                boxes=boxes,
                masks=masks,
            )
            
            print(f"[DEBUG] sparse_embeddings shape: {sparse_embeddings.shape}")
            print(f"[DEBUG] dense_embeddings shape: {dense_embeddings.shape}")
            
            # Get image PE
            image_pe = self.sam.prompt_encoder.get_dense_pe()
            print(f"[DEBUG] image_pe shape: {image_pe.shape}")
            
            # Ensure dense_embeddings has the right shape
            if len(dense_embeddings.shape) != 4:
                # Try to reshape dense_embeddings to match expected 4D format
                if dense_embeddings.numel() > 0:
                    batch_size = image_embeddings.shape[0]
                    # Assuming dense_embeddings should match image_embeddings spatial dimensions
                    expected_h, expected_w = image_embeddings.shape[-2:]
                    
                    if len(dense_embeddings.shape) == 3:
                        # Reshape from (batch, hw, channels) to (batch, channels, h, w)
                        channels = dense_embeddings.shape[-1]
                        dense_embeddings = dense_embeddings.transpose(1, 2).reshape(batch_size, channels, expected_h, expected_w)
                    elif len(dense_embeddings.shape) == 2:
                        # Reshape from (batch, channels) to (batch, channels, h, w)
                        channels = dense_embeddings.shape[-1]
                        dense_embeddings = dense_embeddings.unsqueeze(-1).unsqueeze(-1)
                        dense_embeddings = dense_embeddings.expand(batch_size, channels, expected_h, expected_w)
                else:
                    # Create zero tensor with correct shape
                    dense_embeddings = torch.zeros(
                        image_embeddings.shape[0],
                        image_embeddings.shape[1],
                        image_embeddings.shape[2],
                        image_embeddings.shape[3],
                        device=image_embeddings.device,
                        dtype=image_embeddings.dtype
                    )
            
            print(f"[DEBUG] dense_embeddings final shape: {dense_embeddings.shape}")
            
            # Mask decoder (single mask output)
            masks, iou_predictions = self.sam.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False
            )
            
            return masks, iou_predictions
            
        except Exception as e:
            print(f"[ERROR] Exception in forward pass: {e}")
            print(f"[ERROR] image_embeddings shape: {image_embeddings.shape}")
            print(f"[ERROR] sparse_embeddings shape: {sparse_embeddings.shape if 'sparse_embeddings' in locals() else 'Not computed'}")
            print(f"[ERROR] dense_embeddings shape: {dense_embeddings.shape if 'dense_embeddings' in locals() else 'Not computed'}")
            raise e
    
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
        
        # Process each image to get embeddings
        for i in range(batch_size):
            try:
                img_np = images[i].permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * 255).astype(np.uint8)
                emb = self.sam_lora.get_image_embeddings(img_np)
                image_embeddings.append(emb)
            except Exception as e:
                print(f"[ERROR] Failed to get embeddings for image {i}: {e}")
                # Create a dummy embedding with the right shape
                dummy_emb = torch.zeros((1, 256, 64, 64), device=self.device)  # Adjust dimensions as needed
                image_embeddings.append(dummy_emb)
        
        image_embeddings = torch.stack(image_embeddings, dim=0)
        
        # Generate LoRA parameters for the batch
        lora_params_batch = self.hypernetwork(task_descriptions)
        
        all_masks, all_ious = [], []
        
        # Process each sample in the batch
        for i in range(batch_size):
            try:
                # Get LoRA parameters for this sample
                sample_params = {k: v[i] for k, v in lora_params_batch.items()}
                self.sam_lora.apply_lora(sample_params)
                
                # Prepare inputs for this sample
                sample_image_emb = image_embeddings[i:i+1]
                sample_point_coords = point_coords[i:i+1] if point_coords is not None else None
                sample_point_labels = point_labels[i:i+1] if point_labels is not None else None
                
                # Forward pass
                masks, iou_pred = self.sam_lora(
                    image_embeddings=sample_image_emb,
                    point_coords=sample_point_coords,
                    point_labels=sample_point_labels,
                    multimask_output=False
                )
                
                all_masks.append(masks)
                all_ious.append(iou_pred)
                
            except Exception as e:
                print(f"[ERROR] Failed to process sample {i}: {e}")
                # Create dummy outputs
                dummy_mask = torch.zeros((1, 1, 1024, 1024), device=self.device)  # Adjust dimensions as needed
                dummy_iou = torch.zeros((1, 1), device=self.device)
                all_masks.append(dummy_mask)
                all_ious.append(dummy_iou)
        
        # Concatenate results
        all_masks = torch.cat(all_masks, dim=0)
        all_ious = torch.cat(all_ious, dim=0)
        
        return all_masks, all_ious
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """get all trainable parameters of the hypernetwork"""
        return self.hypernetwork.get_hypernetwork_params()