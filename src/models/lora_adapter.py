import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import math

class LoRALayer(nn.Module):
    """single LoRA layer implementation"""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 32.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        #LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        #dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        #initialize
        self.reset_parameters()
    
    def reset_parameters(self):
        """initialize LoRA param"""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward pass through LoRA layer"""
        original_shape = x.shape
        x = x.view(-1, x.size(-1))
        lora_out = x @ self.lora_A @ self.lora_B * self.scaling
        lora_out = self.dropout(lora_out)
        
        return lora_out.view(original_shape)

class LoRALinear(nn.Module):
    """linear layer with LoRA adaptatio"""
    
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 4,
        alpha: float = 32.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.original_layer = original_layer
        self.lora = LoRALayer(
            in_features=original_layer.in_features,
            out_features=original_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )
        
        #freezing original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward pass with LoRA adaptation"""
        return self.original_layer(x) + self.lora(x)

class LoRAAdapter(nn.Module):
    """LoRA adapter for SAM mask decoder"""
    
    def __init__(
        self,
        target_modules: List[str],
        rank: int = 4,
        alpha: float = 32.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.target_modules = target_modules
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        
        #storing LoRA layers
        self.lora_layers = nn.ModuleDict()
        
    def get_lora_params(self) -> Dict[str, torch.Tensor]:
        """get all LoRA parameters as a dictionary"""
        params = {}
        for name, layer in self.lora_layers.items():
            params[f"{name}.lora_A"] = layer.lora_A
            params[f"{name}.lora_B"] = layer.lora_B
        return params
    
    def set_lora_params(self, params: Dict[str, torch.Tensor]):
        """set LoRA parameters from dictionary"""
        for name, layer in self.lora_layers.items():
            if f"{name}.lora_A" in params:
                layer.lora_A.data = params[f"{name}.lora_A"]
            if f"{name}.lora_B" in params:
                layer.lora_B.data = params[f"{name}.lora_B"]
    
    def add_lora_to_model(self, model: nn.Module):
        """add LoRA layers to target modules in the model"""
        
        def replace_linear_with_lora(module, name=""):
            for child_name, child in module.named_children():
                full_name = f"{name}.{child_name}" if name else child_name
                
                if isinstance(child, nn.Linear) and any(target in full_name for target in self.target_modules):
                    #replace with LoRA linear
                    lora_linear = LoRALinear(
                        original_layer=child,
                        rank=self.rank,
                        alpha=self.alpha,
                        dropout=self.dropout
                    )
                    setattr(module, child_name, lora_linear)
                    
                    self.lora_layers[full_name] = lora_linear.lora
                    
                    print(f"Added LoRA to: {full_name}")
                
                else:
                    replace_linear_with_lora(child, full_name)
        
        replace_linear_with_lora(model)
        
        print(f"added LoRA to {len(self.lora_layers)} layers")
        return model
    
    def enable_lora(self):
        """enable LoRA adaptation"""
        for layer in self.lora_layers.values():
            for param in layer.parameters():
                param.requires_grad = True
    
    def disable_lora(self):
        """disable LoRA adaptation"""
        for layer in self.lora_layers.values():
            for param in layer.parameters():
                param.requires_grad = False
    
    def get_trainable_params(self) -> List[nn.Parameter]:
        """get all trainable LoRA parameters"""
        params = []
        for layer in self.lora_layers.values():
            params.extend(layer.parameters())
        return params
    
    def get_param_count(self) -> int:
        """get total number of LoRA parameters"""
        return sum(p.numel() for p in self.get_trainable_params())

class LoRAConfig:
    """config for LoRA adaptation"""
    
    def __init__(
        self,
        rank: int = 4,
        alpha: float = 32.0,
        dropout: float = 0.1,
        target_modules: Optional[List[str]] = None
    ):
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules or [
            "output_upscaling.0",
            "output_upscaling.1",
            "output_hypernetworks_mlps.0",
            "output_hypernetworks_mlps.1", 
            "output_hypernetworks_mlps.2",
            "output_hypernetworks_mlps.3",
            "iou_prediction_head.0",
            "iou_prediction_head.1",
            "iou_prediction_head.2"
        ]
    
    def create_adapter(self) -> LoRAAdapter:
        """create LoRA adapter with this config"""
        return LoRAAdapter(
            target_modules=self.target_modules,
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout
        )