import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import math
from transformers import AutoTokenizer, AutoModel

class TextEncoder(nn.Module):
    """text encoder for task descriptions"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
    #freeze text encoder
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, texts: List[str]) -> torch.Tensor:
        """encode text descriptions"""
        #tokenize
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        ).to(next(self.model.parameters()).device)
        
        #get embeddings
        with torch.no_grad():
            outputs = self.model(**tokens)
            embeddings = outputs.last_hidden_state.mean(dim=1)  #
        
        return embeddings

class MultiHeadAttention(nn.Module):
    """multi-head attention for hypernetwork"""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim
        
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        #QKV projection
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        #attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        #applying ir to values
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        
        return out

class TransformerBlock(nn.Module):
    """transformer block for hypernetwork"""
    
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class HyperNetwork(nn.Module):
    """hypernetwork that generates LoRA parameters from text"""
    
    def __init__(
        self,
        text_embedding_dim: int = 384,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_lora_params: int = 1000000
    ):
        super().__init__()
        
        self.text_embedding_dim = text_embedding_dim
        self.hidden_dim = hidden_dim
        self.max_lora_params = max_lora_params
        
        #input projection
        self.input_proj = nn.Linear(text_embedding_dim, hidden_dim)
        
        #transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        #uutput projection
        self.output_proj = nn.Linear(hidden_dim, max_lora_params)
        
        #positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        #norm
        self.norm = nn.LayerNorm(hidden_dim)
        
        #initialize
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """initialize weights"""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)
    
    def forward(self, text_embeddings: torch.Tensor) -> torch.Tensor:
        """generate LoRA parameters from text embeddings"""
        B = text_embeddings.shape[0]
        
        #project to hidden dimension
        x = self.input_proj(text_embeddings)  
        x = x.unsqueeze(1) 
        
        #add nd app;y positional encoding
        x = x + self.pos_encoding
        for block in self.blocks:
            x = block(x)
        
        #normalize nd project to output
        x = self.norm(x)
        x = self.output_proj(x.squeeze(1))  # (B, max_lora_params)
        
        return x

class TaskAwareHyperNet(nn.Module):
    """complete hypernetwork system"""
    
    def __init__(
        self,
        lora_config: Dict[str, Tuple[int, int]], 
        lora_rank: int = 4,
        text_encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.lora_config = lora_config
        self.lora_rank = lora_rank
        
        #text encoder
        self.text_encoder = TextEncoder(text_encoder_model)
        text_embedding_dim = self.text_encoder.model.config.hidden_size
        
        #calculate total LoRA parameters needed
        self.param_specs = self._calculate_param_specs()
        total_params = sum(spec['total_params'] for spec in self.param_specs.values())
        
        #hypernetwork
        self.hypernetwork = HyperNetwork(
            text_embedding_dim=text_embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_lora_params=total_params
        )
        
        print(f"TaskAwareHyperNet initialized with {total_params} LoRA parameters")
        for name, spec in self.param_specs.items():
            print(f"  {name}: {spec['in_features']}x{spec['out_features']}, "
                  f"LoRA A: {spec['lora_A_params']}, LoRA B: {spec['lora_B_params']}")
    
    def _calculate_param_specs(self) -> Dict[str, Dict[str, int]]:
        """calculate parameter specifications for each LoRA layer"""
        param_specs = {}
        
        for layer_name, (in_features, out_features) in self.lora_config.items():
            lora_A_params = in_features * self.lora_rank
            lora_B_params = self.lora_rank * out_features
            total_params = lora_A_params + lora_B_params
            
            param_specs[layer_name] = {
                'in_features': in_features,
                'out_features': out_features,
                'lora_A_params': lora_A_params,
                'lora_B_params': lora_B_params,
                'total_params': total_params
            }
        
        return param_specs
    
    def forward(self, task_descriptions: List[str]) -> Dict[str, torch.Tensor]:
        """generate LoRA parameters for given task descriptions"""
        #encode text
        text_embeddings = self.text_encoder(task_descriptions)
        
        #generate LoRA parameters
        lora_params_flat = self.hypernetwork(text_embeddings)
        
        #reshape into LoRA matrices
        lora_params = self._reshape_lora_params(lora_params_flat)
        
        return lora_params
    
    def _reshape_lora_params(self, flat_params: torch.Tensor) -> Dict[str, torch.Tensor]:
        """reshape flat parameters into LoRA matrices"""
        batch_size = flat_params.shape[0]
        lora_params = {}
        
        param_idx = 0
        for layer_name, spec in self.param_specs.items():
            #extract LoRA A parameters
            lora_A_end = param_idx + spec['lora_A_params']
            lora_A_flat = flat_params[:, param_idx:lora_A_end]
            lora_A = lora_A_flat.view(batch_size, spec['in_features'], self.lora_rank)
            
            #extract LoRA B parameters
            param_idx = lora_A_end
            lora_B_end = param_idx + spec['lora_B_params']
            lora_B_flat = flat_params[:, param_idx:lora_B_end]
            lora_B = lora_B_flat.view(batch_size, self.lora_rank, spec['out_features'])
            
            param_idx = lora_B_end
            
            lora_params[f"{layer_name}.lora_A"] = lora_A
            lora_params[f"{layer_name}.lora_B"] = lora_B
        
        return lora_params
    
    def generate_lora_for_task(self, task_description: str) -> Dict[str, torch.Tensor]:
        """generate LoRA parameters for a single task"""
        lora_params = self.forward([task_description])
        
        #remove batch dimension
        single_task_params = {}
        for key, value in lora_params.items():
            single_task_params[key] = value.squeeze(0)
        
        return single_task_params
    
    def get_hypernetwork_params(self) -> List[nn.Parameter]:
        """get hypernetwork parameters (excluding text encoder)"""
        return list(self.hypernetwork.parameters())
    
    def get_param_count(self) -> Dict[str, int]:
        """get parameter counts for different components"""
        hypernet_params = sum(p.numel() for p in self.hypernetwork.parameters())
        text_encoder_params = sum(p.numel() for p in self.text_encoder.parameters())
        
        return {
            'hypernetwork': hypernet_params,
            'text_encoder': text_encoder_params,
            'total_trainable': hypernet_params,  #txt encoder is frozen
            'total': hypernet_params + text_encoder_params
        }
    
    def save_checkpoint(self, path: str, epoch: int = 0, loss: float = 0.0):
        """save hypernetwork checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'loss': loss,
            'hypernetwork_state_dict': self.hypernetwork.state_dict(),
            'lora_config': self.lora_config,
            'lora_rank': self.lora_rank,
            'param_specs': self.param_specs
        }
        torch.save(checkpoint, path)
    
    @classmethod
    def load_checkpoint(cls, path: str, device: str = 'cuda') -> 'TaskAwareHyperNet':
        """load hypernetwork from checkpoint"""
        checkpoint = torch.load(path, map_location=device)
        
        #recreate hypernetwork
        hypernet = cls(
            lora_config=checkpoint['lora_config'],
            lora_rank=checkpoint['lora_rank']
        )
        
        #load state dict
        hypernet.hypernetwork.load_state_dict(checkpoint['hypernetwork_state_dict'])
        hypernet.to(device)
        
        return hypernet

def create_sam_lora_config(mask_decoder) -> Dict[str, Tuple[int, int]]:
    """create LoRA config"""
    lora_config = {}
    
    #define target modules for SAM mask decoder
    target_modules = [
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
    
    #extract layer dimensions
    for module_name in target_modules:
        try:
            #navigate to the module
            module = mask_decoder
            for part in module_name.split('.'):
                if part.isdigit():
                    module = module[int(part)]
                else:
                    module = getattr(module, part)
            
            #check if it's a linear layer
            if isinstance(module, nn.Linear):
                lora_config[module_name] = (module.in_features, module.out_features)
                print(f"Added {module_name}: {module.in_features} -> {module.out_features}")
        
        except (AttributeError, IndexError):
            print(f"Warning: Could not find module {module_name}")
    
    return lora_config