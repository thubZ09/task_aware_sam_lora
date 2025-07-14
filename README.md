# Task-Aware Segment-Anything LoRA

A hypernetwork that generates task-specific LoRA adapters for SAM's mask decoder based on textual descriptions.

## Overview

Implementing a hypernetwork that:
- Takes textual task descriptions like ("segment all round orange fruits")
- Generates LoRA adapters for SAM's mask decoder
- Improves segmentation performance on specific tasks by 3-5 mIoU points
- Runs efficiently on Colab T4 GPUs

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd task_aware_sam_lora

# Install requirements
pip install -r requirements.txt

# Download SAM checkpoint
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P checkpoints/
```

### 2. Data Preparation

```bash
# Download COCO panoptic subset (10k samples)
python scripts/download_coco.py --subset panoptic --size 10000
```

### 3. Training

```bash
# Train hypernetwork
python scripts/train.py --config config/training_config.py --epochs 2
```

### 4. Demo

```bash
# Run interactive demo
python scripts/demo.py --checkpoint checkpoints/best_model.pth
```

## Architecture

### Hypernetwork Design
- **Input**: Text embeddings from task descriptions
- **Architecture**: Tiny transformer (< 0.5M params)
- **Output**: LoRA adapter weights for SAM mask decoder
- **LoRA Rank**: 2-4 (keeping total params minimal)

### SAM Integration
- **Frozen**: ViT-H image encoder (1.1B params)
- **Trainable**: Mask decoder via LoRA (4M params)
- **Memory**: Fits comfortably in T4's 16GB VRAM

## 📊 Expected Results

- **Baseline SAM mIoU**: ~65% on COCO panoptic
- **Task-specific improvement**: +3-5 mIoU points
- **Training time**: ~3 hours on T4 (2 epochs)
- **Inference speed**: Near real-time on T4

## 🔧 Key Features

- **T4 Optimized**: Efficient memory usage and batch processing
- **Modular Design**: Easy to extend and modify
- **Interactive Demo**: Jupyter notebook with live visualization
- **Comprehensive Evaluation**: Multiple metrics and visualizations

## 📝 Usage Examples

```python
from src.models.hypernetwork import TaskAwareHyperNet
from src.models.sam_wrapper import SAMWithLoRA

# Load models
hypernet = TaskAwareHyperNet.load_checkpoint('checkpoints/best_model.pth')
sam = SAMWithLoRA('checkpoints/sam_vit_h_4b8939.pth')

# Generate task-specific LoRA
task_description = "segment all round red fruit"
lora_weights = hypernet.generate_lora(task_description)

# Apply LoRA to SAM
sam.apply_lora(lora_weights)

# Perform segmentation
masks = sam.segment(image, task_description)
```

## 📈 Evaluation Metrics

- **mIoU**: Primary metric for segmentation quality
- **Precision/Recall**: Per-class performance
- **Inference Time**: Speed benchmarks
- **Memory Usage**: VRAM consumption tracking

## 🎨 Visualization

The project includes comprehensive visualization tools:
- Segmentation mask overlays
- Training loss curves
- mIoU improvement charts
- Task-specific performance analysis

## 🔬 Technical Details

### LoRA Implementation
- **Target Layers**: All linear layers in mask decoder
- **Rank**: Adaptive based on layer size (2-4)
- **Alpha**: Learned per-task scaling factor

### Hypernetwork Architecture
- **Encoder**: Text → 512-dim embeddings
- **Transformer**: 4 layers, 8 heads, 512 hidden
- **Decoder**: 512-dim → LoRA weight tensors
- **Total Params**: ~400K

### Training Strategy
- **Curriculum Learning**: Simple → complex tasks
- **Data Augmentation**: Heavy augmentation for robustness
- **Loss Function**: Combined segmentation + regularization
- **Optimizer**: AdamW with cosine scheduling

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Meta AI for the Segment Anything Model
- COCO dataset creators
- LoRA and hypernetwork research communities

---

**Ready to get started?** Open `notebooks/01_demo.ipynb` for an interactive introduction!