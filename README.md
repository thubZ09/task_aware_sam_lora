# Task-Aware Segment Anything with LoRA

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight hypernetwork that generates LoRA adapters for the Segment Anything Model (SAM) based on natural language task descriptions.

> 💡 **Note**: This was implemented on (free-tier) Colab T4 using a subset of COCO 2017 (val) for fast prototyping and learning. The pipeline is designed to scale to full datasets, multi-GPU training, and inference with minimal changes.

## 📌Intuition
- Why - Modern segmentation models are powerful but “one‑size‑fits‑all.” What if you need to “segment all red apples” or “highlight all suitcases” without retraining a massive model from scratch?

- Inspiration - The [Text-to-LoRA: Instant Transformer Adaptation, 2025](https://arxiv.org/abs/2506.06105) paper shows how a small hypernetwork can generate LoRA adapters for transformers directly from text. How about generating LoRA adapters for Meta’s Segment‑Anything Model (SAM) based on task descriptions...

---

## 📌Overview

This repository introduces a task-aware segmentation pipeline by combining:

- A **hypernetwork** that maps **text prompts** → **LoRA weights**
- A **LoRA-injected SAM** (Segment Anything Model) for efficient mask prediction
- An **evaluation module** using COCO-style metrics (mIoU, AP) via `pycocotools`

---

## 📌Architecture

```
          "segment all red apples"
                    │
           [Text Encoder (MiniLM)]
                    │
         ┌─────────────────────────┐
         │  HyperNetwork Transformer│
         └─────────────────────────┘
                    │
        LoRA Adapter Weights (dict)
                    ↓
    LoRA-injected SAM Mask Decoder (ViT-H frozen)
                    ↓
            Segmentation Prediction
```

---

## 📌Packages

| Component            | Description                                                        |
|----------------------|--------------------------------------------------------------------|
| TaskAwareHyperNet    | Transformer-based hypernetwork that maps text embeddings to LoRA weights |
| LoRAAdapter          | Injects low-rank weight updates into SAM’s decoder                 |
| SAMWithLoRA          | Wraps Meta’s official SAM with LoRA support                        |
| TaskAwareDataset     | Loads COCO images + synthetic task prompts                         |
| `notebook/.ipynb`         | train + visualization + eval                                             |

---

## 📌Inference & Evaluation

You can visualize:
- Original image
- Task-specific predicted mask
- Comparison across different prompts (e.g., “segment people”, “segment vehicles”)

Pycocotools were used to compute COCO-style mIoU and AP on COCO val2017.

```python
from pycocotools.cocoeval import COCOeval

#generate predictions.json from val images 
#run COCOeval (included in sam_LoRA_visual.ipynb)
```

**Note:** On COCO val2017, with single-point prompting and LoRA-only tuning, we expect low raw AP but correct qualitative segmentations, especially on well-separated categories like humans, vehicles, fruits.

---
## 📌Learned concepts

- **PyTorch** (Model building, training loops, mixed‑precision `torch.cuda.amp`, DataLoaders).

- **Segment Anything (SAM)** - frozen ViT‑H image encoder + mask decoder, wrapped it in `SamWrapper` to inject LoRA adapters.

- **LoRA (Low-Rank Adaptation)** for efficient tuning.

- **Hypernetworks** that generate weights on-the-fly - tiny transformer (4 layers, 8 heads, 512 d) that ingests text embeddings (from sentence-transformers/all‑MiniLM‑L6‑v2) and outputs a dictionary of LoRA weight tensors.

- COCO instance annotations + pycocotools

- Visualization tools (matplotlib, overlay masks)

- Efficient training on constrained hardware (T4, batch=1)

---

## 📌Takeaways and next steps
- On a small COCO val subset with single‑point prompts, we saw modest AP@0.5. To reach production‑level IoU, you’d integrate multi‑point or box prompts and evaluate on the full COCO split.

- Training takes ~3 hrs for 2 epochs on a single T4 (batch size 1, mixed precision). At inference, generating LoRA + segmentation is near real‑time (~200 ms/image).

- Add box or multi-point prompting for improved AP

- Support full COCO panoptic splits and multi‑point sampling per instance for higher IoU

- Extend hypernetwork to generate adapters for other prompt types (text + mask).

- Benchmark against baseline SAM or segment-anything adapters

---

## 📚 References

> R. Charakorn, E. Cetin, Y. Tang, and R. T. Lange, "Text-to-LoRA: Instant Transformer Adaption," in *Proc. 42nd Int. Conf. Mach. Learn. (ICML)*, Vancouver, Canada, 2025, vol. 267.  
> Repository: (https://github.com/SakanaAI/text-to-lora)  
- [Segment Anything Model (SAM) (Meta AI)](https://github.com/facebookresearch/segment-anything.git)
- [COCO Dataset](https://cocodataset.org/)