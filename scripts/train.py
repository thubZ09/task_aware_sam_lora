import os
import sys
import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import wandb

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

#modules
from config.train_config import get_t4_optimized_config
from src.data.dataset import TaskAwareDataset
from src.models.sam_wrapper import SAMWithLoRA
from src.models.hypernetwork import TaskAwareHyperNet
from src.training.trainer import TaskAwareTrainer
from src.utils.checkpoint import CheckpointManager, create_config_backup

def setup_logger(level: str):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def parse_args():
    p = argparse.ArgumentParser(description="Train Task‑Aware SAM LoRA")
    p.add_argument("--epochs",       type=int,   help="Override num_epochs")
    p.add_argument("--batch-size",   type=int,   help="Override batch_size")
    p.add_argument("--lr",           type=float, help="Override learning rate")
    p.add_argument("--resume",       type=str,   help="Path to checkpoint to resume from")
    p.add_argument("--use-wandb",    action="store_true", help="Enable Weights & Biases")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = get_t4_optimized_config()

    #apply overrides
    if args.epochs: cfg.training.num_epochs = args.epochs
    if args.batch_size:cfg.training.batch_size = args.batch_size
    if args.lr: cfg.training.learning_rate = args.lr

    setup_logger(cfg.system.log_level)
    logger = logging.getLogger("train")
    device = torch.device(cfg.system.device if torch.cuda.is_available() else "cpu")

    logger.info(f"Using device: {device}")

    #init wandb
    if args.use_wandb:
        wandb.init(
            project=cfg.system.wandb_project,
            config=cfg.to_dict(),
            name=f"train_{Path.cwd().name}"
        )

    #make checkpoint dir
    os.makedirs(cfg.system.checkpoint_dir, exist_ok=True)
    create_config_backup(cfg.to_dict(), cfg.system.checkpoint_dir)

    #dataset + dataloader
    logger.info("Building datasets…")
    train_ds = TaskAwareDataset(cfg.data, split="train")
    val_ds   = TaskAwareDataset(cfg.data, split="val")
    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size,
        shuffle=True, num_workers=cfg.system.dataloader_workers,
        pin_memory=cfg.data.pin_memory, collate_fn=train_ds.collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.training.batch_size,
        shuffle=False, num_workers=cfg.system.dataloader_workers,
        pin_memory=cfg.data.pin_memory, collate_fn=val_ds.collate_fn
    )

    #models
    logger.info("Building models…")
    sam_model = SamWrapper(
        sam_model_type=cfg.model.sam_model_type,
        sam_checkpoint=cfg.model.sam_checkpoint,
        lora_rank=cfg.model.lora_rank,
        lora_alpha=cfg.model.lora_alpha,
        lora_dropout=cfg.model.lora_dropout,
        target_modules=cfg.model.lora_target_modules
    ).to(device)

    hypernet = TaskAwareHyperNet(
        text_encoder_model=cfg.model.text_encoder_model,
        text_embedding_dim=cfg.model.text_embedding_dim,
        hidden_dim=cfg.model.hypernetwork_hidden_dim,
        num_layers=cfg.model.hypernetwork_num_layers,
        num_heads=cfg.model.hypernetwork_num_heads,
        dropout=cfg.model.hypernetwork_dropout,
        max_lora_params=cfg.model.hypernetwork_max_lora_params
    ).to(device)

    #trainer & checkpoint manager
    optimizer = torch.optim.AdamW(
        hypernet.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.num_epochs
    )

    trainer = HyperNetTrainer(
        hypernetwork=hypernet,
        sam_model=sam_model,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
        device=device
    )
    ckpt_mgr = CheckpointManager(
        checkpoint_dir=cfg.system.checkpoint_dir,
        max_checkpoints=cfg.training.max_checkpoints
    )

    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        ckpt = ckpt_mgr.load_checkpoint(
            args.resume, hypernet, optimizer, device
        )
        start_epoch = ckpt["epoch"] + 1

    #training loop
    best_loss = float("inf")
    for epoch in range(start_epoch, cfg.training.num_epochs):
        logger.info(f"Epoch {epoch+1}/{cfg.training.num_epochs}")
        train_metrics = trainer.train_epoch(train_loader, epoch)
        val_metrics   = trainer.validate_epoch(val_loader, epoch)
        scheduler.step()

        logger.info(f"Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f} | Val mIoU: {val_metrics['miou']:.4f}")

        is_best = val_metrics["loss"] < best_loss
        if is_best:
            best_loss = val_metrics["loss"]

        ckpt_mgr.save_checkpoint(
            model=hypernet,
            optimizer=optimizer,
            epoch=epoch,
            train_loss=train_metrics["loss"],
            val_loss=val_metrics["loss"],
            metrics=val_metrics,
            is_best=is_best
        )

    #finish
    logger.info("Training complete.")
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
