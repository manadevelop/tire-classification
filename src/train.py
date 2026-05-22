"""
train.py — Entrenamiento de modelos CNN para clasificación de llantas dañadas.

Soporta:
  - CNN desde cero (CustomCNN)
  - Transfer learning con ResNet-50 y EfficientNet-B3
  - Estrategias de mitigación de desbalance de clases
  - Función de pérdida Focal Loss
  - Aumento de datos orientado

Uso:
  python src/train.py --config configs/train_scratch.yaml
  python src/train.py --config configs/train_resnet50.yaml
  python src/train.py --config configs/train_efficientnet.yaml
"""

import argparse
import yaml
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from pathlib import Path

from models.custom_cnn import CustomCNN
from models.transfer_model import TransferModel
from data.dataset import TireDataset
from data.transforms import get_train_transforms, get_val_transforms
from utils.losses import FocalLoss
from utils.metrics import compute_metrics, compute_class_weights
from utils.trainer import Trainer
from utils.logger import setup_logger


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(cfg: dict) -> nn.Module:
    model_name = cfg["model"]["name"]
    num_classes = cfg["model"]["num_classes"]

    if model_name == "custom_cnn":
        return CustomCNN(
            num_classes=num_classes,
            dropout_rate=cfg["model"].get("dropout_rate", 0.5),
            base_channels=cfg["model"].get("base_channels", 32),
        )
    elif model_name in ["resnet50", "efficientnet_b3"]:
        return TransferModel(
            backbone=model_name,
            num_classes=num_classes,
            pretrained=cfg["model"].get("pretrained", True),
            freeze_backbone=cfg["model"].get("freeze_backbone", False),
            dropout_rate=cfg["model"].get("dropout_rate", 0.4),
        )
    else:
        raise ValueError(f"Modelo desconocido: {model_name}")


def build_loss(cfg: dict, class_weights: torch.Tensor = None) -> nn.Module:
    loss_name = cfg["training"]["loss"]
    device = cfg["device"]

    if loss_name == "bce":
        if class_weights is not None:
            pos_weight = class_weights[1] / class_weights[0]
            return nn.BCEWithLogitsLoss(
                pos_weight=pos_weight.to(device)
            )
        return nn.BCEWithLogitsLoss()

    elif loss_name == "focal":
        return FocalLoss(
            alpha=cfg["training"].get("focal_alpha", 0.25),
            gamma=cfg["training"].get("focal_gamma", 2.0),
        )

    elif loss_name == "cross_entropy":
        if class_weights is not None:
            return nn.CrossEntropyLoss(weight=class_weights.to(device))
        return nn.CrossEntropyLoss()

    raise ValueError(f"Función de pérdida desconocida: {loss_name}")


def build_sampler(dataset: TireDataset, cfg: dict):
    """WeightedRandomSampler para mitigar desbalance de clases."""
    if not cfg["training"].get("use_weighted_sampler", False):
        return None

    labels = [dataset[i][1] for i in range(len(dataset))]
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[l] for l in labels]

    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float),
        num_samples=len(sample_weights),
        replacement=True,
    )


def main(cfg_path: str):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    logger = setup_logger(cfg["experiment_name"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg["device"] = device
    logger.info(f"Dispositivo: {device}")

    # ── Datasets ───────────────────────────────────────────────────────────
    data_root = cfg["data"]["root"]
    img_size = cfg["data"]["img_size"]

    train_ds = TireDataset(
        root=os.path.join(data_root, "train"),
        transform=get_train_transforms(
            img_size=img_size,
            augmentation_strategy=cfg["data"].get("augmentation", "standard"),
        ),
    )
    val_ds = TireDataset(
        root=os.path.join(data_root, "val"),
        transform=get_val_transforms(img_size=img_size),
    )
    test_ds = TireDataset(
        root=os.path.join(data_root, "test"),
        transform=get_val_transforms(img_size=img_size),
    )

    logger.info(
        f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}"
    )

    # ── Sampler y DataLoaders ───────────────────────────────────────────────
    sampler = build_sampler(train_ds, cfg)
    shuffle = sampler is None

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg["training"].get("num_workers", 4),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"] * 2,
        shuffle=False,
        num_workers=cfg["training"].get("num_workers", 4),
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["training"]["batch_size"] * 2,
        shuffle=False,
        num_workers=cfg["training"].get("num_workers", 4),
        pin_memory=True,
    )

    # ── Modelo ─────────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    logger.info(f"Parámetros totales: {sum(p.numel() for p in model.parameters()):,}")

    # ── Pérdida ────────────────────────────────────────────────────────────
    class_weights = None
    if cfg["training"].get("use_class_weights", False):
        labels = [train_ds[i][1] for i in range(len(train_ds))]
        class_weights = compute_class_weights(labels)
        logger.info(f"Pesos de clase: {class_weights}")

    criterion = build_loss(cfg, class_weights)

    # ── Optimizador ────────────────────────────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"].get("weight_decay", 1e-4),
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["epochs"],
        eta_min=cfg["training"].get("lr_min", 1e-6),
    )

    # ── W&B ────────────────────────────────────────────────────────────────
    if cfg.get("use_wandb", False):
        wandb.init(project="tire-classification", name=cfg["experiment_name"], config=cfg)

    # ── Entrenamiento ──────────────────────────────────────────────────────
    output_dir = Path(cfg["output_dir"]) / cfg["experiment_name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=output_dir,
        cfg=cfg,
        logger=logger,
    )

    trainer.fit(train_loader, val_loader, epochs=cfg["training"]["epochs"])

    # ── Evaluación final en test ───────────────────────────────────────────
    logger.info("Evaluando en conjunto de prueba...")
    trainer.evaluate(test_loader, split="test")

    if cfg.get("use_wandb", False):
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
