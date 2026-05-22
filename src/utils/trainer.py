"""
utils/trainer.py — Bucle de entrenamiento, checkpointing y evaluación.

Gestiona:
  - Entrenamiento epoch-by-epoch con barra de progreso tqdm
  - Validación con early stopping por F1 macro
  - Guardado del mejor modelo (checkpoint)
  - Evaluación completa en test con métricas y matriz de confusión
  - Registro de curvas de pérdida y métricas en archivos JSON
"""

import json
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.metrics import compute_metrics, print_metrics


class Trainer:
    """
    Entrenador genérico para modelos PyTorch.

    Parámetros
    ----------
    model       : modelo nn.Module
    criterion   : función de pérdida
    optimizer   : optimizador
    scheduler   : scheduler de tasa de aprendizaje
    device      : 'cuda' | 'cpu'
    output_dir  : carpeta donde guardar checkpoints y logs
    cfg         : configuración completa del experimento
    logger      : logger estándar de Python
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer,
        scheduler,
        device: str,
        output_dir: Path,
        cfg: Dict[str, Any],
        logger,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.cfg = cfg
        self.logger = logger

        self.best_f1 = 0.0
        self.patience_counter = 0
        self.patience = cfg.get("early_stopping_patience", 15)
        self.history: Dict[str, list] = {
            "train_loss": [], "val_loss": [],
            "val_accuracy": [], "val_f1": [], "val_auc": [],
        }

    # ─────────────────────────────────────────────────────────────────────
    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in tqdm(loader, desc="  Train", leave=False, ncols=80):
            # Dataset devuelve (img, label, path) — ignoramos path
            imgs, labels = batch[0].to(self.device), batch[1].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(imgs)
            loss   = self.criterion(logits, labels)
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            self.optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        return total_loss / max(n_batches, 1)

    # ─────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader) -> Dict:
        self.model.eval()
        total_loss = 0.0
        all_labels, all_preds, all_proba = [], [], []

        for batch in tqdm(loader, desc="  Val  ", leave=False, ncols=80):
            imgs, labels = batch[0].to(self.device), batch[1].to(self.device)
            logits = self.model(imgs)
            loss   = self.criterion(logits, labels)
            total_loss += loss.item()

            proba = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
            all_proba.extend(proba)

        metrics = compute_metrics(
            y_true=all_labels,
            y_pred=all_preds,
            y_proba=all_proba,
        )
        metrics["loss"] = total_loss / max(len(loader), 1)
        return metrics

    # ─────────────────────────────────────────────────────────────────────
    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        self.logger.info(f"Iniciando entrenamiento por {epochs} épocas...")
        t0 = time.time()

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(train_loader)
            val_metrics = self._val_epoch(val_loader)

            self.scheduler.step()

            val_f1  = val_metrics["f1_macro"]
            val_auc = val_metrics.get("auc_roc", float("nan"))
            val_acc = val_metrics["accuracy"]
            val_loss = val_metrics["loss"]

            # Registro
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_acc)
            self.history["val_f1"].append(val_f1)
            self.history["val_auc"].append(val_auc)

            self.logger.info(
                f"Epoch {epoch:03d}/{epochs} | "
                f"TrainLoss={train_loss:.4f} | "
                f"ValLoss={val_loss:.4f} | "
                f"Acc={val_acc:.4f} | F1={val_f1:.4f} | AUC={val_auc:.4f}"
            )

            # Early stopping y checkpointing
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                self.patience_counter = 0
                self._save_checkpoint("best_model.pt")
                self.logger.info(f"  ✓ Nuevo mejor modelo (F1={val_f1:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    self.logger.info(
                        f"Early stopping activado en época {epoch} "
                        f"(sin mejora en {self.patience} épocas)."
                    )
                    break

        elapsed = (time.time() - t0) / 60
        self.logger.info(f"Entrenamiento completado en {elapsed:.1f} min.")
        self._save_history()

    # ─────────────────────────────────────────────────────────────────────
    def evaluate(self, loader: DataLoader, split: str = "test"):
        """Carga el mejor modelo y evalúa en un split dado."""
        ckpt_path = self.output_dir / "best_model.pt"
        if ckpt_path.exists():
            self.model.load_state_dict(
                torch.load(ckpt_path, map_location=self.device)
            )
            self.logger.info(f"Cargado: {ckpt_path}")

        metrics = self._val_epoch(loader)
        self.logger.info(f"\n=== Resultados en {split.upper()} ===")
        print_metrics(metrics, prefix=split)

        # Guardar métricas en JSON
        out = {k: v.tolist() if hasattr(v, "tolist") else v
               for k, v in metrics.items()
               if k != "classification_report"}
        out["classification_report"] = metrics.get("classification_report", "")

        out_path = self.output_dir / f"{split}_metrics.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2, default=str)
        self.logger.info(f"Métricas guardadas en {out_path}")
        return metrics

    # ─────────────────────────────────────────────────────────────────────
    def _save_checkpoint(self, name: str = "best_model.pt"):
        path = self.output_dir / name
        torch.save(self.model.state_dict(), path)

    def _save_history(self):
        path = self.output_dir / "training_history.json"
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
