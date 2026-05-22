"""
scripts/evaluate.py — Evaluación completa y análisis Grad-CAM de los modelos entrenados.

Genera:
  - Métricas completas en test (accuracy, precision, recall, F1, AUC-ROC)
  - Matriz de confusión
  - Curvas ROC comparativas
  - Grillas de visualización Grad-CAM (ambas clases)
  - Análisis de fallos: 5 peores ejemplos mal clasificados

Uso:
  python scripts/evaluate.py \
    --config configs/eval.yaml \
    --checkpoint outputs/exp_resnet50/best_model.pt \
    --out_dir results/resnet50
"""

import argparse
import json
import sys
import os
from pathlib import Path

import torch
import numpy as np
import yaml

# Añadir src/ al path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from models.custom_cnn import CustomCNN
from models.transfer_model import TransferModel
from data.dataset import TireDataset
from data.transforms import get_val_transforms
from utils.metrics import compute_metrics, print_metrics
from visualization.gradcam import GradCAMVisualizer
from visualization.plots import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_model_comparison,
)
from torch.utils.data import DataLoader


def load_model(cfg: dict, checkpoint_path: str, device: str):
    model_name = cfg["model"]["name"]
    num_classes = cfg["model"]["num_classes"]

    if model_name == "custom_cnn":
        model = CustomCNN(num_classes=num_classes)
    else:
        model = TransferModel(
            backbone=model_name,
            num_classes=num_classes,
            pretrained=False,
        )

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


@torch.no_grad()
def run_inference(model, loader, device) -> tuple:
    all_labels, all_preds, all_proba, all_paths = [], [], [], []

    for batch in loader:
        imgs, labels, paths = batch[0].to(device), batch[1], batch[2]
        logits = model(imgs)
        proba  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds  = logits.argmax(dim=1).cpu().numpy()

        all_labels.extend(labels.numpy())
        all_preds.extend(preds)
        all_proba.extend(proba)
        all_paths.extend(paths)

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_proba),
        all_paths,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out_dir",    type=str, default="results/eval")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Cargar modelo ──────────────────────────────────────────────────────
    model = load_model(cfg, args.checkpoint, device)
    print(f"Modelo cargado: {cfg['model']['name']} desde {args.checkpoint}")

    # ── Dataset de test ────────────────────────────────────────────────────
    img_size = cfg["data"]["img_size"]
    test_ds  = TireDataset(
        root=os.path.join(cfg["data"]["root"], "test"),
        transform=get_val_transforms(img_size),
    )
    class_names = test_ds.class_names
    print(f"Test: {len(test_ds)} imágenes | Clases: {class_names}")

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["training"]["batch_size"] * 2,
        shuffle=False,
        num_workers=2,
    )

    # ── Inferencia ─────────────────────────────────────────────────────────
    y_true, y_pred, y_proba, paths = run_inference(model, test_loader, device)

    # ── Métricas ───────────────────────────────────────────────────────────
    metrics = compute_metrics(y_true, y_pred, y_proba, class_names)
    print_metrics(metrics, prefix=cfg["model"]["name"])

    # Guardar JSON
    out_metrics = {k: v.tolist() if hasattr(v, "tolist") else v
                   for k, v in metrics.items()
                   if k != "classification_report"}
    out_metrics["classification_report"] = metrics["classification_report"]
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(out_metrics, f, indent=2, default=str)

    # ── Visualizaciones ────────────────────────────────────────────────────
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        class_names=class_names,
        model_name=cfg["model"]["name"],
        normalize=True,
        save_path=str(out_dir / "confusion_matrix.png"),
    )

    # Curva ROC del modelo actual
    plot_roc_curves(
        {cfg["model"]["name"]: (y_true, y_proba)},
        save_path=str(out_dir / "roc_curve.png"),
    )

    # ── Grad-CAM ───────────────────────────────────────────────────────────
    viz = GradCAMVisualizer(
        model=model,
        device=device,
        img_size=img_size,
        class_names=class_names,
    )

    # 4 ejemplos correctos de cada clase
    for cls_idx, cls_name in enumerate(class_names):
        cls_paths = [p for p, l in zip(paths, y_true) if l == cls_idx and
                     y_pred[list(paths).index(p) if p in paths else 0] == cls_idx][:4]
        if cls_paths:
            viz.visualize_grid(
                img_paths=cls_paths,
                true_labels=[cls_idx] * len(cls_paths),
                save_path=str(out_dir / f"gradcam_correct_{cls_name}.png"),
            )

    # Análisis de fallos
    viz.analyze_failures(
        img_paths=list(paths),
        true_labels=list(y_true),
        save_dir=str(out_dir / "failures"),
        n_examples=5,
    )

    print(f"\n✓ Resultados guardados en: {out_dir}")


if __name__ == "__main__":
    main()
