"""
visualization/plots.py — Gráficas de resultados para el informe.

Genera:
  - Curvas de pérdida y métricas por época
  - Matrices de confusión comentadas
  - Curvas ROC comparativas entre modelos
  - Análisis de desbalance de clases
  - Comparación de modelos (ablation study)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# Paleta corporativa del proyecto
PALETTE = {
    "custom_cnn":     "#2196F3",
    "resnet50":       "#F44336",
    "efficientnet":   "#4CAF50",
    "baseline":       "#9E9E9E",
}


def plot_training_curves(
    history: Dict[str, List[float]],
    model_name: str,
    save_path: Optional[str] = None,
):
    """Pérdida y F1 por época en dos paneles."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    # ── Pérdida ────────────────────────────────────────────────────────────
    ax1.plot(epochs, history["train_loss"], label="Train", color="#2196F3")
    ax1.plot(epochs, history["val_loss"],   label="Val",   color="#F44336")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Pérdida")
    ax1.set_title(f"{model_name} — Curva de pérdida")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── F1 ────────────────────────────────────────────────────────────────
    ax2.plot(epochs, history["val_f1"],       label="F1 Val",  color="#4CAF50")
    ax2.plot(epochs, history["val_accuracy"], label="Acc Val", color="#FF9800")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Métrica")
    ax2.set_title(f"{model_name} — F1 y Exactitud")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    model_name: str,
    normalize: bool = True,
    save_path: Optional[str] = None,
):
    """Matriz de confusión con anotaciones de conteo y porcentaje."""
    if normalize:
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    else:
        cm_norm = cm

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        ax=ax,
    )

    # Superponer conteos absolutos
    if normalize:
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j + 0.5, i + 0.75,
                    f"n={cm[i, j]}",
                    ha="center", va="center", fontsize=8, color="gray"
                )

    ax.set_xlabel("Clase predicha", fontsize=11)
    ax.set_ylabel("Clase real",     fontsize=11)
    ax.set_title(f"Matriz de confusión — {model_name}", fontsize=12)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_roc_curves(
    roc_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path: Optional[str] = None,
):
    """
    Curvas ROC comparativas.

    Parámetros
    ----------
    roc_data : dict con clave = nombre_modelo, valor = (y_true, y_score)
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    colors = list(PALETTE.values())
    for i, (name, (y_true, y_score)) in enumerate(roc_data.items()):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc     = auc(fpr, tpr)
        c = colors[i % len(colors)]
        ax.plot(fpr, tpr, color=c, lw=2, label=f"{name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Aleatorio (AUC=0.500)")
    ax.set_xlabel("Tasa de Falsos Positivos (FPR)", fontsize=11)
    ax.set_ylabel("Tasa de Verdaderos Positivos (TPR)", fontsize=11)
    ax.set_title("Comparación de curvas ROC", fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_class_distribution(
    distributions: Dict[str, Dict[str, int]],
    save_path: Optional[str] = None,
):
    """
    Distribución de clases por split (train/val/test).

    Parámetros
    ----------
    distributions : {'train': {'good': N, 'cracked': M}, 'val': ..., 'test': ...}
    """
    splits = list(distributions.keys())
    classes = list(next(iter(distributions.values())).keys())
    x = np.arange(len(splits))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, cls in enumerate(classes):
        counts = [distributions[s][cls] for s in splits]
        bars = ax.bar(x + i * width, counts, width, label=cls)
        ax.bar_label(bars, padding=3, fontsize=9)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(splits)
    ax.set_ylabel("Número de imágenes")
    ax.set_title("Distribución de clases por split")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_ablation_study(
    results: Dict[str, Dict[str, float]],
    metric: str = "f1_macro",
    title: str = "Estudio de ablación",
    save_path: Optional[str] = None,
):
    """
    Gráfica de barras horizontal para comparar configuraciones en el estudio de ablación.

    Parámetros
    ----------
    results : {'Config A': {'f1_macro': 0.87, 'auc_roc': 0.92}, ...}
    metric  : métrica a visualizar
    """
    names  = list(results.keys())
    values = [results[n][metric] for n in names]

    fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.6)))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(names)))
    bars = ax.barh(names, values, color=colors, edgecolor="white")

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=9
        )

    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = ("accuracy", "precision", "recall", "f1_macro", "auc_roc"),
    save_path: Optional[str] = None,
):
    """Radar chart comparativo entre modelos."""
    n = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    colors = list(PALETTE.values())

    for i, (model_name, m) in enumerate(results.items()):
        vals  = [m.get(metric, 0) for metric in metrics]
        vals += vals[:1]
        c = colors[i % len(colors)]
        ax.plot(angles, vals, "o-", color=c, linewidth=2, label=model_name)
        ax.fill(angles, vals, color=c, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        [m.replace("_", "\n") for m in metrics], size=9
    )
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("Comparación de modelos (radar)", pad=15, fontsize=12)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    _save_or_show(fig, save_path)


def _save_or_show(fig: plt.Figure, save_path: Optional[str]):
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Guardado: {save_path}")
    else:
        plt.show()
