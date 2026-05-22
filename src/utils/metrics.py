"""
utils/metrics.py — Métricas de evaluación para clasificación de llantas.

Calcula:
  - Exactitud, Precisión, Recall, F1-score (macro y por clase)
  - AUC-ROC
  - Matriz de confusión
  - Resumen en formato de diccionario para logging
"""

from typing import List, Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def compute_metrics(
    y_true: List[int] | np.ndarray,
    y_pred: List[int] | np.ndarray,
    y_proba: List[float] | np.ndarray | None = None,
    class_names: List[str] | None = None,
) -> Dict[str, float | np.ndarray]:
    """
    Calcula métricas completas para clasificación binaria.

    Parámetros
    ----------
    y_true   : etiquetas reales
    y_pred   : predicciones del modelo (clase dura)
    y_proba  : probabilidad de la clase positiva (necesaria para AUC-ROC)
    class_names : nombres de las clases para el reporte

    Devuelve
    --------
    metrics : dict con todas las métricas
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics: Dict = {}

    metrics["accuracy"]  = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["recall"]    = recall_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["f1_macro"]  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["f1_weighted"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Por clase
    metrics["precision_per_class"] = precision_score(
        y_true, y_pred, average=None, zero_division=0
    )
    metrics["recall_per_class"] = recall_score(
        y_true, y_pred, average=None, zero_division=0
    )
    metrics["f1_per_class"] = f1_score(
        y_true, y_pred, average=None, zero_division=0
    )

    # AUC-ROC (requiere probabilidades)
    if y_proba is not None:
        y_proba = np.asarray(y_proba)
        try:
            metrics["auc_roc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics["auc_roc"] = float("nan")

    # Matriz de confusión
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)

    # Reporte textual de sklearn
    names = class_names or [str(i) for i in np.unique(y_true)]
    metrics["classification_report"] = classification_report(
        y_true, y_pred, target_names=names, zero_division=0
    )

    return metrics


def compute_class_weights(labels: List[int]) -> torch.Tensor:
    """Importado desde losses.py para evitar duplicación."""
    from utils.losses import compute_class_weights as _cw
    return _cw(labels)


def print_metrics(metrics: Dict, prefix: str = "") -> None:
    """Imprime las métricas principales de forma legible."""
    p = f"[{prefix}] " if prefix else ""
    print(f"{p}Accuracy : {metrics['accuracy']:.4f}")
    print(f"{p}Precision: {metrics['precision']:.4f}")
    print(f"{p}Recall   : {metrics['recall']:.4f}")
    print(f"{p}F1 Macro : {metrics['f1_macro']:.4f}")
    if "auc_roc" in metrics:
        print(f"{p}AUC-ROC  : {metrics['auc_roc']:.4f}")
    print(f"\n{metrics['classification_report']}")
    print(f"Matriz de confusión:\n{metrics['confusion_matrix']}")
