"""
utils/losses.py — Funciones de pérdida para clasificación con desbalance.

Implementa:
  - FocalLoss: versión multiclase con reducción configurable.
  - compute_class_weights: pesos inversamente proporcionales a la frecuencia de clase.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List


class FocalLoss(nn.Module):
    """
    Focal Loss para clasificación con desbalance de clases.

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Referencia:
        Lin et al., "Focal Loss for Dense Object Detection," ICCV 2017.

    Parámetros
    ----------
    alpha : float | List[float]
        Factor de ponderación por clase.
        - float: mismo alpha para todas las clases.
        - List: un alpha por clase [α_0, α_1, ...].
    gamma : float
        Exponente de modulación (suaviza la pérdida en ejemplos fáciles).
    reduction : str
        'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        alpha: float | List[float] = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([alpha] * 2)
        else:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parámetros
        ----------
        inputs  : (N, C) logits sin softmax
        targets : (N,)   etiquetas enteras

        Devuelve
        --------
        loss : scalar
        """
        alpha = self.alpha.to(inputs.device)
        log_probs = F.log_softmax(inputs, dim=1)     # (N, C)
        probs     = torch.exp(log_probs)              # (N, C)

        # Extraer probabilidad y log-probabilidad de la clase objetivo
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (N,)
        pt     = probs.gather(1, targets.unsqueeze(1)).squeeze(1)       # (N,)
        alpha_t = alpha.gather(0, targets)                               # (N,)

        focal_weight = (1.0 - pt) ** self.gamma
        loss = -alpha_t * focal_weight * log_pt

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss  # 'none'


def compute_class_weights(labels: List[int]) -> torch.Tensor:
    """
    Calcula pesos de clase inversamente proporcionales a su frecuencia.

    w_c = N / (C * n_c)

    donde N es el total de muestras, C el número de clases y n_c el
    número de muestras de la clase c.

    Parámetros
    ----------
    labels : List[int]  Lista de etiquetas enteras.

    Devuelve
    --------
    weights : torch.Tensor  shape (C,)
    """
    labels_arr = np.array(labels)
    n = len(labels_arr)
    classes = np.unique(labels_arr)
    c = len(classes)
    weights = np.zeros(c, dtype=np.float32)

    for cls in classes:
        n_c = np.sum(labels_arr == cls)
        weights[cls] = n / (c * n_c)

    return torch.tensor(weights, dtype=torch.float32)
