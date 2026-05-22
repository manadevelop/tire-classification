"""
models/custom_cnn.py — CNN diseñada desde cero para clasificación de llantas.

Arquitectura:
  4 bloques convolucionales con BatchNorm + ReLU + MaxPool + Dropout espacial
  Clasificador con capas densas y Dropout.
  Soporte para extracción de mapas de activación (Grad-CAM).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Bloque convolucional: Conv → BN → ReLU → Conv → BN → ReLU → MaxPool → Dropout2d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_rate: float = 0.1,
        use_pool: bool = True,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2, 2) if use_pool else nn.Identity()
        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class CustomCNN(nn.Module):
    """
    CNN entrenada desde cero para clasificación binaria de llantas.

    Parámetros
    ----------
    num_classes : int
        Número de clases de salida (2 para clasificación binaria).
    dropout_rate : float
        Tasa de dropout en el clasificador.
    base_channels : int
        Número de filtros en el primer bloque; se duplica en cada bloque.
    """

    def __init__(
        self,
        num_classes: int = 2,
        dropout_rate: float = 0.5,
        base_channels: int = 32,
    ):
        super().__init__()
        c = base_channels

        # ── Feature extractor ──────────────────────────────────────────────
        self.features = nn.Sequential(
            ConvBlock(3, c, dropout_rate=0.05),           # 224→112
            ConvBlock(c, c * 2, dropout_rate=0.05),        # 112→56
            ConvBlock(c * 2, c * 4, dropout_rate=0.1),     # 56→28
            ConvBlock(c * 4, c * 8, dropout_rate=0.1),     # 28→14
        )

        # Capa de activación final (usada por Grad-CAM)
        self.last_conv = nn.Sequential(
            nn.Conv2d(c * 8, c * 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c * 8),
            nn.ReLU(inplace=True),
        )

        # ── Clasificador ───────────────────────────────────────────────────
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(64, num_classes),
        )

        # Inicialización de pesos
        self._init_weights()

        # Hook para Grad-CAM
        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    # ── Hooks para Grad-CAM ────────────────────────────────────────────────
    def _save_gradients(self, grad: torch.Tensor):
        self.gradients = grad

    def _save_activations(self, module, inp, out: torch.Tensor):
        self.activations = out

    def register_gradcam_hooks(self):
        """Registra hooks en la última capa convolucional."""
        handle = self.last_conv[-3].register_forward_hook(self._save_activations)
        return handle

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.last_conv(x)

        # Registrar gradientes si es necesario
        if x.requires_grad:
            x.register_hook(self._save_gradients)

        x = self.global_pool(x)
        x = self.classifier(x)
        return x

    def get_gradcam_map(
        self, x: torch.Tensor, class_idx: int | None = None
    ) -> torch.Tensor:
        """
        Calcula el mapa de calor Grad-CAM para la imagen de entrada.

        Parámetros
        ----------
        x : torch.Tensor  shape (1, 3, H, W)
        class_idx : int | None
            Clase objetivo; si es None, usa la clase predicha.

        Devuelve
        --------
        cam : torch.Tensor  shape (H, W)  valores en [0, 1]
        """
        self.eval()
        x = x.requires_grad_(True)

        handle = self.register_gradcam_hooks()
        logits = self.forward(x)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.zero_grad()
        logits[0, class_idx].backward()
        handle.remove()

        # Pesos α_k: media global de los gradientes sobre el mapa espacial
        grads = self.gradients        # (1, C, H', W')
        acts  = self.activations      # (1, C, H', W')

        weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, H', W')
        cam = F.relu(cam)

        # Normalizar y redimensionar a tamaño de imagen original
        cam = F.interpolate(
            cam, size=x.shape[2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.detach()
