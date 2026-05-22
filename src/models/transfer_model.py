"""
models/transfer_model.py — Modelos de transfer learning para clasificación de llantas.

Backbones soportados:
  - resnet50      : ResNet-50 preentrenado en ImageNet
  - efficientnet_b3: EfficientNet-B3 preentrenado en ImageNet

El clasificador final es reemplazado y entrenado desde cero.
Soporta Grad-CAM mediante hooks en la última capa convolucional del backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import (
    ResNet50_Weights,
    EfficientNet_B3_Weights,
)


class TransferModel(nn.Module):
    """
    Wrapper de transfer learning para clasificación binaria de llantas.

    Parámetros
    ----------
    backbone : str
        Nombre del backbone ('resnet50' o 'efficientnet_b3').
    num_classes : int
        Número de clases de salida.
    pretrained : bool
        Si True carga pesos ImageNet.
    freeze_backbone : bool
        Si True congela todos los parámetros del backbone excepto el clasificador.
    dropout_rate : float
        Dropout antes del último linear.
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.4,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes

        # ── Cargar backbone ────────────────────────────────────────────────
        if backbone == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            base = models.resnet50(weights=weights)
            in_features = base.fc.in_features
            self.feature_extractor = nn.Sequential(*list(base.children())[:-2])
            self._last_conv_layer = self.feature_extractor[-1][-1].conv3

        elif backbone == "efficientnet_b3":
            weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
            base = models.efficientnet_b3(weights=weights)
            in_features = base.classifier[1].in_features
            self.feature_extractor = base.features
            self._last_conv_layer = self.feature_extractor[-1][0]

        else:
            raise ValueError(f"Backbone desconocido: {backbone}")

        # ── Congelar backbone si se requiere ───────────────────────────────
        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        # ── Clasificador personalizado ─────────────────────────────────────
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(256, num_classes),
        )

        # Inicializar solo el clasificador
        self._init_classifier()

        # ── Estado interno para Grad-CAM ───────────────────────────────────
        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None

    def _init_classifier(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    # ── Hooks para Grad-CAM ────────────────────────────────────────────────
    def _fwd_hook(self, module, inp, out: torch.Tensor):
        self.activations = out.detach()

    def _bwd_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def register_gradcam_hooks(self):
        fwd = self._last_conv_layer.register_forward_hook(self._fwd_hook)
        bwd = self._last_conv_layer.register_backward_hook(self._bwd_hook)
        return fwd, bwd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

    def get_gradcam_map(
        self, x: torch.Tensor, class_idx: int | None = None
    ) -> torch.Tensor:
        """
        Calcula el mapa Grad-CAM para la imagen de entrada.

        Parámetros
        ----------
        x : torch.Tensor  shape (1, 3, H, W)
        class_idx : int | None  Clase objetivo; si None usa la predicha.

        Devuelve
        --------
        cam : torch.Tensor  shape (H, W), valores en [0, 1]
        """
        self.eval()
        fwd_h, bwd_h = self.register_gradcam_hooks()

        x = x.requires_grad_(True)
        logits = self.forward(x)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.zero_grad()
        logits[0, class_idx].backward()

        fwd_h.remove()
        bwd_h.remove()

        grads = self.gradients    # (1, C, H', W')  — o (C, H', W') si bwd retorna sin batch
        acts  = self.activations  # (1, C, H', W')

        # Normalizar dimensiones
        if grads.dim() == 3:
            grads = grads.unsqueeze(0)
        if acts.dim() == 3:
            acts = acts.unsqueeze(0)

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam, size=x.shape[2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.detach().cpu()

    def unfreeze_backbone(self, n_layers: int = -1):
        """
        Descongela las últimas n_layers capas del backbone.
        Si n_layers == -1 descongela todo el backbone.
        """
        params = list(self.feature_extractor.parameters())
        if n_layers == -1:
            for p in params:
                p.requires_grad = True
        else:
            for p in params[-n_layers:]:
                p.requires_grad = True
