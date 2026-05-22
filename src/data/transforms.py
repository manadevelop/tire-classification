"""
data/transforms.py — Transformaciones y estrategias de aumento de datos.

Estrategias disponibles:
  - 'minimal'   : solo resize + normalización (línea base)
  - 'standard'  : flip + rotación + color jitter moderado
  - 'aggressive': aumento orientado a texturas (ElasticTransform, GridDistortion,
                  CoarseDropout, RandomGamma) via albumentations

Las transformaciones de validación/test solo aplican resize y normalización.
"""

from typing import Callable

import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image


# Estadísticas ImageNet (usadas para modelos preentrenados y también para CNN desde cero)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# ── Wrapper para integrar albumentations con PyTorch DataLoader ────────────
class AlbumentationsWrapper:
    """Aplica un pipeline de albumentations a una imagen PIL."""

    def __init__(self, transform: A.Compose):
        self.transform = transform

    def __call__(self, img) -> "torch.Tensor":
        if isinstance(img, Image.Image):
            img = np.array(img)
        result = self.transform(image=img)
        return result["image"]


# ── Transformaciones de entrenamiento ─────────────────────────────────────
def get_train_transforms(
    img_size: int = 224,
    augmentation_strategy: str = "standard",
) -> Callable:
    """
    Devuelve las transformaciones de entrenamiento según la estrategia elegida.

    Parámetros
    ----------
    img_size : int
        Tamaño al que se redimensionan las imágenes (cuadrado).
    augmentation_strategy : str
        'minimal' | 'standard' | 'aggressive'
    """

    if augmentation_strategy == "minimal":
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    elif augmentation_strategy == "standard":
        return T.Compose([
            T.Resize((img_size + 32, img_size + 32)),
            T.RandomCrop(img_size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.3),
            T.RandomRotation(degrees=20),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            T.RandomGrayscale(p=0.05),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            T.RandomErasing(p=0.2, scale=(0.02, 0.1)),
        ])

    elif augmentation_strategy == "aggressive":
        # Aumento orientado a texturas con albumentations
        pipeline = A.Compose([
            A.Resize(img_size + 32, img_size + 32),
            A.RandomCrop(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=25, p=0.6),
            # Perturbaciones de textura — relevantes para desgaste de llantas
            A.ElasticTransform(alpha=1.0, sigma=50, p=0.3),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
            A.OpticalDistortion(distort_limit=0.2, p=0.2),
            # Color y brillo — simula condiciones de iluminación industrial
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.4),
            A.RandomGamma(gamma_limit=(70, 130), p=0.4),
            A.CLAHE(clip_limit=4.0, p=0.3),
            # Ruido y oclusión parcial
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.CoarseDropout(
                max_holes=8, max_height=16, max_width=16,
                min_holes=1, fill_value=0, p=0.3,
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
        return AlbumentationsWrapper(pipeline)

    else:
        raise ValueError(
            f"Estrategia de aumento desconocida: '{augmentation_strategy}'. "
            "Usa 'minimal', 'standard' o 'aggressive'."
        )


# ── Transformaciones de validación / test ─────────────────────────────────
def get_val_transforms(img_size: int = 224) -> Callable:
    """
    Transformaciones deterministas para validación y test.
    Solo resize central + normalización.
    """
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
