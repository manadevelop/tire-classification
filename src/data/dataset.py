"""
data/dataset.py — Dataset para clasificación de llantas.

Estructura esperada del directorio raíz:
  root/
    good/          ← imágenes de llantas en buen estado
    cracked/       ← imágenes de llantas dañadas/quebradas

Soporta carga de imágenes en RGB, aplicación de transformaciones y
devuelve (imagen, etiqueta, ruta) para facilitar el análisis de errores.
"""

import os
from pathlib import Path
from typing import Callable, Optional, List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


CLASS_NAMES = ["good", "cracked"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


class TireDataset(Dataset):
    """
    Dataset de imágenes de llantas.

    Parámetros
    ----------
    root : str | Path
        Carpeta raíz con subcarpetas por clase.
    transform : Callable | None
        Transformaciones torchvision a aplicar a la imagen.
    class_names : List[str] | None
        Lista de nombres de clase; si es None infiere de las subcarpetas.
    """

    def __init__(
        self,
        root: str | Path,
        transform: Optional[Callable] = None,
        class_names: Optional[List[str]] = None,
    ):
        self.root = Path(root)
        self.transform = transform

        # Detectar clases desde subcarpetas
        if class_names is None:
            found = sorted(
                d.name for d in self.root.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            )
            self.class_names = found
        else:
            self.class_names = class_names

        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        # Construir lista (ruta, etiqueta)
        self.samples: List[Tuple[Path, int]] = []
        for cls in self.class_names:
            cls_dir = self.root / cls
            if not cls_dir.exists():
                continue
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
                for p in cls_dir.glob(ext):
                    self.samples.append((p, self.class_to_idx[cls]))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No se encontraron imágenes en '{root}'. "
                f"Verifica que las subcarpetas sean: {self.class_names}"
            )

    # ── Estadísticas del dataset ───────────────────────────────────────────
    def class_distribution(self) -> dict:
        """Devuelve el conteo de muestras por clase."""
        counts = {c: 0 for c in self.class_names}
        for _, label in self.samples:
            counts[self.class_names[label]] += 1
        return counts

    def imbalance_ratio(self) -> float:
        """Ratio de desbalance: clase_mayoritaria / clase_minoritaria."""
        dist = self.class_distribution()
        counts = list(dist.values())
        return max(counts) / min(counts)

    # ── Interfaz Dataset ──────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, label, str(path)

    def get_labels(self) -> List[int]:
        """Devuelve lista de etiquetas (útil para WeightedRandomSampler)."""
        return [label for _, label in self.samples]
