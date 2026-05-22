"""
visualization/gradcam.py — Visualización Grad-CAM para interpretabilidad de modelos.

Genera:
  - Mapas de calor Grad-CAM superpuestos sobre la imagen original
  - Grillas de visualización para múltiples imágenes (ambas clases)
  - Análisis cualitativo de fallos (falsos positivos y falsos negativos)

Referencia:
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
    Gradient-based Localization," ICCV 2017.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn as nn
from PIL import Image
import cv2

from data.transforms import get_val_transforms, IMAGENET_MEAN, IMAGENET_STD


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convierte tensor normalizado ImageNet a imagen uint8 HxWx3."""
    mean = np.array(IMAGENET_MEAN)
    std  = np.array(IMAGENET_STD)
    img  = tensor.permute(1, 2, 0).cpu().numpy()
    img  = img * std + mean
    img  = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def overlay_heatmap(
    img_np: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Superpone el mapa de calor Grad-CAM sobre la imagen original.

    Parámetros
    ----------
    img_np   : np.ndarray (H, W, 3) uint8 RGB
    cam      : np.ndarray (H, W)    float [0, 1]
    alpha    : opacidad del heatmap
    colormap : colormap OpenCV

    Devuelve
    --------
    overlay : np.ndarray (H, W, 3) uint8 RGB
    """
    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Redimensionar a tamaño de imagen si es necesario
    if heatmap.shape[:2] != img_np.shape[:2]:
        heatmap = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))

    overlay = (alpha * heatmap + (1 - alpha) * img_np).astype(np.uint8)
    return overlay


class GradCAMVisualizer:
    """
    Visualizador Grad-CAM que trabaja con CustomCNN y TransferModel.

    Parámetros
    ----------
    model     : modelo PyTorch con método get_gradcam_map()
    device    : 'cuda' | 'cpu'
    img_size  : tamaño de imagen esperado por el modelo
    class_names : nombres de clase por índice
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        img_size: int = 224,
        class_names: Optional[List[str]] = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.img_size = img_size
        self.transform = get_val_transforms(img_size)
        self.class_names = class_names or ["good", "cracked"]

    def _prepare_tensor(self, img_path: str) -> Tuple[torch.Tensor, np.ndarray]:
        """Carga imagen PIL y la convierte a tensor + numpy original."""
        img_pil = Image.open(img_path).convert("RGB")
        img_np  = np.array(img_pil.resize((self.img_size, self.img_size)))
        tensor  = self.transform(img_pil).unsqueeze(0).to(self.device)
        return tensor, img_np

    def predict_and_explain(
        self,
        img_path: str,
        class_idx: Optional[int] = None,
    ) -> Tuple[int, float, np.ndarray]:
        """
        Devuelve (clase_predicha, confianza, mapa_cam).
        """
        tensor, img_np = self._prepare_tensor(img_path)
        cam = self.model.get_gradcam_map(tensor, class_idx)

        with torch.no_grad():
            logits = self.model(tensor)
            proba  = torch.softmax(logits, dim=1).squeeze().cpu()

        pred_cls  = proba.argmax().item()
        pred_conf = proba[pred_cls].item()
        return pred_cls, pred_conf, cam.numpy(), img_np

    # ─────────────────────────────────────────────────────────────────────
    def visualize_single(
        self,
        img_path: str,
        save_path: Optional[str] = None,
        true_label: Optional[int] = None,
    ):
        """Visualiza Grad-CAM para una sola imagen."""
        pred_cls, conf, cam, img_np = self.predict_and_explain(img_path)
        overlay = overlay_heatmap(img_np, cam)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img_np)
        axes[0].set_title("Imagen original")
        axes[0].axis("off")

        axes[1].imshow(cam, cmap="jet")
        axes[1].set_title("Mapa de activación (Grad-CAM)")
        axes[1].axis("off")

        axes[2].imshow(overlay)
        title = f"Pred: {self.class_names[pred_cls]} ({conf:.2%})"
        if true_label is not None:
            status = "✓" if pred_cls == true_label else "✗"
            title += f"\nReal: {self.class_names[true_label]} {status}"
        axes[2].set_title(title)
        axes[2].axis("off")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    # ─────────────────────────────────────────────────────────────────────
    def visualize_grid(
        self,
        img_paths: List[str],
        true_labels: Optional[List[int]] = None,
        save_path: Optional[str] = None,
        n_cols: int = 4,
    ):
        """Grilla de visualizaciones Grad-CAM para múltiples imágenes."""
        n = len(img_paths)
        n_rows = (n + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(n_cols * 6, n_rows * 4))
        axes = axes.flatten()

        for i, img_path in enumerate(img_paths):
            pred_cls, conf, cam, img_np = self.predict_and_explain(img_path)
            overlay = overlay_heatmap(img_np, cam)

            ax_img = axes[i * 2]
            ax_cam = axes[i * 2 + 1]

            ax_img.imshow(img_np)
            ax_img.axis("off")
            title = f"Real: {self.class_names[true_labels[i]]}" if true_labels else ""
            ax_img.set_title(title, fontsize=8)

            ax_cam.imshow(overlay)
            ax_cam.axis("off")
            status = ""
            if true_labels:
                ok = "✓" if pred_cls == true_labels[i] else "✗"
                status = f"{ok} Pred: {self.class_names[pred_cls]} ({conf:.1%})"
            ax_cam.set_title(status, fontsize=8)

        # Ocultar ejes vacíos
        for j in range(i * 2 + 2, len(axes)):
            axes[j].axis("off")

        plt.suptitle("Grad-CAM: Activaciones del modelo", fontsize=13, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    # ─────────────────────────────────────────────────────────────────────
    def analyze_failures(
        self,
        img_paths: List[str],
        true_labels: List[int],
        save_dir: str,
        n_examples: int = 5,
    ):
        """
        Identifica y visualiza los N peores ejemplos mal clasificados.
        Guarda imágenes individuales con Grad-CAM en save_dir.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        failures = []
        for path, true_lbl in zip(img_paths, true_labels):
            pred_cls, conf, cam, img_np = self.predict_and_explain(path)
            if pred_cls != true_lbl:
                failures.append(
                    (conf, path, true_lbl, pred_cls, cam, img_np)
                )

        # Ordenar por confianza descendente (los peores primero)
        failures.sort(key=lambda x: -x[0])
        failures = failures[:n_examples]

        print(f"\n{'='*60}")
        print(f"  Análisis de fallos — {len(failures)} ejemplos")
        print(f"{'='*60}")

        for rank, (conf, path, true_lbl, pred_cls, cam, img_np) in enumerate(failures, 1):
            overlay = overlay_heatmap(img_np, cam)
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            axes[0].imshow(img_np); axes[0].axis("off")
            axes[0].set_title(f"Real: {self.class_names[true_lbl]}")

            axes[1].imshow(cam, cmap="jet"); axes[1].axis("off")
            axes[1].set_title("Grad-CAM")

            axes[2].imshow(overlay); axes[2].axis("off")
            axes[2].set_title(
                f"Pred: {self.class_names[pred_cls]} ({conf:.1%})\n"
                f"[Falso {'Positivo' if pred_cls == 1 else 'Negativo'}]"
            )

            fig.suptitle(f"Fallo #{rank}: {Path(path).name}", fontsize=10)
            plt.tight_layout()
            out = save_dir / f"failure_{rank:02d}_{Path(path).stem}.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            plt.close()

            print(f"  [{rank}] {Path(path).name}")
            print(f"      Real: {self.class_names[true_lbl]}  →  "
                  f"Pred: {self.class_names[pred_cls]}  (conf={conf:.1%})")

        print(f"\nImágenes guardadas en: {save_dir}\n")
        return failures
