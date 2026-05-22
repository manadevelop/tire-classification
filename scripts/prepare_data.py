"""
scripts/prepare_data.py — Prepara el dataset Kaggle para entrenamiento.

Uso:
  python scripts/prepare_data.py \
    --raw_dir data/raw \
    --out_dir data/processed \
    --train_ratio 0.70 \
    --val_ratio   0.15

Estructura de entrada esperada (Kaggle):
  data/raw/
    good/          o  train/good/
    cracked/          train/cracked/

Estructura de salida generada:
  data/processed/
    train/ {good/, cracked/}
    val/   {good/, cracked/}
    test/  {good/, cracked/}
    split_stats.json
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(root: Path) -> Dict[str, List[Path]]:
    """Recolecta imágenes por clase desde un directorio raíz."""
    class_images: Dict[str, List[Path]] = {}

    for cls_dir in sorted(root.iterdir()):
        if not cls_dir.is_dir() or cls_dir.name.startswith("."):
            continue
        imgs = [
            p for p in cls_dir.iterdir()
            if p.suffix.lower() in IMG_EXTENSIONS
        ]
        if imgs:
            class_images[cls_dir.name] = imgs

    return class_images


def stratified_split(
    images: List[Path],
    train_ratio: float,
    val_ratio: float,
    seed: int = 42,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """División estratificada train/val/test."""
    random.seed(seed)
    imgs = images.copy()
    random.shuffle(imgs)

    n = len(imgs)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train = imgs[:n_train]
    val   = imgs[n_train:n_train + n_val]
    test  = imgs[n_train + n_val:]
    return train, val, test


def copy_split(
    images: List[Path],
    dest_dir: Path,
    cls_name: str,
):
    cls_dir = dest_dir / cls_name
    cls_dir.mkdir(parents=True, exist_ok=True)
    for img in images:
        shutil.copy2(img, cls_dir / img.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir",     type=str, default="data/raw")
    parser.add_argument("--out_dir",     type=str, default="data/processed")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio",   type=float, default=0.15)
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    # Si el dataset tiene subcarpeta train/ usarla como raíz
    if (raw_dir / "train").exists():
        raw_dir = raw_dir / "train"

    print(f"Leyendo imágenes desde: {raw_dir}")
    class_images = collect_images(raw_dir)

    if not class_images:
        raise RuntimeError(
            f"No se encontraron imágenes en '{raw_dir}'. "
            "Verifica la estructura del dataset."
        )

    stats: Dict = {"splits": {}, "class_distribution": {}}

    for cls_name, imgs in class_images.items():
        print(f"  [{cls_name}] {len(imgs)} imágenes encontradas")
        train, val, test = stratified_split(
            imgs, args.train_ratio, args.val_ratio, args.seed
        )

        copy_split(train, out_dir / "train", cls_name)
        copy_split(val,   out_dir / "val",   cls_name)
        copy_split(test,  out_dir / "test",  cls_name)

        stats["class_distribution"][cls_name] = len(imgs)
        stats["splits"][cls_name] = {
            "train": len(train),
            "val":   len(val),
            "test":  len(test),
        }

    # Imbalance ratio
    counts = list(stats["class_distribution"].values())
    stats["imbalance_ratio"] = max(counts) / min(counts) if min(counts) > 0 else float("inf")

    stats_path = out_dir / "split_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nDataset procesado en: {out_dir}")
    print(f"Ratio de desbalance: {stats['imbalance_ratio']:.2f}")
    print(f"Estadísticas guardadas en: {stats_path}")


if __name__ == "__main__":
    main()
