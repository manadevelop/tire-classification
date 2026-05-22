# Clasificación de Llantas Dañadas — Pregunta 1

**Curso:** Redes Neuronales y Aprendizaje Profundo  
**Dataset:** [Tire Texture Image Recognition — Kaggle](https://www.kaggle.com/datasets/jehanbhathena/tire-texture-image-recognition)

---

## Estructura del proyecto

```
tire_classification/
├── configs/
│   ├── train_scratch.yaml       # CustomCNN desde cero
│   ├── train_resnet50.yaml      # ResNet-50 fine-tuning
│   └── train_efficientnet.yaml  # EfficientNet-B3 fine-tuning
├── data/
│   ├── raw/                     # Dataset original de Kaggle (sin modificar)
│   └── processed/               # Splits train/val/test generados
│       ├── train/{good,cracked}/
│       ├── val/{good,cracked}/
│       └── test/{good,cracked}/
├── outputs/                     # Checkpoints y métricas por experimento
├── results/                     # Figuras, matrices de confusión, Grad-CAM
├── scripts/
│   ├── prepare_data.py          # División del dataset
│   └── evaluate.py              # Evaluación y visualización Grad-CAM
└── src/
    ├── train.py                 # Script principal de entrenamiento
    ├── models/
    │   ├── custom_cnn.py        # CNN desde cero
    │   └── transfer_model.py    # ResNet-50 / EfficientNet-B3
    ├── data/
    │   ├── dataset.py           # TireDataset (PyTorch)
    │   └── transforms.py        # Aumentaciones (minimal/standard/aggressive)
    ├── utils/
    │   ├── losses.py            # FocalLoss + compute_class_weights
    │   ├── metrics.py           # accuracy, precision, recall, F1, AUC-ROC
    │   ├── trainer.py           # Bucle entrenamiento + early stopping
    │   └── logger.py            # Logging estándar
    └── visualization/
        ├── gradcam.py           # Grad-CAM + análisis de fallos
        └── plots.py             # Curvas, matrices, radar, ROC
```

---

## Instalación

```bash
git clone https://github.com/TU_USUARIO/tire-classification.git
cd tire_classification
pip install -r requirements.txt
```

---

## Reproducción completa (un solo comando)

```bash
bash run_all.sh
```

O paso a paso:

### 1. Preparar el dataset

Descarga el dataset de Kaggle y colócalo en `data/raw/`:

```bash
# Requiere API key de Kaggle configurada en ~/.kaggle/kaggle.json
kaggle datasets download -d jehanbhathena/tire-texture-image-recognition -p data/raw --unzip
```

Dividir en train/val/test (70/15/15):

```bash
python scripts/prepare_data.py --raw_dir data/raw --out_dir data/processed
```

### 2. Entrenar los modelos

```bash
# CNN desde cero
python src/train.py --config configs/train_scratch.yaml

# ResNet-50 con fine-tuning
python src/train.py --config configs/train_resnet50.yaml

# EfficientNet-B3 con fine-tuning
python src/train.py --config configs/train_efficientnet.yaml
```

### 3. Evaluar y generar visualizaciones

```bash
# ResNet-50
python scripts/evaluate.py \
  --config configs/train_resnet50.yaml \
  --checkpoint outputs/resnet50_finetune_focal/best_model.pt \
  --out_dir results/resnet50

# EfficientNet-B3
python scripts/evaluate.py \
  --config configs/train_efficientnet.yaml \
  --checkpoint outputs/efficientnet_b3_finetune_focal/best_model.pt \
  --out_dir results/efficientnet

# CustomCNN
python scripts/evaluate.py \
  --config configs/train_scratch.yaml \
  --checkpoint outputs/custom_cnn_aggressive_focal/best_model.pt \
  --out_dir results/custom_cnn
```

---

## Entregables

| Entregable | Ubicación |
|---|---|
| Informe (PDF, formato NeurIPS) | `reports/informe_p1.pdf` |
| Checkpoints de mejores modelos | `outputs/*/best_model.pt` |
| Métricas en test | `results/*/metrics.json` |
| Matrices de confusión | `results/*/confusion_matrix.png` |
| Curvas ROC | `results/*/roc_curve.png` |
| Grad-CAM: clase good | `results/*/gradcam_correct_good.png` |
| Grad-CAM: clase cracked | `results/*/gradcam_correct_cracked.png` |
| Análisis de fallos | `results/*/failures/` |

---

## Detalles técnicos

| Aspecto | Decisión |
|---|---|
| Pérdida | Focal Loss (α=0.75, γ=2.0) vs. BCE con pesos |
| Desbalance | WeightedRandomSampler + Focal Loss |
| Augmentación | Albumentations: ElasticTransform, GridDistortion, CLAHE, CoarseDropout |
| Optimizador | AdamW + CosineAnnealingLR |
| Early stopping | Paciencia 12–15 épocas por F1 macro en validación |
| Interpretabilidad | Grad-CAM sobre última capa convolucional |
