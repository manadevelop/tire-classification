# Clasificación de Llantas Dañadas mediante Reconocimiento de Textura
## Pregunta 1 — Examen Parcial

**Curso:** Redes Neuronales y Aprendizaje Profundo  
**Docente:** Ph.D. Aldo Camargo  
**Universidad Nacional de Ingeniería — Maestría en Inteligencia Artificial**

**Integrantes:**
- Victor Fernando Montes Jaramillo
- Alex Celestino León Pacheco
- Edwin Jhon Minchán Ramos
- Marco Antonio Nina Aguilar

**Dataset:** [Tire Texture Image Recognition — Kaggle](https://www.kaggle.com/datasets/jehanbhathena/tire-texture-image-recognition)  
**Repositorio:** https://github.com/manadevelop/tire-classification

---

## ⚡ Reproducción completa (un solo comando)

```bash
bash run_all.sh
```

Este comando ejecuta automáticamente:
1. Verificación e instalación de dependencias
2. Descarga del dataset desde Kaggle
3. Preparación y división del dataset (70/15/15)
4. Entrenamiento de CustomCNN desde cero
5. Entrenamiento de ResNet-50 con fine-tuning
6. Entrenamiento de EfficientNet-B3 con fine-tuning
7. Estudio de ablación (minimal / standard / aggressive)
8. Evaluación completa con métricas, Grad-CAM y análisis de fallos

> **Recomendación:** Ejecutar en Google Colab con GPU T4 o superior.
> Tiempo estimado: ~1h 45min en GPU T4.

---

## Requisitos previos

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Configurar Kaggle API

```bash
# Descargar kaggle.json desde kaggle.com → Settings → API → Create New Token
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

---

## Resultados obtenidos

| Modelo | Accuracy | F1-macro | AUC-ROC | Params | FN |
|---|---|---|---|---|---|
| CustomCNN (desde cero) | 0.6306 | 0.6262 | 0.6891 | 1.8M | 34 |
| ResNet-50 (fine-tuning) | 0.9682 | 0.9681 | 0.9894 | 24.0M | 3 |
| **EfficientNet-B3 (fine-tuning)** | **0.9682** | **0.9681** | **0.9878** | **11.1M** | **1** |

**FN = Falsos Negativos** (llantas dañadas no detectadas — el error más crítico para seguridad vial)

**Mejor modelo:** EfficientNet-B3 — igual F1 que ResNet-50 con la mitad de parámetros y solo 1 falso negativo en 157 imágenes de test.

---

## Estructura del proyecto

```
tire_classification/
├── configs/
│   ├── train_scratch.yaml        # CustomCNN desde cero
│   ├── train_resnet50.yaml       # ResNet-50 fine-tuning
│   ├── train_efficientnet.yaml   # EfficientNet-B3 fine-tuning
│   ├── ablation_minimal.yaml     # Generado por run_all.sh
│   └── ablation_standard.yaml    # Generado por run_all.sh
├── data/
│   ├── raw/                      # Dataset original de Kaggle
│   ├── raw_organized/            # Reorganizado (normal→good)
│   └── processed/                # Splits train/val/test
│       ├── train/{good,cracked}/
│       ├── val/{good,cracked}/
│       └── test/{good,cracked}/
├── outputs/                      # Checkpoints y métricas
│   ├── custom_cnn_aggressive_focal/
│   │   ├── best_model.pt
│   │   ├── test_metrics.json
│   │   └── training_history.json
│   ├── resnet50_finetune_focal/
│   │   ├── best_model.pt
│   │   ├── test_metrics.json
│   │   └── training_history.json
│   └── efficientnet_b3_finetune_focal/
│       ├── best_model.pt
│       ├── test_metrics.json
│       └── training_history.json
├── reports/
│   ├── figures/                  # Figuras del informe
│   │   ├── roc_comparativo.png
│   │   ├── matrices_confusion.png
│   │   ├── metricas_comparativo.png
│   │   ├── gradcam_good_efficientnet.png
│   │   └── gradcam_cracked_efficientnet.png
│   └── informe_p1.tex            # Informe LaTeX formato NeurIPS
├── results/                      # Resultados generados por evaluate.py
│   ├── custom_cnn/
│   │   ├── gradcam_good.png
│   │   ├── gradcam_cracked.png
│   │   └── failures/
│   ├── resnet50/
│   │   ├── gradcam_good.png
│   │   ├── gradcam_cracked.png
│   │   └── failures/
│   └── efficientnet/
│       ├── gradcam_good.png
│       ├── gradcam_cracked.png
│       └── failures/
├── scripts/
│   ├── prepare_data.py           # División del dataset
│   └── evaluate.py               # Evaluación y Grad-CAM
├── src/
│   ├── train.py                  # Script principal de entrenamiento
│   ├── models/
│   │   ├── custom_cnn.py         # CNN desde cero + Grad-CAM
│   │   └── transfer_model.py     # ResNet-50 / EfficientNet-B3
│   ├── data/
│   │   ├── dataset.py            # TireDataset (PyTorch)
│   │   └── transforms.py         # Augmentaciones
│   ├── utils/
│   │   ├── losses.py             # Focal Loss
│   │   ├── metrics.py            # accuracy, F1, AUC-ROC
│   │   ├── trainer.py            # Bucle entrenamiento + early stopping
│   │   └── logger.py             # Logging
│   └── visualization/
│       ├── gradcam.py            # Grad-CAM + análisis de fallos
│       └── plots.py              # Curvas ROC, matrices, barras
├── colab_training.ipynb          # Notebook para Google Colab
├── run_all.sh                    # Pipeline completo (un solo comando)
└── requirements.txt              # Dependencias Python
```

---

## Entregables requeridos

| # | Entregable | Ubicación |
|---|---|---|
| 1 | 2 arquitecturas CNN | `src/models/custom_cnn.py`, `src/models/transfer_model.py` |
| 2 | Métricas + matriz de confusión | `outputs/*/test_metrics.json`, `results/*/` |
| 3 | Estudio de ablación | `outputs/ablation_*/`, generado por `run_all.sh` |
| 4 | Grad-CAM ambas clases | `results/*/gradcam_good.png`, `results/*/gradcam_cracked.png` |
| 5 | Análisis desbalance de clases | `outputs/*/test_metrics.json` |
| 6 | 5 fallos comentados | `results/*/failures/` |

---

## Detalles técnicos

| Aspecto | Decisión |
|---|---|
| Arquitecturas | CustomCNN (1.8M), ResNet-50 (24.0M), EfficientNet-B3 (11.1M) |
| Pérdida | Focal Loss (α=0.75, γ=2.0) |
| Desbalance | WeightedRandomSampler + Focal Loss |
| Augmentación | albumentations: ElasticTransform, GridDistortion, CLAHE, CoarseDropout |
| Optimizador | AdamW + CosineAnnealingLR |
| Early stopping | 12–15 épocas sin mejora en F1-macro |
| Interpretabilidad | Grad-CAM sobre última capa convolucional |
| Hardware | GPU NVIDIA Tesla T4 (Google Colab) |