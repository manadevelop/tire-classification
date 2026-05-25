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

> ⚠️ **Importante sobre hardware:**
> - `bash run_all.sh` ejecuta en tu **computadora local** usando CPU o GPU local.
> - Si tu computadora **no tiene GPU NVIDIA**, el entrenamiento puede tardar **20+ horas**.
> - Se recomienda ejecutar en **Google Colab con GPU T4** (ver opciones abajo).
> - Tiempo estimado en GPU T4: **~1h 45min**.

---

## Requisitos previos

### Paso 1 — Clonar el repositorio

```bash
git clone https://github.com/manadevelop/tire-classification.git
cd tire-classification
```

### Paso 2 — Instalar dependencias

```bash
pip install -r requirements.txt
```

### Paso 3 — Configurar Kaggle API

El proyecto descarga el dataset automáticamente desde Kaggle.
Para esto necesitas crear un token de acceso:

**3.1 Obtener tu token de Kaggle:**
```
1. Ve a kaggle.com e inicia sesión
2. Haz clic en tu foto de perfil (esquina superior derecha)
3. Selecciona "Settings"
4. Haz clic en la pestaña "API Tokens"
5. En "New Token Name" escribe un nombre (ej: "mi-token")
6. Haz clic en "Generate"
7. Aparece tu token — cópialo, solo se muestra una vez
```

**3.2 Configurar el token en tu computadora:**
```bash
mkdir -p ~/.kaggle
echo '{"username":"TU_USUARIO_KAGGLE","key":"TU_TOKEN_AQUI"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

> Reemplaza `TU_USUARIO_KAGGLE` con tu usuario de Kaggle y
> `TU_TOKEN_AQUI` con el token que copiaste en el paso anterior.

**3.3 Verificar que funciona:**
```bash
kaggle datasets list
```
Si muestra una lista de datasets, está configurado correctamente.

---

## Opciones de ejecución

Hay tres formas de ejecutar el proyecto según el hardware disponible:

---

### OPCIÓN 1 — Ejecución local (CPU/GPU local)

> Usa la CPU o GPU de tu propia computadora.
> Sin GPU NVIDIA el entrenamiento puede tardar 20+ horas.
> Recomendado solo si tienes GPU NVIDIA local.

Después de completar los Requisitos previos:

```bash
bash run_all.sh
```

---

### OPCIÓN 2 — Google Colab desde VS Code (GPU T4 recomendado)

> Ejecuta el código en los servidores de Google con GPU T4
> directamente desde VS Code en tu computadora.
> Requiere la extensión de Google Colab instalada en VS Code.

**Paso 1 — Instalar la extensión de Colab en VS Code:**
```
1. Abre VS Code
2. Cmd + Shift + X (panel de extensiones)
3. Busca "Google Colab"
4. Instala la extensión oficial de Google
```

**Paso 2 — Abrir el notebook:**
```
1. En VS Code abre el archivo: colab_training.ipynb
2. Haz clic en "Select Kernel" (esquina superior derecha)
3. Selecciona "Colab" → "New Colab Server" → "GPU"
4. Inicia sesión con tu cuenta de Google
```

**Paso 3 — Activar GPU en Colab:**
```
En el navegador que se abre automáticamente:
Entorno de ejecución → Cambiar tipo de entorno de ejecución
→ Acelerador: T4 GPU → Guardar
```

**Paso 4 — Ejecutar las celdas del notebook colab_training.ipynb en orden:**

Las celdas del notebook ejecutan automáticamente:
- Clonado del repositorio desde GitHub
- Configuración de Kaggle con usuario y token
- Descarga del dataset
- Pipeline completo (bash run_all.sh)
- Descarga de resultados a tu computadora

---

### OPCIÓN 3 — Google Colab desde el navegador (GPU T4 recomendado)

> La opción más simple. No requiere configuración en VS Code.
> Solo necesitas un navegador web.

**Paso 1 — Ir a Google Colab:**
```
Ve a: colab.research.google.com
Inicia sesión con tu cuenta de Google
```

**Paso 2 — Activar GPU T4:**
```
Entorno de ejecución
→ Cambiar tipo de entorno de ejecución
→ Acelerador de hardware: T4 GPU
→ Guardar
```

**Paso 3 — Crear un nuevo notebook y ejecutar estas celdas:**

Celda 1 — Clonar el repositorio:
```python
import os
!git clone https://github.com/manadevelop/tire-classification.git
os.chdir('/content/tire-classification')
print("✓ Repositorio clonado")
!ls
```

Celda 2 — Instalar dependencias:
```python
!pip install -r requirements.txt -q
print("✓ Dependencias instaladas")
```

Celda 3 — Verificar GPU:
```python
import torch
print("GPU disponible:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
```

Celda 4 — Configurar Kaggle:
```python
import os

# Reemplaza con tu usuario y token de Kaggle
KAGGLE_USERNAME = "TU_USUARIO_KAGGLE"
KAGGLE_KEY      = "TU_TOKEN_KAGGLE"

os.makedirs('/root/.kaggle', exist_ok=True)
with open('/root/.kaggle/kaggle.json', 'w') as f:
    f.write(f'{{"username":"{KAGGLE_USERNAME}","key":"{KAGGLE_KEY}"}}')
!chmod 600 /root/.kaggle/kaggle.json
print("✓ Kaggle configurado")
```

Celda 5 — Ejecutar el pipeline completo:
```python
!bash run_all.sh
```

Celda 6 — Descargar resultados a tu computadora:
```python
import shutil
from google.colab import files

shutil.make_archive('/content/resultados', 'zip',
                    '/content/tire-classification/results')
shutil.make_archive('/content/modelos', 'zip',
                    '/content/tire-classification/outputs')

files.download('/content/resultados.zip')
files.download('/content/modelos.zip')
print("✓ Revisa tu carpeta Descargas")
```

---

## Comparación de opciones

| Opción | Hardware | Tiempo estimado | Dificultad |
|---|---|---|---|
| Local sin GPU | CPU Mac/PC | 20+ horas | Fácil |
| Local con GPU NVIDIA | GPU local | ~2 horas | Fácil |
| Colab desde VS Code | GPU T4 nube | ~1h 45min | Media |
| Colab desde navegador | GPU T4 nube | ~1h 45min | Fácil |

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
│   └── ablation_standard.yaml   # Generado por run_all.sh
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
├── results/                      # Resultados generados
│   ├── custom_cnn/
│   │   ├── gradcam_correct_good.png
│   │   ├── gradcam_correct_cracked.png
│   │   └── failures/
│   ├── resnet50/
│   │   ├── gradcam_correct_good.png
│   │   ├── gradcam_correct_cracked.png
│   │   └── failures/
│   └── efficientnet/
│       ├── gradcam_correct_good.png
│       ├── gradcam_correct_cracked.png
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
| 4 | Grad-CAM ambas clases | `results/*/gradcam_correct_good.png`, `results/*/gradcam_correct_cracked.png` |
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
| Hardware recomendado | GPU NVIDIA Tesla T4 (Google Colab) |