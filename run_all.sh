#!/usr/bin/env bash
# run_all.sh — Reproduce el experimento completo de extremo a extremo
# Uso: bash run_all.sh
# Requisitos:
#   1. pip install -r requirements.txt
#   2. kaggle.json en ~/.kaggle/kaggle.json

set -e

echo "=============================================="
echo "  Clasificación de Llantas — Pipeline Completo"
echo "=============================================="

# ── PASO 0: Verificar entorno ──────────────────────────────────
echo ""
echo "[0/6] Verificando entorno..."

# Instalar dependencias si faltan
if ! python -c "import torch" 2>/dev/null; then
    echo "  Instalando dependencias..."
    pip install -r requirements.txt -q
else
    echo "  ✓ Dependencias ya instaladas"
fi

# Verificar GPU
python - << 'PYEOF'
import torch
if torch.cuda.is_available():
    print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"  ✓ Memoria: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
else:
    print("  ⚠ No hay GPU disponible.")
    print("  ⚠ Se recomienda ejecutar en Google Colab con GPU T4.")
    print("  ⚠ El entrenamiento en CPU puede tardar varias horas.")
PYEOF

# ── PASO 1: Descargar dataset ──────────────────────────────────
echo ""
echo "[1/6] Descargando dataset de Kaggle..."

if [ -d "data/raw/Tire Textures" ]; then
    echo "  ✓ Dataset ya descargado, omitiendo."
else
    if [ ! -f ~/.kaggle/kaggle.json ]; then
        echo ""
        echo "  ERROR: No se encontró ~/.kaggle/kaggle.json"
        echo "  Para obtenerlo:"
        echo "    1. Ve a kaggle.com → perfil → Settings"
        echo "    2. Sección API → Create New Token"
        echo "    3. Copia el archivo a ~/.kaggle/kaggle.json"
        echo "    4. chmod 600 ~/.kaggle/kaggle.json"
        exit 1
    fi

    mkdir -p data/raw
    kaggle datasets download \
        -d jehanbhathena/tire-texture-image-recognition \
        -p data/raw \
        --unzip
    echo "  ✓ Dataset descargado"
fi

# ── PASO 2: Preparar dataset ───────────────────────────────────
echo ""
echo "[2/6] Preparando y dividiendo dataset (70/15/15)..."

if [ -d "data/processed" ]; then
    echo "  ✓ Dataset ya procesado, omitiendo."
else
    # Reorganizar estructura del dataset de Kaggle
    # (Kaggle usa "normal" en lugar de "good" y tiene subcarpetas extra)
    python - << 'PYEOF'
import shutil
from pathlib import Path

raw_base  = Path("data/raw/Tire Textures")
dest_base = Path("data/raw_organized")

clase_map = {"normal": "good", "cracked": "cracked"}

for clase in ["good", "cracked"]:
    (dest_base / clase).mkdir(parents=True, exist_ok=True)

copiadas = {"good": 0, "cracked": 0}
for split in ["training_data", "testing_data"]:
    for clase_orig, clase_dest in clase_map.items():
        src = raw_base / split / clase_orig
        if not src.exists():
            continue
        for img in src.iterdir():
            if img.suffix.lower() in ['.jpg','.jpeg','.png','.bmp']:
                dst = dest_base / clase_dest / f"{split}_{img.name}"
                shutil.copy2(img, dst)
                copiadas[clase_dest] += 1

print(f"  ✓ good: {copiadas['good']} | cracked: {copiadas['cracked']}")
ratio = max(copiadas.values()) / min(copiadas.values())
print(f"  ✓ Ratio de desbalance: {ratio:.2f}x")
PYEOF

    python scripts/prepare_data.py \
        --raw_dir data/raw_organized \
        --out_dir data/processed \
        --train_ratio 0.70 \
        --val_ratio 0.15

    echo "  ✓ Dataset preparado"
fi

# ── PASO 3: Entrenar modelos principales ───────────────────────
echo ""
echo "[3/6] Entrenando modelos principales..."

echo "  >> CustomCNN (desde cero) — ~20 min en GPU T4"
python src/train.py --config configs/train_scratch.yaml

echo "  >> ResNet-50 (fine-tuning) — ~20 min en GPU T4"
python src/train.py --config configs/train_resnet50.yaml

echo "  >> EfficientNet-B3 (fine-tuning) — ~30 min en GPU T4"
python src/train.py --config configs/train_efficientnet.yaml

echo "  ✓ Entrenamiento de modelos principales completado"

# ── PASO 4: Estudio de ablación ────────────────────────────────
echo ""
echo "[4/6] Estudio de ablación..."
echo "  Comparando estrategias de augmentación: minimal / standard / aggressive"
echo "  (aggressive ya entrenado en paso anterior)"

# Crear configs temporales para ablación
python - << 'PYEOF'
import yaml

with open('configs/train_efficientnet.yaml') as f:
    cfg_base = yaml.safe_load(f)

# Ablación 1: augmentación minimal
cfg_min = yaml.safe_load(yaml.dump(cfg_base))
cfg_min['experiment_name'] = 'ablation_minimal'
cfg_min['data']['augmentation'] = 'minimal'
cfg_min['training']['epochs'] = 20
cfg_min['training']['use_weighted_sampler'] = True
with open('configs/ablation_minimal.yaml', 'w') as f:
    yaml.dump(cfg_min, f, default_flow_style=False)

# Ablación 2: augmentación standard
cfg_std = yaml.safe_load(yaml.dump(cfg_base))
cfg_std['experiment_name'] = 'ablation_standard'
cfg_std['data']['augmentation'] = 'standard'
cfg_std['training']['epochs'] = 20
cfg_std['training']['use_weighted_sampler'] = True
with open('configs/ablation_standard.yaml', 'w') as f:
    yaml.dump(cfg_std, f, default_flow_style=False)

print("  ✓ Configs de ablación generados")
PYEOF

echo "  >> Ablación: augmentación minimal — ~10 min"
python src/train.py --config configs/ablation_minimal.yaml

echo "  >> Ablación: augmentación standard — ~10 min"
python src/train.py --config configs/ablation_standard.yaml

# Comparar resultados de ablación
python - << 'PYEOF'
import json, os

configs = [
    ("Minimal",    "outputs/ablation_minimal/test_metrics.json"),
    ("Standard",   "outputs/ablation_standard/test_metrics.json"),
    ("Aggressive", "outputs/efficientnet_b3_finetune_focal/test_metrics.json"),
]

print(f"\n  {'Augmentación':<12} {'F1-macro':>10} {'AUC-ROC':>10}")
print(f"  {'-'*34}")
for nombre, path in configs:
    if os.path.exists(path):
        with open(path) as f:
            m = json.load(f)
        print(f"  {nombre:<12} {m['f1_macro']:>10.4f} {m['auc_roc']:>10.4f}")
print()
PYEOF

echo "  ✓ Estudio de ablación completado"

# ── PASO 5: Evaluación y visualizaciones ───────────────────────
echo ""
echo "[5/6] Evaluando modelos y generando visualizaciones..."
echo "  (matrices de confusión, curvas ROC, Grad-CAM, análisis de fallos)"

python scripts/evaluate.py \
    --config configs/train_scratch.yaml \
    --checkpoint outputs/custom_cnn_aggressive_focal/best_model.pt \
    --out_dir results/custom_cnn

python scripts/evaluate.py \
    --config configs/train_resnet50.yaml \
    --checkpoint outputs/resnet50_finetune_focal/best_model.pt \
    --out_dir results/resnet50

python scripts/evaluate.py \
    --config configs/train_efficientnet.yaml \
    --checkpoint outputs/efficientnet_b3_finetune_focal/best_model.pt \
    --out_dir results/efficientnet

echo "  ✓ Evaluación y visualizaciones completadas"

# ── PASO 6: Resumen final ──────────────────────────────────────
echo ""
echo "[6/6] Pipeline completado exitosamente."
echo ""
echo "=============================================="
echo "  RESUMEN DE RESULTADOS"
echo "=============================================="

python - << 'PYEOF'
import json, os

print()
print(f"  {'Modelo':<20} {'Accuracy':>9} {'F1-macro':>9} {'AUC-ROC':>9} {'FN':>5}")
print(f"  {'-'*56}")

modelos = [
    ("CustomCNN",     "outputs/custom_cnn_aggressive_focal/test_metrics.json"),
    ("ResNet-50",     "outputs/resnet50_finetune_focal/test_metrics.json"),
    ("EfficientNet-B3","outputs/efficientnet_b3_finetune_focal/test_metrics.json"),
]

for nombre, path in modelos:
    if os.path.exists(path):
        with open(path) as f:
            m = json.load(f)
        cm = m['confusion_matrix']
        fn = cm[1][0]
        print(f"  {nombre:<20} {m['accuracy']:>9.4f} {m['f1_macro']:>9.4f} "
              f"{m['auc_roc']:>9.4f} {fn:>5}")

print()
print("  Resultados guardados en:")
print("    results/custom_cnn/     → métricas, Grad-CAM, fallos")
print("    results/resnet50/       → métricas, Grad-CAM, fallos")
print("    results/efficientnet/   → métricas, Grad-CAM, fallos")
print()
print("  Checkpoints guardados en:")
print("    outputs/*/best_model.pt")
print()
PYEOF

echo "=============================================="