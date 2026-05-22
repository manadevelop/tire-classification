#!/usr/bin/env bash
# run_all.sh — Reproduce el experimento completo de extremo a extremo
set -e

echo "=============================================="
echo "  Clasificación de Llantas — Pipeline Completo"
echo "=============================================="

# 1. Preparar datos
echo ""
echo "[1/4] Preparando dataset..."
python scripts/prepare_data.py \
  --raw_dir data/raw \
  --out_dir data/processed \
  --train_ratio 0.70 \
  --val_ratio 0.15

# 2. Entrenar modelos
echo ""
echo "[2/4] Entrenando modelos..."

echo "  >> CustomCNN (desde cero)"
python src/train.py --config configs/train_scratch.yaml

echo "  >> ResNet-50 (fine-tuning)"
python src/train.py --config configs/train_resnet50.yaml

echo "  >> EfficientNet-B3 (fine-tuning)"
python src/train.py --config configs/train_efficientnet.yaml

# 3. Evaluar y generar visualizaciones
echo ""
echo "[3/4] Evaluando modelos y generando visualizaciones..."

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

echo ""
echo "[4/4] Pipeline completado."
echo "  Resultados en: results/"
echo "  Checkpoints en: outputs/"
