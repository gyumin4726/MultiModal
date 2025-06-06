#!/bin/bash

echo "=== VMamba-Enhanced FSCIL Training ==="
echo "VMamba Backbone + MambaNeck + ETF Head"
echo ""

# Check checkpoint file
CHECKPOINT_PATH="./vssm1_base_0229s_ckpt_epoch_225.pth"
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "❌ ERROR: VMamba checkpoint not found!"
    echo "Please download and place the file at: $CHECKPOINT_PATH"
    echo "Download from: https://github.com/MzeroMiko/VMamba/releases"
    exit 1
fi

echo "✅ VMamba checkpoint found: $CHECKPOINT_PATH"
echo ""

GPUS=1
WORK_DIR="work_dirs/vmamba_base_mambafscil_cub"

echo "🚀 Stage 1: Base Session Training (100 classes)"
echo "Flow: Input → VMamba → MambaNeck → ETF Head"
echo ""

python tools/train.py \
    configs/cub/vmamba_base_etf_bs512_80e_cub_mambafscil.py \
    --work-dir ${WORK_DIR} \
    --seed 0 \
    --deterministic

if [ $? -ne 0 ]; then
    echo "❌ Base training failed!"
    exit 1
fi

echo ""
echo "🔄 Stage 2: Incremental Learning Evaluation (10 sessions)"
echo "Flow: VMamba features → Enhanced MambaNeck → Incremental Classification"
echo ""

python tools/fscil.py \
    configs/cub/vmamba_base_etf_bs512_80e_cub_eval_mambafscil.py \
    ${WORK_DIR} \
    ${WORK_DIR}/best.pth \
    --seed 0 \
    --deterministic

echo ""
echo "✅ VMamba-FSCIL Training Complete!"
echo "Results saved in: ${WORK_DIR}" 