#!/bin/bash
# Auto-tuning script for bone_tumor training
# Gradually increases batch_size and num_workers to maximize GPU utilization

cd /home/glcuser/projhighcv/bone_tumor/MulModSeg_2024

echo "=== Starting 20 Epoch Training with Auto-tuning ==="
echo "Device: GPU 0"
echo "Strategy: Start with batch_size=2, num_workers=4, then increase if successful"
echo ""

# Configuration
DATASET="bone_tumor"
DATA_ROOT="/home/glcuser/projhighcv/bone_tumor"
DEVICE=0
BACKBONE="unet"
WITH_TEXT=0
MODALITY="MIX"
MAX_EPOCH=20
NUM_CLASS=2
LR=1e-4
WARMUP=5
LOG_NAME="bone_tumor_20ep_auto"

# Try batch_size=2, num_workers=4 first
echo "=== Attempt 1: batch_size=2, num_workers=4 ==="
python train.py \
  --dataset $DATASET \
  --data_root_path $DATA_ROOT \
  --device $DEVICE \
  --backbone $BACKBONE \
  --with_text_embedding $WITH_TEXT \
  --train_modality $MODALITY \
  --max_epoch $MAX_EPOCH \
  --batch_size 2 \
  --num_workers 4 \
  --num_class $NUM_CLASS \
  --lr $LR \
  --warmup_epoch $WARMUP \
  --log_name ${LOG_NAME}_bs2_nw4 \
  2>&1 | tee training_bs2_nw4.log

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully with batch_size=2, num_workers=4"
    exit 0
fi

# Check if OOM error
if grep -q "out of memory\|CUDA out of memory" training_bs2_nw4.log; then
    echo "❌ OOM with batch_size=2, trying batch_size=1..."

    # Fallback to batch_size=1, num_workers=4
    echo "=== Attempt 2: batch_size=1, num_workers=4 ==="
    python train.py \
      --dataset $DATASET \
      --data_root_path $DATA_ROOT \
      --device $DEVICE \
      --backbone $BACKBONE \
      --with_text_embedding $WITH_TEXT \
      --train_modality $MODALITY \
      --max_epoch $MAX_EPOCH \
      --batch_size 1 \
      --num_workers 4 \
      --num_class $NUM_CLASS \
      --lr $LR \
      --warmup_epoch $WARMUP \
      --log_name ${LOG_NAME}_bs1_nw4 \
      2>&1 | tee training_bs1_nw4.log

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ Training completed successfully with batch_size=1, num_workers=4"
        exit 0
    else
        echo "❌ Training failed with batch_size=1, num_workers=4"
        exit 1
    fi
else
    echo "❌ Training failed with unknown error"
    tail -50 training_bs2_nw4.log
    exit 1
fi
