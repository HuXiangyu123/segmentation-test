#!/bin/bash
# Aggressive training script for RTX 3090 (24GB VRAM)
# Start with larger batch_size to maximize GPU utilization

cd /home/glcuser/projhighcv/bone_tumor/MulModSeg_2024

echo "=== 20 Epoch Training - RTX 3090 Optimization ==="
echo "GPU: RTX 3090 (24GB VRAM)"
echo "Strategy: Start aggressive, reduce if OOM"
echo ""

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
LOG_NAME="bone_tumor_20ep"

# Try batch_size=4, num_workers=8 first (aggressive)
echo "=== Attempt 1: batch_size=4, num_workers=8 ==="
python train.py \
  --dataset $DATASET \
  --data_root_path $DATA_ROOT \
  --device $DEVICE \
  --backbone $BACKBONE \
  --with_text_embedding $WITH_TEXT \
  --train_modality $MODALITY \
  --max_epoch $MAX_EPOCH \
  --batch_size 4 \
  --num_workers 8 \
  --num_class $NUM_CLASS \
  --lr $LR \
  --warmup_epoch $WARMUP \
  --log_name ${LOG_NAME}_bs4 \
  2>&1 | tee training_bs4.log

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed with batch_size=4"
    exit 0
fi

# Check if OOM
if grep -q "out of memory\|CUDA out of memory" training_bs4.log; then
    echo "❌ OOM with batch_size=4, trying batch_size=3..."

    echo "=== Attempt 2: batch_size=3, num_workers=8 ==="
    python train.py \
      --dataset $DATASET \
      --data_root_path $DATA_ROOT \
      --device $DEVICE \
      --backbone $BACKBONE \
      --with_text_embedding $WITH_TEXT \
      --train_modality $MODALITY \
      --max_epoch $MAX_EPOCH \
      --batch_size 3 \
      --num_workers 8 \
      --num_class $NUM_CLASS \
      --lr $LR \
      --warmup_epoch $WARMUP \
      --log_name ${LOG_NAME}_bs3 \
      2>&1 | tee training_bs3.log

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ Training completed with batch_size=3"
        exit 0
    fi

    if grep -q "out of memory\|CUDA out of memory" training_bs3.log; then
        echo "❌ OOM with batch_size=3, trying batch_size=2..."

        echo "=== Attempt 3: batch_size=2, num_workers=8 ==="
        python train.py \
          --dataset $DATASET \
          --data_root_path $DATA_ROOT \
          --device $DEVICE \
          --backbone $BACKBONE \
          --with_text_embedding $WITH_TEXT \
          --train_modality $MODALITY \
          --max_epoch $MAX_EPOCH \
          --batch_size 2 \
          --num_workers 8 \
          --num_class $NUM_CLASS \
          --lr $LR \
          --warmup_epoch $WARMUP \
          --log_name ${LOG_NAME}_bs2 \
          2>&1 | tee training_bs2.log

        EXIT_CODE=$?

        if [ $EXIT_CODE -eq 0 ]; then
            echo "✅ Training completed with batch_size=2"
            exit 0
        fi

        if grep -q "out of memory\|CUDA out of memory" training_bs2.log; then
            echo "❌ OOM with batch_size=2, trying batch_size=1..."

            echo "=== Attempt 4: batch_size=1, num_workers=8 ==="
            python train.py \
              --dataset $DATASET \
              --data_root_path $DATA_ROOT \
              --device $DEVICE \
              --backbone $BACKBONE \
              --with_text_embedding $WITH_TEXT \
              --train_modality $MODALITY \
              --max_epoch $MAX_EPOCH \
              --batch_size 1 \
              --num_workers 8 \
              --num_class $NUM_CLASS \
              --lr $LR \
              --warmup_epoch $WARMUP \
              --log_name ${LOG_NAME}_bs1 \
              2>&1 | tee training_bs1.log

            if [ $? -eq 0 ]; then
                echo "✅ Training completed with batch_size=1"
                exit 0
            else
                echo "❌ Training failed even with batch_size=1"
                exit 1
            fi
        fi
    fi
fi

echo "❌ Training failed with unknown error"
exit 1
