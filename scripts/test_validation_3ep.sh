#!/bin/bash
# Test validation with 3 epochs to verify val_dice changes

cd /home/glcuser/projhighcv/bone_tumor/MulModSeg_2024

echo "=== Testing Validation with 3 Epochs ==="
echo "This will verify if val_dice changes across epochs"
echo ""

DATASET="bone_tumor"
DATA_ROOT="/home/glcuser/projhighcv/bone_tumor"
DEVICE=0
BACKBONE="unet"
WITH_TEXT=0
MODALITY="MIX"
MAX_EPOCH=3
NUM_CLASS=2
LR=1e-4
WARMUP=1
BATCH_SIZE=2
NUM_WORKERS=8
LOG_NAME="test_val_3ep"

python train.py \
  --dataset $DATASET \
  --data_root_path $DATA_ROOT \
  --device $DEVICE \
  --backbone $BACKBONE \
  --with_text_embedding $WITH_TEXT \
  --train_modality $MODALITY \
  --max_epoch $MAX_EPOCH \
  --batch_size $BATCH_SIZE \
  --num_workers $NUM_WORKERS \
  --num_class $NUM_CLASS \
  --lr $LR \
  --warmup_epoch $WARMUP \
  --log_name $LOG_NAME

echo ""
echo "=== Extracting Validation Results ==="
echo ""
echo "Validation Dice scores:"
grep -E "\[Epoch [0-9]+\] Current validation Dice:" out/unet/no_txt/${LOG_NAME}_*/events.out.tfevents.* 2>/dev/null || \
grep -E "\[Epoch [0-9]+\] Current validation Dice:" training_test_val.log 2>/dev/null || \
echo "Check the output above for validation results"
