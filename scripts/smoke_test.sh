#!/bin/bash
# Smoke test for bone_tumor dataset training
# This script runs a minimal training test (1 epoch, small batch) to verify the pipeline works

cd /home/glcuser/projhighcv/bone_tumor/MulModSeg_2024

echo "=== Bone Tumor Dataset Smoke Test ==="
echo "Dataset: bone_tumor"
echo "Modality: MIX (CT_reg + MR)"
echo "Epochs: 1"
echo "Batch size: 1"
echo "ROI size: 96x96x96"
echo "Num classes: 2 (background + tumor)"
echo ""

python train.py \
  --dataset bone_tumor \
  --data_root_path /home/glcuser/projhighcv/bone_tumor \
  --device 0 \
  --backbone unet \
  --with_text_embedding 0 \
  --train_modality MIX \
  --max_epoch 1 \
  --batch_size 1 \
  --num_workers 0 \
  --num_class 2 \
  --roi_x 96 \
  --roi_y 96 \
  --roi_z 96 \
  --num_samples 2 \
  --lr 1e-4 \
  --log_name smoke_test_bone_tumor

echo ""
echo "=== Smoke test completed ==="
