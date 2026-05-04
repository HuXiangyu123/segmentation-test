#!/bin/bash
# =============================================================================
# MulModSeg 主模型启动脚本
# =============================================================================
#
# ==================== 模型架构 ====================
#
# MulModSeg 由 3 个独立模块组成:
#
#   ┌─────────────────────────────────────────────────────┐
#   │  Module 1: SwinUNETR Encoder-Decoder (backbone)     │
#   │  ─────────────────────────────────────────────────   │
#   │  自定义 SwinUNETR (model/SwinUNETR.py)              │
#   │  支持分段 forward:                                    │
#   │    forward_to_dec4(x) → enc0,enc1,enc2,enc3,dec4,hid3│
#   │    forward_from_dec4(dec4, skips) → output           │
#   │  feature_size=48, dec4_dim=768, out_dim=48           │
#   └───────────────────────┬─────────────────────────────┘
#                           │ 48-ch feature map
#                           ▼
#   ┌─────────────────────────────────────────────────────┐
#   │  Module 2: 跨模态融合 (use_cross_attention=True 时)  │
#   │  ─────────────────────────────────────────────────   │
#   │                                                      │
#   │  2a. BidirectionalCrossAttention3D (bottleneck)      │
#   │      CT→MR: CrossAttn(Q=CT, KV=MR) → ct_enhanced    │
#   │      MR→CT: CrossAttn(Q=MR, KV=CT) → mr_enhanced    │
#   │      融合: Concat → Conv1x1 → GroupNorm → LeakyReLU  │
#   │      位置: dec4 (768-dim, 最深层特征)                 │
#   │                                                      │
#   │  2b. GatedSkipFusion3D (各层 skip connection)        │
#   │      enc0(48), enc1(48), enc2(96), enc3(192), hid3(384)│
#   │      融合: AvgPool → Concat → Conv1x1 → Sigmoid      │
#   │      output = α·CT + (1-α)·MR  (channel-wise gate)   │
#   │                                                      │
#   │  注意: 2a 和 2b 是两种不同的融合机制:                 │
#   │    2a = 多头 Q/K/V 注意力 (dec4 bottleneck)          │
#   │    2b = 通道门控 (skip connections)                   │
#   └───────────────────────┬─────────────────────────────┘
#                           │ fused 48-ch feature map
#                           ▼
#   ┌─────────────────────────────────────────────────────┐
#   │  Module 3: DynamicSemanticHead (文本引导 MoE 头)     │
#   │  ─────────────────────────────────────────────────   │
#   │  输入: 48-ch feature + 256-d text feature            │
#   │                                                      │
#   │  文本通路:                                            │
#   │    organ_embedding [2,2,512] (静态 CLIP 词向量)       │
#   │    → text_to_vision: Linear(512→256)                 │
#   │    → routing_feature (256-d)                         │
#   │                                                      │
#   │  MoE 路由:                                            │
#   │    router: Linear(256→64→2) → softmax → expert weights│
#   │    modulator: Linear(256→64→out_c) → sigmoid → gamma  │
#   │                                                      │
#   │  动态卷积:                                            │
#   │    weight_basis: [2, out_c, 48, 3, 3, 3] (2 experts) │
#   │    bias_basis:   [2, out_c]                           │
#   │    combined_W = Σ routing_weight[i] * W_expert[i]    │
#   │    combined_b = Σ routing_weight[i] * b_expert[i]    │
#   │    out = Conv3d(x, combined_W, combined_b)           │
#   │    out = out * (1 + gamma)  (残差语义调制)             │
#   │                                                      │
#   │  输出: final_logits [B, num_class, D, H, W]          │
#   └─────────────────────────────────────────────────────┘
#
# ==================== 启动参数说明 ====================
#
# 必传参数:
#   --with_text_embedding 1    启用 MulModSeg (默认已是 1)
#   --use_cross_attention      启用 CT↔MR 跨模态融合
#   --backbone swinunetr       骨干网络 (默认已是 swinunetr)
#
# 预训练:
#   --pretrain_encoder_only <path>  SSL 预训练 encoder 权重
#   --freeze_level all|stage4|stage34|none
#
# 数据:
#   --train_modality MIX|CT|MR
#   --dataset bone_tumor --data_root_path ./dataset
#
# =============================================================================

cd /work/projhighcv/hzl/bone_tumor

# === MulModSeg + cross-attention + SSL pretrained + freeze all ===
torchrun --nproc_per_node=2 MulModSeg_2024/train.py \
    --distributed \
    --dataset bone_tumor --data_root_path ./dataset \
    --backbone swinunetr \
    --with_text_embedding 1 \
    --use_cross_attention \
    --cross_attn_heads 8 \
    --pretrain_encoder_only ./MulModSeg_2024/pretrained/ssl_pretrained_weights.pth \
    --freeze_level all \
    --max_epoch 150 --warmup_epoch 10 \
    --train_modality MIX --batch_size 1 --num_workers 4 \
    --roi_x 96 --roi_y 96 --roi_z 96 --num_samples 3 \
    --lr 1e-4 --loss_type dicece \
    --log_name M2b_mulmodseg_crossattn_freeze_all
