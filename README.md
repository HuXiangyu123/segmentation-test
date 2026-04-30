# MulModSeg: Multi-Modal Bone Tumor Segmentation

CT+MR multi-modal bone tumor segmentation with SwinUNETR backbone, cross-modal attention fusion, and text-guided MoE dynamic head.

## Model Architecture

```
CT ──→ SwinUNETR.Encoder ──→ enc0(48) enc1(48) enc2(96) enc3(192) dec4(768) hid3(384)
MR ──→ SwinUNETR.Encoder ──→ enc0(48) enc1(48) enc2(96) enc3(192) dec4(768) hid3(384)
                                  │         │         │        │        │
    ┌─────────────────────────────┼─────────┼─────────┼────────┤        │
    │  Skip Fusion (GatedSkipFusion3D)      │         │        │        │
    │  α·CT + (1-α)·MR via channel gate     │         │        │        │
    └─────────────────────────────┼─────────┼─────────┼────────┤        │
                                  │         │         │        │        │
                                  ▼         ▼         ▼        ▼        │
    ┌───────────────────────────────────────────────────────────┤        │
    │  Bottleneck Fusion (BidirectionalCrossAttention3D)        │        │
    │  CT→MR: CrossAttn(Q=CT, KV=MR)                           │        │
    │  MR→CT: CrossAttn(Q=MR, KV=CT)                           │        │
    │  → Concat → Conv1x1 → GroupNorm → LeakyReLU              │        │
    └───────────────────────────────────────────────────────────┤        │
                                                                ▼        │
                                                         fused_dec4     │
                                                                │   fused skips
                                                                ▼        ▼
                                                     SwinUNETR.forward_from_dec4()
                                                                │
                                                                ▼
                                                         48-ch feature map
                                                                │
    CLIP/BiomedCLIP ──→ organ_embedding [2,2,512]                │
                     ──→ text_to_vision (512→256)                │
                     ──→ router (256→64→2)                       │
                     ──→ modulator (256→64→out_c)                │
                                      │                         │
                                      ▼                         ▼
                            DynamicSemanticHead
                            W = Σ routing_weight[i] × W_expert[i]
                            out = Conv3d(x, W, b) × (1 + gamma)
                                      │
                                      ▼
                               final_logits [B, 2, D, H, W]
```

### Three Independent Modules

| Module | What it does | Input | Output |
|--------|-------------|-------|--------|
| **SwinUNETR** | 3D encoder-decoder with skip connections | CT/MR volume (1ch) | 48-ch feature map |
| **CrossModalFusion** | CT↔MR feature fusion at bottleneck + skip | CT/MR features | Fused features |
| **DynamicSemanticHead** | Text-guided MoE routing → dynamic conv kernel | Feature + text (256d) | Segmentation logits |

### Cross-Modal Fusion: Two Mechanisms Stacked

1. **Bottleneck** (`BidirectionalCrossAttention3D`): Full Q/K/V multi-head cross-attention between CT and MR at the deepest feature level (768-dim). Two directions: CT→MR and MR→CT.

2. **Skip Connections** (`GatedSkipFusion3D`): Channel-wise gating at each encoder level (enc0~hid3). Uses adaptive pooling + sigmoid gate: `output = α·CT + (1-α)·MR`.

These are fundamentally different: bottleneck uses dense attention (O(n²)), skip uses lightweight channel gating (O(n)).

### Text-Guided MoE Head

- **Static text**: `organ_embedding [modality, class, 512]` — CLIP/BiomedCLIP embeddings for background/tumor per modality
- **Router**: text feature → 2 expert weights (softmax)
- **Dynamic conv**: Blend 2 expert convolution kernels based on router weights
- **Modulator**: text → sigmoid scaling factor γ, applied as `out × (1 + γ)`

## Quick Start

### Training

```bash
# Main model: MulModSeg + cross-attention + SSL pretrained
torchrun --nproc_per_node=2 MulModSeg_2024/train.py \
    --distributed \
    --dataset bone_tumor --data_root_path ./dataset \
    --backbone swinunetr \
    --with_text_embedding 1 \
    --use_cross_attention \
    --pretrain_encoder_only ./MulModSeg_2024/pretrained/ssl_pretrained_weights.pth \
    --freeze_level all \
    --max_epoch 150 --train_modality MIX \
    --batch_size 1 --num_samples 3 --roi_x 96 --roi_y 96 --roi_z 96 \
    --lr 1e-4 --loss_type dicece \
    --log_name M2b_mulmodseg_crossattn_freeze_all
```

### Inference

```bash
python scripts/infer_checkpoint.py \
    --checkpoint "experiment/infer_ckpt/best_model (3).pt" \
    --dataset bone_tumor --data_root_path ./dataset \
    --backbone swinunetr --train_modality MIX \
    --device 0
```

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--backbone` | `swinunetr` | Encoder backbone: `swinunetr`, `unet` |
| `--with_text_embedding` | `1` | `1` = MulModSeg (MoE head), `0` = plain backbone |
| `--use_cross_attention` | `false` | **Must pass explicitly** to enable CT↔MR fusion |
| `--cross_attn_heads` | `8` | Number of attention heads for cross-attention |
| `--pretrain_encoder_only` | `None` | Path to SSL/BTCV pretrained encoder weights |
| `--freeze_level` | `none` | `all`, `stage4`, `stage34`, `none` |
| `--train_modality` | `MIX` | `CT`, `MR`, `MIX` (paired CT+MR) |
| `--word_embedding` | `text_embedding/bone_tumor_class_embeddings.pth` | Static text embeddings [2,2,512] |
| `--case_text_embedding` | `None` | Per-case LLM caption embeddings |
| `--entropy_weight` | `0.01` | MoE routing entropy regularization |

## Experiment Structure

```
experiment/
├── comp/                          # Comparison baselines (pure models)
│   ├── no_txt/                    # Without text embedding
│   │   └── E0_baseline_swinunetr_... (Dice=0.6310)
│   └── swinunetr/with_txt/CLIP_V3/  # With CLIP text
│       ├── M0_pt_enc_freeze_all_...  (Dice=0.7173)
│       └── M0_pt_enc_freeze_s4_...   (Dice=0.7160)
├── swinunetr/no_txt/              # Main model experiments (text, no cross-attn)
└── infer_ckpt/                    # Checkpoints for inference
```

## Dataset

- 102 paired CT+MR bone tumor patients
- 5-fold cross-validation: `splits/fold5_splits.json`
- 7 patients dropped (registration failure, poor quality)
- Labels: binary (background=0, tumor=1)
