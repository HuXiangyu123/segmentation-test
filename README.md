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
# Default config: MIX modality, dice-only validation starting at epoch 100
# (eval_mode=dice and val_start_epoch=100 are now the built-in defaults)
torchrun --nproc_per_node=2 MulModSeg_2024/train.py \
    --distributed \
    --dataset bone_tumor --data_root_path ./dataset \
    --backbone swinunetr \
    --with_text_embedding 1 \
    --use_cross_attention \
    --pretrain_encoder_only ./MulModSeg_2024/pretrained/ssl_pretrained_weights.pth \
    --freeze_level all \
    --max_epoch 150 --batch_size 1 --num_samples 3 \
    --roi_x 96 --roi_y 96 --roi_z 96 \
    --lr 1e-4 --loss_type dicece \
    --log_name mix_exp
```

### Inference

```bash
python scripts/infer_checkpoint.py \
    --checkpoint "experiment/infer_ckpt/best_model (3).pt" \
    --dataset bone_tumor --data_root_path ./dataset \
    --backbone swinunetr --train_modality MIX \
    --device 0
```

## Train Entry

### Core Parameter Choices

| Argument | Default | Choices / Meaning |
|----------|---------|-------------------|
| `--device` | `0` | Single-GPU device id `0-7` |
| `--distributed` | `false` | Enable DDP via `torchrun` |
| `--dataset` | `data1` | `data1`, `bone_tumor` |
| `--data_root_path` | `./dataset/data1` | Dataset root path |
| `--backbone` | `swinunetr` | `swinunetr`, `unet`, `dints`, `unetpp` |
| `--train_modality` | `MIX` | `CT`, `MR`, `MIX` |
| `--with_text_embedding` | `1` | `1` = MulModSeg (text-guided MoE), `0` = plain backbone |
| `--use_cross_attention` | `false` | Must pass explicitly for paired CT↔MR bottleneck fusion |
| `--cross_attn_heads` | `8` | Number of cross-attention heads |
| `--trans_encoding` | `word_embedding` | `word_embedding`, `rand_embedding` |
| `--word_embedding` | `./text_embedding/bone_tumor_class_embeddings.pth` | Static text embedding file |
| `--case_text_embedding` | `None` | Per-case text embedding `.pth` |
| `--use_case_text_embedding` | `false` | Fuse case-level text embedding with static tumor text |
| `--case_text_alpha` | `0.3` | Fusion weight for case text |
| `--pretrain` | `None` | Full checkpoint preload |
| `--pretrain_encoder_only` | `None` | Encoder-only pretrained weights |
| `--freeze_level` | `none` | `all`, `stage4`, `stage34`, `none` |
| `--resume` | `None` | Resume full training state |
| `--finetune_from` | `None` | Load weights only, reset optimizer/scheduler |
| `--use_ema` | `false` | Evaluate/save EMA-smoothed weights |
| `--use_swa` | `false` | Evaluate/save SWA weights |
| `--max_epoch` | `1000` | Training epochs |
| `--warmup_epoch` | `10` | Warmup epochs |
| `--lr` | `1e-4` | Learning rate |
| `--fixed_lr` | `false` | Disable cosine scheduler and keep fixed LR |
| `--weight_decay` | `1e-5` | Optimizer weight decay |
| `--batch_size` | `1` | Batch size |
| `--num_workers` | `8` | DataLoader workers |
| `--roi_x --roi_y --roi_z` | `96 96 96` | Training / validation crop size |
| `--num_samples` | `2` | Number of random samples per scan during training |
| `--loss_type` | `dicece` | `dicece`, `tversky`, `focal_tversky` |
| `--loss_alpha` | `0.7` | Tversky / focal-Tversky alpha |
| `--loss_beta` | `0.3` | Tversky / focal-Tversky beta |
| `--loss_gamma` | `1.33` | Focal-Tversky gamma |
| `--boundary_loss_weight` | `0.0` | Enable boundary-band Dice auxiliary loss |
| `--boundary_kernel` | `3` | Boundary band kernel size |
| `--boundary_start_epoch` | `140` | First epoch applying boundary loss |
| `--entropy_weight` | `0.01` | MoE routing regularization weight |
| `--pos_neg_ratio` | `None` | Optional positive/negative patch sampling ratio |
| `--fold` | `None` | Fold index `0-4` for `splits/fold5_splits.json` |
| `--split_file` | `None` | Custom split JSON path |
| `--drop_list` | `None` | Patient ids to exclude |
| `--seed` | `42` | Random seed |

### Evaluation Parameters

| Argument | Default | Meaning |
|----------|---------|---------|
| `--eval_mode` | `dice` | `full` = Dice + Precision/Recall/IoU + HD95/ASSD + PR/ROC + visualizations; `dice` = only foreground Dice and bucketed Dice (default, saves ~95% val time) |
| `--val_start_epoch` | `100` | First epoch that actually runs validation. Saves ~2.5min per skipped epoch. Set to `0` for full monitoring |

## Eval Methods

### `--eval_mode full`

Use this when you need the full validation package:

- foreground Dice / IoU / Precision / Recall / F1
- HD95 / ASSD
- PR-AUC / ROC-AUC
- metrics CSV and training curves
- best / worst case visualizations

### `--eval_mode dice`

Use this when validation cost is the bottleneck and you only care about segmentation selection by Dice.

It keeps:

- `foreground_dice_mean`
- `foreground_dice_std`
- bucketed Dice by GT positive ratio: `<2%`, `2-5%`, `>5%`
- best-model selection by foreground Dice

It skips:

- Precision / Recall / IoU
- HD95 / ASSD
- PR / ROC
- heavy best/worst case visualization dumps

This mode is the recommended default for long dual-modality runs when early-epoch validation is mostly noise.

### Recommended Dual-Modality Example

```bash
torchrun --nproc_per_node=2 MulModSeg_2024/train.py \
    --distributed \
    --dataset bone_tumor --data_root_path ./dataset \
    --backbone swinunetr \
    --train_modality MIX \
    --with_text_embedding 1 \
    --use_cross_attention \
    --pretrain_encoder_only ./MulModSeg_2024/pretrained/ssl_pretrained_weights.pth \
    --freeze_level all \
    --batch_size 1 --num_samples 3 \
    --roi_x 96 --roi_y 96 --roi_z 96 \
    --max_epoch 150 --warmup_epoch 10 \
    --lr 1e-4 --loss_type dicece \
    --log_name mix_crossattn
```

Recommended use case:

- `--train_modality MIX`: paired CT+MR training
- `--use_cross_attention`: enable CT↔MR fusion
- `--eval_mode dice` (default): keep validation cheap — Dice only, no HD95/ASSD/PR-ROC/vis
- `--val_start_epoch 100` (default): delay validation until the model has left the unstable early phase

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
