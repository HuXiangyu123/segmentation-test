# Experiment Tracker

**Last Updated**: 2026-04-30
**Total GPU Budget**: ~64h (must-run) to ~97h (complete)
**Hardware**: 2× RTX 3090, all experiments use `torchrun --nproc_per_node=2` (no single GPU)
**Training Config**: batch_size=1/GPU, roi_size=96³, num_samples=2-3, lr=1e-4, effective batch=4-6
**Default**: Single train/val split (80/20). 5-fold CV only when `--fold` is explicitly set.

---

## Status Legend
- ⬜ Pending — not started
- 🔄 Running — currently executing
- ✅ Complete — finished successfully
- ❌ Failed — error or negative result
- ⚠️ Blocked — waiting on dependency or manual action

---

## E0: Baseline (SwinUNETR, no pretrained, DDP)

| Exp ID | Variant | Status | Start | End | GPU-h | Dice (fg) | Dice (pelvis) | Precision | Notes |
|--------|---------|--------|-------|-----|-------|-----------|---------------|-----------|-------|
| E0_baseline_swinunetr | SwinUNETR, DiceCE, MIX | ✅ | 04-30 01:49 | 04-30 10:27 | ~17.0h | **0.6310** | — | 0.651 | DDP 2×3090, 210ep, lr=1e-4, num_samples=3, best ep65 |

## M0: Quick-Win Pilots

| Exp ID | Variant | Status | Start | End | GPU-h | Dice (fg) | Dice (pelvis) | Seed Var | Notes |
|--------|---------|--------|-------|-----|-------|-----------|---------------|----------|-------|
| M0_boundary_dice | boundary_dice_weight=0.3, start_epoch=140 | ⬜ | — | — | — | — | — | — | Config only, boundary at ep140 |
| M0_modality_routing | modality_embedding=true | ⬜ | — | — | — | — | — | — | +10 lines MulModSeg.py |
| M0_pelvis_sample | pelvis_weight=2.0 | ⬜ | — | — | — | — | — | — | dataloader change |
| M0_decoder_se | SE in decoder path | ⬜ | — | — | — | — | — | — | attention_plugins.py |

## M1: CT Baseline Comparison (Inference Only)

| Exp ID | Model | Source | Downloaded | GPU-h | Dice (fg) | Dice (pelvis) | Precision | Recall | Notes |
|--------|-------|--------|-----------|-------|-----------|---------------|-----------|--------|-------|
| M1_unet | 3D U-Net | Self-contained | — | — | — | — | — | — | No download needed |
| M1_segresnet | SegResNet | MONAI | ⬜ | — | — | — | — | — | pip installable |
| M1_swinunetr | SwinUNETR (MONAI) | MONAI zoo | ⬜ | — | — | — | — | — | `pretrained=True` |
| M1_mednext | MedNeXt-v2 | `comparative/MedNeXt` | ⬜ | — | — | — | — | — | ⚠️ git clone |
| M1_nnunet | nnU-Net v2 | `comparative/nnUNet` | ⬜ | — | — | — | — | — | ⚠️ git clone + weights |
| M1_hybridmamba | HybridMamba | `comparative/HybridMamba` | ⬜ | — | — | — | — | — | ⚠️ git clone |
| M1_latest_3d | 2025-2026 top model | `comparative/LatestModel` | ⬜ | — | — | — | — | — | ⚠️ git clone |

## M2a: BiomedCLIP

| Exp ID | Variant | Status | Start | End | GPU-h | Dice (fg) | Dice (pelvis) | Seed Var | Notes |
|--------|---------|--------|-------|-----|-------|-----------|---------------|----------|-------|
| M2a_biomedclip | BiomedCLIP → all text | ⚠️ | — | — | — | — | — | — | Blocked: download weights |

## M2b: Progressive Freeze Strategy

| Exp ID | Freeze Config | Status | Start | End | GPU-h | Dice (fg) | Dice (pelvis) | Precision | Notes |
|--------|--------------|--------|-------|-----|-------|-----------|---------------|-----------|-------|
| M2b_freeze_all | Freeze stages 1-4 | ✅ | 04-29 13:08 | 04-29 18:17 | ~5.0h | **0.7173** | — | 0.765 | SSL pretrained, best ep65 |
| M2b_freeze_s4 | Freeze 1-3, thaw 4 | ✅ | 04-29 10:52 | 04-29 18:17 | ~5.0h | **0.7160** | — | 0.768 | SSL pretrained, best ep65 |
| M2b_freeze_s34 | Freeze 1-2, thaw 3-4 | ⬜ | — | — | — | — | — | — | |
| M2b_full_finetune | No freeze | ⬜ | — | — | — | — | — | — | |
| M2b_no_pretrain | No pretrained weights | ⬜ | — | — | — | — | — | — | |

## M2b+MulModSeg: Freeze + Full Model

| Exp ID | Config | Status | Start | End | GPU-h | Dice (fg) | Dice (pelvis) | Precision | Notes |
|--------|--------|--------|-------|-----|-------|-----------|---------------|-----------|-------|
| M2b_freeze_s4_mulmodseg | freeze_s4 + cross_attn + text, 200ep | 🔄 | 04-30 | — | — | — | — | — | boundary_dice at ep140 |

## M2c: Combined

| Exp ID | Variant | Status | Start | End | GPU-h | Dice (fg) | Dice (pelvis) | Precision | Notes |
|--------|---------|--------|-------|-----|-------|-----------|---------------|-----------|-------|
| M2c_best | M2a + M2b_best + M0_best | ⬜ | — | — | — | — | — | — | Depends on M0, M2a, M2b |

## M3a: LLM-Enhanced Text Description

| Exp ID | Variant | Status | Start | End | GPU-h | Dice (fg) | Dice (pelvis) | Precision | Notes |
|--------|---------|--------|-------|-----|-------|-----------|---------------|-----------|-------|
| M3a_schema_design | Create struct output schema + prompt | ⬜ | — | — | — | — | — | — | Code, no GPU |
| M3a_llm_generate | Run LLM for all 102 patients | ⚠️ | — | — | — | — | — | — | Blocked: API key |
| M3a_biomedclip_encode | Encode to BiomedCLIP embeddings | ⚠️ | — | — | — | — | — | — | Blocked: M3a_generate |
| M3a_train | Train with LLM text features | ⬜ | — | — | — | — | — | — | |

## M3b: Attention Plug-ins

| Exp ID | Plug-in | Status | Start | End | GPU-h | Dice (fg) | Dice (pelvis) | Precision | Notes |
|--------|---------|--------|-------|-----|-------|-----------|---------------|-----------|-------|
| M3b_se | 3D SE in decoder | ⬜ | — | — | — | — | — | — | |
| M3b_cbam | 3D CBAM in decoder | ⬜ | — | — | — | — | — | — | |
| M3b_ema | 3D EMA at bottleneck | ⬜ | — | — | — | — | — | — | |
| M3b_best | Best plug-in + M3a_train | ⬜ | — | — | — | — | — | — | |

## M4: Full Ablation

| Exp ID | Variant | Status | Start | End | GPU-h | Dice (fg) | Dice (pelvis) | Precision | Notes |
|--------|---------|--------|-------|-----|-------|-----------|---------------|-----------|-------|
| M4_ct_only | CT only | ⬜ | — | — | — | — | — | — | |
| M4_mr_only | MR only | ⬜ | — | — | — | — | — | — | |
| M4_add_fusion | CT+MR add | ⬜ | — | — | — | — | — | — | |
| M4_concat_fusion | CT+MR concat | ⬜ | — | — | — | — | — | — | |
| M4_cross_attn | Full cross-attention | ⬜ | — | — | — | — | — | — | |
| M4_best_model | All best combined | ⬜ | — | — | — | — | — | — | |

---

## Manual Action Checklist

### Weight Downloads (pre-M2)
- [ ] **BiomedCLIP**: `AutoModel.from_pretrained('microsoft/BiomedCLIP-PubMedBERT_256_vit_base_patch16')`
- [x] **SwinUNETR SSL pretrained**: `MulModSeg_2024/pretrained/ssl_pretrained_weights.pth` (720MB, already in repo)

### Model Clones (to comparative/)
- [ ] `git clone https://github.com/MIC-DKFZ/MedNeXt.git comparative/MedNeXt`
- [ ] `git clone https://github.com/MIC-DKFZ/nnUNet.git comparative/nnUNet`
- [ ] `git clone <HybridMamba> comparative/HybridMamba`
- [ ] `git clone <latest_2025_3d_model> comparative/LatestModel`

### LLM (M3a)
- [ ] Set API key in `.env`: `OPENAI_API_KEY=...` or `ANTHROPIC_API_KEY=...`
- [ ] Write `scripts/generate_llm_descriptions.py` with struct output schema
- [ ] Run all 102 patients through LLM → store structured JSON
- [ ] Encode with BiomedCLIP → `text_embedding/llm_biomedclip.pth`
