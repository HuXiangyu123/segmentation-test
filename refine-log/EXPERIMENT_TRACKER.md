# Experiment Tracker

**Last Updated**: 2026-04-28
**Total GPU Budget**: ~13h (must-run) to 19h (complete)
**Hardware**: 2× RTX 3090

---

## Status Legend
- ⬜ Pending — not started
- 🔄 Running — currently executing
- ✅ Complete — finished successfully
- ❌ Failed — error or negative result
- ⚠️ Blocked — waiting on dependency or manual action

---

## M0: Quick-Win Pilots

| Exp ID | Variant | Status | Start | End | GPU-h | Dice (fg) | Dice (pelvis) | Seed Var | Notes |
|--------|---------|--------|-------|-----|-------|-----------|---------------|----------|-------|
| M0_boundary_dice | boundary_dice_weight=0.3 | ⬜ | — | — | — | — | — | — | Config only |
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
| M2b_freeze_all | Freeze stages 1-4 | ⚠️ | — | — | — | — | — | — | Blocked: pretrained weights |
| M2b_freeze_s4 | Freeze 1-3, thaw 4 | ⬜ | — | — | — | — | — | — | |
| M2b_freeze_s34 | Freeze 1-2, thaw 3-4 | ⬜ | — | — | — | — | — | — | |
| M2b_full_finetune | No freeze | ⬜ | — | — | — | — | — | — | |
| M2b_no_pretrain | No pretrained weights | ⬜ | — | — | — | — | — | — | |

## M2c: Combined

| Exp ID | Variant | Status | Start | End | GPU-h | Dice (fg) | Dice (pelvis) | Precision | Notes |
|--------|---------|--------|-------|-----|-------|-----------|---------------|-----------|-------|
| M2c_best | M2a + M2b_best + M0_best | ⬜ | — | — | — | — | — | — | Depends on M0, M2a, M2b |

## M3a: LLM-Enhanced Text Description

| Exp ID | Variant | Status | Start | End | GPU-h | Dice (fg) | Dice (pelvis) | Precision | Notes |
|--------|---------|--------|-------|-----|-------|-----------|---------------|-----------|-------|
| M3a_schema_design | Create struct output schema + prompt | ⬜ | — | — | — | — | — | — | Code, no GPU |
| M3a_llm_generate | Run LLM for all 110 cases | ⚠️ | — | — | — | — | — | — | Blocked: API key |
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
- [ ] **SwinUNETR BTCV**: `SwinUNETR(img_size=(96,96,96), in_channels=1, out_channels=14, feature_size=48, pretrained=True)`

### Model Clones (to comparative/)
- [ ] `git clone https://github.com/MIC-DKFZ/MedNeXt.git comparative/MedNeXt`
- [ ] `git clone https://github.com/MIC-DKFZ/nnUNet.git comparative/nnUNet`
- [ ] `git clone <HybridMamba> comparative/HybridMamba`
- [ ] `git clone <latest_2025_3d_model> comparative/LatestModel`

### LLM (M3a)
- [ ] Set API key in `.env`: `OPENAI_API_KEY=...` or `ANTHROPIC_API_KEY=...`
- [ ] Write `scripts/generate_llm_descriptions.py` with struct output schema
- [ ] Run all 110 cases through LLM → store structured JSON
- [ ] Encode with BiomedCLIP → `text_embedding/llm_biomedclip.pth`
