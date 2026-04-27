# Experiment Tracker

**Last Updated**: 2026-04-27
**Total GPU Budget**: 8 hours (core stages)
**Hardware**: 2× RTX 3090

---

## Status Legend
- ⬜ Pending — not started
- 🔄 Running — currently executing
- ✅ Complete — finished successfully
- ❌ Failed — error or negative result
- ⚠️ Blocked — waiting on dependency or manual action

---

## Stage 0: Quick Wins

| Exp ID | Status | Start | End | GPU-h | Dice (fg) | Dice (pelvis) | Precision | Recall | Notes |
|--------|--------|-------|-----|-------|-----------|---------------|-----------|--------|-------|
| P0_boundary_dice | ⬜ | — | — | — | — | — | — | — | Config: boundary_dice_weight=0.3 |
| P0_modality_routing | ⬜ | — | — | — | — | — | — | — | 10-line code change |
| P0_pelvis_sample | ⬜ | — | — | — | — | — | — | — | pelvis_weight=2.0 |
| P0_decoder_se | ⬜ | — | — | — | — | — | — | — | SE after each UnetrUpBlock |

## Stage 1: Encoder + Text Embedding

| Exp ID | Status | Start | End | GPU-h | Dice (fg) | Dice (pelvis) | Precision | Recall | Notes |
|--------|--------|-------|-----|-------|-----------|---------------|-----------|--------|-------|
| E1a_biomedclip | ⬜ | — | — | — | — | — | — | — | ⚠️ MANUAL: download weights |
| E1b_progressive_freeze | ⬜ | — | — | — | — | — | — | — | |
| E1c_combined | ⬜ | — | — | — | — | — | — | — | Depends on E1a+E1b+P0_best |

## Stage 2: Attention & Fusion Plug-ins

| Exp ID | Status | Start | End | GPU-h | Dice (fg) | Dice (pelvis) | Precision | Recall | Notes |
|--------|--------|-------|-----|-------|-----------|---------------|-----------|--------|-------|
| E2a_cbam_decoder | ⬜ | — | — | — | — | — | — | — | |
| E2b_ema_bottleneck | ⬜ | — | — | — | — | — | — | — | |
| E2c_best_attention | ⬜ | — | — | — | — | — | — | — | |

## Stage 3: CT Baselines

| Exp ID | Status | Start | End | GPU-h | Dice (fg) | Dice (pelvis) | Precision | Recall | Notes |
|--------|--------|-------|-----|-------|-----------|---------------|-----------|--------|-------|
| E3a_unet_ct | ⬜ | — | — | — | — | — | — | — | |
| E3b_swinunetr_ct | ⬜ | — | — | — | — | — | — | — | |
| E3c_mamba_ct | ⬜ | — | — | — | — | — | — | — | ⚠️ MANUAL: impl code |

---

## Manual Action Checklist

- [ ] **BiomedCLIP weights**: Download `microsoft/BiomedCLIP-PubMedBERT_256_vit_base_patch16` (~800MB)
- [ ] **SwinUNETR pretrained weights**: Verify cached or download from MONAI model zoo
- [ ] **Mamba implementation**: Clone U-Mamba or nnMamba, extract VSS Block, create encoder wrapper
- [ ] **LLM API key** (Stage 5, deferred): Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in `.env`
- [ ] **Anatomy labels** (for Idea 7, deferred): Derive pelvis/femur labels from case metadata
