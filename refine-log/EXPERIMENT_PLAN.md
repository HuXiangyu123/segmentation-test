# Experiment Plan: MulModSeg_2024 Baseline Improvement

**Date**: 2026-04-27
**Hardware**: 2× RTX 3090 (24GB VRAM each)
**Dataset**: 110 paired CT+MR bone tumor cases (5-fold cross-validation)
**Baseline**: E0_baseline (DiceCE loss, SwinUNETR backbone, default sampling)

---

## Problem Anchor

MulModSeg_2024 achieves reasonable multimodal bone tumor segmentation but has three documented weaknesses:
1. **Text prior instability**: CLIP embeddings produce inconsistent router behavior across seeds
2. **Poor pelvis performance**: Complex pelvic anatomy cases underperform by 10-15% Dice vs. femur cases
3. **No modern baselines**: Missing comparison with 2024-2025 3D segmentation models

## Method Thesis

Domain-specific text embeddings (BiomedCLIP) + progressive encoder freezing + lightweight decoder attention provide the highest gain-to-cost ratio for improving MulModSeg on small-sample bone tumor data.

---

## Experiment Roadmap

### Stage 0: Quick-Win Validation (Pilots)
**Budget**: 2 GPU-hours | **Priority**: Run immediately

| Exp ID | Description | Changes | Est. Time | GPU |
|--------|-------------|---------|-----------|-----|
| **P0_boundary_dice** | Enable boundary Dice loss | Config only: `boundary_dice_weight: 0.3` | 0.5h | 0 |
| **P0_modality_routing** | Modality-aware routing bias | `MulModSeg.py`: +10 lines | 0.5h | 1 |
| **P0_pelvis_sample** | Pelvis oversampling | `dataloader_bone_tumor.py`: case weights | 0.5h | 0 (after P0_boundary) |
| **P0_decoder_se** | SE attention in decoder | `SwinUNETR.py`: +SE blocks in decoder | 0.5h | 1 (after P0_modality) |

**Success criterion**: +2% Dice on pelvis bucket over E0_baseline

---

### Stage 1: Encoder + Text Embedding (Core)
**Budget**: 6 GPU-hours | **Priority**: Highest expected impact

| Exp ID | Description | Dependencies | Est. Time |
|--------|-------------|-------------|-----------|
| **E1a_biomedclip** | Replace CLIP → BiomedCLIP for text embeddings | ⚠️ MANUAL: Download BiomedCLIP weights | 1.5h (precompute) + 0.5h (train) |
| **E1b_progressive_freeze** | Progressive stage-wise encoder freezing | None | 1.0h |
| **E1c_combined** | BiomedCLIP + progressive freeze + best quick-win from Stage 0 | E1a, E1b, P0 best | 1.0h |

**⚠️ MANUAL ACTIONS for E1a**:
```bash
# 1. Download BiomedCLIP model (~800MB)
pip install transformers
python -c "from transformers import AutoModel; AutoModel.from_pretrained('microsoft/BiomedCLIP-PubMedBERT_256_vit_base_patch16')"

# 2. Regenerate all case embeddings with new model
python MulModSeg_2024/text_embedding/generate_embeddings.py \
    --clip_model biomedclip \
    --output text_embedding/bone_tumor_biomedclip_embeddings.pth

# 3. Update config to point to new embeddings
```

**⚠️ MANUAL ACTIONS for E1b**:
```python
# In train.py, add freeze schedule before optimizer setup:
freeze_schedule = {
    0:  [1,2,3,4],  # epoch 0-5: freeze Swin stages 1-4
    5:  [1,2,3],    # epoch 5-10: unfreeze stage 4
    10: [1,2],      # epoch 10-15: unfreeze stage 3
    15: [],         # epoch 15+: unfreeze all
}
```

---

### Stage 2: Attention & Fusion Plug-ins
**Budget**: 4 GPU-hours | **Priority**: Medium

| Exp ID | Description | Dependencies | Est. Time |
|--------|-------------|-------------|-----------|
| **E2a_cbam_decoder** | CBAM in decoder path (vs SE from P0) | None | 1.0h |
| **E2b_ema_bottleneck** | 3D EMA attention after cross-attn fusion | None | 1.0h |
| **E2c_best_attention** | Best attention config + best Stage 1 config | E1c, E2a, E2b | 1.0h |

**Implementation note**: Create `MulModSeg_2024/model/attention_plugins.py`
```python
# 3D-adapted SE, CBAM, EMA blocks
# Each block: [B,C,D,H,W] → [B,C,D,H,W] (shape-preserving)
# Insertion: after each UnetrUpBlock in SwinUNETR.forward_from_dec4
```

---

### Stage 3: 3D Segmentation Baselines (CT-Only)
**Budget**: 4 GPU-hours | **Priority**: Important for paper

| Exp ID | Description | Dependencies | Est. Time |
|--------|-------------|-------------|-----------|
| **E3a_unet_ct** | 3D U-Net baseline (CT-only) | None | 0.5h |
| **E3b_swinunetr_ct** | SwinUNETR baseline (CT-only, no fusion) | None | 0.5h |
| **E3c_mamba_ct** | U-Mamba/nnMamba baseline (CT-only) | ⚠️ MANUAL: Mamba implementation | 1.5h |
| **E3d_baseline_table** | Compile comparison table: all CT-only baselines + MulModSeg full | E3a-c, E1c | — (analysis) |

**⚠️ MANUAL ACTIONS for E3c**:
```bash
# Clone and integrate U-Mamba blocks
git clone https://github.com/JHU-CLSP/UMamba.git references/UMamba
# Or use nnMamba: https://github.com/MrGiovanni/nnMamba

# Extract VSS Block implementation → MulModSeg_2024/model/mamba_encoder.py
# Adapt to same forward_to_dec4/forward_from_dec4 interface
# Determine whether to use pretrained weights (manual download if so)
```

---

### Stage 4: Complete Ablation Study
**Budget**: 6 GPU-hours | **Priority**: Paper-required

| Exp ID | Description | Config Change |
|--------|-------------|---------------|
| **E4a_ct_only** | CT-only performance | `train_modality: CT` |
| **E4b_mr_only** | MR-only performance | `train_modality: MR` |
| **E4c_add_fusion** | CT+MR simple addition (no cross-attn) | `use_cross_attention: false`, add before decoder |
| **E4d_concat_fusion** | CT+MR concatenation (no cross-attn) | `use_cross_attention: false`, concat before decoder |
| **E4e_cross_attn** | Full cross-attention fusion | Current E4_cross_attention config |
| **E4f_best_model** | Best config from all stages | Combined best settings |

---

### Stage 5: LLM-Enhanced Text Description (Optional — High Impact)
**Budget**: 2 GPU-hours + API cost (~$5) | **Priority**: Deferred until API key available

| Exp ID | Description | Est. Time |
|--------|-------------|-----------|
| **E5a_llm_descriptions** | LLM-generated structured descriptions → BiomedCLIP | 1.0h code + 1.0h API calls |
| **E5b_full_text_pipeline** | LLM descriptions + BiomedCLIP + Modality Routing | 1.0h |

**⚠️ MANUAL ACTIONS for E5a**:
```bash
# 1. Set up API key in .env
echo "OPENAI_API_KEY=sk-..." >> .env  # or ANTHROPIC_API_KEY

# 2. Run description generation
python scripts/generate_llm_descriptions.py \
    --api gpt-5.4 \
    --output data/llm_case_descriptions.json

# 3. Encode with BiomedCLIP
python MulModSeg_2024/text_embedding/generate_embeddings.py \
    --clip_model biomedclip \
    --descriptions data/llm_case_descriptions.json \
    --output text_embedding/bone_tumor_llm_biomedclip.pth
```

---

## Ablation Dependency Graph

```
Stage 0 (Quick Wins)          Stage 1 (Core)         Stage 2 (Plug-ins)     Stage 3 (Baselines)
─────────────────────      ─────────────────       ──────────────────      ───────────────────
P0_boundary_dice ──┐                                   
                    ├──→ E1c_combined ──→ E2c_best_attention
P0_modality_routing┘        ↑                        ↑
                    E1a_biomedclip ──────────────────┘
P0_pelvis_sample ──┘        ↑                        E3a_unet_ct ──┐
                    E1b_prog_freeze                    E3b_swin_ct ──┤
P0_decoder_se ─────┘                                    E3c_mamba ───┘
                                                           ↓
                                                      E3d_table
```

---

## GPU Budget Tracking

| Stage | Experiments | Est. GPU-hours | Cumulative |
|-------|------------|----------------|------------|
| Stage 0 | 4 pilots | 2.0h | 2.0h |
| Stage 1 | E1a-c | 3.0h | 5.0h |
| Stage 2 | E2a-c | 3.0h | 8.0h |
| Stage 3 | E3a-c | 2.5h | 10.5h |
| Stage 4 | E4a-f | 6.0h | 16.5h |
| Stage 5 | E5a-b | 2.0h | 18.5h |
| **Total** | | **18.5h** | |

> ⚠️ Total exceeds MAX_TOTAL_GPU_HOURS=8. Stages 4-5 exceed budget — defer to dedicated experiment run.
> Within-budget (8h): Stages 0-2 (8 GPU-hours) covers the core improvement + quick wins + plug-ins.

---

## Experiment Tracker

| Exp ID | Status | GPU-h Used | Foreground Dice | Pelvis Dice | Notes |
|--------|--------|-----------|-----------------|-------------|-------|
| P0_boundary_dice | ⬜ Pending | — | — | — | ⚠️ Run on GPU 0 |
| P0_modality_routing | ⬜ Pending | — | — | — | ⚠️ Run on GPU 1 |
| P0_pelvis_sample | ⬜ Pending | — | — | — | After P0_boundary |
| P0_decoder_se | ⬜ Pending | — | — | — | After P0_modality |
| E1a_biomedclip | ⬜ Pending | — | — | — | ⚠️ MANUAL: weight download |
| E1b_progressive_freeze | ⬜ Pending | — | — | — | |
| E1c_combined | ⬜ Pending | — | — | — | Depends on E1a+E1b+P0 best |
| E2a_cbam_decoder | ⬜ Pending | — | — | — | |
| E2b_ema_bottleneck | ⬜ Pending | — | — | — | |
| E2c_best_attention | ⬜ Pending | — | — | — | |
| E3a_unet_ct | ⬜ Pending | — | — | — | |
| E3b_swinunetr_ct | ⬜ Pending | — | — | — | |
| E3c_mamba_ct | ⬜ Pending | — | — | — | ⚠️ MANUAL: Mamba code |
| E4a-f (ablation) | ⬜ Pending | — | — | — | Deferred |
| E5a-b (LLM text) | ⬜ Pending | — | — | — | ⚠️ MANUAL: API key |

---

## First 3 Runs to Launch

```bash
# Terminal 1 (GPU 0):
python MulModSeg_2024/train.py --config configs/experiments.yaml \
    --experiment P0_boundary_dice \
    --loss.type dicece --loss.boundary_dice_weight 0.3 \
    --device 0

# Terminal 2 (GPU 1):
python MulModSeg_2024/train.py --config configs/experiments.yaml \
    --experiment P0_modality_routing \
    --model.modality_embedding true \
    --device 1

# After P0_boundary_dice completes (GPU 0):
python MulModSeg_2024/train.py --config configs/experiments.yaml \
    --experiment P0_pelvis_sample \
    --sampling.pelvis_weight 2.0 \
    --device 0
```
