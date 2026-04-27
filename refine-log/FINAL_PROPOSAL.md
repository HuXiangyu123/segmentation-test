# Final Proposal: MulModSeg_2024 Baseline Improvement

**Date**: 2026-04-27
**Refinement rounds**: 1 (initial proposal)

---

## Problem Anchor

**Problem**: MulModSeg_2024 is a multimodal (CT+MR) 3D bone tumor segmentation model with a novel text-guided Mixture-of-Experts head. However, it has three documented weaknesses:

1. **Text prior instability**: Generic CLIP embeddings produce inconsistent MoE router behavior across random seeds (identified in `docs/reviews/中期进展改进方向.md`)
2. **Pelvis underperformance**: Complex pelvic bone anatomy cases score 10-15% lower Dice than femur cases (identified in project review)
3. **No modern baselines**: Missing comparison with 2024-2025 3D segmentation models (Mamba-based, foundation models)

**Constraints**:
- 110 paired CT+MR cases (small sample regime)
- 2× RTX 3090 (24GB each, no multi-node)
- Binary segmentation (tumor vs background), 96³ patch size
- Must support atomic (independently testable) experiments

## Method Thesis

**Domain-specific text embeddings (BiomedCLIP) + progressive encoder freezing + lightweight decoder attention provide the highest gain-to-cost ratio for improving MulModSeg on small-sample bone tumor data.**

## Dominant Contribution

The key contribution is demonstrating that **medical-domain vision-language pretraining (BiomedCLIP) combined with progressive encoder unfreezing** substantially stabilizes text-guided MoE routing for small-sample multimodal segmentation — a finding not previously shown for bone tumor imaging or for architectures with dynamic text-conditioned segmentation heads.

## Core Improvements (Priority-Ordered)

### 1. BiomedCLIP Text Embedding Swap
- Replace generic OpenAI CLIP with BiomedCLIP (`microsoft/BiomedCLIP-PubMedBERT_256_vit_base_patch16`)
- Domain-specific biomedical text encoder — trained on 15M PubMed image-text pairs
- Expected gain: +2-3% Dice, reduced variance across seeds

### 2. Progressive Encoder Freezing
- Stage-wise unfreezing of SwinUNETR encoder (freeze stages 1-4 → unfreeze 4 → unfreeze 3 → unfreeze all)
- Prevents small-sample overfitting while leveraging large-scale medical pretraining
- Expected gain: +1-2% Dice on small tumor bucket, improved training stability

### 3. Decoder Channel-Spatial Attention
- Insert 3D-adapted SE or CBAM blocks after each decoder upsampling stage
- <5K extra parameters, negligible FLOPs increase
- Expected gain: +1-2% Dice overall

### 4. Quick Wins (near-zero cost)
- Boundary Dice loss (already implemented, config change only)
- Modality-aware MoE routing bias (~10 lines of code)
- Pelvis-centric oversampling in dataloader

## Must-Run Experiments

| Block | Experiments | GPU-hours | Purpose |
|-------|------------|-----------|---------|
| **Core** | BiomedCLIP swap, Progressive freeze, Combined | 3.0h | Validate method thesis |
| **Quick Wins** | Boundary Dice, Modality Routing, Pelvis Sampling | 1.5h | Low-cost signal |
| **Plug-ins** | CBAM, SE, EMA in decoder | 3.0h | Feature refinement |
| **Baselines** | U-Net CT, SwinUNETR CT, Mamba CT | 2.5h | Modern comparison |
| **Ablations** | CT/MR only, Add/Concat/Cross-Attn | 6.0h | Paper completeness |

## Success Criteria

| Metric | Target | Current (E0_baseline) |
|--------|--------|----------------------|
| Foreground Dice (overall) | >0.75 | ~0.70 (estimated) |
| Foreground Dice (pelvis bucket) | Improve +5% | Underperforming |
| Dice variance (across seeds) | Reduce by 50% | High (text prior unstable) |
| Precision | >0.80 | ~0.75 (estimated) |

## Risks & Mitigations

| Risk | Probability | Mitigation |
|------|------------|------------|
| BiomedCLIP weights incompatible with current embedding pipeline | Low | Fall back to MedCLIP or PMC-CLIP |
| Progressive freezing hurts convergence | Low | Test different unfreeze schedules; fall back to full fine-tuning |
| Mamba encoder integration complex | Medium | Defer to Stage 3; implement only if Stage 1-2 gains confirmed |
| LLM API unavailable | High (short term) | Stage 5 is optional; core gains from Stages 1-2 don't depend on LLM |

## Non-Goals

- DDP/multi-GPU training (engineering, not algorithmic improvement)
- Full-volume inference (out of scope for baseline improvement)
- Multi-class segmentation (current data is binary)
- Real-time inference optimization
