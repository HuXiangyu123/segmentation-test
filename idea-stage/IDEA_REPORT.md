# Idea Discovery Report

**Direction**: Improve MulModSeg_2024 baseline — encoder optimization, multimodal fusion, text-guided routing, 3D comparison baselines
**Date**: 2026-04-27
**Pipeline**: research-lit → idea-creator → novelty-check → research-review → research-refine-pipeline

---

## Executive Summary

Analyzed MulModSeg_2024 (multimodal CT+MR 3D bone tumor segmentation with text-guided MoE head) across 5 improvement axes. Generated 10 atomic improvement ideas, filtered by feasibility on dual 3090 GPUs, novelty, and expected impact. **Top recommendation**: Encoder Embedding Optimization (BiomedCLIP + Progressive Freezing) provides the highest expected gain (~3-5% Dice on pelvis cases) with moderate implementation cost. Two quick-win ideas (Boundary Dice Loss + Modality-Aware Routing) can be tested immediately with near-zero code change.

---

## Literature Landscape

[Full literature review →](LITERATURE_REVIEW.md)

**Key findings**:
- BiomedCLIP (2023, 15M biomedical pairs) consistently outperforms generic CLIP for medical tasks
- nnMamba/U-Mamba (2024) offer efficient alternatives to transformer backbones for 3D medical
- FreqFusion (TPAMI 2024) addresses boundary displacement in multi-scale fusion
- Progressive encoder freezing is the validated strategy for <200 sample medical datasets
- Attention plug-ins (SE/CBAM/EMA) provide cheap channel/spatial refinement

---

## Ranked Ideas

### 🏆 Idea 1: Encoder Embedding Optimization (BiomedCLIP + Progressive Freeze) — RECOMMENDED

**Axis**: Encoder + Text Embedding
**Hypothesis**: Replacing generic CLIP with BiomedCLIP for text features, combined with progressive stage-wise freezing of the pretrained SwinUNETR encoder, improves segmentation Dice by 3-5% on pelvis cases and increases training stability (lower variance across seeds).

**Atomic sub-components** (test independently):
- **1a. BiomedCLIP Swap**: Replace `text_embedding/generate_embeddings.py` CLIP model with BiomedCLIP (`microsoft/BiomedCLIP-PubMedBERT_256_vit_base_patch16`). Regenerate all case embeddings. Compare E0_baseline.
- **1b. Progressive Freeze Schedule**: Start with fully frozen SwinUNETR encoder (stages 1-4), train only decoder + dynamic head for 5 epochs → unfreeze stage 4 for 5 epochs → unfreeze stage 3 for 5 epochs → unfreeze all for remaining epochs.

**Implementation**:
- `text_embedding/generate_embeddings.py`: swap CLIP → BiomedCLIP
- `train.py`: add `freeze_schedule` parameter controlling layer-wise requires_grad
- Config: `freeze_schedule: [0,0,0,0,5,10,15]` (stages to freeze, unfreeze epochs)
- **⚠️ MANUAL**: Download BiomedCLIP weights (`microsoft/BiomedCLIP-PubMedBERT_256_vit_base_patch16`, ~800MB)
- **⚠️ MANUAL**: Download SwinUNETR pretrained weights if not cached

**GPU Cost**: Same as baseline (text encoding is pre-computed; progressive freeze doesn't change forward FLOPs)
**Pilot feasibility**: ✅ Fits 2h/GPU budget

---

### Idea 2: Decoder Channel-Spatial Attention (CBAM/SE Plug-in)

**Axis**: Plug-in Module (Attention)
**Hypothesis**: Inserting 3D-adapted CBAM or SE blocks after each decoder upsampling stage provides +1-2% Dice improvement at negligible parameter cost (<5K extra parameters).

**Atomic variants** (test independently):
- **2a. SE Block**: `AdaptiveAvgPool3d(1) → Conv1d(down) → ReLU → Conv1d(up) → Sigmoid` after each `UnetrUpBlock`
- **2b. CBAM Block**: Channel attention + spatial attention (3D-adapted) after each `UnetrUpBlock`
- **2c. EMA Block**: Multi-scale cross-spatial attention at bottleneck only

**Implementation**:
- Create `MulModSeg_2024/model/attention_plugins.py` with 3D versions of SE, CBAM, EMA
- Modify `SwinUNETR.forward_from_dec4` to apply attention after each decoder block
- Config: `decoder_attention: se | cbam | ema | none`
- **No manual steps required**

**GPU Cost**: +0.1% FLOPs (SE), +0.3% FLOPs (CBAM), +0.5% FLOPs (EMA)
**Pilot feasibility**: ✅ Fits 2h/GPU budget

---

### Idea 3: nnMamba/U-Mamba CT Single-Modality Baseline

**Axis**: General 3D Segmentation Comparison
**Hypothesis**: Replacing SwinUNETR backbone with Mamba-based encoder provides comparable or better CT-only segmentation with lower memory footprint, establishing a modern baseline for comparison. The multimodal fusion gain can then be measured as `MulModSeg_full - Mamba_CT_only`.

**Implementation**:
- Integrate U-Mamba blocks (`VSS Block` from VMamba) as encoder replacement
- Use same `forward_to_dec4` / `forward_from_dec4` interface for fair comparison
- Keep decoder, dynamic head, and loss functions identical
- Config: `backbone: umamba | nnMamba`
- **⚠️ MANUAL**: Decide between U-Mamba (2024) and nnMamba (2024) implementations
- **⚠️ MANUAL**: Download pretrained weights if using pretrained variant

**GPU Cost**: Lower than SwinUNETR (SSM is O(N) vs attention O(N²))
**Pilot feasibility**: ⚠️ May exceed 2h if implementation complex; mark as "validate on paper first"

---

### Idea 4: LLM-Enhanced Structured Text Descriptions

**Axis**: Text Prior Improvement
**Hypothesis**: Replacing simple template-based CLIP prompts with LLM-generated structured anatomical descriptions (GPT-5.4 or medical LLM) produces more discriminative text embeddings, improving the MoE router's ability to select appropriate experts for each case.

**Implementation**:
- Create `scripts/generate_llm_descriptions.py`: for each case, send structured prompt to LLM API requesting JSON with fields: `tumor_location`, `bone_involved`, `size`, `morphology`, `signal_characteristics`
- Feed LLM output → BiomedCLIP text encoder → replace current `case_text_embedding`
- **⚠️ MANUAL**: Set up LLM API key (OpenAI/Anthropic) in `.env`
- **⚠️ MANUAL**: Run description generation script (API cost, ~110 cases × few cents each)
- **⚠️ MANUAL**: Regenerate embeddings with BiomedCLIP

**GPU Cost**: None (text precomputed)
**Pilot feasibility**: ✅ Precomputed, but LLM API dependency makes it "needs manual pilot"

---

### Idea 5: Pelvis-Centric Hard Example Mining

**Axis**: Class Imbalance / Sample Efficiency
**Hypothesis**: Targeted oversampling of pelvic cases (identified as underperforming) combined with case-level focal loss weighting improves pelvis-region Dice by 5-8%.

**Implementation**:
- Identify pelvic cases from metadata → create `pelvis_weight: 2.0` in dataloader
- Modify `dataloader_bone_tumor.py`: `pos_neg_ratio: 5.0` for pelvic cases (vs. 3.0 default)
- Add `case_weight` tensor to loss computation
- **No manual steps required**

**GPU Cost**: None additional
**Pilot feasibility**: ✅ Fits 2h/GPU budget

---

### Idea 6: Multi-Scale FreqFusion for Skip Connections

**Axis**: Fusion Enhancement (Plug-in)
**Hypothesis**: Replacing the current per-level GatedSkipFusion with FreqFusion-style frequency-aware fusion (TPAMI 2024) reduces boundary displacement and improves Dice on small tumors (GT<2% bucket) by 3-5%.

**Implementation**:
- 3D-adapt FreqFusion from `docs/plug/fusion-enhancement-plug-ins.md`
- Replace `GatedSkipFusion3D` with `FreqFusion3D` at each skip level
- Keep cross-attention at bottleneck unchanged
- **No manual steps required** (FreqFusion paper code is MIT licensed)

**GPU Cost**: +5-10% FLOPs (frequency transforms + CARAFE upsampling)
**Pilot feasibility**: ⚠️ Medium — implementation complexity moderate

---

### Idea 7: Anatomy-Aware MoE (2 → 3+ Experts)

**Axis**: Dynamic Head Enhancement
**Hypothesis**: Increasing DynamicSemanticHead from 2 to 3 experts, with routing supervised by anatomical region (pelvis vs femur vs spine), allows more specialized kernel banks and improves segmentation of region-specific tumor morphology.

**Implementation**:
- `num_experts: 3` in model config
- Add anatomy classification head as auxiliary branch (supervise router logits)
- Requires anatomy labels in dataset (derivable from case metadata or segmentation masks)
- **⚠️ MANUAL**: Annotate or derive anatomy region labels for training data

**GPU Cost**: Negligible (+0.1% params, in head only)
**Pilot feasibility**: ✅ Fits 2h/GPU budget (but needs anatomy labels first)

---

### Idea 8: Boundary Dice Loss — QUICK WIN

**Axis**: Loss Function
**Hypothesis**: Adding boundary Dice loss (already implemented in `custom_losses.py`) as auxiliary loss improves edge agreement, particularly for small and pelvic tumors where boundary precision is critical.

**Implementation**:
- Already implemented in `MulModSeg_2024/utils/custom_losses.py`
- Config change only: add `boundary_dice_weight: 0.3` to loss config
- **No manual steps, no code changes required**

**GPU Cost**: +2% compute (morphological gradient per batch)
**Pilot feasibility**: ✅ Trivial — config change only

---

### Idea 9: Frequency-Domain EMA Attention at Bottleneck

**Axis**: Attention Plug-in
**Hypothesis**: Inserting 3D-adapted EMA (Efficient Multi-Scale Attention) after the cross-attention fusion at the bottleneck improves multi-scale feature representation at low computational cost.

**Implementation**:
- Create 3D version of EMA from `docs/plug/attention-plug-ins.md`
- Insert after `self.cross_attention` output (before `cross_attn_proj`)
- Config: `bottleneck_attention: ema | none`
- **No manual steps required**

**GPU Cost**: +0.5% FLOPs
**Pilot feasibility**: ✅ Fits 2h/GPU budget

---

### Idea 10: Modality-Aware Routing Bias — QUICK WIN

**Axis**: Dynamic Head Enhancement
**Hypothesis**: Adding a learnable modality-specific embedding to the MoE router input improves expert selection by making routing sensitive to whether the input is CT-only, MR-only, or fused.

**Implementation**:
- Add `self.modality_embedding = nn.Embedding(3, 256)` (CT, MR, MIX)
- Concatenate to text features before router: `router_input = text_features + modality_emb[modality_idx]`
- ~10 lines of code change in `MulModSeg.py`
- **No manual steps required**

**GPU Cost**: Negligible (768 extra parameters)
**Pilot feasibility**: ✅ Trivial

---

## Pilot Experiment Design

**Hardware**: 2× RTX 3090 (24GB each)
**Budget**: MAX_TOTAL_GPU_HOURS = 8; PILOT_MAX_HOURS = 2 per idea; PILOT_TIMEOUT_HOURS = 3

### Pilots to Run (in parallel, 2 GPUs)

| Pilot | Idea | GPU | Est. Time | Config |
|-------|------|-----|-----------|--------|
| P1 | Idea 8: Boundary Dice Loss | GPU 0 | 0.5h | E0_baseline + boundary_dice_weight=0.3 |
| P2 | Idea 10: Modality-Aware Routing | GPU 1 | 0.5h | E0_baseline + modality_embedding=true |
| P3 (after P1) | Idea 5: Pelvis-Centric Sampling | GPU 0 | 0.5h | E0_baseline + pelvis_weight=2.0 |
| P4 (after P2) | Idea 2: Decoder SE Attention | GPU 1 | 0.5h | E0_baseline + decoder_attention=se |

**Total pilot GPU hours**: ~2h (4 pilots × 0.5h on 2 GPUs)
**Baseline reference**: E0_baseline (DiceCE, default sampling)
**Success criterion**: +2% foreground Dice over baseline on pelvis bucket

---

## Eliminated Ideas

| Idea | Reason Eliminated | Phase |
|------|-------------------|-------|
| Hybrid Swin-Mamba Encoder | Too complex, not atomic, hard to ablate which component helps | Idea filtering |
| Diffusion Augmentation | High GPU cost (>2h/pilot), stochastic training outcomes complicate ablation | Idea filtering |
| Dual-GPU DDP | Engineering effort, not an algorithmic improvement; save for production scaling | Idea filtering |
| Modality-Specific Expert Branches | Premature — need to first validate that modality-aware routing (Idea 10) helps | Deferred |

---

---

## Novelty Verification Results

[Full novelty report →](NOVELTY_CHECK.md)

| Idea | Novelty | Closest Work | Differentiation |
|------|---------|-------------|-----------------|
| 1: BiomedCLIP + Progressive Freeze | ✅ CONFIRMED | BiomedCLIP (2023) | Integration pattern for 3D MoE routing |
| 2: Decoder CBAM/SE | ✅ CONFIRMED | CBAM (ECCV 2018) | 3D + multimodal decoder application |
| 3: nnMamba CT Baseline | ⚠️ WEAK | nnMamba (2024) | Novel application, not novel method |
| 4: LLM Structured Descriptions | ✅ CONFIRMED | VoxelPrompt (2024) | LLM→JSON→BiomedCLIP→MoE chain |
| 5: Pelvis-Centric Sampling | ⚠️ WEAK | Standard techniques | Anatomy-specific targeting |
| 6: 3D FreqFusion Multimodal | ✅ CONFIRMED | FreqFusion (TPAMI 2024) | 3D + multimodal adaptation |
| 7: Anatomy-Aware MoE (3+ experts) | ✅ CONFIRMED | V-MoE (2021) | Anatomy-supervised routing |
| 8: Boundary Dice Loss | ❌ NONE | Boundary loss (2019) | Already implemented |
| 9: 3D EMA Bottleneck Attention | ✅ CONFIRMED | EMA (ICASSP 2023) | 3D adaptation + fusion insertion |
| 10: Modality-Aware Routing Bias | ⚠️ WEAK | Standard modality embeddings | MoE-specific routing |

---

## External Critical Review Summary

**Reviewer**: Simulated senior reviewer (NeurIPS/ICML level)
**Score**: 7/10 (for the overall improvement program)

**Strengths**:
- Systematic approach: document weaknesses → literature review → atomic ideas → pilots → full experiments
- Good constraint awareness: small-sample regime, dual 3090 budget, atomic experiments
- BiomedCLIP + progressive freezing is well-motivated and underexplored for text-guided MoE segmentation
- Quick-win experiments (Boundary Dice, Modality Routing) provide low-cost signal before heavy investment

**Weaknesses**:
- Core method thesis risks being an "integration paper" — BiomedCLIP and progressive freezing are individually known. Need to demonstrate that the combination produces non-obvious gains.
- Missing baseline: should compare against nnU-Net v2 as the strongest non-foundation-model CT baseline
- 110 cases with 96³ patches may be insufficient for stable MoE routing; consider router regularization
- Experiment plan could be strengthened with statistical testing plan (e.g., Wilcoxon signed-rank across folds)

**Minimum Viable Improvements** (to reach score ≥ 8):
1. Add nnU-Net v2 CT-only baseline (strong, reproducible comparison)
2. Add router entropy regularization to prevent expert collapse on small data
3. Report per-fold statistics with confidence intervals, not just means
4. Include failure case analysis (visualizations of worst pelvis cases before/after)

---

## Refined Proposal

- **Proposal**: [`refine-log/FINAL_PROPOSAL.md`](../refine-log/FINAL_PROPOSAL.md)
- **Experiment plan**: [`refine-log/EXPERIMENT_PLAN.md`](../refine-log/EXPERIMENT_PLAN.md)
- **Tracker**: [`refine-log/EXPERIMENT_TRACKER.md`](../refine-log/EXPERIMENT_TRACKER.md)

### Summary

**Problem anchor**: MulModSeg text prior instability + pelvis underperformance + no modern baselines

**Method thesis**: BiomedCLIP + progressive encoder freezing + lightweight decoder attention

**Dominant contribution**: Medical-domain VLP (BiomedCLIP) combined with progressive unfreezing stabilizes text-guided MoE routing for small-sample multimodal 3D segmentation

**Must-run experiments**: 14 experiments across 5 stages, GPU budget: 8-18.5h

**⚠️ Manual operations required**:
- BiomedCLIP weight download (~800MB)
- SwinUNETR pretrained weight verification
- Mamba encoder implementation (for CT baseline)
- LLM API key setup (for Stage 5, deferred)

---

## Next Steps

- [ ] **Run Stage 0 pilots** (P0_boundary_dice, P0_modality_routing, P0_pelvis_sample, P0_decoder_se) — 2 GPU-hours
- [ ] **Download BiomedCLIP weights** (⚠️ MANUAL — see `refine-log/EXPERIMENT_TRACKER.md` checklist)
- [ ] **Run Stage 1 core** (E1a_biomedclip, E1b_progressive_freeze, E1c_combined) — 3 GPU-hours
- [ ] **Run Stage 2 plug-ins** if Stage 1 positive — 3 GPU-hours
- [ ] **Run Stage 3 baselines** for paper comparison table — 2.5 GPU-hours
- [ ] If top idea shows +3% Dice → `/auto-review-loop` to iterate until submission-ready
- [ ] Or invoke `/research-pipeline` for the complete end-to-end flow

### Quickest Path to Result (MVP)
```
1. P0_modality_routing (30 min) — zero risk, quick signal
2. P0_boundary_dice (30 min) — config change only
3. E1a_biomedclip (2h) — highest expected impact
   → If E1a shows +2% Dice: proceed to E1c_combined
   → If E1a shows no gain: pivot to E2 plug-ins or E3 baselines
```
