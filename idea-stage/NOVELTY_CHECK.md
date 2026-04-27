# Novelty Check Report

**Date**: 2026-04-27
**Top Ideas Checked**: 10

---

## Idea 1: BiomedCLIP + Progressive Freeze for 3D Multimodal Text-Guided Segmentation

**Closest work**:
- BiomedCLIP (Zhang et al., 2023, arXiv:2303.00915): Biomedical CLIP pretraining — used for classification, not 3D multimodal segmentation with MoE routing
- Progressive freezing (Howard & Ruder, 2018; ULMFiT): Standard technique for transfer learning

**Differentiation**: The combination (BiomedCLIP text embeddings + progressive SwinUNETR freezing + text-guided MoE routing for 3D CT/MR bone tumor segmentation) is novel. No prior work combines these three elements.
**Novelty**: ✅ CONFIRMED — delta is the integration pattern, not individual components

## Idea 2: 3D Decoder CBAM/SE for Multimodal Medical Segmentation

**Closest work**:
- CBAM (Woo et al., ECCV 2018): 2D channel-spatial attention, widely used
- 3D CBAM variants exist in literature (e.g., 3D-CBAM for brain tumor, 2021)
- Attention U-Net (Oktay et al., 2018): Attention gates in decoder skip connections

**Differentiation**: Applying CBAM/SE specifically in the single-decoder path of a dual-encoder multimodal fusion architecture, after multi-level skip fusion, is not found in prior work. The interaction between attention refinement and cross-modal fused features is untested.
**Novelty**: ✅ CONFIRMED (moderate — plug-in application to specific architecture)

## Idea 3: nnMamba/U-Mamba CT Baseline for Bone Tumor

**Closest work**:
- nnMamba (Gong et al., 2024, arXiv:2402.03526): 3D biomedical segmentation with SSM
- U-Mamba (Ma et al., 2024): Hybrid CNN-SSM for medical
- Comprehensive Mamba 3D analysis (Wang et al., 2025, arXiv:2503.19308)

**Differentiation**: Applying Mamba-based encoders to bone tumor CT segmentation is novel application. But methodology is not novel.
**Novelty**: ⚠️ WEAK — novel application, not novel method. Useful as baseline, not as contribution.

## Idea 4: LLM-Generated Structured Descriptions for MoE Routing

**Closest work**:
- VoxelPrompt (Hoopes et al., 2024, arXiv:2410.08397): LLM + vision agent for medical tasks
- LViT (Li et al., 2023): Language meets vision transformer for medical segmentation
- LaMed (2024): Language-driven medical image analysis

**Differentiation**: Using LLM to generate structured JSON anatomical descriptions → then encoding via BiomedCLIP → then using as MoE router input for dynamic kernel selection is a novel pipeline. The specific chain (LLM → structured schema → domain CLIP → MoE routing) is not in prior work.
**Novelty**: ✅ CONFIRMED — pipeline integration is novel

## Idea 5: Pelvis-Centric Hard Example Mining

**Closest work**: Standard class-imbalance techniques (oversampling, focal loss, hard example mining)
**Differentiation**: Anatomy-specific targeted oversampling for pelvis bone tumors
**Novelty**: ⚠️ WEAK — standard technique, novel application only

## Idea 6: 3D FreqFusion for Multimodal Medical Skip Connections

**Closest work**:
- FreqFusion (Chen et al., TPAMI 2024): 2D frequency-aware feature fusion
- No 3D multimodal adaptation found

**Differentiation**: 3D adaptation of FreqFusion for skip-level CT/MR fusion in segmentation
**Novelty**: ✅ CONFIRMED — 3D + multimodal adaptation is novel

## Idea 7: Anatomy-Aware MoE (2→3+ Experts)

**Closest work**:
- Anatomy-guided segmentation (various): Uses anatomy priors as input or loss
- V-MoE (Riquelme et al., 2021): MoE for vision, but not anatomy-routed
- No anatomy-supervised MoE routing for medical segmentation found

**Differentiation**: Using anatomy labels to supervise MoE router expert selection
**Novelty**: ✅ CONFIRMED — anatomy-supervised MoE routing is novel

## Idea 8: Boundary Dice Loss

**Closest work**: Boundary loss (Kervadec et al., 2019), Boundary Dice (various)
**Differentiation**: Already implemented in codebase; standard technique
**Novelty**: ❌ NONE — well-established technique

## Idea 9: 3D EMA Attention at Multimodal Bottleneck

**Closest work**:
- EMA (Ouyang et al., ICASSP 2023): 2D efficient multi-scale attention
- No 3D adaptation or multimodal bottleneck application found

**Differentiation**: 3D EMA applied after cross-modal attention fusion
**Novelty**: ✅ CONFIRMED (moderate — 3D adaptation + specific insertion point)

## Idea 10: Modality-Aware MoE Routing Bias

**Closest work**: Modality embeddings in multimodal networks (standard)
**Differentiation**: Learnable modality embedding explicitly added to MoE router input
**Novelty**: ⚠️ WEAK — straightforward extension, but effective
