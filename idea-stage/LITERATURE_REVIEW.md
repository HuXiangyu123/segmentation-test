# Literature Review: Multimodal 3D Bone Tumor Segmentation

**Date**: 2026-04-27
**Direction**: Improve MulModSeg_2024 baseline via encoder optimization, multimodal fusion, text-guided routing, and plug-in modules

---

## 1. Research Landscape

### 1.1 Multimodal Medical Image Segmentation (CT + MR Fusion)

The field of multimodal 3D medical segmentation has evolved across several axes:

**Cross-Modal Attention Fusion**: The dominant paradigm uses transformer-based cross-attention between modality-specific encoder branches. MulModSeg_2024 already implements bidirectional cross-attention at the bottleneck (dec4) with gated skip fusion. Recent advances include:
- Frequency-domain fusion (FreqFusion, TPAMI 2024) — addresses intra-category inconsistency and boundary displacement via adaptive low/high-pass filtering
- Deformable cross-attention for handling modality misalignment
- Token-level fusion with learned modality embeddings

**Lightweight Fusion Plug-ins**: From docs/plug, several validated fusion enhancement modules are available:
- **FreqFusion**: Frequency-aware feature fusion for multi-scale scenarios
- **MASAG**: Multi-Scale Adaptive Spatial Attention Gate
- **FCM/FFM**: Feature Correction/Fusion Modules from CFFormer

### 1.2 General 3D Medical Segmentation Models (CT Single-Modality Baselines)

Current state-of-the-art models for comparison on CT-only:
- **nnU-Net v2** (2023): Self-configuring framework, still a strong baseline
- **MedNeXt** (2023/2024): ConvNeXt-style 3D architecture with kernel size scaling
- **SwinUNETR** (2022): Already the backbone of MulModSeg
- **nnMamba** (arXiv:2402.03526, 2024): 3D biomedical segmentation with State Space Models
- **U-Mamba** (2024): Mamba-based U-Net with hybrid SSM-CNN blocks
- **MambaMIM** (arXiv:2408.08070, 2024): Mamba pretraining for medical segmentation
- **Comprehensive Mamba Analysis** (arXiv:2503.19308, 2025): Systematic evaluation of Mamba for 3D volumetric segmentation

**Key insight**: nnMamba (2024) and U-Mamba (2024) demonstrate that SSM blocks can match or exceed transformer performance at lower FLOPs for 3D medical tasks. The 2025 comprehensive analysis (arXiv:2503.19308) provides concrete guidance on Mamba's strengths/limitations in 3D volumetric settings.

### 1.3 Few-Shot / Small-Sample Medical Segmentation

Given only 110 cases (CT+MR paired), small-sample strategies are critical:

- **Transfer Learning**: Pretrained encoders from large-scale medical datasets (e.g., SwinUNETR pretrained on BTCV, AbdomenCT-1K, or Totalsegmentator)
- **Progressive Freezing**: Gradual unfreezing of encoder layers to prevent overfitting, validated in multiple low-data regimes
- **Self-supervised pretraining**: MAE, SimMIM-style masked image modeling on in-domain CT/MR data
- **Prototype-based few-shot**: ADNet (2022) uses anomaly detection with supervoxel self-supervision
- **Foundation model adaptation**: Medical SAM Adapter (arXiv:2304.12620), SAM-Med3D

**Key insight**: The improvement doc correctly identifies progressive stage-wise freezing as the most practical approach for this dataset size. Combined with SwinUNETR's pretrained encoder, this directly addresses the stability issues mentioned.

### 1.4 Text-Guided / Vision-Language Medical Segmentation

The MulModSeg text-guided MoE head is a differentiating feature. Current landscape:

- **BiomedCLIP** (arXiv:2303.00915, 2023): Pretrained on 15M biomedical image-text pairs (PubMed Central). Significantly outperforms standard CLIP on medical tasks. Available via HuggingFace.
- **MedCLIP** (2023): Medical vision-language pretraining with decoupled contrastive learning
- **PMC-CLIP** (2023): Pretrained on PubMed Central articles
- **Medical VLP Survey** (arXiv:2312.06224): Comprehensive review of medical vision-language pretraining
- **VoxelPrompt** (arXiv:2410.08397, 2024): Vision agent integrating LLM with vision network for medical tasks

**Key insight**: The current MulModSeg uses generic CLIP embeddings. Replacing with BiomedCLIP or MedCLIP would provide domain-specific text representations. Combined with LLM-generated structured descriptions (improvement direction #1), this directly addresses the text prior instability issue.

### 1.5 Mixture-of-Experts for Segmentation

MulModSeg's DynamicSemanticHead with 2 experts is a novel contribution. Related work:
- **MoE-Transformer** variants for vision: V-MoE, MoCov3-MoE
- **Dynamic convolution**: Conditional parameter generation (CondConv, DY-CNNs) — conceptually similar to MulModSeg's text-conditioned kernel blending
- **Hypernetworks**: Small network generates weights for main network

**Key insight**: The current 2-expert design could be extended to modality-aware experts (CT-specific vs MR-specific experts) or anatomy-aware experts (pelvis vs femur vs spine).

### 1.6 Encoder Optimization for 3D Medical Imaging

- **SwinUNETR pretrained weights**: Available from BTCV (multi-organ CT, 103 cases) and other public datasets
- **Foundation models**: SAM-Med3D, MedSAM, Totalsegmentator (CT) provide strong pretrained features
- **Progressive layer freezing**: Standard practice: freeze early layers → train head → gradually unfreeze deeper layers
- **LoRA-like adaptation**: Insert trainable low-rank adapters into frozen pretrained encoders

### 1.7 Attention Plug-ins (Available from docs/plug)

Validated attention modules that can be inserted as feature refiners:

| Module | Type | 3D-Adaptable | Cost |
|--------|------|-------------|------|
| **SE** | Channel recalibration | Yes (AdaptiveAvgPool3d) | Very low |
| **CBAM** | Channel + Spatial | Yes | Low |
| **ECA** | Efficient channel attention | Yes | Very low |
| **EMA** | Multi-scale cross-spatial | Yes (reshape groups) | Medium |
| **MCA** | Multi-dimensional collaborative | Yes | Medium |
| **GSA** | Geometry self-attention | Yes (native 3D) | High |
| **LWGA** | Light-weight grouped attention | Yes | Low |

## 2. Key Gaps Identified

1. **Text modality gap**: Generic CLIP vs. medical domain → BiomedCLIP/MedCLIP bridge needed
2. **No modern 3D baselines**: Missing comparison with nnMamba, U-Mamba, MedNeXt on CT single modality
3. **Under-utilized pretraining**: SwinUNETR encoder pretraining not leveraged with progressive freezing
4. **Static MoE routing**: 2 experts fixed — could be modality-aware or anatomically-specialized
5. **No attention refinement**: Decoder path lacks channel/spatial attention; plug-in modules (SE, CBAM, EMA) could enhance features
6. **Pelvis class imbalance**: Complex pelvic structures underperform — needs targeted sampling + hard-example focus
7. **Fusion at single level only**: Cross-attention at bottleneck only; multi-scale fusion (FreqFusion-style) unexplored

## 3. Competing Approaches Summary

| Approach | Strength | Gap vs. MulModSeg |
|----------|----------|-------------------|
| nnU-Net v2 | No manual tuning, strong baseline | No multimodal fusion, no text guidance |
| nnMamba/U-Mamba | Efficient long-range modeling | No multimodal, no text guidance |
| SAM-Med3D | Promptable, strong generalization | 2D foundation, no native 3D text routing |
| BiomedCLIP + UNet | Domain-specific text | No MoE dynamic head, no multimodal fusion |
| FreqFusion-enhanced UNet | Boundary preservation | No text routing, no cross-modal attention |
