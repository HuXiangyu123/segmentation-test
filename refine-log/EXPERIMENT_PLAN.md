# Experiment Plan: MulModSeg_2024 Baseline Improvement

**Date**: 2026-04-28 (v3 — LLM B7 mandatory, M-style milestones, comparative/ baselines)
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

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Milestone |
|-------|----------------|-----------------------------|-----------|
| C1: Domain-specific text stabilizes MoE routing | Text prior is the weakest link (docs review) | +2% Dice on pelvis, -30% seed variance | M2 |
| C2: Pretrained encoder freezing prevents overfit | n=110 demands transfer learning | Freeze variant > full fine-tune; best strategy identified | M2 |
| C3: LLM-structured descriptions enhance router | LLM gives richer anatomical priors vs simple prompts | +1% Dice over BiomedCLIP baseline | M3 |
| C4: Lightweight decoder plug-ins add gain | Must validate, not decorative | +1% Dice at <1% param increase | M3 |
| C5: Our method competes with modern 3D models | Missing baselines was documented gap | At least tied with strongest 2024-2025 models on CT-only | M4 |

---

## Paper Storyline

- **Main paper must prove**: C1 + C2 + C3 (primary), C4 (supporting), C5 (context)
- **Appendix can support**: Full ablation (CT-only, MR-only, fusion variants), failure case analysis, per-bucket breakdown, LLM prompt template design
- **Experiments intentionally cut**: Diffusion augmentation (GPU-heavy), DDP training (engineering), Hybrid Swin-Mamba (not atomic)

---

## Experiment Milestones

### M0: Quick-Win Pilots (Sanity Check)
**Budget**: 2 GPU-hours | **Claims**: Sanity

Quick validation that each code change works. All runnable immediately (no manual downloads).

| Exp ID | Description | Changes | Est. Time | GPU |
|--------|-------------|---------|-----------|-----|
| **M0_boundary_dice** | Enable boundary Dice loss | Config: `boundary_dice_weight: 0.3` | 0.5h | 0 |
| **M0_modality_routing** | Modality-aware routing bias | `MulModSeg.py`: +10 lines | 0.5h | 1 |
| **M0_pelvis_sample** | Pelvis oversampling | `dataloader_bone_tumor.py`: case weights | 0.5h | 0 |
| **M0_decoder_se** | SE attention in decoder | `attention_plugins.py`: new file | 0.5h | 1 |

**Success criterion**: Code runs without error, metrics logged.

---

### M1: CT Baseline Comparison (Inference Only)
**Budget**: 0.5 GPU-hours | **Claims**: C5

Download and run inference for all comparison models on **single CT input**. No training.

#### Model List

| Exp ID | Model | Source | Download To |
|--------|-------|--------|-------------|
| **M1_unet** | 3D U-Net | `MulModSeg_2024/model/Unet.py` (self-contained) | Already in repo |
| **M1_segresnet** | SegResNet | MONAI `SegResNet` | Comparative test script |
| **M1_swinunetr** | Native SwinUNETR | MONAI `SwinUNETR` (pretrained BTCV) | `comparative/MONAI_SwinUNETR/` |
| **M1_mednext** | MedNeXt-v2 | `https://github.com/MIC-DKFZ/MedNeXt` | `comparative/MedNeXt/` |
| **M1_nnunet** | nnU-Net v2 | `https://github.com/MIC-DKFZ/nnUNet` (TotalSegmentator weights) | `comparative/nnUNet/` |
| **M1_hybridmamba** | HybridMamba | `https://github.com/...` (find 2024-2025 repo) | `comparative/HybridMamba/` |
| **M1_latest_3d** | Top-1 2025-2026 3D segmentation model | To be identified via arXiv/scholar search | `comparative/LatestModel/` |

**⚠️ MANUAL ACTIONS**:
```bash
# Clone all models to comparative/
git clone https://github.com/MIC-DKFZ/MedNeXt.git comparative/MedNeXt
git clone https://github.com/MIC-DKFZ/nnUNet.git comparative/nnUNet
# HybridMamba: find the most recent implementation
git clone <hybridmamba_url> comparative/HybridMamba
# Latest 2025-2026 model: search and clone
git clone <latest_model_url> comparative/LatestModel

# Run inference
python scripts/run_comparative_inference.py \
    --model unet,segresnet,swinunetr,mednext,nnunet,hybridmamba,latest_3d \
    --input data/bone_tumor_ct \
    --output output/comparative_results/ \
    --model_dir comparative/
```

**Metrics**: Foreground Dice, Precision, Recall, HD95
**Success criterion**: At least one variant of our model (M2/M3 best) ties or beats the strongest baseline

---

### M2: Core Improvement (BiomedCLIP + Progressive Freeze)
**Budget**: 5.0 GPU-hours | **Claims**: C1, C2

#### M2a: BiomedCLIP Text Embedding (C1)

| Exp ID | Description | Depends On | Est. Time |
|--------|-------------|-----------|-----------|
| **M2a_biomedclip** | Replace CLIP → BiomedCLIP for all text features | ⚠️ Manual: download weights | 0.5h precompute + 1.0h train |

**⚠️ MANUAL**: Download BiomedCLIP from HuggingFace:
```bash
pip install transformers
python -c "from transformers import AutoModel; AutoModel.from_pretrained('microsoft/BiomedCLIP-PubMedBERT_256_vit_base_patch16', trust_remote_code=True)"
```

#### M2b: Progressive Freeze Strategy Comparison (C2)

Each experiment tests **one fixed freeze configuration** to find the best strategy. All use **SwinUNETR BTCV pretrained weights** (manual download required).

**⚠️ MANUAL**: Download SwinUNETR BTCV pretrained weights:
```bash
python -c "
from monai.networks.nets import SwinUNETR
import torch
model = SwinUNETR(img_size=(96,96,96), in_channels=1, out_channels=14, feature_size=48, pretrained=True)
torch.save(model.state_dict(), './pretrained/swinunetr_btcv.pth')
"
```

| Exp ID | Freeze Config | Description | Est. Time |
|--------|--------------|-------------|-----------|
| **M2b_freeze_all** | Freeze stages 1-4, train decoder+head | Worst-case | 1.0h |
| **M2b_freeze_s4** | Freeze 1-3, thaw stage 4 | Gentle thaw | 1.0h |
| **M2b_freeze_s34** | Freeze 1-2, thaw stages 3-4 | Medium thaw | 1.0h |
| **M2b_full_finetune** | No freeze | Full fine-tune | 1.0h |
| **M2b_no_pretrain** | Train from scratch (no pretrained) | Control | 1.0h |

#### M2c: Combined

| Exp ID | Description | Depends On | Est. Time |
|--------|-------------|-----------|-----------|
| **M2c_best** | BiomedCLIP + best freeze config + best M0 quick-win | M2a, M2b_best, M0_best | 1.0h |

---

### M3: Advanced Improvement (LLM Text + Attention Plug-ins)
**Budget**: 5.0 GPU-hours | **Claims**: C3, C4

#### M3a: LLM-Enhanced Structured Text Descriptions (C3) — MANDATORY

Generate structured anatomical descriptions via LLM → BiomedCLIP encoding → MoE router input.

**⚠️ MANUAL**: Requires LLM API key (OpenAI/Anthropic) in `.env`

##### Structured Output (Struct Output) Schema Design Principles

The LLM response must follow a **fixed JSON schema** to ensure:
1. **Reproducibility**: Same case always produces same description (deterministic prompt + temperature=0)
2. **Schema consistency**: Every case produces identical field structure, enabling fair comparison of router behavior
3. **Medical accuracy**: Fields are constrained to radiologically meaningful attributes
4. **Granularity control**: Coarse enough to avoid overfitting to individual cases, fine enough to discriminate tumor types

```json
{
  "tumor_presence": true,
  "anatomy": {
    "bone": "femur | pelvis | sacrum | spine | rib | tibia | humerus | other",
    "side": "left | right | midline | bilateral",
    "compartment": "proximal | diaphyseal | distal | whole_bone | unknown"
  },
  "tumor_morphology": {
    "shape": "round | oval | lobulated | irregular | infiltrative",
    "margin": "well_defined | ill_defined | sclerotic | mixed",
    "matrix": "lytic | blastic | mixed | cystic | soft_tissue | unknown"
  },
  "signal_on_mr": {
    "t1_weighted": "hypointense | isointense | hyperintense | mixed | not_applicable",
    "t2_weighted": "hypointense | isointense | hyperintense | heterogeneous | not_applicable",
    "enhancement": "none | mild | moderate | strong | heterogeneous"
  },
  "size_mm": "small (<30) | medium (30-80) | large (>80)",
  "cortical_involvement": "intact | erosion | penetration | pathological_fracture",
  "differential_priority": [
    "benign_lesion",
    "primary_bone_tumor",
    "metastasis",
    "infection",
    "other"
  ]
}
```

**Design principles for the schema**:
- **Enum constraints**: Every field uses a closed vocabulary (enums), never free text — this ensures the text encoder produces comparable embeddings
- **Hierarchical structure**: Anatomy → morphology → signal → priority — matches radiological reporting order
- **Optional-aware**: Fields like `signal_on_mr` include `not_applicable` for CT-only cases
- **Priority list**: `differential_priority` is a ranked list, allowing the MoE router to weight differentials
- **Granularity**: 5-7 values per field — fine enough to be discriminative, coarse enough to prevent overfitting

**Implementation**:

| Exp ID | Description | Depends On | Est. Time |
|--------|-------------|-----------|-----------|
| **M3a_schema_design** | Create prompt template + schema enforcer | None | 0.5h code |
| **M3a_llm_generate** | ⚠️ MANUAL: Run LLM for all 110 cases | `.env` with API key | ~1h API calls |
| **M3a_biomedclip_encode** | Encode LLM output → BiomedCLIP embeddings | M2a (BiomedCLIP) | 0.5h |
| **M3a_train** | Train with new LLM-enhanced text features | M2c_best | 1.0h |

#### M3b: Decoder Attention Plug-in (C4)

| Exp ID | Plug-in | Insertion Point | Est. Time |
|--------|---------|----------------|-----------|
| **M3b_se** | 3D SE (channel attention) | After each UnetrUpBlock | 0.5h |
| **M3b_cbam** | 3D CBAM (channel + spatial) | After each UnetrUpBlock | 0.5h |
| **M3b_ema** | 3D EMA (multi-scale attention) | After cross-attn bottleneck | 0.5h |
| **M3b_best** | Best plug-in + M3a config | M3a + M3b best | 1.0h |

---

### M4: Full Ablation Study
**Budget**: 6 GPU-hours | **Claims**: C1-C5 (completeness)

| Exp ID | Description | Config |
|--------|-------------|--------|
| **M4_ct_only** | CT-only, no cross-attention | `train_modality: CT` |
| **M4_mr_only** | MR-only, no cross-attention | `train_modality: MR` |
| **M4_add_fusion** | CT+MR addition, no cross-attn | `use_cross_attention: false` + add |
| **M4_concat_fusion** | CT+MR concat, no cross-attn | `use_cross_attention: false` + concat |
| **M4_cross_attn** | Full cross-attention fusion | E4_cross_attention config |
| **M4_best_model** | All best components combined | M2c + M3b_best |

---

## Dependency Graph

```
M0: Quick-Win Pilots
 │
 ├──→ M2a: BiomedCLIP (⚠️ requires manual weight download)
 │
 ├──→ M2b: Progressive Freeze (⚠️ requires manual BTCV pretrained download)
 │         ├── M2b_freeze_all
 │         ├── M2b_freeze_s4
 │         ├── M2b_freeze_s34
 │         ├── M2b_full_finetune
 │         └── M2b_no_pretrain
 │
 ├──→ M2c: M2a + M2b_best + M0_best
 │
 ├──→ M3a: LLM Text (⚠️ requires API key)
 │         └── M3a_schema → M3a_generate → M3a_encode → M3a_train
 │
 ├──→ M3b: Attention Plug-ins
 │         └── M3b_se → M3b_cbam → M3b_ema → M3b_best
 │
 └──→ M4: Full Ablation
          └── M4_best_model = M2c + M3b_best + M3a (if positive)
```

**M1**: CT baseline comparison — independent, runs in parallel with M2-M4 entirely.

```
M1: Comparative Inference
 ├── M1_unet        (self-contained)
 ├── M1_segresnet   (MONAI)
 ├── M1_swinunetr   (MONAI pretrained)
 ├── M1_mednext     (comparative/MedNeXt)
 ├── M1_nnunet      (comparative/nnUNet)
 ├── M1_hybridmamba (comparative/HybridMamba)
 └── M1_latest_3d   (comparative/LatestModel)
```

---

## GPU Budget Tracking

| Milestone | Experiments | GPU-hours | Cumulative | Priority |
|-----------|------------|-----------|------------|----------|
| M0: Quick Wins | 4 pilots | 2.0h | 2.0h | MUST |
| M1: Baselines (inference) | 7 models × inference only | 0.5h | 2.5h | MUST |
| M2a: BiomedCLIP | 1 exp | 1.5h | 4.0h | MUST |
| M2b: Freeze strategy | 5 exps | 5.0h | 9.0h | MUST |
| M2c: Combined | 1 exp | 1.0h | 10.0h | MUST |
| M3a: LLM text | 1 exp (API calls concurrent) | 1.0h | 11.0h | MUST |
| M3b: Attention plug-ins | 4 exps | 2.0h | 13.0h | MUST |
| M4: Full ablation | 6 exps | 6.0h | 19.0h | NICE-TO-HAVE |

---

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|-----------|------|------|---------------|------|------|
| M0 | Sanity check pilots | M0_boundary_dice + M0_modality_routing (parallel) | Code runs? → Proceed | 1h | Low |
| → | | M0_pelvis_sample + M0_decoder_se (parallel) | Code runs? → Proceed | 1h | Low |
| M1 | Comparative baselines | All M1 models (parallel inference) | Results collected → Proceed | 0.5h | Low |
| M2a | BiomedCLIP swap | M2a_biomedclip | Dice improves? → Continue M2b | 1.5h | Med: weight download |
| M2b | Best freeze strategy | M2b_freeze_all → M2b_freeze_s4 → ... (sequential) | Freeze > no freeze? → Best config | 5.0h | Med |
| M2c | Combine best | M2c_best | >+3% Dice vs baseline? → Continue M3 | 1.0h | Low |
| M3a | LLM text descriptions | M3a_schema → M3a_llm_generate → M3a_train | +1% over M2c? → Include in final | 1.5h | High: API key |
| M3b | Attention plug-ins | M3b_se → M3b_cbam → M3b_ema → M3b_best | +1% over M2c? → Include | 2.0h | Low |
| M4 | Ablation + final model | M4_ct_only → ... → M4_best_model | Paper story complete | 6.0h | Budget |

---

## First 3 Runs to Launch

```bash
# Terminal 1 (GPU 0): Quick-win: Boundary Dice
python MulModSeg_2024/train.py --config configs/experiments.yaml \
    --experiment M0_boundary_dice \
    --loss.type dicece --loss.boundary_dice_weight 0.3 \
    --device 0 --log_name M0_boundary_dice

# Terminal 2 (GPU 1): Quick-win: Modality Routing
python MulModSeg_2024/train.py --config configs/experiments.yaml \
    --experiment M0_modality_routing \
    --model.modality_embedding true \
    --device 1 --log_name M0_modality_routing
```

---

## Manual Actions Checklist

### Pre-requisite: Weight Downloads
- [ ] **BiomedCLIP** (for M2a): `AutoModel.from_pretrained('microsoft/BiomedCLIP-PubMedBERT_256_vit_base_patch16')` — ~800MB
- [ ] **SwinUNETR BTCV** (for M2b): `SwinUNETR(img_size=..., pretrained=True)` — ~400MB, auto-cached by MONAI

### Comparative Model Clones (to `comparative/`)
- [ ] `git clone https://github.com/MIC-DKFZ/MedNeXt.git comparative/MedNeXt`
- [ ] `git clone https://github.com/MIC-DKFZ/nnUNet.git comparative/nnUNet`
- [ ] `git clone <HybridMamba_repo> comparative/HybridMamba`
- [ ] `git clone <latest_2025_3d_model> comparative/LatestModel`
- [ ] Write `scripts/run_comparative_inference.py` — unified inference script for all models

### LLM Text (M3a)
- [ ] Set LLM API key in `.env`: `OPENAI_API_KEY=...` or `ANTHROPIC_API_KEY=...`
- [ ] Write `scripts/generate_llm_descriptions.py` with struct output schema
- [ ] Run generation for all 110 cases
- [ ] Encode with BiomedCLIP → store as case text embeddings
