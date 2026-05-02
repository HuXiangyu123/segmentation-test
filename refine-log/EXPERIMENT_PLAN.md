# Experiment Plan: MulModSeg_2024 Baseline Improvement

**Date**: 2026-05-02 (v5 — reordered: M2a -> pelvis -> M3a as priority, freeze marked done/no-benefit, M0 moved to end)
**Hardware**: 2x RTX 3090 (24GB VRAM each)
**Dataset**: 102 paired CT+MR bone tumor patients (5-fold CV, seed=42, 82/10/10 per fold)
**Baseline**: E0_baseline (DiceCE loss, SwinUNETR backbone, default sampling)

### Dataset Details

- **Source**: 5 batch folders (第1批–第5批) + 上海市一
- **Total patients**: 109 raw -> **102 after dropping 7 patients** (see drop list below)
- **5-fold splits**: `splits/fold5_splits.json` — seed=42, each fold 82 train / 10 val / 10 test
- **MIX modality**: Each patient has CT + MR -> 163 train samples, 42 val samples per fold
- **Drop list** (7 patients, 8 entries):
  | Patient | Batch | Reason |
  |---------|-------|--------|
  | 11687281 | unknown | Registration failure (MR/SEG shape/affine mismatch) |
  | 12298737 | unknown | Registration failure |
  | 10180747 | 第4批 | Manually flagged (poor image quality) |
  | 11232743 | 第5批 | Manually flagged |
  | 11744770 | 第4批 | Manually flagged |
  | 11084154 | 第1批 + 第2批 | CT/MR shape/affine mismatch (2 entries, same patient) |
  | 11768711 | 第1批 | CT/MR shape/affine mismatch |

---

## Problem Anchor

MulModSeg_2024 achieves reasonable multimodal bone tumor segmentation but has three documented weaknesses:
1. **Text prior instability**: CLIP embeddings produce inconsistent router behavior across seeds
2. **Poor pelvis performance**: Complex pelvic anatomy cases underperform by 10-15% Dice vs. femur cases
3. **No modern baselines**: Missing comparison with 2024-2025 3D segmentation models

## Method Thesis

Domain-specific text embeddings (BiomedCLIP) + hallucination-controlled LLM lexicon + pelvis-targeted sampling provide the highest gain-to-cost ratio for improving MulModSeg on small-sample bone tumor data.

Progressive freeze experiments (M2b) showed **no significant benefit** — freeze_all and freeze_s4 were near-identical (0.717 vs 0.716 Dice), indicating SSL pretrained weights dominate over freeze strategy. These are archived as completed/negative.

---

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Milestone |
|-------|----------------|-----------------------------|-----------|
| C1: Domain-specific text stabilizes MoE routing | Text prior is the weakest link (docs review) | +2% Dice on pelvis, -30% seed variance | **M2a** (active) |
| C2: Pretrained encoder freezing prevents overfit | n=102 demands transfer learning | ~~Archived~~: freeze_all ~ freeze_s4 (0.717 vs 0.716, no benefit) | Done |
| C3: LLM-built controlled descriptors enhance router | Reliable priors matter more than longer text | +1% Dice over BiomedCLIP baseline | **M3a** (active) |
| C4: Pelvis-targeted sampling improves worst-case | Pelvis underperforms by 10-15% Dice | +2% pelvis Dice, minimal femur regression | **M0_pelvis** (active) |
| C5: Our method competes with modern 3D models | Missing baselines was documented gap | At least tied with strongest 2024-2025 models on CT-only | M1 (optional) |

---

## Paper Storyline

- **Main paper must prove**: C1 + C3 + C4 (primary), C5 (context, optional)
- **Appendix can support**: Full ablation (CT-only, MR-only, fusion variants), failure case analysis, per-bucket breakdown, descriptor-bank and shape-to-text prompt design
- **Experiments intentionally cut**: Diffusion augmentation (GPU-heavy), freeze strategy (no benefit found), decoder attention plug-ins (optional, decorative risk), DDP engineering

---

## Experiment Milestones

### P1: M2a — BiomedCLIP/MedSigLIP Text Embedding
**Budget**: 1.5 GPU-hours | **Claims**: C1 | **Priority**: HIGHEST (prerequisite for M3a)

Replace CLIP with domain-specific text encoders. BiomedCLIP is primary; MedSigLIP is optional control.

**Decision**: BiomedCLIP is the primary text encoder for all downstream M3a plans. MedSigLIP stays as an optional control because its text context is shorter (64 tokens), so it is less suitable for descriptor aggregation.

| Exp ID | Description | Depends On | Est. Time |
|--------|-------------|-----------|-----------|
| **M2a1_biomedclip** | Replace CLIP -> BiomedCLIP for all text features | Manual: download weights | 0.5h precompute + 1.0h train |
| **M2a2_medsiglip** | Replace CLIP -> MedSigLIP for all text features | Manual: download weights | 0.5h precompute + 1.0h train |

**Manual**: Download BiomedCLIP from HuggingFace:
```bash
huggingface-cli download microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 \
  open_clip_pytorch_model.bin \
  open_clip_config.json \
  --local-dir MulModSeg_2024/text_embedding/biomedclip
```

**Manual**: Download MedSigLIP from HuggingFace:
```bash
huggingface-cli download google/medsiglip-448 \
  --local-dir MulModSeg_2024/text_embedding/medsiglip-448
```

---

### P2: M0_pelvis — Pelvis-Targeted Augmentation & Weighted Sampling
**Budget**: 7.0 GPU-hours | **Claims**: C4 | **Priority**: HIGH

Oversample pelvis cases and apply pelvis-weighted loss to close the 10-15% Dice gap vs femur.

#### Detailed Strategy

| Component | Implementation | Expected Dice Gain | Code Location |
|-----------|---------------|-------------------|---------------|
| **Pelvis-weighted sampling** | `WeightedRandomSampler` with pelvis_weight=2.5 | +1-2% on pelvis | `dataloader_bone_tumor.py` |
| **Pelvis-weighted loss** | Per-sample loss scaling: pelvis samples x 2.5 | +0.5-1% | `train.py` `train_mix()` |
| **Higher pelvis RandCrop pos** | pos=5 for pelvis vs pos=3 for femur | +0.5% | `dataloader_bone_tumor.py` |

**Data imbalance**: 31 pelvis (28%) vs 79 femur (72%), batched as 第4-5批 (pelvis) vs 第1-3批+上海市一 (femur).

**Weighted sampling implementation**:
```python
pelvis_weight = len(patients) / len(pelvis_patients)  # ~2.5
patient_weights = [pelvis_weight if batch_is_pelvis(pid) else 1.0 for pid in patient_ids]
sampler = WeightedRandomSampler(patient_weights, len(patient_weights), replacement=True)
```

**Loss weighting in train_mix()**:
```python
pelvis_mask = (route_targets == 0).float()  # 1 for pelvis, 0 for femur
sample_weight = 1.0 + (args.pelvis_loss_weight - 1.0) * pelvis_mask
loss = (loss * sample_weight).mean()
```

**Expected trade-off**: +1-2% pelvis Dice, -0.5% femur Dice (manageable via MoE routing).

---

### P3: M3a — LLM Hallucination-Controlled Lexicon
**Budget**: 14.0-21.0 GPU-hours | **Claims**: C3 | **Priority**: HIGH (depends on M2a)

Two plans for building controlled text descriptors, encoded via BiomedCLIP.

**Manual**: Requires GPT-5.4 API or an OpenAI-compatible vLLM endpoint, deterministic JSON output, and cache/retry logic.

#### Plan A: Controlled Descriptor Bank (default)

Generate 8-16 short English descriptors per canonical metadata key: `(bone, side, compartment, size_category)`. Use only observed fields. No hallucinated radiology findings. Encode each descriptor with BiomedCLIP, normalize and average offline.

| Exp ID | Description | Depends On | Est. Time |
|--------|-------------|-----------|-----------|
| **M3aA_schema_bank** | Build canonical keys + validator + blacklist for forbidden findings | M2a | 0.5h code |
| **M3aA_llm_generate** | Generate 8-16 short descriptors per canonical key via GPT-5.4 / vLLM | .env or local endpoint | ~1h API calls |
| **M3aA_filter_encode** | Filter duplicates/hallucinations, BiomedCLIP encode, aggregate to one case vector | M2a | 0.5h |
| **M3aA_train_k4** | Train with top-4 descriptor aggregation | M2a | 7.0h (200ep) |
| **M3aA_train_k8** | Train with top-8 descriptor aggregation | M2a | 7.0h (200ep) |

#### Plan B: Mask-Grounded Attribute Lexicon (fallback)

Extract measurable morphology first, then let the LLM only rewrite verified bins into short biomedical phrases.

| Exp ID | Description | Depends On | Est. Time |
|--------|-------------|-----------|-----------|
| **M3aB_shape_extract** | Compute shape/extent attributes from tumor masks, bin into closed vocabulary | None | 0.5-1.0d code/CPU |
| **M3aB_template_encode** | Template measured bins directly and encode with BiomedCLIP | M3aB_shape_extract, M2a | 0.5h |
| **M3aB_llm_restyle** | LLM rewrites measured bins into 4-8 short biomedical descriptors | .env or local endpoint | ~1h API calls |
| **M3aB_train** | Train with best Plan B embedding set | M2a | 7.0h (200ep) |

#### Decision Logic

- Run **Plan A** first
- If Plan A does not improve pelvis Dice by at least +0.5 or does not reduce routing instability, move to Plan B
- Carry best of A/B into final model

---

### M2b: Progressive Freeze (COMPLETED — No Benefit Found)
**Budget**: 0 GPU-hours (archived) | **Claims**: C2

| Exp ID | Config | Dice (fg) | Best Epoch | Verdict |
|--------|--------|-----------|------------|---------|
| **M2b_freeze_all** | Freeze stages 1-4 | **0.7173** | 65 | SSL pretrained dominates |
| **M2b_freeze_s4** | Freeze 1-3, thaw stage 4 | **0.7160** | 65 | No benefit over full freeze |
| M2b_freeze_s34 | Freeze 1-2, thaw 3-4 | — | — | Skipped (no benefit signal) |
| M2b_full_finetune | No freeze | — | — | Skipped |
| M2b_no_pretrain | From scratch | — | — | Skipped |

**Conclusion**: freeze_all ~ freeze_s4 (0.717 vs 0.716) — thawing stage 4 provides no benefit. SSL pretrained weights dominate. No further freeze experiments needed.

---

### Optional Experiments

These are de-prioritized and will only be run if time permits after P1-P3.

#### M0: Other Quick-Win Pilots (Sanity Check)
**Budget**: 21 GPU-hours | **Claims**: Sanity

Quick validation that code changes work. Run after primary claims are solid.

| Exp ID | Description | Changes | Est. Time | Notes |
|--------|-------------|---------|-----------|-------|
| **M0_boundary_dice** | Enable boundary Dice loss (late start) | Config: `boundary_dice_weight: 0.3, boundary_start_epoch: 140` | 7h (200ep) | Activates at epoch 140 |
| **M0_modality_routing** | Modality-aware routing bias | `MulModSeg.py`: +10 lines | 7h (200ep) | |
| **M0_decoder_se** | SE attention in decoder | `attention_plugins.py`: new file | 7h (200ep) | |

**Success criterion**: Code runs without error, metrics logged.

#### M1: CT Baseline Comparison (Inference Only)
**Budget**: 0.5 GPU-hours | **Claims**: C5 (optional)

Download and run inference for comparison models on single CT input. No training.

| Exp ID | Model | Source |
|--------|-------|--------|
| **M1_unet** | 3D U-Net | `MulModSeg_2024/model/Unet.py` |
| **M1_segresnet** | SegResNet | MONAI `SegResNet` |
| **M1_swinunetr** | Native SwinUNETR | MONAI (pretrained BTCV) |
| **M1_mednext** | MedNeXt-v2 | `github.com/MIC-DKFZ/MedNeXt` |
| **M1_nnunet** | nnU-Net v2 | `github.com/MIC-DKFZ/nnUNet` |
| **M1_SegMamba** | SegMamba | `github.com/ge-xing/SegMamba` |
| **M1_Primus** | Primus | `github.com/TaWald/nnUNet` |

#### M3b: Decoder Attention Plug-ins (Optional)
**Budget**: 28 GPU-hours | **Claims**: C4 (decorative)

| Exp ID | Plug-in | Insertion Point | Est. Time |
|--------|---------|----------------|-----------|
| **M3b_se** | 3D SE (channel attention) | After each UnetrUpBlock | 7.0h (200ep) |
| **M3b_cbam** | 3D CBAM (channel + spatial) | After each UnetrUpBlock | 7.0h (200ep) |
| **M3b_ema** | 3D EMA (multi-scale attention) | After cross-attn bottleneck | 7.0h (200ep) |

#### M4: Full Ablation Study (Optional)
**Budget**: 42 GPU-hours | **Claims**: C1-C5 (completeness, appendix)

| Exp ID | Description | Config | Est. Time |
|--------|-------------|--------|-----------|
| **M4_ct_only** | CT-only, no cross-attention | `train_modality: CT` | 7.0h (200ep) |
| **M4_mr_only** | MR-only, no cross-attention | `train_modality: MR` | 7.0h (200ep) |
| **M4_add_fusion** | CT+MR addition, no cross-attn | `use_cross_attention: false` + add | 7.0h (200ep) |
| **M4_concat_fusion** | CT+MR concat, no cross-attn | `use_cross_attention: false` + concat | 7.0h (200ep) |
| **M4_cross_attn** | Full cross-attention fusion | cross_attention config | 7.0h (200ep) |

---

### Run Notes

**Time optimization**: Use `--val_start_epoch 100` to skip validation for first 100 epochs (~10min) and only monitor the second half (~3.5h). Full 200 epochs with val every epoch takes ~7h.

**All experiments use `torchrun --nproc_per_node=2`** (2x RTX 3090 DDP).

---

## Dependency Graph

```
P1: M2a BiomedCLIP (manual weight download required)
 |
 |-- depends on nothing (start here)
 |
 |---> P2: M0_pelvis (independent, can run parallel to P1)
 |
 |---> P3: M3a LLM Lexicon (depends on M2a for BiomedCLIP encoding)
          |
          +-- Plan A: descriptor_bank -> filter_encode -> train_k4/k8
          +-- Plan B: shape_extract -> template_or_llm -> train

Optional (only if time permits):
  M0: boundary_dice, modality_routing, decoder_se
  M1: CT baselines (independent, runs in parallel)
  M3b: attention plug-ins (SE -> CBAM -> EMA)
  M4: full ablation (CT-only, MR-only, fusion variants)
```

**M1**: CT baseline comparison — independent, runs in parallel with all other work.

---

## GPU Budget Tracking

All experiments use **2x RTX 3090 DDP** (`torchrun --nproc_per_node=2`). Time estimates based on actual runs.

| Milestone | Experiments | GPU-hours | Cumulative | Priority | Notes |
|-----------|------------|-----------|------------|----------|-------|
| E0: Baseline | 1 exp (210ep) | 17.0h | 17.0h | Done | |
| **P1: M2a BiomedCLIP** | 1-2 exps (200ep) | **1.5h** | **18.5h** | **#1** | 0.5h precompute + 1.0h train |
| **P2: M0_pelvis** | 1 exp (200ep) | **7.0h** | **25.5h** | **#2** | Use val_start_epoch=100 |
| **P3: M3a LLM Lexicon** | 2-5 exps + API calls | **14.0-21.0h** | **39.5-46.5h** | **#3** | Plan A first, Plan B if needed |
| M2b: Freeze strategy | Already completed | 0.0h | 39.5-46.5h | Archived | No benefit found |
| M0: Other quick-wins | 3 exps (200ep each) | 21.0h | 60.5-67.5h | Optional | |
| M1: Baselines (inference) | 7 models | 0.5h | 61.0-68.0h | Optional | |
| M3b: Attention plug-ins | 3 exps (200ep each) | 21.0h | 82.0-89.0h | Optional | |
| M4: Full ablation | 5 exps (200ep each) | 35.0h | 117.0-124.0h | Optional | Appendix-only |

---

## Run Order and Milestones

| Order | Milestone | Goal | Runs | Decision Gate | Cost |
|-------|-----------|------|------|---------------|------|
| 1 | P1: M2a | BiomedCLIP swap | M2a1_biomedclip (+ M2a2_medsiglip if needed) | Dice improves? -> Continue P3 | 1.5h |
| 2 | P2: M0_pelvis | Pelvis sampling | M0_pelvis_sample | Pelvis Dice +1-2%? -> Keep | 7.0h |
| 3 | P3: M3a | LLM lexicon | Plan A first -> if flat, Plan B | Plan A +0.5 pelvis? -> Keep best | 14.0-21.0h |
| — | M2b | Freeze | Done — no benefit | Archived | 0h |
| — | M0 | Other pilots | boundary_dice, modality_routing, decoder_se | Only if time permits | 21.0h |
| — | M1 | Baselines | All CT models inference | Only if needed for paper | 0.5h |
| — | M3b | Decoder plug-ins | SE, CBAM, EMA | Only if time permits | 21.0h |
| — | M4 | Ablation | CT-only, MR-only, fusion | Only if needed for appendix | 35.0h |

---

## First 3 Runs to Launch

All experiments use **2x RTX 3090 DDP** via `torchrun --nproc_per_node=2`.

```bash
# Run 1: M2a1_biomedclip (Priority 1 — prerequisite for M3a)
torchrun --nproc_per_node=2 MulModSeg_2024/train.py \
    --distributed \
    --dataset bone_tumor --data_root_path ./dataset \
    --backbone swinunetr \
    --with_text_embedding 1 \
    --use_cross_attention \
    --cross_attn_heads 8 \
    --pretrain_encoder_only ./MulModSeg_2024/pretrained/ssl_pretrained_weights.pth \
    --word_embedding ./MulModSeg_2024/text_embedding/biomedclip/bone_tumor_class_embeddings.pth \
    --max_epoch 200 --warmup_epoch 10 \
    --train_modality MIX --batch_size 1 --num_workers 4 \
    --roi_x 96 --roi_y 96 --roi_z 96 --num_samples 2 \
    --lr 1e-4 --loss_type dicece \
    --val_start_epoch 100 \
    --log_name M2a1_biomedclip

# Run 2: M0_pelvis_sample (Priority 2 — independent of M2a)
torchrun --nproc_per_node=2 MulModSeg_2024/train.py \
    --distributed \
    --dataset bone_tumor --data_root_path ./dataset \
    --backbone swinunetr \
    --with_text_embedding 1 \
    --use_cross_attention \
    --cross_attn_heads 8 \
    --pretrain_encoder_only ./MulModSeg_2024/pretrained/ssl_pretrained_weights.pth \
    --word_embedding ./MulModSeg_2024/text_embedding/bone_tumor_class_embeddings.pth \
    --max_epoch 200 --warmup_epoch 10 \
    --train_modality MIX --batch_size 1 --num_workers 4 \
    --roi_x 96 --roi_y 96 --roi_z 96 --num_samples 2 \
    --lr 1e-4 --loss_type dicece \
    --pelvis_weight 2.5 \
    --val_start_epoch 100 \
    --log_name M0_pelvis_sample

# Run 3: M3aA_train_k4 (Priority 3 — depends on M2a embedding)
# Run after M2a1_biomedclip completes and LLM descriptors are generated
torchrun --nproc_per_node=2 MulModSeg_2024/train.py \
    --distributed \
    --dataset bone_tumor --data_root_path ./dataset \
    --backbone swinunetr \
    --with_text_embedding 1 \
    --use_cross_attention \
    --cross_attn_heads 8 \
    --pretrain_encoder_only ./MulModSeg_2024/pretrained/ssl_pretrained_weights.pth \
    --word_embedding ./MulModSeg_2024/text_embedding/biomedclip/llm_lexicon_k4.pth \
    --max_epoch 200 --warmup_epoch 10 \
    --train_modality MIX --batch_size 1 --num_workers 4 \
    --roi_x 96 --roi_y 96 --roi_z 96 --num_samples 2 \
    --lr 1e-4 --loss_type dicece \
    --val_start_epoch 100 \
    --log_name M3aA_train_k4
```

---

## Manual Actions Checklist

### Pre-requisite: Weight Downloads
- [ ] **BiomedCLIP** (for M2a, M3a): `huggingface-cli download microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` to `MulModSeg_2024/text_embedding/biomedclip/`
- [ ] **MedSigLIP** (optional, for M2a2): `huggingface-cli download google/medsiglip-448` to `MulModSeg_2024/text_embedding/medsiglip-448/`

### LLM Text (M3a)
- [ ] Set LLM API key in `.env`: `OPENAI_API_KEY=...` or `ANTHROPIC_API_KEY=...`
- [ ] Write `scripts/generate_llm_descriptions.py` with structured output schema
- [ ] Build schema bank: canonical keys + validator + blacklist for forbidden findings
- [ ] Run LLM generation for all 102 patients
- [ ] Encode with BiomedCLIP -> store as case text embeddings
- [ ] If Plan B needed: write `scripts/extract_shape_attributes.py` for tumor mask morphology

### Optional (if time permits)
- [ ] **SwinUNETR BTCV** (for M1_swinunetr): download via MONAI
- [ ] Clone comparative models to `comparative/` for M1 baselines
- [ ] Write `scripts/run_comparative_inference.py` for M1
