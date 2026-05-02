# Preprocessing & Postprocessing Analysis + Pelvis Few-Shot Augmentation Strategy

**Date**: 2026-05-01
**Context**: Systematic analysis of current data preprocessing, postprocessing, and the feasibility of pelvis-specific few-shot augmentation as outlined in M0_pelvis_sample.

---

## 1. Preprocessing Pipeline

### 1.1 Overview

There are 3 distinct transform pipelines depending on training mode:

| Pipeline | Used When | Input Keys |
|----------|-----------|------------|
| **Single CT** | `--train_modality CT`, or `--train_modality MIX` (no cross-attn) | image, label |
| **Single MR** | `--train_modality MR`, or `--train_modality MIX` (no cross-attn) | image, label |
| **Paired CT+MR** | `--train_modality MIX --use_cross_attention` | ct, mr, label |

### 1.2 CT Training Pipeline (single modality)

```
LoadImaged                    ← .nii.gz / .nii 文件，shape: (C, D, H, W)
    ↓
Spacingd                      ← pixdim=(1.5, 1.5, 2.0) mm, bilinear(图像) / nearest(标签)
    ↓
Orientationd                  ← axcodes="RAS" (右→左, 前→后, 下→上)
    ↓
ScaleIntensityRanged          ← CT窗宽: a_min=-125, a_max=275, b_min=0.0, b_max=1.0, clip=True
    ↓                         [骨肿瘤典型窗宽：-125~275 HU]
CropForegroundd               ← 以CT为前景源(source_key="image")，裁剪背景
    ↓
SpatialPadd                   ← pad到 roi_size=(96,96,96)，不足补零
    ↓
======== [缓存分离点：以上为 cache_transforms，PersistentDataset 缓存] ========
    ↓
RandCropByPosNegLabeld        ← pos=1, neg=1, num_samples=2, 前景vs背景平衡采样
    ↓
RandShiftIntensityd           ← offset=0.10, prob=0.50 (亮度偏移 ±10%)
    ↓
RandAffined                   ← prob=1.0, rotate=(0,0,π/30), scale=(0.1,0.1,0.1)
    ↓                          bilinear(图像) / nearest(标签), 输出尺寸 roi_size
ToTensord
```

**Key observations**:
- CT intensity normalization uses **fixed window** (-125~275 HU). This is a bone CT window, appropriate for osteosarcoma.
- **No Z-score normalization** for CT — only min-max scaling to [0,1].
- `SpatialPadd` before `RandCrop` ensures uniform tensor shape but means crops may sample from padded (zero) regions.
- `RandAffine` includes small rotation (π/30 ≈ 6°) and scale (10%), which provides mild spatial augmentation.

### 1.3 MR Training Pipeline (single modality)

```
LoadImaged                    ← shape: (C, D, H, W)
    ↓
Spacingd                      ← pixdim=(1.5, 1.5, 2.0) mm
    ↓
Orientationd                  ← axcodes="RAS"
    ↓
NormalizeIntensityd           ← nonzero=True, channel_wise=True (zero-mean, unit-std)
    ↓                         [MR信号无绝对单位，Z-score标准化]
CropForegroundd
    ↓
SpatialPadd
    ↓
======== [缓存分离点] ========
    ↓
RandCropByPosNegLabeld        ← pos=1, neg=1, num_samples=2
    ↓
RandScaleIntensityd           ← factors=0.1, prob=0.5 (强度缩放 ±10%)
    ↓
RandShiftIntensityd           ← offsets=0.1, prob=0.5 (亮度偏移 ±10%)
    ↓
ToTensord
```

**Key differences from CT**:
- MR uses **Z-score normalization** (zero-mean, unit-variance) — standard for MR where signal has no absolute scale.
- MR has **two independent intensity augmentations**: RandScaleIntensityd + RandShiftIntensityd.
- No RandAffine for MR single pipeline — wait, actually there is NO RandAffine in the MR aug transforms. CT has it but MR doesn't. This is asymmetric.

**Key observation**: The MR training augmentation in single-modality mode does NOT include RandAffine (only CT has it). The paired pipeline (section 1.4) DOES apply RandAffine to MR through shared keys. This inconsistency means:
- In single-modality MIX mode (no cross-attention), MR training lacks spatial augmentation.
- In paired mode, MR inherits spatial augmentation through shared key grouping.

### 1.4 Paired CT+MR Training Pipeline (cross-attention mode)

This is the primary pipeline used for `--use_cross_attention` training.

```
LoadImaged(keys=["ct","mr","label"])     ← 同时加载 CT、MR、标签
    ↓
Orientationd(keys=["ct","mr","label"])   ← 统一到 RAS 方向
    ↓
ReplaceZeroEdgeBackgroundd(keys=["ct"])   ← CT 边缘零值背景 → -2048 HU
    [↓ 独特设计：先检测边缘连通零区域，再替换为 -2048，避免 Spacing 插值引入伪影]
    [使用两种算法：contagion (边缘传播 5次膨胀) + connectivity (连通域标记)]
    ↓
Spacingd(keys=["ct","mr","label"])       ← pixdim=(1.5,1.5,2.0), bilinear/bilinear/nearest
    ↓
ScaleIntensityRanged(keys=["ct"])        ← CT: [-125, 275] → [0, 1]
    ↓
ScaleIntensityRangePercentilesd(keys=["mr"]) ← MR: [0.5%, 99.5%] → [0, 1], clip=True
    [↓ 不同于单模态 MR 的 NormalizeIntensityd(z-score)，
       这里使用百分位数归一化到 [0,1]，与 CT 的数值范围对齐]
    ↓
CropForegroundd(keys=["ct","mr","label"], source_key="ct")
    [↓ 以 CT 为前景参考，保证 CT/MR/label 的裁剪边界一致]
    ↓
SpatialPadd(keys=["ct","mr","label"])   ← pad 到 (96,96,96)
    ↓
======== [缓存分离点：以上为 cache_transforms] ========
    ↓
RandCropByPosNegLabeld(keys=["ct","mr","label"])  ← pos=3, neg=1, num_samples=2
    [↓ 注意：pos=3 比单模态的 pos=1 更强调肿瘤区域采样]
    [image_key="ct", image_threshold=0 → 避免采到纯黑空气区域]
    ↓
RandAffined(keys=["ct","mr","label"])    ← prob=0.8, rotate=(0,0,π/30), scale=(0.1,0.1,0.1)
    [↓ 共享随机参数，保证 CT/MR 空间对齐]
    ↓
RandShiftIntensityd(keys=["ct"])         ← CT 独立强度偏移 10%
    ↓
RandScaleIntensityd(keys=["mr"])         ← MR 独立强度缩放 10%
    ↓
RandShiftIntensityd(keys=["mr"])         ← MR 独立强度偏移 10%
    ↓
ToTensord(keys=["ct","mr","label"])
```

**Key observations about paired pipeline**:
1. **CT window varies from single pipeline** — paired uses `ScaleIntensityRangePercentilesd` for MR (instead of `NormalizeIntensityd`), ensuring CT and MR both normalize to [0,1] range.
2. **`ReplaceZeroEdgeBackgroundd`** is unique to the paired pipeline — a sophisticated transform that detects zero-valued background regions in CT (connected to image edges) and replaces them with -2048 HU. This prevents Spacing interpolation from mixing scanner-bed background (-2048) with soft-tissue window values.
3. **pos=3 in RandCrop** — the paired pipeline triple-weights tumor-positive regions vs pos=1 in single modality. This means paired training gives 3× more crop samples containing tumor.
4. **RandAffine prob=0.8** — in paired mode, spatial augmentation is applied to 80% of samples (vs 100% in CT single mode).

### 1.5 Validation Pipeline (all modes)

Validation pipelines are **identical** to the cache portion of training pipelines, with only `ToTensord` appended — no random cropping, no augmentation. Key properties:
- Entire volume is used (no cropping — SpatialPad ensures minimum size, and sliding window inference covers the full volume).
- Deterministic and reproducible across runs.

### 1.6 Data Organization

```
dataset/
  ├── 第1批/  ← patient folders, each containing .nii.gz files
  │   ├── 11084154/
  │   │   ├── 11084154.nii.gz       (label/segmentation)
  │   │   ├── 11084154_ct_reg.nii.gz (CT image)
  │   │   └── 11084154_mr.nii.gz    (MR image)
  │   └── ...
  ├── 第2批/
  ├── 第3批/  ← These 3 batches = femur cases (Expert 1)
  ├── 第4批/  ← Pelvis cases (Expert 0)
  ├── 第5批/  ← Pelvis cases (Expert 0)
  └── 上海市一/
```

**File naming conventions**:
- CT: `{patient_id}_ct_reg.nii.gz` or `{patient_id}-DF_ct_reg.nii.gz` or `{patient_id}-DFKN_ct_reg.nii.gz`
- MR: `{patient_id}_mr.nii.gz` or `{patient_id}-DF_mr.nii.gz` or `{patient_id}-DFKN_mr.nii.gz`
- Label: `{patient_id}.nii.gz` (with fallback to `{patient_id}.nii`)

---

## 2. Postprocessing Pipeline

### 2.1 Current State: Minimal

The current postprocessing is **effectively nonexistent**:

```
┌──────────┐     ┌──────────┐     ┌──────────────┐
│ logit_map │────→│ softmax  │────→│ argmax(dim=1)│────→ binary mask
└──────────┘     └──────────┘     └──────────────┘
```

That's it. There is:
- **No morphological cleanup** (no hole-filling, no isolated-component removal)
- **No CRF** (Conditional Random Field refinement)
- **No connected-component filtering** (no keeping only the largest component)
- **No threshold tuning** (fixed argmax at 0.5)
- **No test-time augmentation** (TTA)

This is implemented in:
- `MulModSeg_2024/utils/enhanced_validation.py` line ~255: `pred_binary = (pred_argmax == 1).float()`
- `scripts/infer_checkpoint.py` line ~154: same pattern
- `MulModSeg_2024/train.py` validation loop: uses `enhanced_validation()` which handles postprocessing internally

### 2.2 Impact of Missing Postprocessing

For bone tumor segmentation, lack of postprocessing means:
1. **Small false-positive clusters** in background regions are counted as tumor — inflates FP, hurts precision.
2. **Holes within tumor predictions** are left unfilled — may slightly reduce Dice but the impact is small since loss function already encourages connected predictions via spatial smoothness.
3. **No volume-based filtering**: In clinical practice, osteosarcoma is a single cohesive mass. Any predicted region below a minimum volume threshold (e.g., <50 voxels) is almost certainly noise.

### 2.3 Recommended Postprocessing Steps

For a <50 line addition to `enhanced_validation.py`, we could add:

```python
def postprocess(pred_binary, min_volume=50, fill_holes=True):
    """Lightweight postprocessing for bone tumor predictions."""
    import cc3d  # or use MONAI's components
    # 1. Connected-component labeling
    labeled = cc3d.connected_components(pred_binary)
    # 2. Remove small components (< min_volume voxels)
    for comp_id in range(1, labeled.max() + 1):
        if (labeled == comp_id).sum() < min_volume:
            pred_binary[labeled == comp_id] = 0
    # 3. Fill holes (binary fill)
    if fill_holes:
        from scipy import ndimage
        pred_binary = ndimage.binary_fill_holes(pred_binary)
    return pred_binary
```

However, the expected gain is **small (<1% Dice)** since:
- MONAI's sliding_window_inference with Gaussian blending already reduces boundary artifacts
- The DiceCE loss function already penalizes disconnected predictions
- Tumor-to-background ratio is low (1-5%), so small FP regions have outsized impact on precision

---

## 3. Pelvis Few-Shot Augmentation Analysis

### 3.1 The Data Imbalance Problem

| Category | Count | Percentage | Batch Source | MoE Expert |
|----------|-------|------------|-------------|------------|
| Femur (distal) | ~79 | ~72% | 第1-3批 + 上海市一 | Expert 1 |
| Pelvis | ~31 | ~28% | 第4-5批 | Expert 0 |

The 79:31 ≈ 2.5:1 imbalance means the model sees ~2.5× more femur samples than pelvis samples during training.

### 3.2 Current Mitigations Already in Place

| Mechanism | Effect | Location |
|-----------|--------|----------|
| **pos=3 in RandCrop** | Paired loader triple-weights tumor regions | `dataloader_bone_tumor.py` line 1016 |
| **MoE routing** | Separate expert for pelvis (Expert 0) | `train.py` `train_mix()` lines 381-397 |
| **Routing CE loss** | Forces router to specialize (α=0.3 weight) | `train.py` line 406 |
| **Separate pelvis group** | Batch-based assignment avoids wrong expert | `train.py` lines 393-397 |

### 3.3 Gaps in Current Pipeline

#### Gap 1: No pelvis-class weighting in loss
The segmentation loss (`DiceCELoss`) is computed per-sample uniformly — pelvis samples contribute equally to femur samples. A pelvis-weighted loss (2.5× weight on pelvis samples) would directly address the imbalance.

```python
# Idea: weighted loss for pelvis samples (not implemented)
pelvis_weight = 2.5  # proportional to femur/pelvis ratio
sample_weight = pelvis_weight if is_pelvis_batch else 1.0
loss = sample_weight * (term_seg_Dice + term_seg_BCE)
```

#### Gap 2: No oversampling in dataloader
Current dataloader uses uniform patient sampling. There's no weighted sampler that increases pelvis patient selection probability:

```python
# Idea: weighted sampler (not implemented)
weights = [pelvis_weight if is_pelvis(pid) else 1.0 for pid in patient_ids]
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
```

#### Gap 3: No pelvis-specific augmentation
Current augmentation applies the same `RandAffine(π/30, 0.1)` to all samples regardless of anatomy. Pelvis cases could benefit from:
- Larger rotation range (pelvis has more complex geometry)
- Elastic deformation (pelvis is more anatomically variable)
- CutMix/MixUp between pelvis cases

#### Gap 4: Small pelvis absolute count
Even with oversampling, with only ~31 pelvis cases (25 train after 80/20 split), there's limited anatomical diversity. Data augmentation cannot create new anatomical variations.

### 3.4 Experiment Plan: M0_pelvis_sample

The experiment plan (EXPERIMENT_PLAN.md) lists `M0_pelvis_sample` as a quick-win pilot:

```
M0_pelvis_sample: Pelvis oversampling via case weights
- Changes: dataloader_bone_tumor.py: case weights
- Est. Time: 1.5h (150ep)
```

This is described as **Pelvis oversampling** but the specifics are not elaborated. The minimal implementation would:

1. Compute per-patient weights: `1.0` for femur, `~2.5` for pelvis
2. Pass weights to `WeightedRandomSampler` in the DataLoader
3. This ensures pelvis samples appear at roughly equal frequency to femur samples

**Expected impact**: +1-2% Dice on pelvis, at the cost of slightly reduced femur performance due to relative under-sampling.

### 3.5 Advanced Pelvis Augmentation Strategies

#### Strategy A: Pelvis-Specific Data Augmentation (Recommended for M0)

| Technique | Implementation | Expected Gain | Risk |
|-----------|---------------|---------------|------|
| **Weighted sampling** | `WeightedRandomSampler` with pelvis_weight=2.5 | +1-2% pelvis Dice | Femur regression (-0.5%) |
| **Higher RandCrop pos** | Increase pos=5 (from 3) for pelvis via per-sample config | +0.5% | Minimal |
| **Larger rotation** | Increase rotate_range for pelvis samples | +0.5% | May generate unrealistic geometry |

Implementation in dataloader:

```python
# In get_loader_paired_bone_tumor:
# 1. Per-sample weight based on batch
patient_weights = []
for pid in patient_ids:
    if pid_batch_map.get(pid, '') in ['第4批', '第5批']:
        patient_weights.append(args.pelvis_weight)  # e.g., 2.5
    else:
        patient_weights.append(1.0)

# 2. Use WeightedRandomSampler
sampler = WeightedRandomSampler(patient_weights, len(patient_weights), replacement=True)
loader = DataLoader(..., sampler=sampler, shuffle=False)  # shuffle=False when using sampler
```

#### Strategy B: Pelvis-Specific Expert Loss Weighting (Low Effort, Moderate Gain)

Add a pelvis-loss multiplier in `train_mix()`:

```python
# Modify train_mix() line ~355:
pelvis_loss_weight = getattr(args, 'pelvis_loss_weight', 1.0)
if num_pelvis > 0:  # batch contains pelvis samples
    # Scale loss for pelvis samples
    pelvis_mask = (route_targets == 0).float()
    sample_weight = 1.0 + (pelvis_loss_weight - 1.0) * pelvis_mask
    loss = (loss * sample_weight).mean()
```

#### Strategy C: Pelvis Cross-Validation Ensemble (Offline, Highest Gain)

Train 5 folds with pelvis-weighted sampling, ensemble the 5 models at inference. Ensembling typically gives +2-3% Dice on the worst-performing subgroup.

#### Strategy D: Pseudo-Labeling on Pelvis (Semi-Supervised)

1. Train initial model (current pipeline) → predict on pelvis val set
2. Select high-confidence pelvis predictions as pseudo-labels
3. Add pseudo-labeled cases to training set
4. Retrain with expanded pelvis set

Requires careful threshold tuning to avoid confirmation bias. Risk: if initial model is bad at pelvis, pseudo-labels reinforce errors.

### 3.6 Implementation Priority

| Priority | Strategy | Effort | Pelvis Dice Gain | Code Location |
|----------|----------|--------|------------------|---------------|
| **P0** | Pelvis-weighted sampling (M0_pelvis_sample) | 1h | +1-2% | `dataloader_bone_tumor.py` + `train.py` |
| **P1** | Pelvis-weighted loss (per-sample scaling) | 0.5h | +0.5-1% | `train.py` `train_mix()` |
| **P2** | Larger pelvis rotation range | 0.5h | +0.5% | `dataloader_bone_tumor.py` transform config |
| **P3** | Minimal postprocessing (connected component) | 0.5h | +0.5% | `enhanced_validation.py` |
| **P4** | 5-fold cross-validation ensemble | 3h | +2-3% | `scripts/ensemble.py` |
| **P5** | Pseudo-labeling | 4h | +1-2% | Additional pipeline |

### 3.7 Note on Train.py Bug

The `train_mix()` function in `train.py` has a **hardcoded path** for loading the split file:

```python
json_path = os.path.join('/root/autodl-tmp/segmentation-test-main/splits/split_seed42.json')
```

This will fail when running from `/work/projhighcv/hzl/bone_tumor/`. The fix should use `args.split_file` (which defaults to `'splits/split_seed42.json'`):

```python
json_path = getattr(args, 'split_file', 'splits/split_seed42.json')
if not os.path.isabs(json_path):
    json_path = os.path.join(os.path.dirname(__file__), '..', json_path)
```

---

## 4. Summary

### Current Pipeline Strengths
- Cache separation (deterministic → cache, random → on-the-fly) ensures efficient training
- ReplaceZeroEdgeBackgroundd prevents interpolation artifacts from scanner bed
- Paired pipeline shares spatial augmentation parameters across CT and MR
- MoE routing with batch-based expert assignment is appropriate for the femur/pelvis split

### Current Pipeline Weaknesses
- **No pelvis oversampling** → pelvis samples under-represented by 2.5×
- **No postprocessing** → small FP clusters inflate metrics
- **No pelvis-specific augmentation** → uniform augmentation treats femur and pelvis identically
- **MR single-modality lacks RandAffine** → asymmetry between CT and MR augmentation
- **Hardcoded path** in `train_mix()` for split file loading

### Recommended Actions
1. Implement pelvis-weighted sampling (M0_pelvis_sample) as the highest ROI change
2. Add minimal postprocessing (connected-component filtering) — low effort, guaranteed improvement
3. Add pelvis loss weighting — near-zero effort extension to M0_pelvis_sample
4. Fix the hardcoded path bug in train.py line 361
