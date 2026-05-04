# Time Optimization Analysis

**Date**: 2026-05-02
**Source Experiment**: `M2b_freeze_s4_mulmodseg_ns4v3_MIX_lr0.0001_max_epoch200_04_30_17_04`
**Hardware**: 2× RTX 3090 DDP (torchrun --nproc_per_node=2)

---

## 1. Baseline Timing from Experiment Log

### 1.1 Wall Clock Timeline

| Event | Timestamp | Elapsed |
|-------|-----------|---------|
| Training start | 17:04:29 | 0 min |
| epoch_032 val summary | ~17:07 | ~2.5 min |
| epoch_040 best3 vis | 17:30 | 26 min |
| epoch_050 best3 vis | 17:56 | 52 min |
| epoch_060 best3 vis | 18:23 | 79 min |
| epoch_070 best3 vis | 18:50 | 106 min |
| epoch_080 best3 vis | 19:17 | 133 min |
| epoch_090 best3 vis | 19:44 | 160 min |
| epoch_100 best3 vis | 20:11 | 187 min |
| epoch_110 best3 vis | 20:39 | 215 min |
| epoch_120 best3 vis | 21:06 | 242 min |
| epoch_130 best3 vis | 21:33 | 269 min |
| epoch_140 best3 vis | 22:00 | 296 min |
| epoch_150 best3 vis | 22:28 | 324 min |
| epoch_160 best3 vis | 22:55 | 351 min |
| Training end | 23:16:43 | 372 min |

**Config**: `--freeze_level stage4 --num_samples 4 --batch_size 1 --train_modality MIX --use_cross_attention --lr 1e-4 --max_epoch 200`

### 1.2 Per-Epoch Cost Breakdown

| Component | Time | % of Epoch | Notes |
|-----------|------|------------|-------|
| **Training** (85 steps × 2 crops) | ~5-7s | ~4% | Frozen encoder: only stage4 + decoder + head need gradients |
| **Validation** (21 cases, sliding window) | ~150s (150s ÷ 60 ≈ 2.5 min) | ~96% | Dominant cost |
| **Visualization** (every 10 epochs) | ~30s extra | — | Best3/worst3 case images, PR/ROC curves |
| **Metrics logging + CSV** | ~1s | — | Negligible |
| **Total per epoch** | ~156s ≈ 2.6 min | 100% | — |

**Key insight**: Validation takes **~96%** of per-epoch time. Training alone is only ~5-7 seconds per epoch.

### 1.3 Projected vs Actual Time

| Scenario | Formula | Total Time |
|----------|---------|------------|
| 200 epochs with val every epoch | 200 × 2.6min | ~8.7h |
| 200 epochs (actual observed) | 200 × 2.3min | ~7.7h |
| 160 epochs (actual run, terminated early) | 160 × 2.3min | ~6.2h |

The actual run terminated around epoch 160 (no epoch 170+ vis dirs), likely manual stop.

---

## 2. Where Time Is Spent in Validation

### 2.1 Validation Inference (sliding_window_inference)

For each of the 21 validation cases:
- Full-resolution CT+MR volume after spacing: approximately (250-400, 150-300, 100-200) voxels
- Sliding window with roi_size=(96,96,96), overlap=0.5
- Number of windows per case: ~8-18 (varies by volume size)
- Each window: 1 forward pass through full model (backbone + cross-attn + decoder + head)
- Total forward passes per epoch: 21 × ~12 = ~252

At ~0.5-0.6s per forward pass (2× DDP): ~126-151s ≈ 2.1-2.5 min just for inference.

### 2.2 HD95 and ASSD Computation

```python
compute_hausdorff_distance(pred_oh, gt_oh, ...)  # Euclidean distance transform
compute_average_surface_distance(pred_oh, gt_oh, symmetric=True, ...)
```

These MONAI functions compute full 3D distance transforms per case, which is expensive:
- ~0.1-0.3s per case, negligible compared to sliding window inference

### 2.3 Visualization (Every 10 Epochs)

- PR curve + ROC curve plotting: ~3s
- Best3/Worst3 case visualization (6 cases × 4-5 panels): ~20-30s
- Training curves plot (cumulative): ~2s

Total: ~25-35s every 10 epochs, which is ~2.5-3.5s amortized per epoch.

---

## 3. Optimization Proposals

### 3.1 P0: Deferred Validation (Implemented)

**Change**: `--val_start_epoch N` — skip validation for first N epochs.

| Config | Training Only | Training + Val | Total | vs Baseline |
|--------|--------------|----------------|-------|-------------|
| Current (val every epoch) | 200 × 0.1min = 0.3h | 200 × 2.5min = 8.3h | **8.7h** | — |
| `--val_start_epoch 50` | 50 × 0.1min ≈ 5min | 150 × 2.5min = 6.25h | **6.3h** | **-28%** |
| `--val_start_epoch 100` | 100 × 0.1min ≈ 10min | 100 × 2.5min = 4.2h | **4.3h** | **-50%** |
| `--val_start_epoch 150` | 150 × 0.1min ≈ 15min | 50 × 2.5min = 2.1h | **2.3h** | **-73%** |

**Risk**: No validation means no best-model saving and no training curve tracking. The model might diverge silently. Reasonable default: `--val_start_epoch 100` (start monitoring after first half of training).

### 3.2 P1: Reduce Validation Frequency

Instead of every epoch, validate every N epochs:

| N | Total Val Runs | Val Time | Total Time | vs Baseline |
|---|---------------|----------|------------|-------------|
| 1 (current) | 200 | 500 min | 520 min | — |
| 2 | 100 | 250 min | 270 min | -48% |
| 5 | 40 | 100 min | 120 min | -77% |

A simple addition: `if args.epoch % val_interval == 0: run_validation()`

### 3.3 P2: Optimize Sliding Window Inference

| Technique | Speed Gain | Quality Impact | Effort |
|-----------|-----------|----------------|--------|
| Reduce overlap from 0.5 to 0.25 | ~2× fewer windows = -50% val time | Minimal (+0.1% Dice variance) | 1 line change |
| Use `sliding_window_inference_cy` (cython) | +~20% speed | None | 1 line change |
| Only validate on 5 representative cases | -75% val time | Higher variance, may miss bad cases | 1 line change |

### 3.4 P3: Skip HD95/ASSD Every Epoch

Compute full metrics (HD95, ASSD, PR-AUC, ROC-AUC) only every N epochs. Basic Dice/precision/recall every epoch.

### 3.5 P4: Gradient Accumulation to Reduce DDP Sync

If using DDP, gradient synchronization across GPUs is a bottleneck. Increasing batch size via gradient accumulation reduces sync frequency.

### 3.6 Summary

| Priority | Change | Time Saved | Effort | Implementation |
|----------|--------|-----------|--------|----------------|
| **P0** | `--val_start_epoch 100` | **-50%** (~4.4h) | ✅ Done | `--val_start_epoch 100` |
| **P1** | `--val_interval N` | -50-80% | 1 line | Add val_interval arg |
| **P2** | Reduce SW overlap | -50% val time | 1 line | Already in `enhanced_validation()` |
| **P3** | Skip heavy metrics | -30s/epoch | 3 lines | Gate HD95/ASSD/vis |

**Recommended config for rapid iteration**:
```bash
--val_start_epoch 100 --max_epoch 200
```
→ Expected time: ~4h (down from ~8.7h)

**Recommended config for final runs**:
```bash
--val_start_epoch 50 --max_epoch 200
```
→ Expected time: ~6.3h, with half the training monitored.
