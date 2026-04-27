#!/usr/bin/env python
"""
Enhanced validation with detailed metrics and visualization.

Legacy version for models whose `forward()` does not accept
`case_text_embedding`, such as `0MulModSeg.py`.
"""

import torch
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from tqdm import tqdm
from monai.inferers import sliding_window_inference
from monai.metrics import compute_average_surface_distance, compute_hausdorff_distance
import csv

from utils.case_text_embedding import get_case_text_embedding_from_batch
from utils.plot_style import BASE_PALETTE, apply_publication_style, save_figure, style_axis


def compute_foreground_case_dice(pred_binary, gt_binary, smooth: float = 1e-8):
    """
    Case-level foreground Dice for binary segmentation.

    Rules:
    - GT empty and Pred empty -> 1.0
    - GT empty and Pred non-empty -> 0.0
    - otherwise -> standard Dice on foreground
    """
    gt_sum = gt_binary.sum().item()
    pred_sum = pred_binary.sum().item()
    if gt_sum == 0:
        return 1.0 if pred_sum == 0 else 0.0

    inter = (pred_binary * gt_binary).sum().item()
    return (2.0 * inter + smooth) / (gt_sum + pred_sum + smooth)


def compute_foreground_case_iou(pred_binary, gt_binary, smooth: float = 1e-8):
    """
    Case-level foreground IoU (Jaccard), same empty-GT convention as Dice.
    """
    gt_sum = gt_binary.sum().item()
    pred_sum = pred_binary.sum().item()
    if gt_sum == 0:
        return 1.0 if pred_sum == 0 else 0.0
    inter = (pred_binary * gt_binary).sum().item()
    union = gt_sum + pred_sum - inter
    return _safe_div(inter, union + smooth)


EPOCH_METRIC_COLUMNS = [
    "epoch",
    "foreground_dice_mean",
    "f1_mean",
    "precision_mean",
    "recall_mean",
    "iou_mean",
    "hd95_mean",
    "assd_mean",
    "voxel_dice",
    "voxel_f1",
    "voxel_precision",
    "voxel_recall",
    "voxel_iou",
    "pr_auc",
    "roc_auc",
    "loss",
    "lr",
]

PLOT_COLORS = {
    "dice": BASE_PALETTE["blue"],
    "f1": BASE_PALETTE["green"],
    "iou": "#B07AA1",
    "precision": BASE_PALETTE["orange"],
    "recall": BASE_PALETTE["purple"],
    "hd95": BASE_PALETTE["red"],
    "assd": BASE_PALETTE["purple"],
    "pr_auc": BASE_PALETTE["blue"],
    "roc_auc": BASE_PALETTE["red"],
    "loss": BASE_PALETTE["red"],
    "lr": BASE_PALETTE["gray"],
    "gt": BASE_PALETTE["green"],
    "pred": BASE_PALETTE["orange"],
    "tp": BASE_PALETTE["blue"],
    "fp": BASE_PALETTE["red"],
    "fn": BASE_PALETTE["purple"],
    "chance": BASE_PALETTE["gray"],
    "text": BASE_PALETTE["black"],
}

# enhanced_validation 内对二值 Pred 的后处理（先闭运算再保留最大连通域）。
# 闭运算: 3×3×3 立方结构元素, iterations=1（约可桥接最宽约 1 个体素的缝隙，视形状而定）。

def _safe_div(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def compute_binary_metrics_from_masks(pred_binary, gt_binary, smooth: float = 1e-8):
    pred_np = np.asarray(pred_binary, dtype=np.float32) > 0.5
    gt_np = np.asarray(gt_binary, dtype=np.float32) > 0.5

    tp = float(np.logical_and(pred_np, gt_np).sum())
    fp = float(np.logical_and(pred_np, np.logical_not(gt_np)).sum())
    fn = float(np.logical_and(np.logical_not(pred_np), gt_np).sum())
    tn = float(np.logical_and(np.logical_not(pred_np), np.logical_not(gt_np)).sum())

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall + smooth)
    dice = _safe_div(2.0 * tp, 2.0 * tp + fp + fn + smooth)
    iou = _safe_div(tp, tp + fp + fn + smooth)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "dice": dice,
        "iou": iou,
    }


def update_probability_histograms(prob_fg, gt_binary, pos_hist, neg_hist):
    scores = np.asarray(prob_fg, dtype=np.float32).reshape(-1)
    labels = np.asarray(gt_binary, dtype=np.float32).reshape(-1) > 0.5

    scores = np.clip(scores, 0.0, 1.0)
    num_bins = len(pos_hist)
    bin_indices = np.minimum((scores * num_bins).astype(np.int32), num_bins - 1)

    if np.any(labels):
        np.add.at(pos_hist, bin_indices[labels], 1)
    if np.any(~labels):
        np.add.at(neg_hist, bin_indices[~labels], 1)


def compute_curves_from_histograms(pos_hist, neg_hist):
    pos_hist = np.asarray(pos_hist, dtype=np.float64)
    neg_hist = np.asarray(neg_hist, dtype=np.float64)

    total_pos = pos_hist.sum()
    total_neg = neg_hist.sum()
    prevalence = _safe_div(total_pos, total_pos + total_neg)

    if total_pos <= 0 or total_neg <= 0:
        return {
            "precision": np.asarray([1.0, prevalence], dtype=np.float64),
            "recall": np.asarray([0.0, 1.0], dtype=np.float64),
            "pr_auc": float("nan"),
            "fpr": np.asarray([0.0, 1.0], dtype=np.float64),
            "tpr": np.asarray([0.0, 1.0], dtype=np.float64),
            "roc_auc": float("nan"),
            "prevalence": prevalence,
        }

    tp = np.cumsum(pos_hist[::-1])
    fp = np.cumsum(neg_hist[::-1])

    recall = tp / total_pos
    precision = tp / np.maximum(tp + fp, 1e-12)
    fpr = fp / total_neg
    tpr = recall.copy()

    recall_curve = np.concatenate(([0.0], recall))
    precision_curve = np.concatenate(([1.0], precision))
    fpr_curve = np.concatenate(([0.0], fpr, [1.0]))
    tpr_curve = np.concatenate(([0.0], tpr, [1.0]))

    pr_auc = float(np.trapz(precision_curve, recall_curve))
    roc_auc = float(np.trapz(tpr_curve, fpr_curve))

    return {
        "precision": precision_curve,
        "recall": recall_curve,
        "pr_auc": pr_auc,
        "fpr": fpr_curve,
        "tpr": tpr_curve,
        "roc_auc": roc_auc,
        "prevalence": prevalence,
    }


def _apply_publication_style():
    apply_publication_style()


def _style_axis(ax, grid_axis="y"):
    style_axis(ax, grid_axis=grid_axis)


def _annotate_best(ax, x, y, color, label):
    if len(x) == 0 or np.all(np.isnan(y)):
        return
    idx = int(np.nanargmax(y))
    ax.scatter([x[idx]], [y[idx]], s=60, color=color, edgecolors="white", linewidths=1.0, zorder=5)
    ax.annotate(
        f"{label}: {y[idx]:.3f}",
        (x[idx], y[idx]),
        xytext=(8, 10),
        textcoords="offset points",
        fontsize=9,
        color=color,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=color, alpha=0.92),
    )


def _annotate_best_lower(ax, x, y, color, label):
    """Mark global minimum (for distance metrics where lower is better)."""
    if len(x) == 0 or np.all(np.isnan(y)):
        return
    idx = int(np.nanargmin(y))
    ax.scatter([x[idx]], [y[idx]], s=60, color=color, edgecolors="white", linewidths=1.0, zorder=5)
    ax.annotate(
        f"{label}: {y[idx]:.3f}",
        (x[idx], y[idx]),
        xytext=(8, 10),
        textcoords="offset points",
        fontsize=9,
        color=color,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=color, alpha=0.92),
    )


def compute_case_hd95_assd(pred_fg, gt_fg, spacing_xyz):
    """
    Per-case HD95 (mm) and symmetric ASSD (mm) for binary foreground.
    pred_fg / gt_fg: (1, 1, *spatial) float tensor, values in {0, 1}.
    spacing_xyz: (sx, sy, sz) matching spatial dimensions of the tensors.
    """
    pred_fg = pred_fg.detach().float()
    gt_fg = gt_fg.detach().float()
    pred_oh = torch.cat([1.0 - pred_fg, pred_fg], dim=1)
    gt_oh = torch.cat([1.0 - gt_fg, gt_fg], dim=1)

    hd = compute_hausdorff_distance(
        pred_oh,
        gt_oh,
        include_background=False,
        distance_metric="euclidean",
        percentile=95.0,
        directed=False,
        spacing=spacing_xyz,
    )
    asd = compute_average_surface_distance(
        pred_oh,
        gt_oh,
        include_background=False,
        symmetric=True,
        distance_metric="euclidean",
        spacing=spacing_xyz,
    )
    hd_v = float(hd.reshape(-1)[0].detach().cpu().item())
    asd_v = float(asd.reshape(-1)[0].detach().cpu().item())
    return hd_v, asd_v


def _plot_metric_series(ax, x, y, *, label, color, linestyle="-", annotate_best=False):
    if len(x) == 0 or np.all(np.isnan(y)):
        return
    ax.plot(
        x,
        y,
        color=color,
        linewidth=2.0,
        linestyle=linestyle,
        label=label,
    )
    if annotate_best:
        _annotate_best(ax, x, y, color, f"Best {label}")


def _select_representative_slice(label_volume, pred_volume):
    combined = (label_volume > 0) | (pred_volume > 0)
    slice_scores = combined.reshape(-1, combined.shape[-1]).sum(axis=0)
    if np.all(slice_scores == 0):
        return int(combined.shape[-1] // 2)
    return int(np.argmax(slice_scores))


def _normalize_slice(image_slice):
    image_slice = np.asarray(image_slice, dtype=np.float32)
    p1, p99 = np.percentile(image_slice, [1.0, 99.0])
    if p99 <= p1:
        return image_slice
    return np.clip((image_slice - p1) / (p99 - p1), 0.0, 1.0)


def _draw_contours(ax, mask, color, linewidth=1.7, fill_holes: bool = False):
    """Draw foreground contour at 0.5 level.

    If fill_holes=True (for GT visualization only): fill interior holes before contouring
    so spurious inner rings from label errors do not appear. Pred should use fill_holes=False.
    """
    if np.any(mask > 0):
        m = np.asarray(mask, dtype=bool)
        to_draw = ndimage.binary_fill_holes(m) if fill_holes else m
        ax.contour(
            to_draw.astype(np.float32),
            levels=[0.5],
            colors=[color],
            linewidths=linewidth,
        )


def _add_panel_label(ax, label):
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        color=PLOT_COLORS["text"],
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.9),
    )


def visualize_case(image, label, pred, case_name, output_path, mr_image=None):
    """Create a publication-style case panel with contours and error map.

    If mr_image is provided, an extra MR panel is inserted after the CT panel.
    Layout (CT+MR): CT | MR | Ground Truth | Prediction | Agreement and Errors
    (GT / Pred / Error panels use MR as background; contours still match the same slice index.)
    Layout (CT only): CT | Ground Truth | Prediction | Agreement and Errors
    """
    image_volume = image[0, 0].cpu().numpy()
    label_volume = label[0, 0].cpu().numpy()
    pred_volume = pred[0, 0].cpu().numpy()

    slice_idx = _select_representative_slice(label_volume, pred_volume)
    image_slice = _normalize_slice(image_volume[:, :, slice_idx])
    label_slice = label_volume[:, :, slice_idx] > 0
    pred_slice = pred_volume[:, :, slice_idx] > 0

    has_mr = mr_image is not None
    if has_mr:
        mr_volume = mr_image[0, 0].cpu().numpy()
        mr_slice = _normalize_slice(mr_volume[:, :, slice_idx])

    tp_mask = np.logical_and(label_slice, pred_slice)
    fp_mask = np.logical_and(~label_slice, pred_slice)
    fn_mask = np.logical_and(label_slice, ~pred_slice)

    case_metrics = compute_binary_metrics_from_masks(pred_slice, label_slice)

    _apply_publication_style()
    n_panels = 5 if has_mr else 4
    fig, axes = plt.subplots(1, n_panels, figsize=(n_panels * 3.0, 3.2))

    if has_mr:
        # A/B: CT、MR 原图；C/D/E：以 MR 为底图勾画 GT/Pred/误差（与 MR 定边缘的阅片习惯一致）
        panel_titles = ["CT", "MR", "Ground Truth", "Prediction", "Agreement and Errors"]
        bg_slices = [image_slice, mr_slice, mr_slice, mr_slice, mr_slice]
        gt_ax, pred_ax, err_ax = axes[2], axes[3], axes[4]
    else:
        panel_titles = ["Image", "Ground Truth", "Prediction", "Agreement and Errors"]
        bg_slices = [image_slice, image_slice, image_slice, image_slice]
        gt_ax, pred_ax, err_ax = axes[1], axes[2], axes[3]

    for idx, ax in enumerate(axes):
        ax.imshow(bg_slices[idx], cmap="gray", interpolation="nearest")
        ax.set_title(panel_titles[idx], pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        _add_panel_label(ax, chr(ord("A") + idx))

    _draw_contours(gt_ax, label_slice, PLOT_COLORS["gt"], fill_holes=True)
    gt_ax.text(0.03, 0.05, "GT contour", transform=gt_ax.transAxes, color=PLOT_COLORS["gt"], fontsize=9)

    _draw_contours(pred_ax, pred_slice, PLOT_COLORS["pred"], fill_holes=False)
    pred_ax.text(0.03, 0.05, "Pred contour", transform=pred_ax.transAxes, color=PLOT_COLORS["pred"], fontsize=9)

    error_overlay = np.zeros((*label_slice.shape, 4), dtype=np.float32)
    error_overlay[tp_mask] = [76 / 255.0, 120 / 255.0, 168 / 255.0, 0.55]
    error_overlay[fp_mask] = [228 / 255.0, 87 / 255.0, 86 / 255.0, 0.65]
    error_overlay[fn_mask] = [178 / 255.0, 121 / 255.0, 162 / 255.0, 0.65]
    err_ax.imshow(error_overlay, interpolation="nearest")
    _draw_contours(err_ax, label_slice, PLOT_COLORS["gt"], linewidth=1.2, fill_holes=True)
    _draw_contours(err_ax, pred_slice, PLOT_COLORS["pred"], linewidth=1.2, fill_holes=False)

    legend_items = [
        Line2D([0], [0], color=PLOT_COLORS["gt"], lw=1.8, label="GT"),
        Line2D([0], [0], color=PLOT_COLORS["pred"], lw=1.8, label="Pred"),
        Line2D([0], [0], color=PLOT_COLORS["tp"], lw=5.0, alpha=0.8, label="TP"),
        Line2D([0], [0], color=PLOT_COLORS["fp"], lw=5.0, alpha=0.8, label="FP"),
        Line2D([0], [0], color=PLOT_COLORS["fn"], lw=5.0, alpha=0.8, label="FN"),
    ]
    err_ax.legend(handles=legend_items, loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=5)

    fig.suptitle(
        f"{case_name}  |  slice={slice_idx}  |  Dice={case_metrics['dice']:.3f}  |  "
        f"Precision={case_metrics['precision']:.3f}  |  Recall={case_metrics['recall']:.3f}",
        y=1.02,
        fontsize=12,
        fontweight="bold",
        color=PLOT_COLORS["text"],
    )
    fig.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def _append_epoch_metrics(
    metrics_path,
    epoch,
    foreground_dice_mean,
    f1_mean,
    precision_mean,
    recall_mean,
    iou_mean=None,
    hd95_mean=None,
    assd_mean=None,
    voxel_dice=None,
    voxel_f1=None,
    voxel_precision=None,
    voxel_recall=None,
    voxel_iou=None,
    pr_auc=None,
    roc_auc=None,
    loss=None,
    lr=None,
):
    """
    追加每个 epoch 的整体指标到 CSV，便于后续画曲线。
    """
    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    write_mode = "a"
    write_header = not metrics_path.exists()
    old_rows = []
    if metrics_path.exists():
        try:
            with metrics_path.open("r", newline="") as f_in:
                reader = csv.DictReader(f_in)
                fieldnames = reader.fieldnames
                if not fieldnames or any(col not in fieldnames for col in EPOCH_METRIC_COLUMNS):
                    for row in reader:
                        old_rows.append(row)
                    write_mode = "w"
                    write_header = True
        except Exception:
            write_mode = "w"
            write_header = True

    with metrics_path.open(write_mode, newline="") as f:
        writer = csv.writer(f)

        def _safe_float(x):
            if x is None:
                return ""
            try:
                return float(x)
            except Exception:
                return ""

        def _finite_or_empty(x):
            if x is None:
                return ""
            try:
                v = float(x)
                return "" if np.isnan(v) else v
            except (TypeError, ValueError):
                return ""

        if write_header:
            writer.writerow(EPOCH_METRIC_COLUMNS)
            for row in old_rows:
                writer.writerow([row.get(col, "") for col in EPOCH_METRIC_COLUMNS])

        writer.writerow(
            [
                int(epoch),
                float(foreground_dice_mean),
                float(f1_mean),
                float(precision_mean),
                float(recall_mean),
                _safe_float(iou_mean),
                _finite_or_empty(hd95_mean),
                _finite_or_empty(assd_mean),
                _safe_float(voxel_dice),
                _safe_float(voxel_f1),
                _safe_float(voxel_precision),
                _safe_float(voxel_recall),
                _safe_float(voxel_iou),
                _safe_float(pr_auc),
                _safe_float(roc_auc),
                _safe_float(loss),
                _safe_float(lr),
            ]
        )


def plot_epoch_curves(metrics_csv, output_dir=None):
    """
    从 metrics CSV 中读取 epoch 级别的 loss / dice / lr，并画出随 epoch 变化的曲线。

    Args:
        metrics_csv: enhanced_validation 写出的 CSV 路径
        output_dir: 曲线图输出目录（默认与 CSV 同目录）
    """
    metrics_csv = Path(metrics_csv)
    if not metrics_csv.exists():
        print(f"[Visualization] Metrics CSV not found: {metrics_csv}")
        return

    if output_dir is None:
        output_dir = metrics_csv.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    epochs = []
    metric_arrays = {
        "foreground_dice_mean": [],
        "f1_mean": [],
        "precision_mean": [],
        "recall_mean": [],
        "iou_mean": [],
        "hd95_mean": [],
        "assd_mean": [],
        "pr_auc": [],
        "roc_auc": [],
        "loss": [],
        "lr": [],
    }

    with metrics_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                epoch = int(row.get("epoch", 0))
            except ValueError:
                continue

            epochs.append(epoch)

            for metric_name in metric_arrays:
                raw_value = row.get(metric_name, "")
                try:
                    metric_arrays[metric_name].append(float(raw_value) if raw_value not in ("", None) else np.nan)
                except ValueError:
                    metric_arrays[metric_name].append(np.nan)

    if len(epochs) == 0:
        print(f"[Visualization] No data in metrics CSV: {metrics_csv}")
        return

    epochs = np.asarray(epochs, dtype=np.int32)
    metric_arrays = {name: np.asarray(values, dtype=float) for name, values in metric_arrays.items()}

    _apply_publication_style()
    fig, axes = plt.subplots(3, 2, figsize=(8.6, 9.6))
    axes = axes.reshape(-1)

    ax = axes[0]
    _plot_metric_series(ax, epochs, metric_arrays["foreground_dice_mean"], label="Dice", color=PLOT_COLORS["dice"], annotate_best=True)
    _plot_metric_series(ax, epochs, metric_arrays["f1_mean"], label="F1", color=PLOT_COLORS["f1"], linestyle="--")
    _plot_metric_series(ax, epochs, metric_arrays["iou_mean"], label="IoU", color=PLOT_COLORS["iou"], linestyle=":")
    ax.set_title("Validation Dice, F1, and IoU")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.02)
    _style_axis(ax)
    ax.legend(loc="lower right")

    ax = axes[1]
    _plot_metric_series(ax, epochs, metric_arrays["precision_mean"], label="Precision", color=PLOT_COLORS["precision"])
    _plot_metric_series(ax, epochs, metric_arrays["recall_mean"], label="Recall", color=PLOT_COLORS["recall"], linestyle="--")
    ax.set_title("Validation Precision and Recall")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.02)
    _style_axis(ax)
    ax.legend(loc="lower right")

    ax = axes[2]
    _plot_metric_series(ax, epochs, metric_arrays["pr_auc"], label="PR-AUC", color=PLOT_COLORS["pr_auc"])
    _plot_metric_series(ax, epochs, metric_arrays["roc_auc"], label="ROC-AUC", color=PLOT_COLORS["roc_auc"], linestyle="--")
    ax.set_title("Ranking Metrics")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.0, 1.02)
    _style_axis(ax)
    ax.legend(loc="lower right")

    ax = axes[3]
    if not np.all(np.isnan(metric_arrays["loss"])):
        _plot_metric_series(ax, epochs, metric_arrays["loss"], label="Loss", color=PLOT_COLORS["loss"])
    ax.set_title("Optimization Schedule")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss", color=PLOT_COLORS["loss"])
    ax.tick_params(axis="y", colors=PLOT_COLORS["loss"])
    _style_axis(ax)

    if not np.all(np.isnan(metric_arrays["lr"])):
        ax2 = ax.twinx()
        ax2.plot(epochs, metric_arrays["lr"], color=PLOT_COLORS["lr"], linewidth=1.8, linestyle="--", label="LR")
        ax2.set_ylabel("Learning Rate", color=PLOT_COLORS["lr"])
        ax2.tick_params(axis="y", colors=PLOT_COLORS["lr"])
        ax2.spines["top"].set_visible(False)

    ax = axes[4]
    _plot_metric_series(
        ax,
        epochs,
        metric_arrays["hd95_mean"],
        label="HD95",
        color=PLOT_COLORS["hd95"],
        annotate_best=False,
    )
    _annotate_best_lower(ax, epochs, metric_arrays["hd95_mean"], PLOT_COLORS["hd95"], "Best HD95")
    ax.set_title("Validation HD95 (95% Hausdorff, mm)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Distance (mm)")
    _style_axis(ax)

    ax = axes[5]
    _plot_metric_series(
        ax,
        epochs,
        metric_arrays["assd_mean"],
        label="ASSD",
        color=PLOT_COLORS["assd"],
        annotate_best=False,
    )
    _annotate_best_lower(ax, epochs, metric_arrays["assd_mean"], PLOT_COLORS["assd"], "Best ASSD")
    ax.set_title("Validation ASSD (symmetric, mm)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Distance (mm)")
    _style_axis(ax)

    fig.suptitle("Validation Metrics Across Epochs", fontsize=13, fontweight="bold", color=PLOT_COLORS["text"])
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    fig_path = output_dir / "training_curves.png"
    save_figure(fig, fig_path)
    plt.close(fig)

    print(f"[Visualization] Saved epoch curves to: {fig_path}")


def plot_validation_summary(metrics, curve_data, epoch, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _apply_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.8))

    ax = axes[0]
    ax.plot(
        curve_data["recall"],
        curve_data["precision"],
        color=PLOT_COLORS["pr_auc"],
        linewidth=2.2,
        label=f"PR curve (AUC={curve_data['pr_auc']:.3f})" if np.isfinite(curve_data["pr_auc"]) else "PR curve",
    )
    ax.axhline(
        curve_data["prevalence"],
        color=PLOT_COLORS["chance"],
        linewidth=1.2,
        linestyle="--",
        label=f"Prevalence={curve_data['prevalence']:.3f}",
    )
    ax.scatter(
        [metrics["voxel_recall"]],
        [metrics["voxel_precision"]],
        s=70,
        color=PLOT_COLORS["precision"],
        edgecolors="white",
        linewidths=1.0,
        zorder=6,
        label="Threshold=0.5",
    )
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    _style_axis(ax, grid_axis="both")
    ax.legend(loc="lower left")

    ax = axes[1]
    ax.plot(
        curve_data["fpr"],
        curve_data["tpr"],
        color=PLOT_COLORS["roc_auc"],
        linewidth=2.2,
        label=f"ROC curve (AUC={curve_data['roc_auc']:.3f})" if np.isfinite(curve_data["roc_auc"]) else "ROC curve",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color=PLOT_COLORS["chance"], linewidth=1.2, label="Chance")
    ax.scatter(
        [_safe_div(metrics["voxel_fp"], metrics["voxel_fp"] + metrics["voxel_tn"])],
        [metrics["voxel_recall"]],
        s=70,
        color=PLOT_COLORS["recall"],
        edgecolors="white",
        linewidths=1.0,
        zorder=6,
        label="Threshold=0.5",
    )
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    _style_axis(ax, grid_axis="both")
    ax.legend(loc="lower right")

    ax = axes[2]
    bar_labels = ["Dice", "F1", "IoU", "Prec.", "Recall"]
    bar_values = [
        metrics["foreground_dice_mean"],
        metrics["f1_mean"],
        metrics["iou_mean"],
        metrics["precision_mean"],
        metrics["recall_mean"],
    ]
    bar_colors = [
        PLOT_COLORS["dice"],
        PLOT_COLORS["f1"],
        PLOT_COLORS["iou"],
        PLOT_COLORS["precision"],
        PLOT_COLORS["recall"],
    ]
    bars = ax.bar(bar_labels, bar_values, color=bar_colors, width=0.5, edgecolor="white", linewidth=0.8)
    for bar, value in zip(bars, bar_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=PLOT_COLORS["text"],
        )
    ax.set_title("Epoch Metric Summary")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.08)
    _style_axis(ax)

    fig.suptitle(f"Validation Summary at Epoch {epoch}", fontsize=13, fontweight="bold", color=PLOT_COLORS["text"])
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig_path = output_dir / f"epoch_{int(epoch):03d}_validation_summary.png"
    save_figure(fig, fig_path)
    plt.close(fig)

    print(f"[Visualization] Saved validation summary to: {fig_path}")


def enhanced_validation(args, val_loader, model, epoch, output_dir=None, loss=None, lr=None):
    """
    Enhanced validation with detailed metrics and visualization.
    """
    model.eval()

    print(f"\n[Enhanced Validation] Epoch {epoch}")
    print(f"[Validation] Total batches: {len(val_loader)}")

    spacing_xyz = (float(args.space_x), float(args.space_y), float(args.space_z))

    case_results = []
    prob_pos_hist = np.zeros(256, dtype=np.int64)
    prob_neg_hist = np.zeros(256, dtype=np.int64)
    voxel_tp = 0.0
    voxel_fp = 0.0
    voxel_fn = 0.0
    voxel_tn = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation")):
            use_paired = 'ct' in batch and 'mr' in batch

            if use_paired:
                x_ct = batch['ct'].to(args.device)
                x_mr = batch['mr'].to(args.device)
                y = batch['label'].to(args.device)
                x_vis = x_ct
                case_text_embedding = get_case_text_embedding_from_batch(
                    batch,
                    getattr(args, "case_text_store", None),
                    args.device,
                    modality='CT',
                ) if getattr(args, "use_case_text_embedding", False) else None
            else:
                x_ct = batch["image"].to(args.device)
                x_mr = None
                y = batch["label"].to(args.device)
                x_vis = x_ct
                batch_modality = batch['modality'][0] if 'modality' in batch else None
                case_text_embedding = get_case_text_embedding_from_batch(
                    batch,
                    getattr(args, "case_text_store", None),
                    args.device,
                    modality=batch_modality,
                ) if getattr(args, "use_case_text_embedding", False) else None

            if 'name' in batch:
                case_name = batch['name'][0] if isinstance(batch['name'], list) else str(batch['name'])
            elif 'sample_id' in batch:
                case_name = str(batch['sample_id'][0] if isinstance(batch['sample_id'], list) else batch['sample_id'])
            else:
                case_name = f'case_{batch_idx}'

            roi_size = (args.roi_x, args.roi_y, args.roi_z)

            if use_paired:
                x_combined = torch.cat([x_ct, x_mr], dim=1)

                def paired_predictor(combined_patch):
                    ct_patch = combined_patch[:, 0:1]
                    mr_patch = combined_patch[:, 1:2]
                    # MulModSeg.forward() 可能返回 (final_logits, router_logits, weights)
                    # sliding_window_inference 需要 logits 张量，因此要解包取第一个输出。
                    outputs = model(
                        ct_patch,
                        'CT',
                        x_in_mr=mr_patch,
                        case_text_embedding=case_text_embedding,
                    )
                    return outputs[0] if isinstance(outputs, (tuple, list)) else outputs

                logit_map = sliding_window_inference(
                    x_combined, roi_size, 1, paired_predictor, overlap=0.5
                )
            elif args.with_text_embedding == 1:
                z = batch['modality']
                def single_predictor_with_text(x):
                    outputs = model(x, z[0], case_text_embedding=case_text_embedding)
                    return outputs[0] if isinstance(outputs, (tuple, list)) else outputs

                logit_map = sliding_window_inference(
                    x_ct, roi_size, 1,
                    single_predictor_with_text,
                    overlap=0.5,
                )
            else:
                def single_predictor_no_text(x):
                    outputs = model(x)
                    return outputs[0] if isinstance(outputs, (tuple, list)) else outputs

                logit_map = sliding_window_inference(
                    x_ct, roi_size, 1, single_predictor_no_text, overlap=0.5
                )

            prob_map = torch.softmax(logit_map, dim=1)
            prob_fg = prob_map[:, 1:2]
            pred_argmax = torch.argmax(logit_map, dim=1, keepdim=True)

            gt_binary = (y > 0.5).float()
            pred_binary = (pred_argmax == 1).float()

            if batch_idx == 0:
                try:
                    y_cpu = y.detach().cpu()
                    unique_y = torch.unique(y_cpu)
                    print(f"\n[DEBUG][Val] Epoch {epoch} batch0: raw y shape={tuple(y_cpu.shape)}, unique={unique_y.tolist()[:20]}")
                    if y_cpu.numel() > 0:
                        raw_pos_ratio = (y_cpu > 0).float().mean().item()
                        print(f"[DEBUG][Val] Epoch {epoch} batch0: raw y (>0) pos_ratio={raw_pos_ratio:.6f}")

                    gt_fg = gt_binary.detach().cpu()
                    pr_fg = pred_binary.detach().cpu()
                    gt_ratio = gt_fg.mean().item()
                    pr_ratio = pr_fg.mean().item()
                    inter = (gt_fg * pr_fg).sum().item()
                    gt_sum = gt_fg.sum().item()
                    pr_sum = pr_fg.sum().item()
                    manual_dice = (2.0 * inter) / (gt_sum + pr_sum + 1e-8) if (gt_sum + pr_sum) > 0 else 1.0
                    print(f"[DEBUG][Val] Epoch {epoch} batch0: gt_fg_ratio={gt_ratio:.6f}, pred_fg_ratio={pr_ratio:.6f}")
                    print(f"[DEBUG][Val] Epoch {epoch} batch0: gt_fg_sum={gt_sum:.1f}, pred_fg_sum={pr_sum:.1f}, inter={inter:.1f}, manual_dice_fg={manual_dice:.6f}")
                except Exception as e:
                    print(f"[DEBUG][Val] DEBUG block failed: {e}")

            gt_fg_np = gt_binary[0, 0].cpu().numpy()
            pred_fg_np = pred_binary[0, 0].cpu().numpy()
            prob_fg_np = prob_fg[0, 0].detach().cpu().numpy()
            case_metric = compute_binary_metrics_from_masks(pred_fg_np, gt_fg_np)
            update_probability_histograms(prob_fg_np, gt_fg_np, prob_pos_hist, prob_neg_hist)
            voxel_tp += case_metric["tp"]
            voxel_fp += case_metric["fp"]
            voxel_fn += case_metric["fn"]
            voxel_tn += case_metric["tn"]

            gt_positive_ratio = gt_fg_np.sum() / gt_fg_np.size
            pred_positive_ratio = pred_fg_np.sum() / pred_fg_np.size
            is_empty_gt = float(gt_positive_ratio == 0.0)
            foreground_dice = compute_foreground_case_dice(pred_binary[0, 0], gt_binary[0, 0])
            foreground_iou = compute_foreground_case_iou(pred_binary[0, 0], gt_binary[0, 0])

            if is_empty_gt:
                hd95_v, assd_v = float("nan"), float("nan")
            else:
                try:
                    hd95_v, assd_v = compute_case_hd95_assd(pred_binary, gt_binary, spacing_xyz)
                except Exception as ex:
                    print(f"[HD95/ASSD] case={case_name} failed: {ex}")
                    hd95_v, assd_v = float("nan"), float("nan")

            case_results.append({
                'case_name': case_name,
                'foreground_dice': foreground_dice,
                'foreground_iou': foreground_iou,
                'precision': case_metric['precision'],
                'recall': case_metric['recall'],
                'f1': case_metric['f1'],
                'case_dice_thresholded': case_metric['dice'],
                'gt_positive_ratio': gt_positive_ratio,
                'pred_positive_ratio': pred_positive_ratio,
                'is_empty_gt': bool(is_empty_gt),
                'hd95': hd95_v,
                'assd': assd_v,
                'image': x_vis.cpu(),
                'mr_image': x_mr.cpu() if x_mr is not None else None,
                'label': y.cpu(),
                'pred': pred_argmax.cpu()
            })

    nonempty_cases = [r for r in case_results if not r['is_empty_gt']]
    empty_cases = [r for r in case_results if r['is_empty_gt']]

    foreground_dices = [r['foreground_dice'] for r in nonempty_cases]
    precisions = [r['precision'] for r in nonempty_cases]
    recalls = [r['recall'] for r in nonempty_cases]
    f1_scores = [r['f1'] for r in nonempty_cases]
    ious = [r['foreground_iou'] for r in nonempty_cases]

    foreground_dice_mean = np.mean(foreground_dices) if len(foreground_dices) > 0 else 0.0
    foreground_dice_std = np.std(foreground_dices) if len(foreground_dices) > 0 else 0.0
    precision_mean = np.mean(precisions) if len(precisions) > 0 else 0.0
    recall_mean = np.mean(recalls) if len(recalls) > 0 else 0.0
    f1_mean = np.mean(f1_scores) if len(f1_scores) > 0 else 0.0
    iou_mean = np.mean(ious) if len(ious) > 0 else 0.0
    iou_std = np.std(ious) if len(ious) > 0 else 0.0

    if len(nonempty_cases) > 0:
        hd95_arr = np.asarray([r["hd95"] for r in nonempty_cases], dtype=np.float64)
        assd_arr = np.asarray([r["assd"] for r in nonempty_cases], dtype=np.float64)
        hd95_mean = float(np.nanmean(hd95_arr))
        hd95_std = float(np.nanstd(hd95_arr))
        assd_mean = float(np.nanmean(assd_arr))
        assd_std = float(np.nanstd(assd_arr))
    else:
        hd95_mean = float("nan")
        hd95_std = float("nan")
        assd_mean = float("nan")
        assd_std = float("nan")

    voxel_precision = _safe_div(voxel_tp, voxel_tp + voxel_fp)
    voxel_recall = _safe_div(voxel_tp, voxel_tp + voxel_fn)
    voxel_f1 = _safe_div(2.0 * voxel_precision * voxel_recall, voxel_precision + voxel_recall + 1e-8)
    voxel_dice = _safe_div(2.0 * voxel_tp, 2.0 * voxel_tp + voxel_fp + voxel_fn + 1e-8)
    voxel_iou = _safe_div(voxel_tp, voxel_tp + voxel_fp + voxel_fn + 1e-8)
    curve_data = compute_curves_from_histograms(prob_pos_hist, prob_neg_hist)

    bucket_lt2 = [r['foreground_dice'] for r in nonempty_cases if 0.0 < r['gt_positive_ratio'] < 0.02]
    bucket_2to5 = [r['foreground_dice'] for r in nonempty_cases if 0.02 <= r['gt_positive_ratio'] <= 0.05]
    bucket_gt5 = [r['foreground_dice'] for r in nonempty_cases if r['gt_positive_ratio'] > 0.05]

    bucket_lt2_dice = np.mean(bucket_lt2) if len(bucket_lt2) > 0 else 0.0
    bucket_2to5_dice = np.mean(bucket_2to5) if len(bucket_2to5) > 0 else 0.0
    bucket_gt5_dice = np.mean(bucket_gt5) if len(bucket_gt5) > 0 else 0.0

    empty_case_accuracy = (
        np.mean([1.0 if r['pred_positive_ratio'] == 0.0 else 0.0 for r in empty_cases])
        if len(empty_cases) > 0 else 0.0
    )
    empty_case_fp_rate = (
        np.mean([1.0 if r['pred_positive_ratio'] > 0.0 else 0.0 for r in empty_cases])
        if len(empty_cases) > 0 else 0.0
    )
    empty_case_pred_positive_ratio_mean = (
        np.mean([r['pred_positive_ratio'] for r in empty_cases])
        if len(empty_cases) > 0 else 0.0
    )

    print(f"\n[Validation Metrics]")
    print(f"  Non-empty GT cases: {len(nonempty_cases)}")
    print(f"  Empty GT cases: {len(empty_cases)}")
    print(f"  Foreground Dice (non-empty GT only): {foreground_dice_mean:.4f} +/- {foreground_dice_std:.4f}")
    print(f"  F1 (non-empty GT only): {f1_mean:.4f}")
    print(f"  Precision (non-empty GT only): {precision_mean:.4f}")
    print(f"  Recall (non-empty GT only): {recall_mean:.4f}")
    print(f"  IoU / Jaccard (non-empty GT only): {iou_mean:.4f} +/- {iou_std:.4f}")
    if np.isfinite(hd95_mean):
        print(f"  HD95 (non-empty GT, mm): {hd95_mean:.4f} +/- {hd95_std:.4f}")
        print(f"  ASSD (non-empty GT, mm): {assd_mean:.4f} +/- {assd_std:.4f}")
    else:
        print(f"  HD95 / ASSD: n/a (no non-empty GT cases)")
    print(f"  Voxel Dice @0.5: {voxel_dice:.4f}")
    print(f"  Voxel IoU @0.5: {voxel_iou:.4f}")
    print(f"  Voxel F1 @0.5: {voxel_f1:.4f}")
    print(f"  PR-AUC: {curve_data['pr_auc']:.4f}" if np.isfinite(curve_data['pr_auc']) else "  PR-AUC: nan")
    print(f"  ROC-AUC: {curve_data['roc_auc']:.4f}" if np.isfinite(curve_data['roc_auc']) else "  ROC-AUC: nan")
    if len(empty_cases) > 0:
        print(f"\n[Empty-GT Case Metrics]")
        print(f"  Empty case accuracy: {empty_case_accuracy:.4f}")
        print(f"  Empty case FP rate: {empty_case_fp_rate:.4f}")
        print(f"  Empty case mean pred-positive ratio: {empty_case_pred_positive_ratio_mean:.6f}")
    print(f"\n[Bucketed Dice by GT Positive Ratio]")
    print(f"  <2%:   {bucket_lt2_dice:.4f} (n={len(bucket_lt2)})")
    print(f"  2-5%:  {bucket_2to5_dice:.4f} (n={len(bucket_2to5)})")
    print(f"  >5%:   {bucket_gt5_dice:.4f} (n={len(bucket_gt5)})")

    ranking_cases = nonempty_cases if len(nonempty_cases) > 0 else case_results
    sorted_cases = sorted(ranking_cases, key=lambda x: x['foreground_dice'])
    worst_3 = sorted_cases[:3]
    best_3 = sorted_cases[-3:][::-1] if len(sorted_cases) >= 3 else sorted_cases[::-1]

    print(f"\n[Worst-3 Cases]")
    for i, case in enumerate(worst_3):
        print(
            f"  {i+1}. {case['case_name']}: Dice={case['foreground_dice']:.4f}, "
            f"GT%={case['gt_positive_ratio']*100:.2f}%, "
            f"Pred%={case['pred_positive_ratio']*100:.2f}%"
        )

    print(f"\n[Best-3 Cases]")
    for i, case in enumerate(best_3):
        print(
            f"  {i+1}. {case['case_name']}: Dice={case['foreground_dice']:.4f}, "
            f"GT%={case['gt_positive_ratio']*100:.2f}%, "
            f"Pred%={case['pred_positive_ratio']*100:.2f}%"
        )

    if output_dir is not None:
        # 最差三个可视化
        vis_dir_worst = Path(output_dir) / f'epoch_{epoch:03d}_worst3'
        vis_dir_worst.mkdir(parents=True, exist_ok=True)

        for i, case in enumerate(worst_3):
            safe_name = case['case_name'].replace('/', '_').replace('\\', '_')
            output_path = vis_dir_worst / f'worst{i+1}_{safe_name}_dice{case["foreground_dice"]:.3f}.png'
            visualize_case(
                case['image'],
                case['label'],
                case['pred'],
                f"Worst-{i+1}: {case['case_name']} (Dice={case['foreground_dice']:.3f})",
                output_path,
                mr_image=case.get('mr_image', None),
            )

        print(f"\n[Visualization] Saved worst-3 cases to: {vis_dir_worst}")

        # 最好三个可视化
        vis_dir_best = Path(output_dir) / f'epoch_{epoch:03d}_best3'
        vis_dir_best.mkdir(parents=True, exist_ok=True)

        for i, case in enumerate(best_3):
            safe_name = case['case_name'].replace('/', '_').replace('\\', '_')
            output_path = vis_dir_best / f'best{i+1}_{safe_name}_dice{case["foreground_dice"]:.3f}.png'
            visualize_case(
                case['image'],
                case['label'],
                case['pred'],
                f"Best-{i+1}: {case['case_name']} (Dice={case['foreground_dice']:.3f})",
                output_path,
                mr_image=case.get('mr_image', None),
            )

        print(f"[Visualization] Saved best-3 cases to: {vis_dir_best}")

        # 记录 epoch 级别指标，并更新 Loss / Dice / LR 曲线
        metrics_csv = Path(output_dir) / "metrics_epoch.csv"
        _append_epoch_metrics(
            metrics_csv,
            epoch,
            foreground_dice_mean,
            f1_mean,
            precision_mean,
            recall_mean,
            iou_mean=iou_mean,
            hd95_mean=hd95_mean,
            assd_mean=assd_mean,
            voxel_dice=voxel_dice,
            voxel_f1=voxel_f1,
            voxel_precision=voxel_precision,
            voxel_recall=voxel_recall,
            voxel_iou=voxel_iou,
            pr_auc=curve_data['pr_auc'],
            roc_auc=curve_data['roc_auc'],
            loss=loss,
            lr=lr,
        )
        plot_epoch_curves(metrics_csv, output_dir=output_dir)
        plot_validation_summary(
            {
                'foreground_dice_mean': foreground_dice_mean,
                'f1_mean': f1_mean,
                'precision_mean': precision_mean,
                'recall_mean': recall_mean,
                'iou_mean': iou_mean,
                'voxel_precision': voxel_precision,
                'voxel_recall': voxel_recall,
                'voxel_tp': voxel_tp,
                'voxel_fp': voxel_fp,
                'voxel_tn': voxel_tn,
            },
            curve_data,
            epoch=epoch,
            output_dir=output_dir,
        )

    model.train()

    return {
        'foreground_dice_mean': foreground_dice_mean,
        'foreground_dice_std': foreground_dice_std,
        'f1_mean': f1_mean,
        'precision_mean': precision_mean,
        'recall_mean': recall_mean,
        'iou_mean': iou_mean,
        'iou_std': iou_std,
        'hd95_mean': hd95_mean,
        'hd95_std': hd95_std,
        'assd_mean': assd_mean,
        'assd_std': assd_std,
        'voxel_dice': voxel_dice,
        'voxel_f1': voxel_f1,
        'voxel_precision': voxel_precision,
        'voxel_recall': voxel_recall,
        'voxel_iou': voxel_iou,
        'pr_auc': curve_data['pr_auc'],
        'roc_auc': curve_data['roc_auc'],
        'nonempty_case_count': len(nonempty_cases),
        'empty_case_count': len(empty_cases),
        'empty_case_accuracy': empty_case_accuracy,
        'empty_case_fp_rate': empty_case_fp_rate,
        'empty_case_pred_positive_ratio_mean': empty_case_pred_positive_ratio_mean,
        'bucket_lt2_dice': bucket_lt2_dice,
        'bucket_lt2_count': len(bucket_lt2),
        'bucket_2to5_dice': bucket_2to5_dice,
        'bucket_2to5_count': len(bucket_2to5),
        'bucket_gt5_dice': bucket_gt5_dice,
        'bucket_gt5_count': len(bucket_gt5),
        'worst_cases': [(c['case_name'], c['foreground_dice'], c['gt_positive_ratio']) for c in worst_3],
        'best_cases': [(c['case_name'], c['foreground_dice'], c['gt_positive_ratio']) for c in best_3],
    }
