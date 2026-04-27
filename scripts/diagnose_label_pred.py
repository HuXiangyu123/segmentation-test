#!/usr/bin/env python
"""
Diagnostic script to check label values and prediction issues.
Samples 3 cases from train and val, prints:
- Label unique values
- GT positive ratio
- Pred positive ratio
- Visualizes one case (GT vs Pred)
"""

import sys
import os
sys.path.insert(0, '/home/glcuser/projhighcv/bone_tumor/MulModSeg_2024')

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# Import from train.py
from dataset.dataloader_bone_tumor import get_loader_bone_tumor
from model.MulModSeg import UNet3D_cy
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete


def diagnose_case(batch, model, args, case_idx, phase, output_dir):
    """Diagnose a single case."""
    model.eval()

    with torch.no_grad():
        x = batch["image"].to(args.device)
        y = batch["label"].to(args.device)

        # Get case name
        case_name = batch.get('name', [f'case_{case_idx}'])[0]
        if isinstance(case_name, torch.Tensor):
            case_name = str(case_name.item())

        print(f"\n{'='*60}")
        print(f"[{phase.upper()}] Case {case_idx}: {case_name}")
        print(f"{'='*60}")

        # Check label statistics
        label_np = y.cpu().numpy()
        unique_values = np.unique(label_np)
        print(f"Label shape: {label_np.shape}")
        print(f"Label unique values: {unique_values}")
        print(f"Label dtype: {label_np.dtype}")
        print(f"Label min/max: {label_np.min():.4f} / {label_np.max():.4f}")

        # GT positive ratio
        total_voxels = label_np.size
        positive_voxels = (label_np > 0).sum()
        gt_positive_ratio = positive_voxels / total_voxels
        print(f"GT positive voxels: {positive_voxels} / {total_voxels}")
        print(f"GT positive ratio: {gt_positive_ratio:.8f} ({gt_positive_ratio*100:.4f}%)")

        # Get prediction
        roi_size = (args.roi_x, args.roi_y, args.roi_z)
        logit_map = sliding_window_inference(
            x, roi_size, sw_batch_size=1, predictor=model, overlap=0.5
        )

        # Check logit statistics
        logit_np = logit_map.cpu().numpy()
        print(f"\nLogit shape: {logit_np.shape}")
        print(f"Logit channels: {logit_np.shape[1]}")
        for ch in range(logit_np.shape[1]):
            ch_data = logit_np[0, ch]
            print(f"  Channel {ch}: min={ch_data.min():.4f}, max={ch_data.max():.4f}, mean={ch_data.mean():.4f}")

        # Convert to prediction (argmax)
        pred_class = torch.argmax(logit_map, dim=1, keepdim=True)
        pred_np = pred_class.cpu().numpy()

        print(f"\nPrediction shape: {pred_np.shape}")
        pred_unique = np.unique(pred_np)
        print(f"Prediction unique values: {pred_unique}")

        # Pred positive ratio
        pred_positive_voxels = (pred_np > 0).sum()
        pred_positive_ratio = pred_positive_voxels / pred_np.size
        print(f"Pred positive voxels: {pred_positive_voxels} / {pred_np.size}")
        print(f"Pred positive ratio: {pred_positive_ratio:.8f} ({pred_positive_ratio*100:.4f}%)")

        # Compute Dice for this case
        from monai.metrics import compute_dice

        # Convert to one-hot
        post_label = AsDiscrete(to_onehot=args.num_class)
        post_pred = AsDiscrete(to_onehot=args.num_class)

        y_onehot = post_label(y)
        pred_onehot = post_pred(pred_class)

        dice_score = compute_dice(pred_onehot, y_onehot, include_background=True)
        print(f"\nDice score (with background): {dice_score.mean().item():.8f}")
        if args.num_class == 2:
            # dice_score shape: [batch, num_classes] or [num_classes]
            print(f"  Dice score shape: {dice_score.shape}")
            if dice_score.numel() >= 2:
                # Flatten and get both values
                dice_flat = dice_score.flatten()
                print(f"  Background Dice: {dice_flat[0].item():.8f}")
                if len(dice_flat) > 1:
                    print(f"  Foreground Dice: {dice_flat[1].item():.8f}")

        # Visualize middle slice
        if case_idx == 0:  # Only visualize first case
            visualize_slice(label_np, pred_np, logit_np, case_name, phase, output_dir)

        return {
            'case_name': case_name,
            'phase': phase,
            'label_unique': unique_values.tolist(),
            'gt_positive_ratio': gt_positive_ratio,
            'pred_positive_ratio': pred_positive_ratio,
            'dice': dice_score.mean().item()
        }


def visualize_slice(label_np, pred_np, logit_np, case_name, phase, output_dir):
    """Visualize middle slice of GT vs Pred."""
    # Get middle slice (axial view)
    mid_slice = label_np.shape[-1] // 2

    label_slice = label_np[0, 0, :, :, mid_slice]
    pred_slice = pred_np[0, 0, :, :, mid_slice]

    # Get logit slices for both channels
    logit_ch0_slice = logit_np[0, 0, :, :, mid_slice]
    logit_ch1_slice = logit_np[0, 1, :, :, mid_slice]

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: GT, Pred, Overlay
    axes[0, 0].imshow(label_slice, cmap='gray')
    axes[0, 0].set_title(f'GT Label (slice {mid_slice})\nUnique: {np.unique(label_slice)}')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(pred_slice, cmap='gray')
    axes[0, 1].set_title(f'Prediction (slice {mid_slice})\nUnique: {np.unique(pred_slice)}')
    axes[0, 1].axis('off')

    # Overlay
    overlay = np.zeros((*label_slice.shape, 3))
    overlay[label_slice > 0] = [0, 1, 0]  # GT in green
    overlay[pred_slice > 0] = [1, 0, 0]   # Pred in red
    overlay[(label_slice > 0) & (pred_slice > 0)] = [1, 1, 0]  # Overlap in yellow
    axes[0, 2].imshow(overlay)
    axes[0, 2].set_title('Overlay\nGreen=GT, Red=Pred, Yellow=Both')
    axes[0, 2].axis('off')

    # Row 2: Logit channels
    im0 = axes[1, 0].imshow(logit_ch0_slice, cmap='viridis')
    axes[1, 0].set_title(f'Logit Channel 0 (Background)\nmin={logit_ch0_slice.min():.2f}, max={logit_ch0_slice.max():.2f}')
    axes[1, 0].axis('off')
    plt.colorbar(im0, ax=axes[1, 0])

    im1 = axes[1, 1].imshow(logit_ch1_slice, cmap='viridis')
    axes[1, 1].set_title(f'Logit Channel 1 (Foreground)\nmin={logit_ch1_slice.min():.2f}, max={logit_ch1_slice.max():.2f}')
    axes[1, 1].axis('off')
    plt.colorbar(im1, ax=axes[1, 1])

    # Difference map
    diff = logit_ch1_slice - logit_ch0_slice
    im2 = axes[1, 2].imshow(diff, cmap='RdBu_r')
    axes[1, 2].set_title(f'Logit Diff (Ch1 - Ch0)\nmin={diff.min():.2f}, max={diff.max():.2f}')
    axes[1, 2].axis('off')
    plt.colorbar(im2, ax=axes[1, 2])

    plt.suptitle(f'{phase.upper()}: {case_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save figure - sanitize filename to remove invalid characters
    safe_case_name = case_name.replace('/', '_').replace('\\', '_')
    output_path = output_dir / f'{phase}_case_{safe_case_name}_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[Visualization] Saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_path', type=str, required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--num_class', type=int, default=2)
    parser.add_argument('--roi_x', type=int, default=96)
    parser.add_argument('--roi_y', type=int, default=96)
    parser.add_argument('--roi_z', type=int, default=96)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='diagnosis_output')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Setup device
    args.device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {args.device}")

    # Load model
    print(f"\nLoading model from: {args.model_path}")
    model = UNet3D_cy(out_channels=args.num_class, act='relu')

    checkpoint = torch.load(args.model_path, map_location=args.device, weights_only=False)
    if 'net' in checkpoint:
        model.load_state_dict(checkpoint['net'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(args.device)
    model.eval()
    print("Model loaded successfully!")

    # Create data loaders
    print("\nCreating data loaders...")
    roi_size = (args.roi_x, args.roi_y, args.roi_z)

    train_loader = get_loader_bone_tumor(
        root_dir=args.data_root_path,
        modality='MIX',
        phase='train',
        batch_size=1,
        roi_size=roi_size,
        train_num_samples=1,
        persistent=False,
        num_workers=args.num_workers,
    )

    val_loader = get_loader_bone_tumor(
        root_dir=args.data_root_path,
        modality='MIX',
        phase='val',
        batch_size=1,
        roi_size=roi_size,
        train_num_samples=1,
        persistent=False,
        num_workers=args.num_workers,
    )

    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")

    # Diagnose 3 cases from each phase
    results = []

    print("\n" + "="*80)
    print("DIAGNOSING TRAIN SET (3 cases)")
    print("="*80)

    for i, batch in enumerate(train_loader):
        if i >= 3:
            break
        result = diagnose_case(batch, model, args, i, 'train', output_dir)
        results.append(result)

    print("\n" + "="*80)
    print("DIAGNOSING VAL SET (3 cases)")
    print("="*80)

    for i, batch in enumerate(val_loader):
        if i >= 3:
            break
        result = diagnose_case(batch, model, args, i, 'val', output_dir)
        results.append(result)

    # Save summary
    summary_path = output_dir / 'diagnosis_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DIAGNOSIS SUMMARY\n")
        f.write("="*80 + "\n\n")

        for result in results:
            f.write(f"Case: {result['case_name']} ({result['phase'].upper()})\n")
            f.write(f"  Label unique values: {result['label_unique']}\n")
            f.write(f"  GT positive ratio: {result['gt_positive_ratio']:.8f} ({result['gt_positive_ratio']*100:.4f}%)\n")
            f.write(f"  Pred positive ratio: {result['pred_positive_ratio']:.8f} ({result['pred_positive_ratio']*100:.4f}%)\n")
            f.write(f"  Dice score: {result['dice']:.8f}\n")
            f.write("\n")

    print(f"\n[Summary] Saved to: {summary_path}")
    print("\n" + "="*80)
    print("DIAGNOSIS COMPLETE!")
    print("="*80)
    print(f"Output directory: {output_dir.absolute()}")
    print(f"  - diagnosis_summary.txt")
    print(f"  - train_case_*_visualization.png")
    print(f"  - val_case_*_visualization.png")


if __name__ == '__main__':
    main()
