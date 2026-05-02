"""
Inference script for MulModSeg checkpoint.
Loads a trained model checkpoint and runs validation to reproduce metrics.

Usage:
    python scripts/infer_checkpoint.py \
        --checkpoint "experiment/infer_ckpt/best_model (3).pt" \
        --dataset bone_tumor --data_root_path ./dataset \
        --backbone swinunetr --train_modality MIX \
        --roi_x 96 --roi_y 96 --roi_z 96
"""
import os
import sys
import argparse
import json
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'MulModSeg_2024')))

from model.MulModSeg import MulModSeg
from monai.networks.nets import SwinUNETR as m_SwinUNETR
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    ScaleIntensityRanged, CropForegroundd, SpatialPadd, ToTensord,
    NormalizeIntensityd,
)
from monai.data import DataLoader, Dataset, list_data_collate


def load_model_from_checkpoint(checkpoint_path, device):
    """Load MulModSeg model from checkpoint, handling architecture differences."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    net_state = ckpt['net']

    # Detect architecture from checkpoint keys
    has_cross_attn = any('cross_attention' in k for k in net_state)
    has_skip_fusion = any('skip_fusion' in k for k in net_state)
    has_cross_attn_proj = any('cross_attn_proj' in k for k in net_state)

    print(f"[INFO] Checkpoint architecture:")
    print(f"  cross_attention: {has_cross_attn}")
    print(f"  skip_fusion: {has_skip_fusion}")
    print(f"  cross_attn_proj: {has_cross_attn_proj}")
    print(f"  epoch: {ckpt.get('epoch', '?')}")
    print(f"  best_dice: {ckpt.get('best_dice', '?')}")
    if 'val_metrics' in ckpt:
        print(f"  val_metrics.foreground_dice_mean: {ckpt['val_metrics'].get('foreground_dice_mean', '?')}")

    # Create model with cross_attention enabled if checkpoint has it
    model = MulModSeg(
        img_size=(96, 96, 96),
        in_channels=1,
        out_channels=2,
        backbone='swinunetr',
        encoding='word_embedding',
        use_cross_attention=has_cross_attn,
        cross_attn_heads=8,
        num_experts=2,
        case_text_alpha=0.3,
    )

    # Fix organ_embedding shape mismatch (checkpoint may be [2,2,512] vs model [2,512])
    if 'organ_embedding' in net_state and net_state['organ_embedding'].shape != model.organ_embedding.shape:
        print(f"[INFO] Resizing organ_embedding: {model.organ_embedding.shape} → {net_state['organ_embedding'].shape}")
        # Re-register buffer with correct shape
        del model.organ_embedding
        model.register_buffer('organ_embedding', net_state['organ_embedding'].clone())

    # Load weights with strict=False to handle missing/extra keys
    missing, unexpected = model.load_state_dict(net_state, strict=False)
    if missing:
        print(f"[WARN] Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"[WARN] Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

    # If checkpoint has cross_attn but missing skip_fusion/cross_attn_proj,
    # we need to ensure inference doesn't crash
    if has_cross_attn and (not has_skip_fusion or not has_cross_attn_proj):
        print("[INFO] Legacy checkpoint: missing skip_fusion/cross_attn_proj.")
        print("[INFO] Inference will use single-CT path (no paired CT+MR input).")

    model.to(device)
    model.eval()
    return model


def get_val_transforms(modality='CT', roi_size=(96, 96, 96)):
    """Get validation transforms for CT or MR."""
    if modality == 'CT':
        return Compose([
            LoadImaged(keys=["image", "label"], ensure_channel_first=True, image_only=True),
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["image"], a_min=-125, a_max=275, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            SpatialPadd(keys=["image", "label"], spatial_size=roi_size, mode='constant'),
            ToTensord(keys=["image", "label"]),
        ])
    else:  # MR
        return Compose([
            LoadImaged(keys=["image", "label"], ensure_channel_first=True, image_only=True),
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            SpatialPadd(keys=["image", "label"], spatial_size=roi_size, mode='constant'),
            ToTensord(keys=["image", "label"]),
        ])


def compute_dice(pred, gt, smooth=1e-8):
    """Compute foreground Dice coefficient."""
    inter = (pred * gt).sum()
    return (2.0 * inter + smooth) / (pred.sum() + gt.sum() + smooth)


def run_inference(model, val_loader, device, roi_size=(96, 96, 96), paired=False):
    """Run sliding window inference on validation set."""
    results = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            y = batch['label'].to(device)
            case_name = batch.get('name', [f'case_{batch_idx}'])[0] if 'name' in batch else f'case_{batch_idx}'

            if paired and 'ct' in batch and 'mr' in batch:
                # Paired CT+MR inference
                x_ct = batch['ct'].to(device)
                x_mr = batch['mr'].to(device)
                x_combined = torch.cat([x_ct, x_mr], dim=1)  # [B, 2, D, H, W]

                def paired_predictor(combined_patch):
                    ct_patch = combined_patch[:, 0:1]
                    mr_patch = combined_patch[:, 1:2]
                    outputs = model(ct_patch, 'CT', x_in_mr=mr_patch)
                    return outputs[0] if isinstance(outputs, (tuple, list)) else outputs

                logit_map = sliding_window_inference(x_combined, roi_size, 1, paired_predictor, overlap=0.5)
            else:
                # Single-CT inference
                x = batch['image'].to(device) if 'image' in batch else batch['ct'].to(device)

                def predictor(x_patch):
                    outputs = model(x_patch, 'CT')
                    return outputs[0] if isinstance(outputs, (tuple, list)) else outputs

                logit_map = sliding_window_inference(x, roi_size, 1, predictor, overlap=0.5)

            prob_map = torch.softmax(logit_map, dim=1)
            pred_binary = (torch.argmax(logit_map, dim=1, keepdim=True) == 1).float()
            gt_binary = (y > 0.5).float()

            dice = compute_dice(pred_binary, gt_binary).item()
            gt_ratio = gt_binary.mean().item()
            pred_ratio = pred_binary.mean().item()

            results.append({
                'case': case_name,
                'dice': dice,
                'gt_ratio': gt_ratio,
                'pred_ratio': pred_ratio,
            })
            print(f"  [{batch_idx+1}/{len(val_loader)}] {case_name}: Dice={dice:.4f}, GT%={gt_ratio:.4f}, Pred%={pred_ratio:.4f}")

    # Summary
    dices = [r['dice'] for r in results]
    print(f"\n{'='*60}")
    print(f"Mean Dice: {np.mean(dices):.4f} ± {np.std(dices):.4f}")
    print(f"Median Dice: {np.median(dices):.4f}")
    print(f"Min/Max: {np.min(dices):.4f} / {np.max(dices):.4f}")
    print(f"{'='*60}")

    # Bucket by GT ratio
    for bucket_name, lo, hi in [("<2%", 0, 0.02), ("2-5%", 0.02, 0.05), (">5%", 0.05, 1.0)]:
        bucket = [r['dice'] for r in results if lo <= r['gt_ratio'] < hi or (hi == 1.0 and r['gt_ratio'] >= lo)]
        if bucket:
            print(f"  {bucket_name}: {np.mean(bucket):.4f} (n={len(bucket)})")

    # Worst/Best 3
    sorted_results = sorted(results, key=lambda r: r['dice'])
    print(f"\nWorst-3:")
    for r in sorted_results[:3]:
        print(f"  {r['case']}: Dice={r['dice']:.4f}, GT%={r['gt_ratio']:.4f}")
    print(f"Best-3:")
    for r in sorted_results[-3:]:
        print(f"  {r['case']}: Dice={r['dice']:.4f}, GT%={r['gt_ratio']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='MulModSeg checkpoint inference')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint .pt file')
    parser.add_argument('--dataset', default='bone_tumor', help='Dataset name')
    parser.add_argument('--data_root_path', default='./dataset', help='Data root path')
    parser.add_argument('--backbone', default='swinunetr', help='Backbone network')
    parser.add_argument('--train_modality', default='MIX', choices=['CT', 'MR', 'MIX'])
    parser.add_argument('--roi_x', type=int, default=96)
    parser.add_argument('--roi_y', type=int, default=96)
    parser.add_argument('--roi_z', type=int, default=96)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='0', help='GPU device')
    parser.add_argument('--fold', type=int, default=-1, help='Fold index (0-4), -1 for fixed-split (no fold)')
    parser.add_argument('--split_file', type=str, default='splits/fold5_splits.json', help='Split file path')
    parser.add_argument('--output_json', type=str, default=None, help='Save results to JSON')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    roi_size = (args.roi_x, args.roi_y, args.roi_z)

    # Load model
    print(f"[INFO] Loading checkpoint: {args.checkpoint}")
    model = load_model_from_checkpoint(args.checkpoint, device)

    # Load drop list from split file
    with open(args.split_file) as f:
        split_data = json.load(f)
    drop_list = split_data.get('drop_list', [])
    print(f"[INFO] Drop list: {len(drop_list)} patients")

    # Load data splits
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'MulModSeg_2024'))
    from dataloader_bone_tumor import get_loader_bone_tumor, get_loader_paired_bone_tumor

    fold = None if args.fold < 0 else args.fold

    paired = args.train_modality == 'MIX'
    if paired:
        val_loader = get_loader_paired_bone_tumor(
            root_dir=args.data_root_path,
            phase='val',
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            fold=fold,
            split_file=args.split_file,
            drop_list=drop_list,
            distributed=False,
        )
    else:
        val_loader = get_loader_bone_tumor(
            root_dir=args.data_root_path,
            modality='CT',
            phase='val',
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            fold=fold,
            split_file=args.split_file,
            drop_list=drop_list,
            distributed=False,
        )
    print(f"[INFO] Val samples: {len(val_loader.dataset)}, paired={paired}")

    # Run inference
    print(f"\n[INFO] Running inference (mode={'paired CT+MR' if paired else 'single CT'})...")
    results = run_inference(model, val_loader, device, roi_size, paired=paired)

    # Also run CT-only if paired, for comparison
    if paired:
        print(f"\n{'='*60}")
        print(f"[INFO] Also running CT-only inference for comparison...")
        ct_val_loader = get_loader_bone_tumor(
            root_dir=args.data_root_path,
            modality='CT',
            phase='val',
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            fold=fold,
            split_file=args.split_file,
            drop_list=drop_list,
            distributed=False,
        )
        ct_results = run_inference(model, ct_val_loader, device, roi_size, paired=False)

    # Save results
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"[INFO] Results saved to {args.output_json}")


if __name__ == '__main__':
    main()
