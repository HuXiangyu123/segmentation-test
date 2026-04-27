#!/usr/bin/env python
"""
Custom loss functions for bone tumor segmentation.

Implements:
- TverskyLoss: Addresses class imbalance by controlling FP/FN trade-off
- FocalTverskyLoss: Adds focal term to focus on hard examples
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def morphological_boundary_mask_3d(
    fg_mask: torch.Tensor, kernel_size: int = 3
) -> torch.Tensor:
    """
    Foreground boundary band via morphological gradient: dilate(fg) - erode(fg).

    Args:
        fg_mask: [B, 1, D, H, W] float in {0, 1}
        kernel_size: odd integer, effective band width ~ kernel_size//2 voxels

    Returns:
        boundary: [B, 1, D, H, W] float in {0, 1}
    """
    if kernel_size % 2 == 0:
        kernel_size = kernel_size + 1
    pad = kernel_size // 2
    dil = F.max_pool3d(fg_mask, kernel_size=kernel_size, stride=1, padding=pad)
    ero = 1.0 - F.max_pool3d(1.0 - fg_mask, kernel_size=kernel_size, stride=1, padding=pad)
    grad = (dil - ero).clamp(min=0.0)
    return (grad > 1e-6).to(fg_mask.dtype)


def boundary_dice_loss(
    logits: torch.Tensor,
    target_onehot: torch.Tensor,
    kernel_size: int = 3,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """
    Dice loss on foreground channel, evaluated only on voxels in the GT boundary band.
    Encourages sharp agreement with tumor edges (e.g. MR-defined margins).

    Args:
        logits: [B, C, D, H, W]
        target_onehot: [B, C, D, H, W] one-hot (C>=2)
        kernel_size: boundary thickness control (odd)

    Returns:
        scalar loss (1 - dice_on_boundary); 0 if no boundary voxels (grad-connected).
    """
    fg = target_onehot[:, 1:2].detach()
    boundary = morphological_boundary_mask_3d(fg, kernel_size=kernel_size)
    if boundary.sum() < 1:
        return logits.sum() * 0.0

    pred_fg = F.softmax(logits, dim=1)[:, 1:2]
    inter = (pred_fg * fg * boundary).sum()
    p_sum = (pred_fg * boundary).sum()
    g_sum = (fg * boundary).sum()
    dice = (2.0 * inter + smooth) / (p_sum + g_sum + smooth)
    return 1.0 - dice


def _flatten_predictions(pred, target):
    """Flatten spatial dims while keeping batch and channel dims."""
    pred = pred.view(pred.size(0), pred.size(1), -1)  # [B, C, N]
    target = target.view(target.size(0), target.size(1), -1)  # [B, C, N]
    return pred, target


def _select_loss_channels(score_tensor, include_background):
    """
    For binary tumor segmentation, the training target should align with
    foreground-only validation. Excluding background avoids diluting the loss.
    """
    if include_background or score_tensor.size(1) <= 1:
        return score_tensor
    return score_tensor[:, 1:]


class TverskyLoss(nn.Module):
    """
    Tversky Loss for imbalanced segmentation.

    Args:
        alpha: Weight for false positives (higher alpha = penalize FP more)
        beta: Weight for false negatives (higher beta = penalize FN more)
        smooth: Smoothing constant to avoid division by zero

    For our case (over-prediction, FP > FN):
        Use alpha > beta (e.g., alpha=0.7, beta=0.3) to penalize FP more
    """

    def __init__(self, alpha=0.7, beta=0.3, smooth=1.0, include_background=False):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.include_background = include_background

    def forward(self, pred, target):
        """
        Args:
            pred: [B, C, H, W, D] predicted logits (before softmax)
            target: [B, C, H, W, D] one-hot encoded target

        Returns:
            Tversky loss (scalar)
        """
        # This module expects logits and applies softmax internally.
        pred = F.softmax(pred, dim=1)

        # Flatten spatial dimensions
        pred, target = _flatten_predictions(pred, target)

        # Compute TP, FP, FN for each class
        tp = (pred * target).sum(dim=2)  # [B, C]
        fp = (pred * (1 - target)).sum(dim=2)  # [B, C]
        fn = ((1 - pred) * target).sum(dim=2)  # [B, C]

        # Tversky index
        tversky_index = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        tversky_index = _select_loss_channels(tversky_index, self.include_background)

        # Average over batch and classes
        tversky_loss = 1 - tversky_index.mean()

        return tversky_loss


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss for imbalanced segmentation with hard example mining.

    Combines Tversky loss with focal term to focus on hard examples.

    Args:
        alpha: Weight for false positives
        beta: Weight for false negatives
        gamma: Focal parameter (higher gamma = focus more on hard examples)
               Typical values: 1.0, 1.33, 1.5
        smooth: Smoothing constant
    """

    def __init__(self, alpha=0.7, beta=0.3, gamma=1.33, smooth=1.0, include_background=False):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.include_background = include_background

    def forward(self, pred, target):
        """
        Args:
            pred: [B, C, H, W, D] predicted logits (before softmax)
            target: [B, C, H, W, D] one-hot encoded target

        Returns:
            Focal Tversky loss (scalar)
        """
        # This module expects logits and applies softmax internally.
        pred = F.softmax(pred, dim=1)

        # Flatten spatial dimensions
        pred, target = _flatten_predictions(pred, target)

        # Compute TP, FP, FN for each class
        tp = (pred * target).sum(dim=2)  # [B, C]
        fp = (pred * (1 - target)).sum(dim=2)  # [B, C]
        fn = ((1 - pred) * target).sum(dim=2)  # [B, C]

        # Tversky index
        tversky_index = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        tversky_index = _select_loss_channels(tversky_index, self.include_background)

        # Standard focal Tversky form: larger gamma increases focus on hard cases.
        focal_tversky = torch.pow(1 - tversky_index, self.gamma)

        # Average over batch and classes
        focal_tversky_loss = focal_tversky.mean()

        return focal_tversky_loss


class CombinedLoss(nn.Module):
    """
    Combined loss: Tversky/FocalTversky + CrossEntropy

    Args:
        tversky_weight: Weight for Tversky loss
        ce_weight: Weight for CrossEntropy loss
        use_focal: Whether to use Focal Tversky (True) or regular Tversky (False)
        **tversky_kwargs: Arguments for Tversky/FocalTversky loss
    """

    def __init__(self, tversky_weight=1.0, ce_weight=0.5, use_focal=False, **tversky_kwargs):
        super(CombinedLoss, self).__init__()
        self.tversky_weight = tversky_weight
        self.ce_weight = ce_weight

        if use_focal:
            self.tversky_loss = FocalTverskyLoss(**tversky_kwargs)
        else:
            self.tversky_loss = TverskyLoss(**tversky_kwargs)

        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        """
        Args:
            pred: [B, C, H, W, D] predicted logits
            target: [B, C, H, W, D] one-hot encoded target

        Returns:
            Combined loss (scalar)
        """
        # Tversky loss (expects one-hot target)
        tversky = self.tversky_loss(pred, target)

        # CrossEntropy loss (expects class indices)
        target_indices = torch.argmax(target, dim=1)  # [B, H, W, D]
        ce = self.ce_loss(pred, target_indices)

        # Combined
        total_loss = self.tversky_weight * tversky + self.ce_weight * ce

        return total_loss


def get_loss_function(loss_type='dicece', **kwargs):
    """
    Factory function to get loss function.

    Args:
        loss_type: 'dicece', 'tversky', 'focal_tversky'
        **kwargs: Loss-specific parameters

    Returns:
        Loss function
    """
    if loss_type == 'dicece':
        from monai.losses import DiceCELoss
        return DiceCELoss(to_onehot_y=False, softmax=True)

    elif loss_type == 'tversky':
        alpha = kwargs.get('alpha', 0.7)
        beta = kwargs.get('beta', 0.3)
        include_background = kwargs.get('include_background', False)
        return CombinedLoss(
            tversky_weight=1.0,
            ce_weight=0.5,
            use_focal=False,
            alpha=alpha,
            beta=beta,
            include_background=include_background,
        )

    elif loss_type == 'focal_tversky':
        alpha = kwargs.get('alpha', 0.7)
        beta = kwargs.get('beta', 0.3)
        gamma = kwargs.get('gamma', 1.33)
        include_background = kwargs.get('include_background', False)
        return CombinedLoss(
            tversky_weight=1.0,
            ce_weight=0.5,
            use_focal=True,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            include_background=include_background,
        )

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == '__main__':
    # Test losses
    print("Testing loss functions...")

    # Create dummy data
    B, C, H, W, D = 2, 2, 32, 32, 32
    pred = torch.randn(B, C, H, W, D)
    target = torch.zeros(B, C, H, W, D)
    target[:, 0, :, :, :] = 1  # All background

    # Test Tversky
    tversky_loss = TverskyLoss(alpha=0.7, beta=0.3)
    loss = tversky_loss(pred, target)
    print(f"TverskyLoss: {loss.item():.4f}")

    # Test Focal Tversky
    focal_tversky_loss = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=1.33)
    loss = focal_tversky_loss(pred, target)
    print(f"FocalTverskyLoss: {loss.item():.4f}")

    # Test Combined
    combined_loss = CombinedLoss(use_focal=False, alpha=0.7, beta=0.3)
    loss = combined_loss(pred, target)
    print(f"CombinedLoss (Tversky): {loss.item():.4f}")

    combined_loss = CombinedLoss(use_focal=True, alpha=0.7, beta=0.3, gamma=1.33)
    loss = combined_loss(pred, target)
    print(f"CombinedLoss (FocalTversky): {loss.item():.4f}")

    print("\nAll tests passed!")
