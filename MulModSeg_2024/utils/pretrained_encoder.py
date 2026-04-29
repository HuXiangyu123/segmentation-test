"""
Utility for loading pretrained SwinUNETR encoder weights.

Supported formats:
  1. SSL self-supervised (ssl_pretrained_weights.pth):
     top-level structure: {'model': OrderedDict, 'optimizer': ..., 'epoch': ...}
     encoder keys: encoder.patch_embed.*, encoder.layers1-4.*, encoder.norm.*
     → remap to: swinViT.patch_embed.*, swinViT.layers1-4.*, swinViT.norm.*

  2. BTCV/MONAI pretrained (e.g. model.pt from MONAI model zoo):
     top-level structure: {'state_dict': OrderedDict, ...}  or raw OrderedDict
     keys already match swinViT.* — load directly into swinViT sub-module

Both formats load only encoder keys (swinViT.*) and skip decoder keys.
"""

from __future__ import annotations

import collections
import os
from typing import Tuple

import torch
import torch.nn as nn


_ENCODER_PREFIXES = (
    "swinViT.patch_embed",
    "swinViT.layers1",
    "swinViT.layers2",
    "swinViT.layers3",
    "swinViT.layers4",
    "swinViT.norm",
)


def _remap_ssl_keys(raw_state: dict) -> dict:
    """Remap SSL pretrained keys: encoder.X → swinViT.X (encoder-only).

    Also collapses the extra nesting level in layer paths:
      SSL:  encoder.layers1.0.0.blocks.X → swinViT.layers1.0.blocks.X
    """
    import re

    remapped = {}
    for k, v in raw_state.items():
        if not k.startswith("encoder."):
            continue
        # Skip mask_token (no equivalent in model)
        if k == "encoder.mask_token":
            continue
        new_k = "swinViT." + k[len("encoder."):]
        # Collapse layers<N>.0.0 → layers<N>.0 (SSL has extra Sequential nesting)
        new_k = re.sub(r"(layers\d+)\.0\.0\.", r"\1.0.", new_k)
        remapped[new_k] = v
    return remapped


def _extract_state_dict(ckpt: object) -> dict:
    """Extract the raw state dict from various checkpoint formats."""
    if isinstance(ckpt, dict):
        # SSL format: top-level 'model' key
        if "model" in ckpt and isinstance(ckpt["model"], (dict, collections.OrderedDict)):
            return dict(ckpt["model"])
        # MONAI format: top-level 'state_dict' key
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], (dict, collections.OrderedDict)):
            return dict(ckpt["state_dict"])
        # Also check 'net' (finetune checkpoint format)
        if "net" in ckpt and isinstance(ckpt["net"], (dict, collections.OrderedDict)):
            return dict(ckpt["net"])
        # Assume it is already a flat state dict
        return dict(ckpt)
    raise ValueError(f"Unsupported checkpoint type: {type(ckpt)}")


def _detect_format(state_dict: dict) -> str:
    """Return 'ssl' if keys start with 'encoder.', 'direct' if 'swinViT.*'."""
    keys = list(state_dict.keys())
    if any(k.startswith("encoder.") for k in keys):
        return "ssl"
    if any(k.startswith("swinViT.") for k in keys):
        return "direct"
    return "unknown"


def load_pretrained_encoder(
    model: nn.Module,
    ckpt_path: str,
    strict: bool = False,
    verbose: bool = True,
) -> Tuple[int, int, int]:
    """
    Load encoder weights from a pretrained checkpoint into model.

    The function:
      1. Loads the checkpoint, extracts the state dict.
      2. Detects format (SSL vs BTCV/MONAI) and remaps keys if needed.
      3. Filters to encoder-only keys (swinViT.*).
      4. Loads into the model with strict=False (non-encoder keys are ignored).

    Returns:
        (loaded_count, missing_count, unexpected_count)
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    raw_state = _extract_state_dict(ckpt)
    fmt = _detect_format(raw_state)

    if fmt == "ssl":
        encoder_state = _remap_ssl_keys(raw_state)
        if verbose:
            print(f"[pretrained_encoder] Detected SSL format. Remapped {len(encoder_state)} encoder keys.")
    elif fmt == "direct":
        encoder_state = {k: v for k, v in raw_state.items() if k.startswith("swinViT.")}
        if verbose:
            print(f"[pretrained_encoder] Detected BTCV/MONAI format. Found {len(encoder_state)} swinViT.* keys.")
    else:
        raise ValueError(
            f"Cannot determine checkpoint format. Expected keys starting with 'encoder.' (SSL) "
            f"or 'swinViT.' (BTCV/MONAI). First 5 keys: {list(raw_state.keys())[:5]}"
        )

    # Filter to only encoder prefixes
    filtered = {
        k: v for k, v in encoder_state.items()
        if any(k.startswith(p) for p in _ENCODER_PREFIXES)
    }
    if verbose:
        print(f"[pretrained_encoder] Filtered to {len(filtered)} encoder-prefix keys.")

    # Detect key prefix: model state dict may have backbone.swinViT.* not swinViT.*
    model_swin_keys = [k for k in model.state_dict().keys() if "swinViT." in k]
    key_prefix = ""
    if model_swin_keys:
        # Extract prefix from first swinViT key, e.g. "backbone.swinViT.xxx" → prefix "backbone."
        first = model_swin_keys[0]
        idx = first.index("swinViT.")
        key_prefix = first[:idx]  # e.g. "backbone." or ""
        if verbose and key_prefix:
            print(f"[pretrained_encoder] Detected model key prefix: '{key_prefix}'")

    if key_prefix:
        filtered = {key_prefix + k: v for k, v in filtered.items()}

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    # 'missing' here = model keys not in filtered (expected: all non-swinViT keys)
    # 'unexpected' should be empty since we filtered
    encoder_missing = [k for k in missing if any(k.startswith(p) for k in _ENCODER_PREFIXES
                                                  for p in _ENCODER_PREFIXES
                                                  if k.startswith(p))]
    loaded_count = len(filtered) - len(unexpected)
    if verbose:
        print(f"[pretrained_encoder] Loaded: {loaded_count}, Unexpected: {len(unexpected)}, "
              f"Total model missing (non-encoder expected): {len(missing)}")

    return loaded_count, len(missing), len(unexpected)


def freeze_encoder(model: nn.Module, freeze_level: str) -> None:
    """
    Freeze SwinUNETR encoder stages according to freeze_level.

    freeze_level values:
      'all'    — freeze all 4 stages + patch_embed + norm (train decoder only)
      'stage4' — freeze stages 1-3 + patch_embed, thaw stage 4
      'stage34'— freeze stages 1-2 + patch_embed, thaw stages 3-4
      'none'   — no freezing (full fine-tune)

    For MulModSeg, the swinViT is accessed via model.backbone.swinViT
    or model.swinViT depending on backbone type.
    """
    # Locate the swinViT sub-module
    swin = None
    for attr in ("backbone", ""):
        candidate = getattr(model, attr, None) if attr else model
        if candidate is not None and hasattr(candidate, "swinViT"):
            swin = candidate.swinViT
            break
    if swin is None:
        print(f"[pretrained_encoder] WARNING: swinViT not found in model — skip freezing.")
        return

    if freeze_level == "none":
        return

    # Always freeze patch_embed for all non-'none' levels
    _set_requires_grad(swin.patch_embed, requires_grad=False)

    if freeze_level == "all":
        for stage in [swin.layers1, swin.layers2, swin.layers3, swin.layers4]:
            _set_requires_grad(stage, requires_grad=False)
        if hasattr(swin, "norm"):
            _set_requires_grad(swin.norm, requires_grad=False)
    elif freeze_level == "stage4":
        for stage in [swin.layers1, swin.layers2, swin.layers3]:
            _set_requires_grad(stage, requires_grad=False)
        if hasattr(swin, "norm"):
            _set_requires_grad(swin.norm, requires_grad=False)
    elif freeze_level == "stage34":
        for stage in [swin.layers1, swin.layers2]:
            _set_requires_grad(stage, requires_grad=False)

    frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    total = sum(1 for p in model.parameters())
    print(f"[pretrained_encoder] freeze_level='{freeze_level}': {frozen}/{total} param tensors frozen.")


def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(requires_grad)
