#!/usr/bin/env python
"""
单张 CT 或 MR 图像肿瘤分割推理：读入一个 NIfTI 文件，用训练好的模型预测肿瘤分割并保存为 NIfTI。

用法（在项目根目录 g:\\segmentation-test-main 下）：
  python scripts/segment_single_image.py ^
    --image path/to/your/ct_or_mr.nii.gz ^
    --checkpoint MulModSeg_2024/outputs/your_exp/best_model.pt ^
    --modality CT ^
    --output path/to/output_seg.nii.gz

  # MR 图像
  python scripts/segment_single_image.py ^
    --image path/to/mr.nii.gz ^
    --checkpoint MulModSeg_2024/outputs/your_exp/best_model.pt ^
    --modality MR ^
    --output path/to/mr_seg.nii.gz

预处理与训练时验证集一致（spacing 1.5×1.5×2，RAS，CT 窗宽窗位 / MR 归一化，前景裁剪+padding）。
"""

# 避免 libgomp: Invalid value for environment variable OMP_NUM_THREADS
import os
if not os.environ.get("OMP_NUM_THREADS", "").strip().isdigit():
    os.environ["OMP_NUM_THREADS"] = "1"

import sys
import argparse
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MULMOD_ROOT = Path(__file__).resolve().parent.parent / "MulModSeg_2024"
if str(MULMOD_ROOT) not in sys.path:
    sys.path.insert(0, str(MULMOD_ROOT))


def get_preprocess_transform(modality: str, roi_size: tuple, with_label: bool = False):
    """与 bone_tumor 验证集一致的预处理。with_label=True 时同时加载并重采样 label，与 image 同空间。"""
    from monai.transforms import (
        Compose,
        LoadImaged,
        Spacingd,
        Orientationd,
        ScaleIntensityRanged,
        NormalizeIntensityd,
        CropForegroundd,
        SpatialPadd,
        ToTensord,
    )
    keys = ["image", "label"] if with_label else ["image"]
    if modality.upper() == "CT":
        load = [LoadImaged(keys=keys, ensure_channel_first=True, image_only=True)]
        spacing = [Spacingd(keys=keys, pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest") if with_label else "bilinear")]
        orient = [Orientationd(keys=keys, axcodes="RAS")]
        intensity = [
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-125,
                a_max=275,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
        ]
        crop = [CropForegroundd(keys=keys, source_key="image")]
        pad = [SpatialPadd(keys=keys, spatial_size=roi_size, mode="constant")]
        to_t = [ToTensord(keys=keys)]
        t = Compose(load + spacing + orient + intensity + crop + pad + to_t)
    else:
        load = [LoadImaged(keys=keys, ensure_channel_first=True, image_only=True)]
        spacing = [Spacingd(keys=keys, pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest") if with_label else "bilinear")]
        orient = [Orientationd(keys=keys, axcodes="RAS")]
        intensity = [NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True)]
        crop = [CropForegroundd(keys=keys, source_key="image")]
        pad = [SpatialPadd(keys=keys, spatial_size=roi_size, mode="constant")]
        to_t = [ToTensord(keys=keys)]
        t = Compose(load + spacing + orient + intensity + crop + pad + to_t)
    return t


def _is_mulmodseg_checkpoint(state):
    """根据 state_dict 的 key 判断是 MulModSeg 还是纯 UNet/SwinUNETR。"""
    keys = list(state.keys())
    if not keys:
        return False
    # MulModSeg 的权重有 backbone. 前缀或 organ_embedding
    if "organ_embedding" in state:
        return True
    if any(k.startswith("backbone.") for k in keys):
        return True
    # 纯 UNet：down_tr64.ops、out_tr.final_conv 等，无 backbone 前缀
    return False


def build_model(args, use_text_embedding=None):
    """use_text_embedding=None 时用 args.with_text_embedding。"""
    from model.MulModSeg import MulModSeg, UNet3D_cy, SwinUNETR_cy
    use_txt = (use_text_embedding if use_text_embedding is not None else args.with_text_embedding) == 1
    if use_txt:
        use_cross_attention = getattr(args, "use_cross_attention", False)
        model = MulModSeg(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=1,
            out_channels=args.num_class,
            backbone=args.backbone,
            encoding=args.trans_encoding,
            use_cross_attention=use_cross_attention,
            cross_attn_heads=getattr(args, "cross_attn_heads", 8),
        )
    else:
        if args.backbone == "unet":
            model = UNet3D_cy(out_channels=args.num_class)
        else:
            model = SwinUNETR_cy(
                img_size=(args.roi_x, args.roi_y, args.roi_z),
                out_channels=args.num_class,
            )
    return model


def load_model_from_checkpoint(args, path, device):
    """
    根据 checkpoint 里 state_dict 的 key 自动判断是 MulModSeg 还是纯 UNet，
    构建对应模型并加载权重。返回 (model, use_text_embedding_for_inference)。
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("net", ckpt)
    if not isinstance(state, dict):
        state = ckpt  # 有的保存格式直接是 state_dict

    is_mulmod = _is_mulmodseg_checkpoint(state)
    model = build_model(args, use_text_embedding=1 if is_mulmod else 0)
    model.load_state_dict(state, strict=True)
    return model, is_mulmod


def compute_dice(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-5) -> float:
    """二值分割 Dice（前景类），pred/gt 为 0/1 或任意非负，>0 视为前景。"""
    p = (pred > 0).astype(np.float32).ravel()
    g = (gt > 0).astype(np.float32).ravel()
    intersection = (p * g).sum()
    return float((2.0 * intersection + smooth) / (p.sum() + g.sum() + smooth))


def save_visualization(
    img_np: np.ndarray,
    pred_np: np.ndarray,
    save_path: str,
    num_slices: int = 5,
    overlay_alpha: float = 0.35,
    label_np: np.ndarray = None,
):
    """
    生成并保存 3 个方向的切片叠加图（轴向 / 矢状 / 冠状）。
    img_np: [D, H, W] 预处理后的图像；pred_np: [D, H, W] 预测分割。
    label_np: 可选 [D, H, W] 真值（与 img 同空间）；若有则三张图各多一行「图像+真值」并标出 Dice。
    """
    D, H, W = img_np.shape
    img = img_np.astype(np.float32)
    if img.max() > img.min():
        img = (img - img.min()) / (img.max() - img.min())
    else:
        img = np.zeros_like(img)
    pred_mask = (pred_np > 0).astype(np.float32)
    has_gt = label_np is not None and label_np.size > 0
    dice_score = None
    if has_gt:
        gt_mask = (label_np.astype(np.float32) > 0).astype(np.float32)
        if gt_mask.shape != img.shape:
            has_gt = False
        else:
            dice_score = compute_dice(pred_mask, gt_mask)

    def _slice_indices(n, num):
        if num >= n:
            return list(range(n))
        step = max(1, (n - 1) // max(1, num - 1))
        return list(range(0, n, step))[:num]

    iz = _slice_indices(D, num_slices)
    iy = _slice_indices(H, num_slices)
    ix = _slice_indices(W, num_slices)
    base = save_path.rsplit(".", 1)[0] if "." in save_path else save_path

    def _overlay_2d(im_slice, m_slice, color=(1, 0, 0)):
        disp = im_slice.T
        m = m_slice.T.astype(np.float32)
        rgb = np.zeros((*disp.shape, 3))
        for ch, val in enumerate(color):
            rgb[..., ch] = val * m
        return disp, rgb, m * overlay_alpha

    n_cols = min(num_slices, len(iz))
    n_rows = 3 if has_gt else 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    for c, i in enumerate(iz):
        disp, rgb_pred, alpha_p = _overlay_2d(img[i], pred_mask[i], (1, 0, 0))
        axes[0, c].imshow(disp, cmap="gray", origin="lower")
        axes[0, c].set_title(f"Axial {i+1}/{D}")
        axes[0, c].axis("off")
        axes[1, c].imshow(disp, cmap="gray", origin="lower")
        axes[1, c].imshow(rgb_pred, alpha=alpha_p, origin="lower")
        axes[1, c].set_title(f"Pred {i+1}/{D}")
        axes[1, c].axis("off")
        if has_gt:
            _, rgb_gt, alpha_gt = _overlay_2d(img[i], gt_mask[i], (0, 1, 0))
            axes[2, c].imshow(disp, cmap="gray", origin="lower")
            axes[2, c].imshow(rgb_gt, alpha=alpha_gt, origin="lower")
            axes[2, c].set_title(f"GT {i+1}/{D}")
            axes[2, c].axis("off")
    title_axial = "Axial: image | pred(red) | GT(green)" if has_gt else "Axial: image | pred(red)"
    if has_gt and dice_score is not None:
        title_axial += f"  Dice={dice_score:.4f}"
    plt.suptitle(title_axial)
    plt.tight_layout()
    plt.savefig(f"{base}_axial.png", dpi=120, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    for c, j in enumerate(ix):
        disp, rgb_pred, alpha_p = _overlay_2d(img[:, :, j], pred_mask[:, :, j], (1, 0, 0))
        axes[0, c].imshow(disp, cmap="gray", origin="lower")
        axes[0, c].set_title(f"Sagittal {j+1}/{W}")
        axes[0, c].axis("off")
        axes[1, c].imshow(disp, cmap="gray", origin="lower")
        axes[1, c].imshow(rgb_pred, alpha=alpha_p, origin="lower")
        axes[1, c].set_title(f"Pred {j+1}/{W}")
        axes[1, c].axis("off")
        if has_gt:
            _, rgb_gt, alpha_gt = _overlay_2d(img[:, :, j], gt_mask[:, :, j], (0, 1, 0))
            axes[2, c].imshow(disp, cmap="gray", origin="lower")
            axes[2, c].imshow(rgb_gt, alpha=alpha_gt, origin="lower")
            axes[2, c].set_title(f"GT {j+1}/{W}")
            axes[2, c].axis("off")
    title_sag = "Sagittal: image | pred(red) | GT(green)" if has_gt else "Sagittal: image | pred(red)"
    if has_gt and dice_score is not None:
        title_sag += f"  Dice={dice_score:.4f}"
    plt.suptitle(title_sag)
    plt.tight_layout()
    plt.savefig(f"{base}_sagittal.png", dpi=120, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    for c, k in enumerate(iy):
        disp, rgb_pred, alpha_p = _overlay_2d(img[:, k, :], pred_mask[:, k, :], (1, 0, 0))
        axes[0, c].imshow(disp, cmap="gray", origin="lower")
        axes[0, c].set_title(f"Coronal {k+1}/{H}")
        axes[0, c].axis("off")
        axes[1, c].imshow(disp, cmap="gray", origin="lower")
        axes[1, c].imshow(rgb_pred, alpha=alpha_p, origin="lower")
        axes[1, c].set_title(f"Pred {k+1}/{H}")
        axes[1, c].axis("off")
        if has_gt:
            _, rgb_gt, alpha_gt = _overlay_2d(img[:, k, :], gt_mask[:, k, :], (0, 1, 0))
            axes[2, c].imshow(disp, cmap="gray", origin="lower")
            axes[2, c].imshow(rgb_gt, alpha=alpha_gt, origin="lower")
            axes[2, c].set_title(f"GT {k+1}/{H}")
            axes[2, c].axis("off")
    title_cor = "Coronal: image | pred(red) | GT(green)" if has_gt else "Coronal: image | pred(red)"
    if has_gt and dice_score is not None:
        title_cor += f"  Dice={dice_score:.4f}"
    plt.suptitle(title_cor)
    plt.tight_layout()
    plt.savefig(f"{base}_coronal.png", dpi=120, bbox_inches="tight")
    plt.close()

    if has_gt and dice_score is not None:
        print(f"Dice (pred vs GT) = {dice_score:.4f}")
    print(f"可视化已保存: {base}_axial.png, {base}_sagittal.png, {base}_coronal.png" + (" (含真值对比)" if has_gt else ""))


def parse_args():
    p = argparse.ArgumentParser(description="单张 CT/MR 图像肿瘤分割推理")
    p.add_argument("--image", type=str, required=True, help="输入 CT 或 MR 的 NIfTI 路径（.nii 或 .nii.gz）")
    p.add_argument("--checkpoint", type=str, required=True, help="训练好的权重，如 best_model.pt 或 epoch_100.pt")
    p.add_argument("--output", type=str, default=None, help="输出分割 NIfTI 路径；默认在输入同目录下加 _seg.nii.gz")
    p.add_argument("--modality", type=str, required=True, choices=["CT", "MR"], help="输入图像模态：CT 或 MR")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--backbone", type=str, default="unet", choices=["unet", "swinunetr"])
    p.add_argument("--num_class", type=int, default=2)
    p.add_argument("--roi_x", type=int, default=96)
    p.add_argument("--roi_y", type=int, default=96)
    p.add_argument("--roi_z", type=int, default=96)
    p.add_argument("--with_text_embedding", type=int, default=1, choices=[0, 1])
    p.add_argument("--use_cross_attention", action="store_true")
    p.add_argument("--cross_attn_heads", type=int, default=8)
    p.add_argument("--trans_encoding", type=str, default="word_embedding", choices=["rand_embedding", "word_embedding"])
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--sw_batch_size", type=int, default=1)
    # 可视化
    p.add_argument("--save_viz", action="store_true", default=True, help="保存切片叠加可视化 PNG")
    p.add_argument("--no_save_viz", action="store_false", dest="save_viz", help="不保存可视化")
    p.add_argument("--viz_dir", type=str, default=None, help="可视化 PNG 保存目录，默认与 --output 同目录")
    p.add_argument("--viz_slices", type=int, default=5, help="每个方向显示的切片数量（取中心附近）")
    p.add_argument("--label", type=str, default=None, help="真值 NIfTI 路径；不传则按约定从 --image 推断：同目录下 登记号.nii.gz（如 10947254_ct_reg.nii.gz -> 10947254.nii.gz）")
    return p.parse_args()


def infer_label_path(image_path: str) -> Optional[str]:
    """
    根据图像路径推断真值路径：同目录下，去掉 _ct_reg / _mr 等后缀得到 登记号.nii.gz。
    例如 .../10947254/10947254_ct_reg.nii.gz -> .../10947254/10947254.nii.gz
    """
    p = Path(image_path)
    stem = p.stem
    if stem.endswith(".nii"):
        stem = Path(stem).stem
    for suffix in ("_ct_reg", "_mr", "_ct", "_MR", "_CT"):
        if stem.lower().endswith(suffix.lower()):
            stem = stem[: -len(suffix)]
            break
    candidate = p.parent / f"{stem}.nii.gz"
    if candidate.exists():
        return str(candidate)
    candidate2 = p.parent / f"{stem}.nii"
    if candidate2.exists():
        return str(candidate2)
    return None


def main():
    args = parse_args()
    args.image = str(Path(args.image).resolve())
    if args.output is None:
        base = Path(args.image).stem
        if base.endswith(".nii"):
            base = base[:-4]
        args.output = str(Path(args.image).parent / f"{base}_seg.nii.gz")
    else:
        args.output = str(Path(args.output).resolve())

    os.chdir(MULMOD_ROOT)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    roi_size = (args.roi_x, args.roi_y, args.roi_z)

    # 真值路径：显式 --label 或按约定从图像路径推断（同目录 登记号.nii.gz）
    label_path = args.label
    if not label_path:
        label_path = infer_label_path(args.image)
    if label_path and not os.path.exists(label_path):
        label_path = None
    use_label = bool(label_path)

    # 预处理：有真值时 image+label 一起做空间变换，保证与预测同空间
    transform = get_preprocess_transform(args.modality, roi_size, with_label=use_label)
    if use_label:
        data = transform({"image": args.image, "label": label_path})
        x = data["image"].unsqueeze(0).float().to(device)
        label_np = data["label"].numpy()
        if label_np.ndim == 4:
            label_np = label_np[0]
        label_np = label_np.astype(np.uint8)
        print(f"已加载真值: {label_path}")
    else:
        data = transform({"image": args.image})
        x = data["image"].unsqueeze(0).float().to(device)
        label_np = None

    # 根据 checkpoint 自动识别模型类型并加载（支持 MulModSeg 或纯 UNet3D）
    model, use_text_embedding = load_model_from_checkpoint(args, args.checkpoint, device)
    model = model.to(device)
    model.eval()

    from monai.inferers import sliding_window_inference

    with torch.no_grad():
        if use_text_embedding:
            # 传入当前模态，模型内部会选对应 organ_embedding
            logit_map = sliding_window_inference(
                x, roi_size, args.sw_batch_size,
                lambda inp: model(inp, args.modality),
                overlap=args.overlap,
            )
        else:
            logit_map = sliding_window_inference(
                x, roi_size, args.sw_batch_size, model, overlap=args.overlap,
            )

    pred = torch.argmax(logit_map, dim=1, keepdim=True)
    pred_np = pred[0, 0].cpu().numpy().astype(np.uint8)
    img_np = x[0, 0].cpu().numpy()  # [D,H,W] 与 pred_np 同空间

    # 保存：与预处理后的图像同空间（1.5×1.5×2 mm, RAS）
    out_nii = nib.Nifti1Image(pred_np, np.eye(4))
    nib.save(out_nii, args.output)
    print(f"分割已保存: {args.output}")

    # 可视化：三个方向切片 + 分割叠加
    if args.save_viz:
        if args.viz_dir:
            os.makedirs(args.viz_dir, exist_ok=True)
            base = Path(args.output).stem
            if base.endswith(".nii"):
                base = base[:-4]
            viz_path = str(Path(args.viz_dir) / base)
        else:
            viz_path = str(Path(args.output).with_suffix(""))
        save_visualization(
            img_np, pred_np, viz_path,
            num_slices=args.viz_slices,
            label_np=label_np,
        )


if __name__ == "__main__":
    main()
