"""
可视化 dataloader_bone_tumor 的输出，用于检查 CT、MR、label 及预处理效果。
结果保存到指定目录，适用于无图形界面的服务器环境。
--step_by_step: 逐步展示每步变换后的结果。
"""
import argparse
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from monai.transforms import (
    Compose,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    ScaleIntensityRangePercentilesd, # <-- 新增：引入百分位归一化
    Spacingd,
    NormalizeIntensityd,
    CopyItemsd,
    MaskIntensityd,
    ThresholdIntensityd,
    CropForegroundd,
    SpatialPadd,
    CenterSpatialCropd,
    RandCropByPosNegLabeld,
    RandAffined,
    RandFlipd,
    RandShiftIntensityd,
    RandScaleIntensityd,
    ToTensord,
)
from dataset.dataloader_bone_tumor import (
    get_loader_paired_bone_tumor,
    get_loader_bone_tumor,
    get_paired_data_dicts,
    ReplaceZeroEdgeBackgroundd,
)


def _safe_filename(s):
    """将名称转为安全文件名"""
    return re.sub(r"[^\w\-.]", "_", str(s))[:64]


def _to_np(x):
    """tensor/list -> numpy"""
    if hasattr(x, 'cpu'):
        return x.cpu().numpy()
    if isinstance(x, list):
        return np.stack([_to_np(t) for t in x])
    return np.asarray(x)


def _clim(v, p=(1, 99)):
    """用百分位限制显示范围，避免异常值导致全黑/全白"""
    v = np.asarray(v)
    lo, hi = np.nanpercentile(v, p)
    if hi <= lo:
        return 0, 1
    return float(lo), float(hi)


def save_slices_step(ct, mr, label, out_path, step_name="", name="", ct_normed=False):
    """保存某一步之后的切片：核心修改 -> 动态锁定肿瘤最大切面（保证同一物理深度）"""
    ct = _to_np(ct).squeeze()
    mr = _to_np(mr).squeeze()
    label = _to_np(label).squeeze()
    if ct.ndim != 3:
        print(f"[WARN] ct shape {ct.shape}, skip")
        return
        
    n_slices = ct.shape[2]
    
    # --- 核心对齐逻辑开始 ---
    if label.max() > 0:
        si = int(np.argmax(np.sum(label, axis=(0, 1))))
    else:
        si = n_slices // 2
    # --- 核心对齐逻辑结束 ---

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    slc_ct = ct[:, :, si]
    slc_mr = mr[:, :, si]
    slc_lab = label[:, :, si]
    
    ct_vmin, ct_vmax = (0, 1) if ct_normed else _clim(slc_ct, (5, 95))
    axes[0].imshow(slc_ct, cmap="gray", vmin=ct_vmin, vmax=ct_vmax)
    axes[0].set_title(f"CT (Normed={ct_normed})")
    axes[0].axis("off")
    
    # MR 显示范围自动适应，如果背景是 0 就会显示纯黑
    axes[1].imshow(slc_mr, cmap="gray")
    axes[1].set_title(f"MR (Min:{slc_mr.min():.2f}, Max:{slc_mr.max():.2f})")
    axes[1].axis("off")
    
    axes[2].imshow(slc_lab, cmap="gray", vmin=0, vmax=max(1, slc_lab.max()))
    axes[2].set_title("Label")
    axes[2].axis("off")
    
    lab2 = np.ma.masked_where(slc_lab < 0.5, slc_lab)
    axes[3].imshow(slc_ct, cmap="gray", vmin=ct_vmin, vmax=ct_vmax)
    axes[3].imshow(lab2, cmap="Reds", alpha=0.5, vmin=0, vmax=1)
    axes[3].set_title("CT+Label")
    axes[3].axis("off")
    
    title = f"{step_name} | {name} | shape={ct.shape} | Anatomical Slice: {si}"
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()


def save_slices_paired(ct, mr, label, out_path, name="", slice_indices=None):
    """保存配对样本的 CT、MR、label 若干切片（轴状面/axial，RAS 下沿第 3 维）"""
    ct = _to_np(ct).squeeze()
    mr = _to_np(mr).squeeze()
    label = _to_np(label).squeeze()
    if ct.ndim != 3:
        print(f"[WARN] ct shape {ct.shape}, skip")
        return
    D, H, W = ct.shape
    ax_dim = 2  
    n_slices = ct.shape[ax_dim]
    if slice_indices is None:
        slice_indices = [n_slices // 4, n_slices // 2, 3 * n_slices // 4]
    n = len(slice_indices)
    fig, axes = plt.subplots(3, n + 1, figsize=(4 * (n + 1), 10))
    for col, si in enumerate(slice_indices):
        si = min(max(0, si), n_slices - 1)
        slc = ct[:, :, si]
        axes[0, col].imshow(slc, cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(f"CT axial {si}")
        axes[0, col].axis("off")
        m = mr[:, :, si]
        axes[1, col].imshow(m, cmap="gray", vmin=_clim(m)[0], vmax=_clim(m)[1])
        axes[1, col].set_title(f"MR axial {si}")
        axes[1, col].axis("off")
        lab = label[:, :, si]
        axes[2, col].imshow(lab, cmap="gray", vmin=0, vmax=max(1, lab.max()))
        axes[2, col].set_title(f"Label axial {si}")
        axes[2, col].axis("off")
    slc_mid = ct[:, :, n_slices // 2]
    axes[0, n].imshow(slc_mid, cmap="gray", vmin=0, vmax=1)
    lab2 = label[:, :, n_slices // 2]
    lab2 = np.ma.masked_where(lab2 < 0.5, lab2)
    axes[0, n].imshow(lab2, cmap="Reds", alpha=0.5, vmin=0, vmax=1)
    axes[0, n].set_title("CT+Label overlay")
    axes[0, n].axis("off")
    axes[1, n].axis("off")
    axes[2, n].axis("off")
    fig.suptitle(f"{name or 'sample'} | shape={ct.shape}", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()


def save_slices_single(img, label, out_path, name="", slice_indices=None):
    """保存单模态样本的 image、label 若干切片（轴状面/axial）"""
    img = _to_np(img).squeeze()
    label = _to_np(label).squeeze()
    if img.ndim != 3:
        print(f"[WARN] img shape {img.shape}, skip")
        return
    D, H, W = img.shape
    ax_dim = 2
    n_slices = img.shape[ax_dim]
    if slice_indices is None:
        slice_indices = [n_slices // 4, n_slices // 2, 3 * n_slices // 4]
    n = len(slice_indices)
    fig, axes = plt.subplots(2, n + 1, figsize=(4 * (n + 1), 6))
    for col, si in enumerate(slice_indices):
        si = min(max(0, si), n_slices - 1)
        m = img[:, :, si]
        axes[0, col].imshow(m, cmap="gray", vmin=_clim(m)[0], vmax=_clim(m)[1])
        axes[0, col].set_title(f"Image axial {si}")
        axes[0, col].axis("off")
        lab = label[:, :, si]
        axes[1, col].imshow(lab, cmap="gray", vmin=0, vmax=max(1, lab.max()))
        axes[1, col].set_title(f"Label axial {si}")
        axes[1, col].axis("off")
    m_mid = img[:, :, n_slices // 2]
    axes[0, n].imshow(m_mid, cmap="gray", vmin=_clim(m_mid)[0], vmax=_clim(m_mid)[1])
    lab2 = np.ma.masked_where(label[:, :, n_slices // 2] < 0.5, label[:, :, n_slices // 2])
    axes[0, n].imshow(lab2, cmap="Reds", alpha=0.5, vmin=0, vmax=1)
    axes[0, n].set_title("Image+Label overlay")
    axes[0, n].axis("off")
    axes[1, n].axis("off")
    fig.suptitle(f"{name or 'sample'} | shape={img.shape}", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()


def _find_case_with_zero_edge_effect(data_list, max_scan=8, min_replaced=800):
    load = LoadImaged(keys=["ct", "mr", "label"], ensure_channel_first=True, image_only=True)
    orient = Orientationd(keys=["ct", "mr", "label"], axcodes="RAS")
    replace = ReplaceZeroEdgeBackgroundd(keys=["ct"], black_value=-2048)
    best_idx, best_count = 0, 0
    for i, s in enumerate(data_list[:max_scan]):
        try:
            d = load(dict(s))
            d = orient(d)
            ct_before = _to_np(d["ct"]).squeeze()
            d = replace(d)
            ct_after = _to_np(d["ct"]).squeeze()
            replaced = np.sum((ct_before == 0) & (np.abs(ct_after - (-2048)) < 1))
            if replaced >= min_replaced:
                return i, replaced
            if replaced > best_count:
                best_count, best_idx = replaced, i
        except Exception:
            continue
    return best_idx, best_count


def _run_step_by_step_for_sample(sample, args, out_dir, roi_size, subdir_suffix=""):
    """对单个 sample 执行 step-by-step 并保存到 subdir"""
    name = sample.get("name", "sample")
    subdir = os.path.join(out_dir, "step_by_step", _safe_filename(name) + subdir_suffix)
    os.makedirs(subdir, exist_ok=True)
    
    steps = [
        ("01_LoadImaged", LoadImaged(keys=["ct", "mr", "label"], ensure_channel_first=True, image_only=True)),
        ("02_Orientationd_RAS", Orientationd(keys=["ct", "mr", "label"], axcodes="RAS")),
        ("03_ReplaceZeroEdgeBackgroundd", ReplaceZeroEdgeBackgroundd(keys=["ct"], black_value=-2048)),
        ("04_Spacingd", Spacingd(keys=["ct", "mr", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "bilinear", "nearest"))),
        ("05_ScaleIntensityRanged_CT", ScaleIntensityRanged(keys=["ct"], a_min=-125, a_max=275, b_min=0.0, b_max=1.0, clip=True)),
        
        # <-- 修改点：用极简的百分位归一化替换掉原来的4步掩码操作 -->
        ("06_ScaleIntensityRangePercentilesd_MR", ScaleIntensityRangePercentilesd(
            keys=["mr"], lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True, channel_wise=True
        )),
        
        ("07_CropForegroundd", CropForegroundd(keys=["ct", "mr", "label"], source_key="ct")),
        ("08_SpatialPadd", SpatialPadd(keys=["ct", "mr", "label"], spatial_size=roi_size, mode="constant")),
    ]

    def _shapes(d):
        return tuple(_to_np(d["ct"]).shape), tuple(_to_np(d["mr"]).shape), tuple(_to_np(d["label"]).shape)

    d = dict(sample)
    ct_normed = False
    for i, (step_name, t) in enumerate(steps):
        d = t(d)
        sc, sm, sl = _shapes(d)
        if sc != sm or sc != sl:
            print(f"  [WARN] {step_name} 形状不一致: ct={sc} mr={sm} label={sl}")
        else:
            print(f"  {step_name}: ct={sc}")
        ct_normed = "ScaleIntensity" in step_name or ct_normed
        save_slices_step(d["ct"], d["mr"], d["label"], os.path.join(subdir, f"{i+1:02d}_{step_name}.png"), step_name, name, ct_normed=ct_normed)

    if args.phase == "train":
        # 删除了 CenterSpatialCropd，直接执行动态随机裁剪
        
        # <-- 修改点：更新 RandCropByPosNegLabeld 的参数 (pos=3) -->
        t_crop = RandCropByPosNegLabeld(
            keys=["ct", "mr", "label"], 
            label_key="label", 
            spatial_size=roi_size, 
            pos=3, 
            neg=1, 
            num_samples=1, 
            image_key="ct", 
            image_threshold=0
        )
        t_crop.set_random_state(seed=42) # 强制定死位置，方便对比
        
        out_list = t_crop(d)
        d = out_list[0] if isinstance(out_list, list) else out_list
        sc9, sm9, sl9 = _shapes(d)
        if sc9 != sl9:
            print(f"  [WARN] RandCrop后 ct={sc9} label={sl9} 不一致")
        else:
            print(f"  [09] RandCrop后 shape: ct={sc9} label={sl9}")
            
        save_slices_step(d["ct"], d["mr"], d["label"], os.path.join(subdir, "09_RandCropByPosNegLabeld.png"), "09_RandCropByPosNegLabeld", name, ct_normed=True)
        
        d = RandAffined(keys=["ct", "mr", "label"], mode=("bilinear", "bilinear", "nearest"), prob=1.0, spatial_size=roi_size, rotate_range=(0, 0, np.pi / 30), scale_range=(0.1, 0.1, 0.1))(d)
        for ax in range(3):
            d = RandFlipd(keys=["ct", "mr", "label"], prob=1.0, spatial_axis=ax)(d)
        d = RandShiftIntensityd(keys=["ct"], offsets=0.10, prob=1.0)(d)
        d = RandScaleIntensityd(keys=["mr"], factors=0.1, prob=1.0)(d)
        d = RandShiftIntensityd(keys=["mr"], offsets=0.1, prob=1.0)(d)
        save_slices_step(d["ct"], d["mr"], d["label"], os.path.join(subdir, "10_IntensityAug.png"), "10_IntensityAug", name, ct_normed=True)
    print(f"  已保存到 {subdir}")


def run_step_by_step(args, out_dir):
    roi_size = (96, 96, 96)
    data_list = get_paired_data_dicts(
        root_dir=args.root_dir,
        phase=args.phase,
        train_ratio=0.8,
        random_seed=42,
    )
    if not data_list:
        print("没有配对数据")
        return

    print("扫描 ReplaceZeroEdgeBackgroundd 效果明显的 case...")
    effect_idx, effect_count = _find_case_with_zero_edge_effect(data_list)
    sample_first = data_list[0]
    sample_effect = data_list[effect_idx]
    print(f"  data_list[0]: {sample_first.get('name')}")
    print(f"  ReplaceZeroEdgeBackgroundd 效果最佳 (idx={effect_idx}, 替换体素≈{effect_count}): {sample_effect.get('name')}")

    print("输出 1: 第一个 case")
    _run_step_by_step_for_sample(sample_first, args, out_dir, roi_size, subdir_suffix="_first")
    print("输出 2: ReplaceZeroEdgeBackgroundd 效果明显的 case")
    _run_step_by_step_for_sample(sample_effect, args, out_dir, roi_size, subdir_suffix="_zero_edge_effect")


def main():
    parser = argparse.ArgumentParser(description="可视化 bone tumor dataloader，结果保存到目录")
    parser.add_argument("--root_dir", default="./dataset", help="数据根目录")
    parser.add_argument("--step_by_step", action="store_true", help="逐步展示每步变换后的结果")
    parser.add_argument("--loader", default="paired", choices=["paired", "bone_tumor"],
                        help="paired=CT+MR 配重新找配对, bone_tumor=单模态/MIX")
    parser.add_argument("--phase", default="train", choices=["train", "val"])
    parser.add_argument("--num_batches", type=int, default=1, help="保存前几个 batch")
    parser.add_argument("--persistent", action="store_true", default=True, help="使用缓存")
    parser.add_argument("--no_persistent", dest="persistent", action="store_false", help="不使用缓存")
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)

    if args.step_by_step:
        if args.loader != "paired":
            print("[WARN] --step_by_step 仅支持 paired，已切换")
        print("Step-by-step 可视化（第一个样本）...")
        run_step_by_step(args, out_dir)
        return

    if args.loader == "paired":
        loader = get_loader_paired_bone_tumor(
            root_dir=args.root_dir,
            phase=args.phase,
            batch_size=1,
            train_num_samples=1,
            persistent=args.persistent,
            num_workers=0,
        )
    else:
        loader = get_loader_bone_tumor(
            root_dir=args.root_dir,
            modality="MIX",
            phase=args.phase,
            batch_size=1,
            train_num_samples=1,
            persistent=args.persistent,
            num_workers=0,
        )

    shown = 0
    for batch_idx, batch in enumerate(loader):
        if shown >= args.num_batches:
            break

        if args.loader == "paired":
            ct_list = batch["ct"]
            mr_list = batch["mr"]
            label_list = batch["label"]
            names = batch["name"]
            if isinstance(ct_list, list):
                for i, (ct, mr, lab, nm) in enumerate(zip(ct_list, mr_list, label_list, names)):
                    out_path = os.path.join(out_dir, f"batch{batch_idx}_sample{i}_{_safe_filename(nm)}.png")
                    save_slices_paired(ct, mr, lab, out_path, name=str(nm))
                    print(f"Saved {out_path}")
                    shown += 1
                    if shown >= args.num_batches:
                        break
            else:
                ct = _to_np(batch["ct"]).squeeze()
                if ct.ndim == 4:
                    for i in range(ct.shape[0]):
                        nm = batch["name"][i] if i < len(batch["name"]) else f"sample{i}"
                        out_path = os.path.join(out_dir, f"batch{batch_idx}_sample{i}_{_safe_filename(nm)}.png")
                        save_slices_paired(
                            batch["ct"][i : i + 1],
                            batch["mr"][i : i + 1],
                            batch["label"][i : i + 1],
                            out_path,
                            name=str(nm),
                        )
                        print(f"Saved {out_path}")
                        shown += 1
                        if shown >= args.num_batches:
                            break
                else:
                    nm = batch["name"][0] if batch["name"] else "sample"
                    out_path = os.path.join(out_dir, f"batch{batch_idx}_sample0_{_safe_filename(nm)}.png")
                    save_slices_paired(batch["ct"], batch["mr"], batch["label"], out_path, name=str(nm))
                    print(f"Saved {out_path}")
                    shown += 1
        else:
            img_list = batch["image"]
            label_list = batch["label"]
            names = batch["name"]
            if isinstance(img_list, list):
                for i, (img, lab, nm) in enumerate(zip(img_list, label_list, names)):
                    out_path = os.path.join(out_dir, f"batch{batch_idx}_sample{i}_{_safe_filename(nm)}.png")
                    save_slices_single(img, lab, out_path, name=str(nm))
                    print(f"Saved {out_path}")
                    shown += 1
                    if shown >= args.num_batches:
                        break
            else:
                nm = batch["name"][0] if batch["name"] else "sample"
                out_path = os.path.join(out_dir, f"batch{batch_idx}_sample0_{_safe_filename(nm)}.png")
                save_slices_single(batch["image"], batch["label"], out_path, name=str(nm))
                print(f"Saved {out_path}")
                shown += 1

    if shown == 0:
        print("没有可保存的数据，请检查 root_dir 和 loader 配置。")
    else:
        print(f"共保存 {shown} 张图到 {out_dir}")


if __name__ == "__main__":
    main()