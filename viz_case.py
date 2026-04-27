#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, CheckButtons

CASE_ID_PATTERN = re.compile(r"^(?:N)?\d+$", re.IGNORECASE)


def is_nii(p: Path) -> bool:
    name = p.name.lower()
    return p.is_file() and (name.endswith(".nii") or name.endswith(".nii.gz"))


def pick_best(cands: List[Path], prefer_keywords: List[str]) -> Optional[Path]:
    if not cands:
        return None

    def score(p: Path) -> Tuple[int, int]:
        name = p.name.lower()
        hits = sum(1 for k in prefer_keywords if k in name)
        # hits 越多越好；名字越短越好（更“干净”）
        return (hits, -len(name))

    return sorted(cands, key=score, reverse=True)[0]


def infer_case_id_from_dir(case_dir: Path) -> Optional[str]:
    # 目录名若是 11687281 或 N12896530
    name = case_dir.name.strip()
    if CASE_ID_PATTERN.fullmatch(name):
        return name
    # 目录名不规范时，尝试从目录名里提取前缀
    m = re.match(r"^(N?\d+)", name, flags=re.IGNORECASE)
    if m and CASE_ID_PATTERN.fullmatch(m.group(1)):
        return m.group(1)
    return None


def auto_pick_from_case_dir(case_dir: Path, case_id: Optional[str] = None) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    files = [p for p in case_dir.iterdir() if is_nii(p)]
    if not files:
        return None, None, None

    cid = (case_id or infer_case_id_from_dir(case_dir) or "").lower()

    # 如果 case_id 能推断出来，优先用它筛一遍（避免把别的文件误选进来）
    related = files
    if cid:
        rel = [p for p in files if cid in p.name.lower()]
        if len(rel) > 0:
            related = rel

    ct_cands = [p for p in related if "ct" in p.name.lower()]
    mr_cands = [p for p in related if "mr" in p.name.lower()]

    # seg：排除 ct/mr
    seg_cands = [p for p in related if ("ct" not in p.name.lower() and "mr" not in p.name.lower())]

    ct = pick_best(ct_cands, prefer_keywords=["ct_reg", "ctreg", "reg", "_ct_"])
    mr = pick_best(mr_cands, prefer_keywords=["_mr_", "mr"])

    # seg 常见：包含 seg/mask/label/gt，或刚好是 {id}.nii(.gz)
    prefer = ["seg", "mask", "label", "gt"]
    if cid:
        prefer += [f"{cid}.nii", f"{cid}.nii.gz", f"{cid}_", f"{cid}-"]
    seg = pick_best(seg_cands, prefer_keywords=prefer)

    return ct, mr, seg


def robust_percentile_window(x: np.ndarray, p_low=1.0, p_high=99.0) -> Tuple[float, float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(x, [p_low, p_high])
    if lo == hi:
        lo2, hi2 = float(np.min(x)), float(np.max(x))
        if hi2 <= lo2:
            hi2 = lo2 + 1.0
        return lo2, hi2
    return float(lo), float(hi)


def affine_max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def load_nii(path: Path) -> nib.Nifti1Image:
    return nib.load(str(path))


def get_spacing(img: nib.Nifti1Image) -> Tuple[float, float, float]:
    zooms = img.header.get_zooms()
    zooms = zooms[:3] if len(zooms) >= 3 else tuple(list(zooms) + [1.0] * (3 - len(zooms)))
    return tuple(float(z) for z in zooms)


def axcodes(img: nib.Nifti1Image) -> Tuple[str, str, str]:
    try:
        from nibabel.orientations import aff2axcodes
        return aff2axcodes(img.affine)
    except Exception:
        return ("?", "?", "?")


def safe_get_data(img: nib.Nifti1Image, max_bytes: int = 2_000_000_000) -> np.ndarray:
    shape = img.shape[:3]
    dtype = np.dtype(img.get_data_dtype())
    est = int(np.prod(shape)) * dtype.itemsize
    if est > max_bytes:
        raise MemoryError(f"Data too large (~{est/1e9:.2f} GB). Consider downsampling first.")
    return np.asanyarray(img.dataobj)


def maybe_resample_to_ct(
    ct_img: nib.Nifti1Image,
    src_img: nib.Nifti1Image,
    is_label: bool = False
) -> nib.Nifti1Image:
    """
    把 src_img 重采样到 ct_img 的空间（shape + affine）。
    注意：这通常需要 SciPy（nibabel.processing 内部常依赖）。
    """
    try:
        from nibabel.processing import resample_from_to
    except Exception as e:
        raise RuntimeError("resample 模式需要 nibabel.processing（通常依赖 SciPy）。") from e

    order = 0 if is_label else 1
    return resample_from_to(src_img, (ct_img.shape, ct_img.affine), order=order)


def extract_slice(vol: np.ndarray, axis: int, idx: int) -> np.ndarray:
    if axis == 0:      # sagittal: x fixed -> (y,z)
        sl = vol[idx, :, :]
        return np.flipud(sl.T)
    elif axis == 1:    # coronal: y fixed -> (x,z)
        sl = vol[:, idx, :]
        return np.flipud(sl.T)
    else:              # axial: z fixed -> (x,y)
        sl = vol[:, :, idx]
        return np.flipud(sl.T)


def make_seg_overlay(seg2d: np.ndarray, label_value: Optional[int], alpha: float) -> np.ndarray:
    if seg2d is None:
        return None
    if label_value is None:
        mask = seg2d != 0
    else:
        mask = seg2d == label_value
    overlay = np.zeros((seg2d.shape[0], seg2d.shape[1], 4), dtype=np.float32)
    # 红色 overlay
    overlay[..., 0] = 1.0
    overlay[..., 3] = alpha * mask.astype(np.float32)
    return overlay


def to_3d(img: nib.Nifti1Image) -> np.ndarray:
    data = safe_get_data(img)
    if data.ndim == 3:
        return np.asarray(data)
    if data.ndim > 3:
        # 默认取第一个 volume
        return np.asarray(data[..., 0])
    raise ValueError("Volume is not 3D/4D.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case_dir", type=str, default=None, help="样本文件夹路径（自动识别 CT/MR/SEG）")
    ap.add_argument("--case_id", type=str, default=None, help="可选：指定 case_id（用于更准地筛选文件）")

    ap.add_argument("--ct", type=str, default=None, help="手动指定 CT .nii/.nii.gz（会覆盖自动识别）")
    ap.add_argument("--mr", type=str, default=None, help="手动指定 MR .nii/.nii.gz（会覆盖自动识别）")
    ap.add_argument("--seg", type=str, default=None, help="手动指定 SEG/mask .nii/.nii.gz（可选）")

    ap.add_argument("--mode", type=str, default="native", choices=["native", "resample"],
                    help="native: 不重采样直接叠加；resample: MR/SEG 重采样到 CT 空间后叠加（更严格）")
    ap.add_argument("--alpha_mr", type=float, default=0.35, help="MR overlay alpha")
    ap.add_argument("--alpha_seg", type=float, default=0.35, help="Seg overlay alpha")
    ap.add_argument("--label", type=int, default=None, help="只显示某个 label 值（默认 None=显示所有非零）")

    ap.add_argument("--p_low", type=float, default=1.0, help="window low percentile")
    ap.add_argument("--p_high", type=float, default=99.0, help="window high percentile")
    args = ap.parse_args()

    ct_path = Path(args.ct) if args.ct else None
    mr_path = Path(args.mr) if args.mr else None
    seg_path = Path(args.seg) if args.seg else None

    # 自动从 case_dir 选文件（如果没手动指定）
    if (ct_path is None or mr_path is None) and args.case_dir:
        case_dir = Path(args.case_dir)
        if not case_dir.exists() or not case_dir.is_dir():
            raise FileNotFoundError(f"case_dir 不存在或不是文件夹：{case_dir}")

        auto_ct, auto_mr, auto_seg = auto_pick_from_case_dir(case_dir, case_id=args.case_id)

        if ct_path is None:
            ct_path = auto_ct
        if mr_path is None:
            mr_path = auto_mr
        if seg_path is None and args.seg is None:
            seg_path = auto_seg

    if ct_path is None or mr_path is None:
        raise SystemExit("必须提供 CT 和 MR。你可以用 --case_dir 自动识别，或手动用 --ct --mr。")

    # 打印选择结果
    print("=== SELECTED FILES ===")
    print(f"CT : {ct_path}")
    print(f"MR : {mr_path}")
    print(f"SEG: {seg_path if seg_path else 'None'}")
    print(f"MODE: {args.mode}")

    ct_img = load_nii(ct_path)
    mr_img = load_nii(mr_path)
    seg_img = load_nii(seg_path) if seg_path else None

    # 基本信息打印
    print("\n=== BASIC INFO ===")
    print(f"CT shape   = {ct_img.shape}  spacing={get_spacing(ct_img)}  axcodes={axcodes(ct_img)}")
    print(f"MR shape   = {mr_img.shape}  spacing={get_spacing(mr_img)}  axcodes={axcodes(mr_img)}")
    print(f"CT vs MR shape same       = {ct_img.shape == mr_img.shape}")
    print(f"CT vs MR affine max|diff| = {affine_max_abs_diff(ct_img.affine, mr_img.affine):.6g}")

    if seg_img is not None:
        print(f"SEG shape  = {seg_img.shape} spacing={get_spacing(seg_img)} axcodes={axcodes(seg_img)}")
        print(f"SEG vs CT shape same       = {seg_img.shape == ct_img.shape}")
        print(f"SEG vs CT affine max|diff| = {affine_max_abs_diff(seg_img.affine, ct_img.affine):.6g}")

    # 读取数据
    ct = to_3d(ct_img)
    mr = to_3d(mr_img)
    seg = to_3d(seg_img) if seg_img is not None else None

    # 可选重采样到 CT 空间
    mode_used = args.mode
    if args.mode == "resample":
        try:
            mr_r = maybe_resample_to_ct(ct_img, mr_img, is_label=False)
            mr = to_3d(mr_r)
            if seg_img is not None:
                seg_r = maybe_resample_to_ct(ct_img, seg_img, is_label=True)
                seg = to_3d(seg_r)
            print("\n[RESAMPLE] MR/SEG 已重采样到 CT 空间用于叠加检查。")
        except Exception as e:
            print("\n[RESAMPLE] 失败，已回退到 native。原因：", repr(e))
            mode_used = "native"

    if ct.ndim != 3 or mr.ndim != 3:
        raise ValueError("CT/MR must be 3D volumes after loading/resampling.")
    if seg is not None and seg.ndim != 3:
        raise ValueError("SEG must be 3D volume after loading/resampling.")

    # intensity window
    ct_lo, ct_hi = robust_percentile_window(ct, args.p_low, args.p_high)
    mr_lo, mr_hi = robust_percentile_window(mr, args.p_low, args.p_high)

    # 初始切片索引
    ix = ct.shape[0] // 2
    iy = ct.shape[1] // 2
    iz = ct.shape[2] // 2

    # matplotlib UI
    plt.rcParams["figure.figsize"] = (14, 6)
    fig, axes = plt.subplots(1, 3)
    plt.subplots_adjust(left=0.06, right=0.88, bottom=0.18, top=0.90, wspace=0.08)

    axis_names = ["Sagittal (X)", "Coronal (Y)", "Axial (Z)"]

    state = {
        "alpha_mr": float(args.alpha_mr),
        "alpha_seg": float(args.alpha_seg),
        "show_mr": True,
        "show_seg": True,
        "mode": mode_used,
        "label": args.label,
        "ct_lo": ct_lo, "ct_hi": ct_hi,
        "mr_lo": mr_lo, "mr_hi": mr_hi,
    }

    def redraw(ix_, iy_, iz_):
        idxs = [ix_, iy_, iz_]
        for a in range(3):
            ax = axes[a]
            ax.clear()
            ax.set_title(axis_names[a])

            ct2d = extract_slice(ct, a, idxs[a])
            mr2d = extract_slice(mr, a, idxs[a])
            seg2d = extract_slice(seg, a, idxs[a]) if seg is not None else None

            ax.imshow(ct2d, cmap="gray", vmin=state["ct_lo"], vmax=state["ct_hi"])

            if state["show_mr"]:
                ax.imshow(mr2d, cmap="gray", vmin=state["mr_lo"], vmax=state["mr_hi"], alpha=state["alpha_mr"])

            if seg2d is not None and state["show_seg"]:
                overlay = make_seg_overlay(seg2d, state["label"], state["alpha_seg"])
                ax.imshow(overlay)

            ax.axis("off")

        fig.canvas.draw_idle()

    redraw(ix, iy, iz)

    # sliders
    axcolor = "lightgoldenrodyellow"
    ax_x = plt.axes([0.10, 0.10, 0.70, 0.03], facecolor=axcolor)
    ax_y = plt.axes([0.10, 0.06, 0.70, 0.03], facecolor=axcolor)
    ax_z = plt.axes([0.10, 0.02, 0.70, 0.03], facecolor=axcolor)

    s_x = Slider(ax_x, "X", 0, ct.shape[0] - 1, valinit=ix, valstep=1)
    s_y = Slider(ax_y, "Y", 0, ct.shape[1] - 1, valinit=iy, valstep=1)
    s_z = Slider(ax_z, "Z", 0, ct.shape[2] - 1, valinit=iz, valstep=1)

    # overlay toggles
    ax_checks = plt.axes([0.89, 0.60, 0.10, 0.18], facecolor=axcolor)
    checks = CheckButtons(ax_checks, ["Show MR", "Show SEG"], [state["show_mr"], state["show_seg"]])

    # alpha sliders
    ax_amr = plt.axes([0.89, 0.48, 0.10, 0.03], facecolor=axcolor)
    ax_aseg = plt.axes([0.89, 0.42, 0.10, 0.03], facecolor=axcolor)
    s_amr = Slider(ax_amr, "MR α", 0.0, 1.0, valinit=state["alpha_mr"])
    s_aseg = Slider(ax_aseg, "SEG α", 0.0, 1.0, valinit=state["alpha_seg"])

    # mode radio (显示为主，真正 resample 已在启动时决定)
    ax_radio = plt.axes([0.89, 0.28, 0.10, 0.12], facecolor=axcolor)
    radio = RadioButtons(ax_radio, ["native", "resample"], active=0 if state["mode"] == "native" else 1)

    def on_slide(_):
        redraw(int(s_x.val), int(s_y.val), int(s_z.val))

    def on_check(label):
        if label == "Show MR":
            state["show_mr"] = not state["show_mr"]
        elif label == "Show SEG":
            state["show_seg"] = not state["show_seg"]
        redraw(int(s_x.val), int(s_y.val), int(s_z.val))

    def on_alpha(_):
        state["alpha_mr"] = float(s_amr.val)
        state["alpha_seg"] = float(s_aseg.val)
        redraw(int(s_x.val), int(s_y.val), int(s_z.val))

    def on_radio(_label):
        redraw(int(s_x.val), int(s_y.val), int(s_z.val))

    s_x.on_changed(on_slide)
    s_y.on_changed(on_slide)
    s_z.on_changed(on_slide)
    checks.on_clicked(on_check)
    s_amr.on_changed(on_alpha)
    s_aseg.on_changed(on_alpha)
    radio.on_clicked(on_radio)

    fig.suptitle(
        f"Overlay Check | mode={state['mode']} | label={state['label']} | "
        f"CT axcodes={axcodes(ct_img)} MR axcodes={axcodes(mr_img)}",
        fontsize=12
    )

    print("\n[HOW TO USE]")
    print("  - 用 X/Y/Z 滑条切片")
    print("  - 勾选 Show MR / Show SEG 控制叠加")
    print("  - 调 MR α / SEG α 看错位边缘")
    print("  - 若 mode=resample 成功：是在 CT 物理空间下叠加；否则是 voxel-space 直观叠加")

    plt.show()


if __name__ == "__main__":
    main()


'''


python viz_case.py --case_dir dataset/第3批/11519753 --mode native   / resample

dataset/第4批/11642786

dataset/第1批/11687281


python viz_case.py --case_dir dataset/第4批/11642786 --mode native   / resample
python viz_case.py --case_dir     --mode native   / resample

'''