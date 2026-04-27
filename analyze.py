import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import nibabel as nib
except ImportError as e:
    raise SystemExit("缺少 nibabel。请安装：pip install nibabel pandas openpyxl") from e


# 你的病例ID形态：纯数字（如 10075157）或 N+数字（如 N12896530）
CASE_ID_PATTERN = re.compile(r"^(?:N)?\d+$", re.IGNORECASE)


@dataclass
class NiftiInfo:
    path: str
    shape: Tuple[int, ...]
    spacing: Tuple[float, ...]
    dtype: str
    affine: List[List[float]]  # 4x4
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None


@dataclass
class CaseRecord:
    case_id: str
    diagnosis: Optional[str]
    site: Optional[str]
    note: Optional[str]

    batch: Optional[str] = None        # 第X批 / 上海市一
    case_dir: Optional[str] = None     # 病例文件夹路径（若存在）

    ct_path: Optional[str] = None
    mr_path: Optional[str] = None
    seg_path: Optional[str] = None

    ct_info: Optional[Dict] = None
    mr_info: Optional[Dict] = None
    seg_info: Optional[Dict] = None

    ct_mr_shape_same: Optional[bool] = None
    ct_mr_affine_max_abs_diff: Optional[float] = None
    seg_ct_shape_same: Optional[bool] = None
    seg_ct_affine_max_abs_diff: Optional[float] = None

    seg_unique_labels: Optional[str] = None
    seg_foreground_voxels: Optional[int] = None
    seg_bbox: Optional[str] = None

    issues: Optional[str] = None


def normalize_case_id(x) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    s = re.sub(r"\.0$", "", s)  # 防止 excel 读成 float
    if CASE_ID_PATTERN.fullmatch(s):
        return s
    return None


def read_registry(excel_path: Path) -> pd.DataFrame:
    df = pd.read_excel(excel_path, sheet_name=0)

    # 标准化列名
    df = df.rename(columns={c: str(c).strip() for c in df.columns})

    # 兼容：如果列名不存在就补空列
    for col in ["诊断", "部位", "登记号", "备注1"]:
        if col not in df.columns:
            df[col] = None

    df["case_id"] = df["登记号"].apply(normalize_case_id)
    df = df[df["case_id"].notna()].copy()

    # 去重：台账同一ID可能重复
    dup_counts = df["case_id"].value_counts()
    df["is_dup_in_registry"] = df["case_id"].map(lambda x: dup_counts.get(x, 0) > 1)

    df = df.drop_duplicates(subset=["case_id"], keep="first").reset_index(drop=True)
    df = df.rename(columns={"诊断": "diagnosis", "部位": "site", "备注1": "note"})
    return df[["case_id", "diagnosis", "site", "note", "is_dup_in_registry"]]


def is_nii(p: Path) -> bool:
    name = p.name.lower()
    return p.is_file() and (name.endswith(".nii") or name.endswith(".nii.gz"))


def pick_best(cands: List[Path], prefer_keywords: List[str]) -> Optional[Path]:
    if not cands:
        return None

    def score(p: Path) -> Tuple[int, int]:
        name = p.name.lower()
        hits = sum(1 for k in prefer_keywords if k in name)
        return (hits, -len(name))

    return sorted(cands, key=score, reverse=True)[0]


def match_case_files_from_list(case_id: str, files: List[Path]) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    cid = case_id.lower()
    related = [p for p in files if cid in p.name.lower()]

    ct_cands = [p for p in related if "ct" in p.name.lower()]
    mr_cands = [p for p in related if "mr" in p.name.lower()]

    # seg：排除 ct/mr
    seg_cands = [p for p in related if ("ct" not in p.name.lower() and "mr" not in p.name.lower())]

    ct = pick_best(ct_cands, prefer_keywords=["ct_reg", "ctreg", "reg", "_ct_"])
    mr = pick_best(mr_cands, prefer_keywords=["_mr_", "mr"])

    # seg 常见情况：{id}.nii(.gz) 或包含 seg/mask/label/gt
    seg = pick_best(
        seg_cands,
        prefer_keywords=["seg", "mask", "label", "gt", f"{cid}.nii", f"{cid}.nii.gz"]
    )
    return ct, mr, seg


def load_nifti_info(path: Path, with_stats: bool) -> NiftiInfo:
    img = nib.load(str(path))
    shape = tuple(int(x) for x in img.shape)
    zooms = img.header.get_zooms()
    spacing = tuple(float(z) for z in zooms[: min(3, len(zooms))])
    dtype = str(img.header.get_data_dtype())
    affine = img.affine.astype(float).tolist()

    info = NiftiInfo(
        path=str(path),
        shape=shape,
        spacing=spacing,
        dtype=dtype,
        affine=affine,
    )

    if with_stats:
        data = np.asanyarray(img.dataobj)
        finite = data[np.isfinite(data)]
        if finite.size > 0:
            info.min = float(np.min(finite))
            info.max = float(np.max(finite))
            info.mean = float(np.mean(finite))
    return info


def affine_max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def seg_stats(seg_path: Path) -> Tuple[str, int, str]:
    img = nib.load(str(seg_path))
    data = np.asanyarray(img.dataobj)

    uniq = np.unique(data)
    uniq_str = ",".join([str(int(u)) if np.issubdtype(uniq.dtype, np.integer) else str(u) for u in uniq[:50]])
    if uniq.size > 50:
        uniq_str += ",..."

    fg = int(np.sum(data != 0))

    if fg == 0:
        bbox = "EMPTY"
    else:
        coords = np.argwhere(data != 0)
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        bbox = f"min={mins.tolist()} max={maxs.tolist()} size={(maxs - mins + 1).tolist()}"

    return uniq_str, fg, bbox


def build_case_index(root: Path, shanghai_folder_name: str = "上海市一") -> Dict[str, Dict]:
    """
    返回：case_id -> {"batch": str, "case_dir": Optional[str], "files": List[Path]}
    支持两种结构：
      1) root/第X批/病例ID/*.nii(.gz)
      2) root/上海市一/*.nii(.gz)  (平铺，按文件名前缀 ID 分组)
    """
    index: Dict[str, Dict] = {}

    if not root.exists():
        raise FileNotFoundError(f"root 不存在: {root}")

    # 遍历 root 下所有一级目录
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue

        batch_name = child.name

        # 特例：上海市一 平铺文件
        if batch_name == shanghai_folder_name:
            flat_files = [p for p in child.iterdir() if is_nii(p)]
            # 通过文件名前缀提取 case_id（如 N12896530_ct_reg.nii.gz -> N12896530）
            for p in flat_files:
                m = re.match(r"^(N?\d+)", p.stem, flags=re.IGNORECASE)  # p.stem 对 .nii.gz 会变成 'xxx.nii'
                # 修正 .nii.gz 的 stem 问题：用 name 更稳
                m = re.match(r"^(N?\d+)", p.name, flags=re.IGNORECASE)
                if not m:
                    continue
                cid = m.group(1)
                if not CASE_ID_PATTERN.fullmatch(cid):
                    continue
                index.setdefault(cid, {"batch": batch_name, "case_dir": str(child), "files": []})
                index[cid]["files"].append(p)
            continue

        # 常规：第X批 下面是病例文件夹
        for case_dir in sorted(child.iterdir()):
            if not case_dir.is_dir():
                continue
            cid = case_dir.name.strip()
            if not CASE_ID_PATTERN.fullmatch(cid):
                # 如果目录名不规范，可按需放宽；目前你截图是规范的，先严格
                continue
            files = []
            for p in case_dir.iterdir():
                if is_nii(p):
                    files.append(p)
            index.setdefault(cid, {"batch": batch_name, "case_dir": str(case_dir), "files": []})
            # 如果同一 cid 多处出现：合并文件列表（后面匹配会自动挑最优）
            index[cid]["files"].extend(files)

    return index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="数据集根目录（包含 第1批.. / 上海市一 等）")
    ap.add_argument("--registry", type=str, required=True, help="信息登记表 xlsx 路径")
    ap.add_argument("--out", type=str, default="dataset_audit", help="输出目录")
    ap.add_argument("--with_stats", action="store_true", help="额外计算CT/MR min/max/mean（慢一些）")
    ap.add_argument("--with_seg_stats", action="store_true", help="额外计算mask标签/前景体素/bbox（慢一些）")
    ap.add_argument("--shanghai_name", type=str, default="上海市一", help="平铺文件夹名称（默认 上海市一）")
    args = ap.parse_args()

    root = Path(args.root)
    registry_path = Path(args.registry)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    reg = read_registry(registry_path)
    print(f"[Registry] unique cases from registry: {len(reg)}")

    case_index = build_case_index(root, shanghai_folder_name=args.shanghai_name)
    print(f"[Index] cases found in filesystem: {len(case_index)}")

    issues_counter: Dict[str, int] = {}
    records: List[CaseRecord] = []

    for _, r in reg.iterrows():
        case_id = str(r["case_id"])
        rec = CaseRecord(
            case_id=case_id,
            diagnosis=(None if pd.isna(r["diagnosis"]) else str(r["diagnosis"])),
            site=(None if pd.isna(r["site"]) else str(r["site"])),
            note=(None if pd.isna(r["note"]) else str(r["note"])),
        )

        flags = []

        if case_id not in case_index:
            flags.append("NOT_FOUND_IN_FS")
            rec.issues = "|".join(flags)
            for f in flags:
                issues_counter[f] = issues_counter.get(f, 0) + 1
            records.append(rec)
            continue

        meta = case_index[case_id]
        rec.batch = meta.get("batch")
        rec.case_dir = meta.get("case_dir")

        files = meta.get("files", [])
        ct, mr, seg = match_case_files_from_list(case_id, files)

        if ct is None:
            flags.append("MISSING_CT")
        else:
            rec.ct_path = str(ct)

        if mr is None:
            flags.append("MISSING_MR")
        else:
            rec.mr_path = str(mr)

        if seg is None:
            flags.append("MISSING_SEG")
        else:
            rec.seg_path = str(seg)

        # header & alignment
        try:
            if ct is not None:
                rec.ct_info = asdict(load_nifti_info(ct, with_stats=args.with_stats))
            if mr is not None:
                rec.mr_info = asdict(load_nifti_info(mr, with_stats=args.with_stats))
            if seg is not None:
                rec.seg_info = asdict(load_nifti_info(seg, with_stats=False))

            if ct is not None and mr is not None:
                ct_img = nib.load(str(ct))
                mr_img = nib.load(str(mr))
                rec.ct_mr_shape_same = (ct_img.shape == mr_img.shape)
                rec.ct_mr_affine_max_abs_diff = affine_max_abs_diff(ct_img.affine, mr_img.affine)
                if not rec.ct_mr_shape_same:
                    flags.append("CT_MR_SHAPE_MISMATCH")
                if rec.ct_mr_affine_max_abs_diff is not None and rec.ct_mr_affine_max_abs_diff > 1e-3:
                    flags.append("CT_MR_AFFINE_DIFF")

            if seg is not None and ct is not None:
                seg_img = nib.load(str(seg))
                ct_img = nib.load(str(ct))
                rec.seg_ct_shape_same = (seg_img.shape == ct_img.shape)
                rec.seg_ct_affine_max_abs_diff = affine_max_abs_diff(seg_img.affine, ct_img.affine)
                if not rec.seg_ct_shape_same:
                    flags.append("SEG_CT_SHAPE_MISMATCH")
                if rec.seg_ct_affine_max_abs_diff is not None and rec.seg_ct_affine_max_abs_diff > 1e-3:
                    flags.append("SEG_CT_AFFINE_DIFF")

            if args.with_seg_stats and seg is not None:
                uniq_str, fg, bbox = seg_stats(seg)
                rec.seg_unique_labels = uniq_str
                rec.seg_foreground_voxels = fg
                rec.seg_bbox = bbox
                if fg == 0:
                    flags.append("EMPTY_MASK")

        except Exception as e:
            flags.append(f"READ_ERROR:{type(e).__name__}")

        if bool(r.get("is_dup_in_registry", False)):
            flags.append("DUP_IN_REGISTRY")

        rec.issues = "|".join(flags) if flags else ""
        for f in flags:
            issues_counter[f] = issues_counter.get(f, 0) + 1

        records.append(rec)

    # 输出
    df_out = pd.DataFrame([asdict(x) for x in records])
    csv_path = out_dir / "dataset_audit.csv"
    json_path = out_dir / "dataset_audit.json"
    df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([asdict(x) for x in records], f, ensure_ascii=False, indent=2)

    print(f"\n[Output] {csv_path}")
    print(f"[Output] {json_path}")

    print("\n[Issues Summary]")
    for k, v in sorted(issues_counter.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    dangerous = df_out[df_out["issues"].astype(str).str.contains(
        "NOT_FOUND_IN_FS|MISSING_|SHAPE_MISMATCH|AFFINE_DIFF|EMPTY_MASK|READ_ERROR", regex=True, na=False
    )]
    print(f"\n[Dangerous Cases] {len(dangerous)}")
    if len(dangerous) > 0:
        cols = ["case_id", "batch", "case_dir", "ct_path", "mr_path", "seg_path", "issues"]
        cols = [c for c in cols if c in dangerous.columns]
        print(dangerous[cols].head(50).to_string(index=False))


if __name__ == "__main__":
    main()


'''

python analyze.py \
  --root dataset \
  --registry dataset/信息登记.xlsx \
  --out audit_out \
  --with_seg_stats

'''