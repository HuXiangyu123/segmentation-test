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
    from nibabel.orientations import aff2axcodes
except ImportError as e:
    raise SystemExit("缺少 nibabel。请安装：pip install nibabel pandas openpyxl numpy") from e


CASE_ID_PATTERN = re.compile(r"^(?:N)?\d+$", re.IGNORECASE)


@dataclass
class NiftiInfo:
    path: str
    shape: Tuple[int, ...]
    spacing: Tuple[float, ...]
    axcodes: Tuple[str, str, str]
    dtype: str
    affine: List[List[float]]


@dataclass
class SampleRecord:
    sample_id: str              # batch/case_id
    batch: str                  # 第1批... / 上海市一
    patient_id: str             # case_id (病人ID/登记号)
    case_dir: str               # 病例目录（上海市一是平铺目录）

    # optional registry info (aggregated)
    diagnosis: Optional[str] = None
    site: Optional[str] = None
    note: Optional[str] = None
    registry_rows: Optional[int] = None

    # file paths
    mr_path: Optional[str] = None
    seg_path: Optional[str] = None
    ct_raw_path: Optional[str] = None
    ct_reg_path: Optional[str] = None

    # chosen CT
    chosen_ct_path: Optional[str] = None
    chosen_ct_type: Optional[str] = None  # raw / reg / None
    chosen_ct_reason: Optional[str] = None

    # header info
    mr_info: Optional[Dict] = None
    seg_info: Optional[Dict] = None
    ct_raw_info: Optional[Dict] = None
    ct_reg_info: Optional[Dict] = None

    # consistency checks
    mr_seg_shape_same: Optional[bool] = None
    mr_seg_affine_max_abs_diff: Optional[float] = None

    ct_raw_vs_mr_shape_same: Optional[bool] = None
    ct_raw_vs_mr_affine_max_abs_diff: Optional[float] = None
    ct_raw_vs_seg_shape_same: Optional[bool] = None
    ct_raw_vs_seg_affine_max_abs_diff: Optional[float] = None

    ct_reg_vs_mr_shape_same: Optional[bool] = None
    ct_reg_vs_mr_affine_max_abs_diff: Optional[float] = None
    ct_reg_vs_seg_shape_same: Optional[bool] = None
    ct_reg_vs_seg_affine_max_abs_diff: Optional[float] = None

    # seg stats (optional)
    seg_unique_labels: Optional[str] = None
    seg_foreground_voxels: Optional[int] = None
    seg_bbox: Optional[str] = None

    # patient-level info
    patient_occurrences: Optional[int] = None  # same patient_id appears in how many samples

    # flags
    issues: Optional[str] = None


def is_nii(p: Path) -> bool:
    n = p.name.lower()
    return p.is_file() and (n.endswith(".nii") or n.endswith(".nii.gz"))


def affine_max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def load_nifti_info(path: Path) -> NiftiInfo:
    img = nib.load(str(path))
    shape = tuple(int(x) for x in img.shape)
    zooms = img.header.get_zooms()
    spacing = tuple(float(z) for z in zooms[: min(3, len(zooms))])
    axcodes = tuple(aff2axcodes(img.affine))
    dtype = str(img.header.get_data_dtype())
    affine = img.affine.astype(float).tolist()
    return NiftiInfo(
        path=str(path),
        shape=shape,
        spacing=spacing,
        axcodes=axcodes,
        dtype=dtype,
        affine=affine,
    )


def seg_stats(seg_path: Path) -> Tuple[str, int, str]:
    img = nib.load(str(seg_path))
    data = np.asanyarray(img.dataobj)

    uniq = np.unique(data)
    uniq_str = ",".join(
        [str(int(u)) if np.issubdtype(uniq.dtype, np.integer) else str(u) for u in uniq[:50]]
    )
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


def pick_best(cands: List[Path], prefer_keywords: List[str]) -> Optional[Path]:
    if not cands:
        return None

    def score(p: Path) -> Tuple[int, int]:
        name = p.name.lower()
        hits = sum(1 for k in prefer_keywords if k in name)
        return (hits, -len(name))

    return sorted(cands, key=score, reverse=True)[0]


def match_files_in_case(patient_id: str, files: List[Path]) -> Tuple[Optional[Path], Optional[Path], Optional[Path], Optional[Path]]:
    """
    返回：mr, seg, ct_raw, ct_reg
    - MR：含 mr
    - CT_reg：含 ct_reg 或 (ct & reg)
    - CT_raw：含 ct 但不含 reg
    - SEG：优先 seg/mask/label/gt；否则在“非ct非mr”里优先 {id}.nii(.gz)
    """
    mr_cands = [p for p in files if "mr" in p.name.lower()]

    ct_reg_cands, ct_raw_cands, seg_cands = [], [], []

    for p in files:
        n = p.name.lower()
        is_ct = "ct" in n
        is_mr = "mr" in n
        is_reg = ("reg" in n) or ("registered" in n)
        is_ct_reg = ("ct_reg" in n) or (is_ct and is_reg)

        is_seg_kw = any(k in n for k in ["seg", "mask", "label", "gt"])
        if is_seg_kw:
            seg_cands.append(p)
            continue

        if is_ct_reg:
            ct_reg_cands.append(p)
            continue

        if is_ct and (not is_reg):
            ct_raw_cands.append(p)
            continue

        if (not is_ct) and (not is_mr):
            seg_cands.append(p)

    mr = pick_best(mr_cands, prefer_keywords=["-df_mr", "_mr", "mr"])
    ct_reg = pick_best(ct_reg_cands, prefer_keywords=["ct_reg", "-df_ct_reg", "ctreg", "reg"])
    ct_raw = pick_best(ct_raw_cands, prefer_keywords=["_ct", "-df_ct", "ct"])

    # seg：优先匹配以 patient_id 开头的 nii/nii.gz（支持 11294725.nii.gz 或 11294725-xxx.nii.gz）
    pid = patient_id.lower()
    seg = None
    if seg_cands:
        exact = [p for p in seg_cands if p.name.lower().startswith(pid) and (p.name.lower().endswith(".nii") or p.name.lower().endswith(".nii.gz"))]
        seg = pick_best(exact, prefer_keywords=[f"{pid}.nii", f"{pid}.nii.gz"]) if exact else pick_best(seg_cands, prefer_keywords=["seg", "mask", "label", "gt", f"{pid}.nii", f"{pid}.nii.gz"])

    return mr, seg, ct_raw, ct_reg


def choose_ct(patient_id: str, mr: Optional[Path], seg: Optional[Path], ct_raw: Optional[Path], ct_reg: Optional[Path]) -> Tuple[Optional[Path], Optional[str], str]:
    """
    选用哪个 CT（raw 或 reg），依据它与 MR/SEG 的几何一致性。
    打分（越大越好）：
      +2: shape 与 MR 相同
      +2: shape 与 SEG 相同
      -0.1 * affine_diff(MR)
      -0.1 * affine_diff(SEG)
    """
    if ct_raw is None and ct_reg is None:
        return None, None, "no CT"

    def score(ct: Path) -> Tuple[float, str]:
        s = 0.0
        details = []
        ct_img = nib.load(str(ct))

        if mr is not None:
            mr_img = nib.load(str(mr))
            same = (ct_img.shape == mr_img.shape)
            diff = affine_max_abs_diff(ct_img.affine, mr_img.affine)
            if same:
                s += 2.0
            s -= 0.1 * diff
            details.append(f"mr_shape_same={same}, mr_affine_diff={diff:.4f}")

        if seg is not None:
            seg_img = nib.load(str(seg))
            same = (ct_img.shape == seg_img.shape)
            diff = affine_max_abs_diff(ct_img.affine, seg_img.affine)
            if same:
                s += 2.0
            s -= 0.1 * diff
            details.append(f"seg_shape_same={same}, seg_affine_diff={diff:.4f}")

        return s, "; ".join(details)

    if ct_raw is None:
        sc, det = score(ct_reg)
        return ct_reg, "reg", f"only ct_reg | score={sc:.3f} | {det}"
    if ct_reg is None:
        sc, det = score(ct_raw)
        return ct_raw, "raw", f"only ct_raw | score={sc:.3f} | {det}"

    sc_raw, det_raw = score(ct_raw)
    sc_reg, det_reg = score(ct_reg)
    if sc_reg >= sc_raw:
        return ct_reg, "reg", f"choose ct_reg | score_reg={sc_reg:.3f} >= score_raw={sc_raw:.3f} | reg: {det_reg} | raw: {det_raw}"
    else:
        return ct_raw, "raw", f"choose ct_raw | score_raw={sc_raw:.3f} > score_reg={sc_reg:.3f} | raw: {det_raw} | reg: {det_reg}"


def build_samples(root: Path, shanghai_folder_name: str = "上海市一") -> List[Dict]:
    """
    返回 samples 列表，每个元素：
      {"batch":..., "patient_id":..., "case_dir":..., "files":[...], "sample_id":...}
    样本唯一键：sample_id = batch/patient_id
    """
    samples: List[Dict] = []

    for batch_dir in sorted(root.iterdir()):
        if not batch_dir.is_dir():
            continue
        batch = batch_dir.name

        # 平铺目录：上海市一
        if batch == shanghai_folder_name:
            files = [p for p in batch_dir.iterdir() if is_nii(p)]
            groups: Dict[str, List[Path]] = {}
            for p in files:
                m = re.match(r"^(N?\d+)", p.name, flags=re.IGNORECASE)
                if not m:
                    continue
                pid = m.group(1)
                if not CASE_ID_PATTERN.fullmatch(pid):
                    continue
                groups.setdefault(pid, []).append(p)

            for pid, gfiles in groups.items():
                samples.append(
                    {
                        "batch": batch,
                        "patient_id": pid,
                        "case_dir": str(batch_dir),
                        "files": gfiles,
                        "sample_id": f"{batch}/{pid}",
                    }
                )
            continue

        # 常规：第X批/病例ID/
        for case_dir in sorted(batch_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            pid = case_dir.name.strip()
            if not CASE_ID_PATTERN.fullmatch(pid):
                continue
            files = [p for p in case_dir.iterdir() if is_nii(p)]
            samples.append(
                {
                    "batch": batch,
                    "patient_id": pid,
                    "case_dir": str(case_dir),
                    "files": files,
                    "sample_id": f"{batch}/{pid}",
                }
            )

    return samples


def normalize_case_id(x) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    s = re.sub(r"\.0$", "", s)
    if CASE_ID_PATTERN.fullmatch(s):
        return s
    return None


def load_registry_aggregated(excel_path: Path) -> Dict[str, Dict]:
    """
    case_id -> aggregated info:
      {"diagnosis": "...", "site":"...", "note":"...", "rows": n}
    如果同一 case_id 在表里多行，不去重丢掉，而是聚合。
    """
    df = pd.read_excel(excel_path, sheet_name=0)
    df = df.rename(columns={c: str(c).strip() for c in df.columns})

    for col in ["诊断", "部位", "登记号", "备注1"]:
        if col not in df.columns:
            df[col] = None

    df["case_id"] = df["登记号"].apply(normalize_case_id)
    df = df[df["case_id"].notna()].copy()

    def agg_unique(series: pd.Series) -> str:
        vals = []
        for v in series:
            if pd.isna(v):
                continue
            sv = str(v).strip()
            if sv and sv not in vals:
                vals.append(sv)
        return " ; ".join(vals) if vals else ""

    grouped = df.groupby("case_id", as_index=False).agg(
        diagnosis=("诊断", agg_unique),
        site=("部位", agg_unique),
        note=("备注1", agg_unique),
        rows=("case_id", "count"),
    )

    mp = {row["case_id"]: {"diagnosis": row["diagnosis"], "site": row["site"], "note": row["note"], "rows": int(row["rows"])}
          for _, row in grouped.iterrows()}
    return mp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="数据根目录（第1批.. / 上海市一）")
    ap.add_argument("--out", type=str, default="audit_ctreg_v2", help="输出目录")
    ap.add_argument("--registry", type=str, default="", help="信息登记表 xlsx（可选，不填则不合并）")
    ap.add_argument("--with_seg_stats", action="store_true", help="计算mask标签/前景体素/bbox（慢一些）")
    ap.add_argument("--shanghai_name", type=str, default="上海市一", help="平铺文件夹名（默认 上海市一）")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = build_samples(root, shanghai_folder_name=args.shanghai_name)
    print(f"[Scan] samples found in filesystem: {len(samples)}")

    # 统计同一 patient_id 出现次数（跨批次出现是允许的）
    patient_counts: Dict[str, int] = {}
    for s in samples:
        pid = s["patient_id"]
        patient_counts[pid] = patient_counts.get(pid, 0) + 1

    # 合并登记表（可选）
    registry_map: Dict[str, Dict] = {}
    if args.registry:
        registry_map = load_registry_aggregated(Path(args.registry))
        print(f"[Registry] unique patient_id in registry: {len(registry_map)}")

    records: List[SampleRecord] = []
    issues_counter: Dict[str, int] = {}

    # 检查同一 batch 内是否有重复 patient_id（这才是真重复）
    seen_in_batch: Dict[Tuple[str, str], int] = {}

    for s in samples:
        batch = s["batch"]
        pid = s["patient_id"]
        case_dir = s["case_dir"]
        sample_id = s["sample_id"]
        files = s["files"]

        rec = SampleRecord(sample_id=sample_id, batch=batch, patient_id=pid, case_dir=case_dir)
        rec.patient_occurrences = patient_counts.get(pid, 1)

        # 合并登记表信息（按 patient_id）
        if pid in registry_map:
            rec.diagnosis = registry_map[pid]["diagnosis"] or None
            rec.site = registry_map[pid]["site"] or None
            rec.note = registry_map[pid]["note"] or None
            rec.registry_rows = registry_map[pid]["rows"]
        else:
            rec.registry_rows = 0

        flags = []

        k = (batch, pid)
        seen_in_batch[k] = seen_in_batch.get(k, 0) + 1
        if seen_in_batch[k] > 1:
            # 同一批次同一ID出现两次才需要警惕（目录结构异常）
            flags.append("DUP_IN_SAME_BATCH")

        mr, seg, ct_raw, ct_reg = match_files_in_case(pid, files)

        if mr is None:
            flags.append("MISSING_MR")
        else:
            rec.mr_path = str(mr)

        if seg is None:
            flags.append("MISSING_SEG")
        else:
            rec.seg_path = str(seg)

        if ct_raw is not None:
            rec.ct_raw_path = str(ct_raw)
        if ct_reg is not None:
            rec.ct_reg_path = str(ct_reg)
        if ct_raw is None and ct_reg is None:
            flags.append("MISSING_CT")

        # 选用哪个 CT
        try:
            chosen_ct, chosen_type, reason = choose_ct(pid, mr, seg, ct_raw, ct_reg)
            rec.chosen_ct_path = str(chosen_ct) if chosen_ct is not None else None
            rec.chosen_ct_type = chosen_type
            rec.chosen_ct_reason = reason
        except Exception as e:
            flags.append(f"CHOOSE_CT_ERROR:{type(e).__name__}")

        # header & consistency
        try:
            if mr is not None:
                rec.mr_info = asdict(load_nifti_info(mr))
            if seg is not None:
                rec.seg_info = asdict(load_nifti_info(seg))
            if ct_raw is not None:
                rec.ct_raw_info = asdict(load_nifti_info(ct_raw))
            if ct_reg is not None:
                rec.ct_reg_info = asdict(load_nifti_info(ct_reg))

            if mr is not None and seg is not None:
                mr_img = nib.load(str(mr))
                seg_img = nib.load(str(seg))
                rec.mr_seg_shape_same = (mr_img.shape == seg_img.shape)
                rec.mr_seg_affine_max_abs_diff = affine_max_abs_diff(mr_img.affine, seg_img.affine)
                if not rec.mr_seg_shape_same:
                    flags.append("MR_SEG_SHAPE_MISMATCH")
                if rec.mr_seg_affine_max_abs_diff is not None and rec.mr_seg_affine_max_abs_diff > 1e-3:
                    flags.append("MR_SEG_AFFINE_DIFF")

            if ct_raw is not None and mr is not None:
                ct_img = nib.load(str(ct_raw))
                mr_img = nib.load(str(mr))
                rec.ct_raw_vs_mr_shape_same = (ct_img.shape == mr_img.shape)
                rec.ct_raw_vs_mr_affine_max_abs_diff = affine_max_abs_diff(ct_img.affine, mr_img.affine)
                if not rec.ct_raw_vs_mr_shape_same:
                    flags.append("CTRAW_MR_SHAPE_MISMATCH")
                if rec.ct_raw_vs_mr_affine_max_abs_diff is not None and rec.ct_raw_vs_mr_affine_max_abs_diff > 1e-3:
                    flags.append("CTRAW_MR_AFFINE_DIFF")

            if ct_raw is not None and seg is not None:
                ct_img = nib.load(str(ct_raw))
                seg_img = nib.load(str(seg))
                rec.ct_raw_vs_seg_shape_same = (ct_img.shape == seg_img.shape)
                rec.ct_raw_vs_seg_affine_max_abs_diff = affine_max_abs_diff(ct_img.affine, seg_img.affine)
                if not rec.ct_raw_vs_seg_shape_same:
                    flags.append("CTRAW_SEG_SHAPE_MISMATCH")
                if rec.ct_raw_vs_seg_affine_max_abs_diff is not None and rec.ct_raw_vs_seg_affine_max_abs_diff > 1e-3:
                    flags.append("CTRAW_SEG_AFFINE_DIFF")

            if ct_reg is not None and mr is not None:
                ct_img = nib.load(str(ct_reg))
                mr_img = nib.load(str(mr))
                rec.ct_reg_vs_mr_shape_same = (ct_img.shape == mr_img.shape)
                rec.ct_reg_vs_mr_affine_max_abs_diff = affine_max_abs_diff(ct_img.affine, mr_img.affine)
                if not rec.ct_reg_vs_mr_shape_same:
                    flags.append("CTREG_MR_SHAPE_MISMATCH")
                if rec.ct_reg_vs_mr_affine_max_abs_diff is not None and rec.ct_reg_vs_mr_affine_max_abs_diff > 1e-3:
                    flags.append("CTREG_MR_AFFINE_DIFF")

            if ct_reg is not None and seg is not None:
                ct_img = nib.load(str(ct_reg))
                seg_img = nib.load(str(seg))
                rec.ct_reg_vs_seg_shape_same = (ct_img.shape == seg_img.shape)
                rec.ct_reg_vs_seg_affine_max_abs_diff = affine_max_abs_diff(ct_img.affine, seg_img.affine)
                if not rec.ct_reg_vs_seg_shape_same:
                    flags.append("CTREG_SEG_SHAPE_MISMATCH")
                if rec.ct_reg_vs_seg_affine_max_abs_diff is not None and rec.ct_reg_vs_seg_affine_max_abs_diff > 1e-3:
                    flags.append("CTREG_SEG_AFFINE_DIFF")

            if args.with_seg_stats and seg is not None:
                uniq_str, fg, bbox = seg_stats(seg)
                rec.seg_unique_labels = uniq_str
                rec.seg_foreground_voxels = fg
                rec.seg_bbox = bbox
                if fg == 0:
                    flags.append("EMPTY_MASK")

        except Exception as e:
            flags.append(f"READ_ERROR:{type(e).__name__}")

        rec.issues = "|".join(flags) if flags else ""
        for f in flags:
            issues_counter[f] = issues_counter.get(f, 0) + 1

        records.append(rec)

    df_out = pd.DataFrame([asdict(x) for x in records])
    csv_path = out_dir / "dataset_audit.csv"
    json_path = out_dir / "dataset_audit.json"
    df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([asdict(x) for x in records], f, ensure_ascii=False, indent=2)

    print(f"\n[Output] {csv_path}")
    print(f"[Output] {json_path}")

    # 统计三文件/四文件
    def has(x): return 0 if (x is None or (isinstance(x, float) and np.isnan(x)) or x == "") else 1
    df_out["has_ct_raw"] = df_out["ct_raw_path"].apply(has)
    df_out["has_ct_reg"] = df_out["ct_reg_path"].apply(has)
    four_file = df_out[(df_out["has_ct_raw"] == 1) & (df_out["has_ct_reg"] == 1)]
    three_file = df_out[(df_out["has_ct_raw"] + df_out["has_ct_reg"] == 1)]
    print(f"\n[CT variants] both ct & ct_reg: {len(four_file)}")
    print(f"[CT variants] only one CT (raw or reg): {len(three_file)}")

    # 统计同一病人跨批次出现情况
    multi_patient = df_out[df_out["patient_occurrences"].fillna(1).astype(int) > 1]
    print(f"[Patient] patient_id appears in >1 samples: {multi_patient['patient_id'].nunique() if 'patient_id' in multi_patient.columns else 0}")
    if len(multi_patient) > 0:
        top = (
            multi_patient.groupby("patient_id")["sample_id"]
            .count()
            .sort_values(ascending=False)
            .head(20)
        )
        print("\n[Patient] top repeated patient_id (count of samples):")
        print(top.to_string())

    print("\n[Issues Summary]")
    for k, v in sorted(issues_counter.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    dangerous = df_out[df_out["issues"].astype(str).str.contains(
        "MISSING_|_MISMATCH|_AFFINE_DIFF|EMPTY_MASK|READ_ERROR|CHOOSE_CT_ERROR|DUP_IN_SAME_BATCH", regex=True, na=False
    )]
    print(f"\n[Dangerous Samples] {len(dangerous)}")
    if len(dangerous) > 0:
        cols = ["sample_id", "batch", "patient_id", "case_dir", "mr_path", "seg_path",
                "ct_raw_path", "ct_reg_path", "chosen_ct_type", "chosen_ct_path", "issues"]
        cols = [c for c in cols if c in dangerous.columns]
        print(dangerous[cols].head(60).to_string(index=False))


if __name__ == "__main__":
    main()

'''

python scan_dataset_ctreg.py \
  --root dataset \
  --registry dataset/信息登记.xlsx \
  --out audit_ctreg \
  --with_seg_stats

'''