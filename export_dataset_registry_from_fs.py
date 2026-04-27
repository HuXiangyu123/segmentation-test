import argparse
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


CASE_ID_PATTERN = re.compile(r"^(?:N)?\d+$", re.IGNORECASE)


def is_nii(p: Path) -> bool:
    n = p.name.lower()
    return p.is_file() and (n.endswith(".nii") or n.endswith(".nii.gz"))


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
    - MR：文件名含 mr
    - CT_reg：含 ct_reg 或 (ct & reg/registered)
    - CT_raw：含 ct 但不含 reg
    - SEG：优先 seg/mask/label/gt；否则在“非ct非mr”里兜底
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
        exact = [
            p for p in seg_cands
            if p.name.lower().startswith(pid) and (p.name.lower().endswith(".nii") or p.name.lower().endswith(".nii.gz"))
        ]
        seg = pick_best(exact, prefer_keywords=[f"{pid}.nii", f"{pid}.nii.gz"]) if exact else pick_best(
            seg_cands, prefer_keywords=["seg", "mask", "label", "gt", f"{pid}.nii", f"{pid}.nii.gz"]
        )

    return mr, seg, ct_raw, ct_reg


def infer_site_by_batch(batch_name: str, shanghai_name: str = "上海市一") -> str:
    if batch_name in ["第1批", "第2批", "第3批", shanghai_name]:
        return "左右骨远端"
    if batch_name in ["第4批", "第5批"]:
        return "骨盆"
    return "未知"


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


@dataclass
class ExportRow:
    诊断: str
    部位: str
    登记号: str
    批次: str
    sample_id: str
    case_dir: str

    mr_path: str = ""
    seg_path: str = ""
    ct_raw_path: str = ""
    ct_reg_path: str = ""

    # 简单标记
    file_variant: str = ""        # 3-file / 4-file / missing
    issues: str = ""              # MISSING_MR|MISSING_SEG|MISSING_CT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="数据根目录（包含 第1批..第5批/上海市一）")
    ap.add_argument("--out_xlsx", type=str, default="信息登记_实际生成.xlsx", help="输出xlsx路径")
    ap.add_argument("--diagnosis", type=str, default="骨肉瘤", help="诊断默认值（默认全填骨肉瘤）")
    ap.add_argument("--shanghai_name", type=str, default="上海市一", help="平铺文件夹名（默认 上海市一）")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"root not exists: {root}")

    samples = build_samples(root, shanghai_folder_name=args.shanghai_name)
    print(f"[Scan] samples found: {len(samples)}")

    rows: List[ExportRow] = []
    issue_counter: Dict[str, int] = {}

    for s in samples:
        batch = s["batch"]
        pid = s["patient_id"]
        case_dir = s["case_dir"]
        sample_id = s["sample_id"]
        files: List[Path] = s["files"]

        site = infer_site_by_batch(batch, shanghai_name=args.shanghai_name)

        mr, seg, ct_raw, ct_reg = match_files_in_case(pid, files)

        issues = []
        if mr is None:
            issues.append("MISSING_MR")
        if seg is None:
            issues.append("MISSING_SEG")
        if ct_raw is None and ct_reg is None:
            issues.append("MISSING_CT")

        has_raw = ct_raw is not None
        has_reg = ct_reg is not None
        if has_raw and has_reg:
            variant = "4-file(ct+ct_reg)"
        elif has_raw or has_reg:
            variant = "3-file(only_one_ct)"
        else:
            variant = "missing_ct"

        for it in issues:
            issue_counter[it] = issue_counter.get(it, 0) + 1

        r = ExportRow(
            诊断=args.diagnosis,
            部位=site,
            登记号=pid,
            批次=batch,
            sample_id=sample_id,
            case_dir=case_dir,
            mr_path=str(mr) if mr is not None else "",
            seg_path=str(seg) if seg is not None else "",
            ct_raw_path=str(ct_raw) if ct_raw is not None else "",
            ct_reg_path=str(ct_reg) if ct_reg is not None else "",
            file_variant=variant,
            issues="|".join(issues),
        )
        rows.append(r)

    df = pd.DataFrame([asdict(x) for x in rows])

    # summary
    summary_batch = df.groupby(["批次", "部位"], as_index=False).agg(
        samples=("sample_id", "count"),
        unique_patient=("登记号", "nunique"),
        missing_mr=("issues", lambda s: (s.astype(str).str.contains("MISSING_MR")).sum()),
        missing_seg=("issues", lambda s: (s.astype(str).str.contains("MISSING_SEG")).sum()),
        missing_ct=("issues", lambda s: (s.astype(str).str.contains("MISSING_CT")).sum()),
        four_file=("file_variant", lambda s: (s == "4-file(ct+ct_reg)").sum()),
        three_file=("file_variant", lambda s: (s == "3-file(only_one_ct)").sum()),
    )

    summary_issue = pd.DataFrame(
        [{"issue": k, "count": v} for k, v in sorted(issue_counter.items(), key=lambda x: -x[1])]
    )

    out_xlsx = Path(args.out_xlsx)
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="samples")
        summary_batch.to_excel(w, index=False, sheet_name="summary")
        summary_issue.to_excel(w, index=False, sheet_name="issues")

    print(f"[OK] wrote: {out_xlsx}")
    print("[Tip] samples sheet = 真实样本清单；summary/issues sheet = 统计与缺失概览。")


if __name__ == "__main__":
    main()
