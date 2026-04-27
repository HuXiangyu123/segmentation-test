#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build English caption prompts from Excel registry.

Outputs (default out_dir=prompts):
- prompts/prompts.csv:
    case_key, sample_id, batch, reg_id, diagnosis_zh, site_zh, diagnosis_en, site_en, caption
- prompts/template.md:
    template + rules + drop list
- prompts/untranslated_sites.csv (optional but recommended):
    site_zh, count   (for sites not covered by dictionary)

Key ideas:
- batch/sample_id are NOT part of caption; they are only used to uniquely identify cases for joining with files.
- "左/右 + 骨远端/股骨远端" -> left/right distal femur
- other sites: translate via dictionary; if unknown, keep original Chinese and record it.

Usage:
  python scripts/build_prompts.py --excel dataset/信息登记_实际生成.xlsx --out_dir prompts
  python scripts/build_prompts.py --excel dataset/信息登记_实际生成.xlsx --out_dir prompts --strict_site
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd


# Must drop (CT_reg vs MR shape mismatch)
DEFAULT_DROP_REG_IDS = ["11687281", "12298737"]


def _clean_str(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "<na>"}:
        return ""
    # handle "11009626.0"
    if re.fullmatch(r"\d+\.0", s):
        s = s[:-2]
    return s


def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns)
    for c in candidates:
        if c in cols:
            return c
    # fuzzy: remove whitespace
    norm = {re.sub(r"\s+", "", str(c)): c for c in cols}
    for c in candidates:
        key = re.sub(r"\s+", "", c)
        if key in norm:
            return norm[key]
    return None


def load_registry(excel_path: str) -> pd.DataFrame:
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel not found: {excel_path}")

    df = pd.read_excel(excel_path, engine="openpyxl")
    if df.empty:
        raise ValueError(f"Excel loaded but empty: {excel_path}")

    col_diag = _pick_col(df, ["诊断", "diagnosis", "Dx", "Diagnosis"])
    col_site = _pick_col(df, ["部位", "部位信息", "site", "Site", "Location"])
    col_reg = _pick_col(df, ["登记号", "登记号（去重）", "reg_id", "RegID", "ID"])
    col_batch = _pick_col(df, ["批次", "batch", "Batch"])
    col_sample = _pick_col(df, ["sample_id", "SampleID", "样本ID", "样本id"])

    missing = []
    if col_diag is None:
        missing.append("诊断/diagnosis")
    if col_site is None:
        missing.append("部位/site")
    if col_reg is None:
        missing.append("登记号/reg_id")
    if col_batch is None and col_sample is None:
        missing.append("批次/batch (or sample_id)")
    if missing:
        raise KeyError(f"Missing required columns: {missing}. Columns found: {list(df.columns)}")

    out = pd.DataFrame()
    out["diagnosis_zh"] = df[col_diag].map(_clean_str)
    out["site_zh"] = df[col_site].map(_clean_str)
    out["reg_id"] = df[col_reg].map(_clean_str)
    out["batch"] = df[col_batch].map(_clean_str) if col_batch is not None else ""
    out["sample_id"] = df[col_sample].map(_clean_str) if col_sample is not None else ""

    # Build sample_id if missing: "批次/登记号" (used as stable join key)
    need_sample = out["sample_id"].eq("")
    if need_sample.any():
        if col_batch is None:
            raise KeyError("sample_id missing/empty and batch column not found; cannot construct sample_id.")
        out.loc[need_sample, "sample_id"] = (
            out.loc[need_sample, "batch"].map(_clean_str) + "/" + out.loc[need_sample, "reg_id"].map(_clean_str)
        )

    out = out[out["sample_id"].ne("") & out["reg_id"].ne("")].copy()
    out = out.drop_duplicates(subset=["sample_id"], keep="first").reset_index(drop=True)
    return out


# ---- Translation rules ----

DX_MAP: Dict[str, str] = {
    "骨肉瘤": "osteosarcoma",
    "软骨肉瘤": "chondrosarcoma",
    "尤文肉瘤": "ewing sarcoma",
    "骨巨细胞瘤": "giant cell tumor of bone",
}

# Common site map (extend whenever you see untranslated sites)
SITE_MAP: Dict[str, str] = {
    "骨盆": "pelvis",
    "髋臼": "acetabulum",
    "股骨近端": "proximal femur",
    "股骨远端": "distal femur",
    "胫骨近端": "proximal tibia",
    "胫骨远端": "distal tibia",
    "腓骨": "fibula",
    "肱骨": "humerus",
    "桡骨": "radius",
    "尺骨": "ulna",
    "肩胛骨": "scapula",
    "髂骨": "ilium",
    "骶骨": "sacrum",
}


def _detect_laterality(site_zh: str) -> Tuple[str, str]:
    """
    Returns (laterality, site_wo_lat)
    laterality: 'left' | 'right' | ''
    """
    s = _clean_str(site_zh)
    lat = ""
    if "左" in s:
        lat = "left"
        s = s.replace("左", "")
    elif "右" in s:
        lat = "right"
        s = s.replace("右", "")
    return lat, s.strip()


def translate_diagnosis(dx_zh: str) -> str:
    dx_zh = _clean_str(dx_zh)
    if dx_zh in DX_MAP:
        return DX_MAP[dx_zh]
    # fallback: keep original
    return dx_zh


def translate_site(site_zh: str) -> Tuple[str, bool]:
    """
    Translate Chinese anatomical site into English.
    Returns (site_en, translated_ok)
    """
    raw = _clean_str(site_zh)
    if raw == "":
        return "", True

    lat, core = _detect_laterality(raw)

    # Special rule you requested:
    # "左/右 + 骨远端/股骨远端" => left/right distal femur
    # - treat any core containing "骨远端" as distal femur (even if femur not explicitly written)
    if "骨远端" in core or "股骨远端" in core:
        base = "distal femur"
        site_en = f"{lat} {base}".strip() if lat else base
        return site_en, True

    # Direct dictionary match
    if core in SITE_MAP:
        base = SITE_MAP[core]
        site_en = f"{lat} {base}".strip() if lat else base
        return site_en, True

    # Partial match: if any key appears in core, map to that
    for k, v in SITE_MAP.items():
        if k and (k in core):
            base = v
            site_en = f"{lat} {base}".strip() if lat else base
            return site_en, True

    # If already looks like English, keep it
    if re.search(r"[A-Za-z]", raw):
        return raw, True

    # Unknown Chinese site: keep raw (so you don't lose info), mark untranslated
    # Still keep laterality in front if exists
    site_en = raw
    if lat:
        site_en = f"{lat} {raw}"
    return site_en, False


def build_caption(diagnosis_en: str, site_en: str) -> str:
    diagnosis_en = _clean_str(diagnosis_en)
    site_en = _clean_str(site_en)
    if diagnosis_en and site_en:
        return f"{diagnosis_en} in {site_en}"
    if diagnosis_en:
        return diagnosis_en
    if site_en:
        return f"lesion in {site_en}"
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", type=str, required=True, help="Path to Excel registry (.xlsx)")
    ap.add_argument("--out_dir", type=str, default="prompts", help="Output directory (will be created)")
    ap.add_argument(
        "--drop_reg_ids",
        type=str,
        default=",".join(DEFAULT_DROP_REG_IDS),
        help="Comma-separated reg_id list to drop (shape mismatch).",
    )
    ap.add_argument(
        "--key_mode",
        type=str,
        default="sample_id",
        choices=["sample_id", "reg_id"],
        help="case_key used to join other files. Use sample_id to avoid collisions across batches.",
    )
    ap.add_argument(
        "--strict_site",
        action="store_true",
        help="If set, raise error when encountering untranslated site_zh.",
    )
    args = ap.parse_args()

    df = load_registry(args.excel)

    drop_ids = [_clean_str(x) for x in args.drop_reg_ids.split(",") if _clean_str(x)]
    before = len(df)
    df = df[~df["reg_id"].isin(drop_ids)].reset_index(drop=True)
    after = len(df)

    # Translate diagnosis + site
    dx_en_list = []
    site_en_list = []
    ok_list = []

    for dx_zh, site_zh in zip(df["diagnosis_zh"], df["site_zh"]):
        dx_en = translate_diagnosis(dx_zh)
        site_en, ok = translate_site(site_zh)
        dx_en_list.append(dx_en)
        site_en_list.append(site_en)
        ok_list.append(ok)

    df["diagnosis_en"] = dx_en_list
    df["site_en"] = site_en_list
    df["site_translated_ok"] = ok_list

    # Caption (English only)
    df["caption"] = [build_caption(d, s) for d, s in zip(df["diagnosis_en"], df["site_en"])]

    # Decide join key
    df["case_key"] = df[args.key_mode]

    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, "prompts.csv")

    df_out = df[
        ["case_key", "sample_id", "batch", "reg_id", "diagnosis_zh", "site_zh", "diagnosis_en", "site_en", "caption"]
    ].copy()
    df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # Untranslated sites report
    untranslated = df.loc[~df["site_translated_ok"], "site_zh"].map(_clean_str)
    untranslated = untranslated[untranslated.ne("")]
    untranslated_csv = os.path.join(args.out_dir, "untranslated_sites.csv")
    if len(untranslated) > 0:
        untranslated.value_counts().rename_axis("site_zh").reset_index(name="count").to_csv(
            untranslated_csv, index=False, encoding="utf-8-sig"
        )

    if args.strict_site and len(untranslated) > 0:
        raise ValueError(
            f"Found untranslated site_zh values. See: {untranslated_csv}. "
            f"Add mappings into SITE_MAP or adjust rules."
        )

    # Write template doc
    template_md = os.path.join(args.out_dir, "template.md")
    with open(template_md, "w", encoding="utf-8") as f:
        f.write("# Prompt Template (caption)\n\n")
        f.write("## Caption template (EN)\n")
        f.write('- "{diagnosis_en} in {site_en}"\n\n')
        f.write("## Rules\n")
        f.write("- batch/sample_id are NOT part of caption; they are join keys only.\n")
        f.write("- Special: '左/右 + 骨远端/股骨远端' -> left/right distal femur.\n")
        f.write("- Other sites translated by SITE_MAP; unknown Chinese sites are kept as-is and listed in untranslated_sites.csv.\n\n")
        f.write("## Drop list (shape mismatch)\n")
        for rid in drop_ids:
            f.write(f"- {rid}\n")

    print(f"[OK] Loaded {before} rows; dropped {before - after} rows by reg_id; wrote: {out_csv}")
    print(f"[OK] Wrote: {template_md}")
    if os.path.exists(untranslated_csv):
        print(f"[WARN] Untranslated sites found. Review and extend SITE_MAP: {untranslated_csv}")


if __name__ == "__main__":
    main()
