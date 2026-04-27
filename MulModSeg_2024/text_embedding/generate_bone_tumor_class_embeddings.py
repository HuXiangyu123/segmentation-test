#!/usr/bin/env python
"""
为每个病例生成个性化的空间描述 CLIP 文本嵌入（内存优化版）。

输入：
  1. prompts_updated.csv (推荐，由 xlsx 另存为) 或 .xlsx，提供 reg_id/case_key 与 site_zh
  2. tumor_each_case_proportion.csv，提供 Case_ID 与 Size_Category

输出：
  一个 .pth 文件，格式为：
    {
      "embeddings": Tensor[N_cases, 2, 512],  # modality 顺序固定为 [MR, CT]
      "id_map": list[dict],                   # 每个 case 的元信息
      "modality_order": ["MR", "CT"],
    }
"""

import argparse
import os
import re
import gc
from typing import Any, Dict, Iterable, List, Optional

import clip
import pandas as pd
import torch
import torch.nn.functional as F

MODALITY_ORDER = ["MR", "CT"]
SIZE_TRANSLATION_MAP = {
    "小": "small focal",
    "中": "medium-sized",
    "大": "large massive",
    "small": "small focal",
    "medium": "medium-sized",
    "large": "large massive",
}
GENERIC_SITE_DESCRIPTION = "affected skeletal region"
EXACT_SITE_MAP = {
    "左股骨远端": "left distal femur",
    "右股骨远端": "right distal femur",
    "左股骨近端": "left proximal femur",
    "右股骨近端": "right proximal femur",
    "左侧骨盆": "left pelvis",
    "右侧骨盆": "right pelvis",
    "左盆骨": "left pelvis",
    "右盆骨": "right pelvis",
    "骨盆": "pelvis",
    "盆骨": "pelvis",
    "双侧骨盆": "bilateral pelvis",
    "双侧盆骨": "bilateral pelvis",
}
SIDE_TOKENS = [
    ("双侧", "bilateral"),
    ("左侧", "left"),
    ("右侧", "right"),
    ("左", "left"),
    ("右", "right"),
]
REGION_TOKENS = [
    ("远端", "distal"),
    ("近端", "proximal"),
    ("中段", "midshaft"),
    ("骨干", "diaphyseal"),
    ("干骺端", "metaphyseal"),
]
ANATOMY_TOKENS = [
    ("股骨", "femur"),
    ("胫骨", "tibia"),
    ("腓骨", "fibula"),
    ("肱骨", "humerus"),
    ("尺骨", "ulna"),
    ("桡骨", "radius"),
    ("骨盆", "pelvis"),
    ("盆骨", "pelvis"),
    ("髂骨", "ilium"),
    ("骶骨", "sacrum"),
    ("肩胛骨", "scapula"),
    ("锁骨", "clavicle"),
]


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except TypeError:
        return False


def normalize_case_id(value: Any) -> str:
    """统一提取病例主键中的数字部分。"""
    if is_missing(value):
        return ""
    match = re.search(r"\d+", str(value))
    return match.group() if match else str(value).strip()


def clean_text(value: Any) -> str:
    if is_missing(value):
        return ""
    return str(value).strip()


def read_prompt_table(input_path: str) -> pd.DataFrame:
    """读取提示词表格，优先推荐使用 CSV 以降低内存。"""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Prompt metadata not found: {input_path}")
    ext = os.path.splitext(input_path)[1].lower()
    
    if ext == ".csv":
        # 如果是 CSV，读取效率高且省内存
        return pd.read_csv(input_path, dtype=str)
    elif ext in {".xlsx", ".xls"}:
        print("[INFO] 正在读取 Excel 文件，这可能会消耗大量内存...")
        return pd.read_excel(input_path, dtype=str)
    
    raise ValueError(f"Unsupported prompt metadata format: {input_path}")


def resolve_case_id_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    existing = set(columns)
    for candidate in candidates:
        if candidate in existing:
            return candidate
    return None


def translate_size_label(raw_value: Any) -> str:
    raw = clean_text(raw_value)
    if not raw:
        raise ValueError("Encountered empty Size_Category.")

    normalized = raw.lower()
    if normalized in SIZE_TRANSLATION_MAP:
        return SIZE_TRANSLATION_MAP[normalized]
    if raw in SIZE_TRANSLATION_MAP:
        return SIZE_TRANSLATION_MAP[raw]

    raise ValueError(
        f"Unsupported Size_Category: {raw!r}. "
        f"Supported values include {sorted(SIZE_TRANSLATION_MAP)}."
    )


def strip_punctuation(text: str) -> str:
    return re.sub(r"[\s,，;；:：/\\()（）\-]+", "", text)


def translate_site_label(site_zh: Any) -> str:
    raw = clean_text(site_zh)
    if not raw:
        return GENERIC_SITE_DESCRIPTION

    compact = strip_punctuation(raw)
    if compact in EXACT_SITE_MAP:
        return EXACT_SITE_MAP[compact]

    working = compact
    side_en = ""
    region_en = ""
    anatomy_en = ""

    for zh, en in SIDE_TOKENS:
        if zh in working:
            side_en = en
            working = working.replace(zh, "", 1)
            break

    for zh, en in REGION_TOKENS:
        if zh in working:
            region_en = en
            working = working.replace(zh, "", 1)
            break

    for zh, en in ANATOMY_TOKENS:
        if zh in working:
            anatomy_en = en
            working = working.replace(zh, "", 1)
            break

    if anatomy_en:
        parts = [part for part in [side_en, region_en, anatomy_en] if part]
        return " ".join(parts)

    return GENERIC_SITE_DESCRIPTION


def build_case_caption(size_en: str, site_en: str) -> str:
    location = site_en or GENERIC_SITE_DESCRIPTION
    return (
        "A multi-modal CT and MR scan showing a single "
        f"{size_en} osteosarcoma located in the {location}."
    )


def prepare_case_metadata(prompts_file: str, size_csv: str) -> pd.DataFrame:
    # 1. 加载并清洗 prompt 数据
    prompt_df = read_prompt_table(prompts_file).copy()
    if "site_zh" not in prompt_df.columns:
        raise ValueError("Prompt metadata must contain 'site_zh'.")

    prompt_id_col = resolve_case_id_column(prompt_df.columns, ["reg_id", "case_key", "sample_id"])
    if prompt_id_col is None:
        raise ValueError("Prompt metadata must contain one of: reg_id, case_key, sample_id.")

    prompt_df["case_id"] = prompt_df[prompt_id_col].apply(normalize_case_id)
    prompt_df["site_zh"] = prompt_df["site_zh"].apply(clean_text)
    prompt_df = prompt_df.loc[prompt_df["case_id"] != ""].copy()
    prompt_df = prompt_df.drop_duplicates(subset=["case_id"], keep="first")

    # 2. 加载并清洗 size 数据
    size_df = pd.read_csv(size_csv, dtype=str).copy()
    if "Size_Category" not in size_df.columns:
        raise ValueError("Size metadata must contain 'Size_Category'.")

    size_id_col = resolve_case_id_column(
        size_df.columns,
        ["Case_ID", "ID", "reg_id", "case_key", "case_id", "patient_id"],
    )
    if size_id_col is None:
        raise ValueError("Size metadata must contain an ID column such as Case_ID or ID.")

    size_df["case_id"] = size_df[size_id_col].apply(normalize_case_id)
    size_df["size_zh"] = size_df["Size_Category"].apply(clean_text)
    size_df["size_en"] = size_df["Size_Category"].apply(translate_size_label)
    size_df = size_df.loc[size_df["case_id"] != ""].copy()
    size_df = size_df.drop_duplicates(subset=["case_id"], keep="first")

    # 3. 合并数据，仅保留 size_df 中需要的列
    merged = prompt_df.merge(
        size_df[["case_id", "size_zh", "size_en"]],
        on="case_id",
        how="left",
        validate="one_to_one",
    )

    # 4. 主动释放内存 (关键修复 OOM)
    del prompt_df
    del size_df
    gc.collect()

    # 5. 处理缺失值与生成英文描述
    missing_size = merged["size_en"].isna()
    if missing_size.any():
        missing_ids = merged.loc[missing_size, "case_id"].tolist()
        print(
            "[WARN] Skipping cases without matched Size_Category: "
            f"{missing_ids[:10]}"
            + (" ..." if len(missing_ids) > 10 else "")
        )
        merged = merged.loc[~missing_size].copy()

    merged["site_en"] = merged["site_zh"].apply(translate_site_label)
    merged["caption"] = merged.apply(
        lambda row: build_case_caption(row["size_en"], row["site_en"]),
        axis=1,
    )
    
    return merged.reset_index(drop=True)


def encode_texts(clip_model, texts: List[str], device: str) -> torch.Tensor:
    tokens = clip.tokenize(texts, truncate=True).to(device)
    with torch.no_grad():
        features = clip_model.encode_text(tokens).float().cpu()
    return F.normalize(features, dim=-1)


def build_case_embeddings(
    prompts_file: str,
    size_csv: str,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    # 生成元数据
    case_df = prepare_case_metadata(prompts_file, size_csv)
    
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loaded {len(case_df)} cases from {prompts_file}")
    print(f"Merged size metadata from {size_csv}")
    print(f"Using device: {device}")

    # 加载 CLIP 模型
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    all_embeddings: List[torch.Tensor] = []
    id_map: List[Dict[str, Any]] = []

    for _, row in case_df.iterrows():
        caption = row["caption"]
        # 保留 [MR, CT, 512] 结构，即使两种模态使用同一空间描述文本。
        case_embeddings = encode_texts(clip_model, [caption, caption], device)
        all_embeddings.append(case_embeddings)
        id_map.append(
            {
                "patient_id": row["case_id"],
                "reg_id": clean_text(row.get("reg_id", row["case_id"])),
                "case_key": clean_text(row.get("case_key", "")),
                "site_zh": row["site_zh"],
                "site_en": row["site_en"],
                "size_category_zh": row["size_zh"],
                "size_category_en": row["size_en"],
                "caption": caption,
            }
        )
        print(f"[{row['case_id']}] {caption}")

    embeddings_tensor = (
        torch.stack(all_embeddings, dim=0)
        if all_embeddings
        else torch.empty(0, len(MODALITY_ORDER), 512)
    )
    embeddings_tensor = F.normalize(embeddings_tensor, dim=-1)

    return {
        "embeddings": embeddings_tensor,
        "id_map": id_map,
        "modality_order": MODALITY_ORDER,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate per-case personalized CLIP text embeddings with shape [N_cases, 2, 512]."
    )
    parser.add_argument(
        "--prompts_xlsx",
        type=str,
        required=True,
        help="Path to prompts_updated.csv or .xlsx",
    )
    parser.add_argument(
        "--size_csv",
        type=str,
        required=True,
        help="Path to tumor_each_case_proportion.csv.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save output .pth file.",
    )
    args = parser.parse_args()

    output_data = build_case_embeddings(args.prompts_xlsx, args.size_csv)
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(output_data, args.output_file)

    print(f"\nSaved case-level embeddings to: {args.output_file}")
    print(f"Tensor shape: {tuple(output_data['embeddings'].shape)}")
    print("Dimension meaning:")
    print("  dim0 = case index")
    print(f"  dim1 = modality {MODALITY_ORDER}")
    print("  dim2 = CLIP embedding dim (512)")


if __name__ == "__main__":
    main()