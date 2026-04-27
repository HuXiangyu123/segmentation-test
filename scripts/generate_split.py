#!/usr/bin/env python3
"""
Generate train/val split for bone_tumor dataset.
- Reads prompts/prompts.csv to get all valid sample_ids
- Excludes drop_list (shape mismatch samples)
- Splits into train/val with fixed random seed
- Outputs splits/split_seed{seed}.json
"""
import argparse
import json
import random
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts_csv", type=str, default="prompts/prompts.csv", help="Path to prompts.csv")
    ap.add_argument("--out_dir", type=str, default="splits", help="Output directory for split files")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for split")
    ap.add_argument("--val_ratio", type=float, default=0.2, help="Validation ratio (default: 0.2)")
    ap.add_argument("--drop_reg_ids", type=str, default="11687281,12298737", help="Comma-separated reg_ids to drop")
    args = ap.parse_args()

    # Parse drop list
    drop_list = [x.strip() for x in args.drop_reg_ids.split(",") if x.strip()]
    print(f"[Drop list] {len(drop_list)} samples: {drop_list}")

    # Read prompts.csv
    df = pd.read_csv(args.prompts_csv)
    print(f"[Prompts] Total samples in prompts.csv: {len(df)}")

    # Filter out drop_list
    df_filtered = df[~df["reg_id"].astype(str).isin(drop_list)].copy()
    print(f"[Filtered] Remaining samples after drop: {len(df_filtered)}")

    # Get sample_ids
    sample_ids = df_filtered["sample_id"].tolist()

    # Shuffle with fixed seed
    random.seed(args.seed)
    random.shuffle(sample_ids)

    # Split
    num_val = max(1, int(len(sample_ids) * args.val_ratio))
    val_ids = sample_ids[:num_val]
    train_ids = sample_ids[num_val:]

    print(f"[Split] Train: {len(train_ids)}, Val: {len(val_ids)}")

    # Build output
    split_data = {
        "train": train_ids,
        "val": val_ids,
        "drop_list": drop_list,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "total_samples": len(sample_ids),
    }

    # Write to file
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"split_seed{args.seed}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(split_data, f, ensure_ascii=False, indent=2)

    print(f"\n[Output] {out_path}")
    print(f"  - train: {len(train_ids)} samples")
    print(f"  - val: {len(val_ids)} samples")
    print(f"  - dropped: {len(drop_list)} samples")


if __name__ == "__main__":
    main()
