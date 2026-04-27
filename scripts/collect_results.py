#!/usr/bin/env python
"""
Collect and summarize experiment results.

Reads validation logs from each experiment and generates:
1. experiments_summary.csv
2. Updated EXPERIMENTS_LOSS_SAMPLING.md with results
"""

import re
import pandas as pd
from pathlib import Path
import json


def parse_validation_log(log_file):
    """
    Parse validation log to extract metrics.

    Expected format:
    [Validation Metrics]
      Foreground Dice: 0.7234 ± 0.1234
      Precision: 0.8123
      Recall: 0.7456

    [Bucketed Dice by GT Positive Ratio]
      <2%:   0.5123 (n=5)
      2-5%:  0.7234 (n=8)
      >5%:   0.8345 (n=8)
    """
    if not log_file.exists():
        return None

    with open(log_file, 'r') as f:
        content = f.read()

    # Extract metrics using regex
    metrics = {}

    # Foreground Dice
    match = re.search(r'Foreground Dice:\s+([\d.]+)\s+±\s+([\d.]+)', content)
    if match:
        metrics['foreground_dice_mean'] = float(match.group(1))
        metrics['foreground_dice_std'] = float(match.group(2))

    # Precision
    match = re.search(r'Precision:\s+([\d.]+)', content)
    if match:
        metrics['precision'] = float(match.group(1))

    # Recall
    match = re.search(r'Recall:\s+([\d.]+)', content)
    if match:
        metrics['recall'] = float(match.group(1))

    # Bucketed Dice
    match = re.search(r'<2%:\s+([\d.]+)\s+\(n=(\d+)\)', content)
    if match:
        metrics['dice_lt2'] = float(match.group(1))
        metrics['count_lt2'] = int(match.group(2))

    match = re.search(r'2-5%:\s+([\d.]+)\s+\(n=(\d+)\)', content)
    if match:
        metrics['dice_2to5'] = float(match.group(1))
        metrics['count_2to5'] = int(match.group(2))

    match = re.search(r'>5%:\s+([\d.]+)\s+\(n=(\d+)\)', content)
    if match:
        metrics['dice_gt5'] = float(match.group(1))
        metrics['count_gt5'] = int(match.group(2))

    return metrics if metrics else None


def collect_experiment_results(base_dir):
    """
    Collect results from all experiments.

    Args:
        base_dir: Path to MulModSeg_2024/out/unet/no_txt/

    Returns:
        DataFrame with experiment results
    """
    experiments = {
        'E0_baseline': 'Baseline (DiceCE)',
        'E1_tversky': 'TverskyLoss',
        'E2_focal_tversky': 'FocalTverskyLoss',
        'E3_enhanced_sampling': 'Enhanced Sampling',
    }

    results = []

    for exp_name, exp_desc in experiments.items():
        # Find experiment directory
        exp_dirs = list(Path(base_dir).glob(f'{exp_name}*'))

        if not exp_dirs:
            print(f"[WARNING] No directory found for {exp_name}")
            results.append({
                'experiment': exp_name,
                'description': exp_desc,
                'status': 'Not Run',
            })
            continue

        # Use the most recent directory
        exp_dir = sorted(exp_dirs, key=lambda x: x.stat().st_mtime)[-1]
        print(f"[INFO] Found {exp_name}: {exp_dir}")

        # Look for validation log or tensorboard events
        log_file = exp_dir / 'validation_final.log'

        if not log_file.exists():
            # Try to find any log file
            log_files = list(exp_dir.glob('*.log'))
            if log_files:
                log_file = log_files[0]
            else:
                print(f"[WARNING] No log file found for {exp_name}")
                results.append({
                    'experiment': exp_name,
                    'description': exp_desc,
                    'status': 'No Log',
                })
                continue

        # Parse metrics
        metrics = parse_validation_log(log_file)

        if metrics is None:
            print(f"[WARNING] Could not parse metrics for {exp_name}")
            results.append({
                'experiment': exp_name,
                'description': exp_desc,
                'status': 'Parse Failed',
            })
            continue

        # Add to results
        result = {
            'experiment': exp_name,
            'description': exp_desc,
            'status': 'Completed',
            **metrics
        }
        results.append(result)

    return pd.DataFrame(results)


def generate_summary_csv(df, output_path):
    """Generate CSV summary."""
    # Reorder columns
    columns = [
        'experiment', 'description', 'status',
        'foreground_dice_mean', 'foreground_dice_std',
        'precision', 'recall',
        'dice_lt2', 'count_lt2',
        'dice_2to5', 'count_2to5',
        'dice_gt5', 'count_gt5',
    ]

    # Keep only existing columns
    columns = [c for c in columns if c in df.columns]

    df_out = df[columns].copy()

    # Save
    df_out.to_csv(output_path, index=False, float_format='%.4f')
    print(f"\n[SUCCESS] Saved summary to: {output_path}")


def generate_markdown_table(df):
    """Generate markdown table for documentation."""
    if df.empty or 'foreground_dice_mean' not in df.columns:
        return "| Exp | Status |\n|-----|--------|\n" + "\n".join(
            f"| {row['experiment']} | {row['status']} |"
            for _, row in df.iterrows()
        )

    # Create table
    lines = [
        "| Exp | Foreground Dice | Precision | Recall | Dice<2% | Dice 2-5% | Dice>5% | Description |",
        "|-----|----------------|-----------|--------|---------|-----------|---------|-------------|"
    ]

    for _, row in df.iterrows():
        exp = row['experiment']
        desc = row['description']

        if row['status'] != 'Completed':
            lines.append(f"| {exp} | - | - | - | - | - | - | {desc} ({row['status']}) |")
            continue

        dice = f"{row.get('foreground_dice_mean', 0):.4f}"
        prec = f"{row.get('precision', 0):.4f}"
        rec = f"{row.get('recall', 0):.4f}"
        d_lt2 = f"{row.get('dice_lt2', 0):.4f}"
        d_2to5 = f"{row.get('dice_2to5', 0):.4f}"
        d_gt5 = f"{row.get('dice_gt5', 0):.4f}"

        lines.append(f"| {exp} | {dice} | {prec} | {rec} | {d_lt2} | {d_2to5} | {d_gt5} | {desc} |")

    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str,
                        default='/root/autodl-tmp/segmentation-test-main/MulModSeg_2024/out/unet/no_txt',
                        help='Base directory for experiment outputs')
    parser.add_argument('--output_csv', type=str,
                        default='/root/autodl-tmp/segmentation-test-main/MulModSeg_2024/experiments_summary.csv',
                        help='Output CSV file')
    args = parser.parse_args()

    print("="*80)
    print("Collecting Experiment Results")
    print("="*80)

    # Collect results
    df = collect_experiment_results(args.base_dir)

    # Generate CSV
    generate_summary_csv(df, args.output_csv)

    # Generate markdown table
    print("\n" + "="*80)
    print("Markdown Table for Documentation")
    print("="*80 + "\n")
    print(generate_markdown_table(df))

    print("\n" + "="*80)
    print("Summary Statistics")
    print("="*80)

    if 'foreground_dice_mean' in df.columns:
        completed = df[df['status'] == 'Completed']
        if not completed.empty:
            print(f"\nCompleted experiments: {len(completed)}")
            print(f"Best Foreground Dice: {completed['foreground_dice_mean'].max():.4f} "
                  f"({completed.loc[completed['foreground_dice_mean'].idxmax(), 'experiment']})")
            print(f"Best Precision: {completed['precision'].max():.4f} "
                  f"({completed.loc[completed['precision'].idxmax(), 'experiment']})")
            print(f"Best Recall: {completed['recall'].max():.4f} "
                  f"({completed.loc[completed['recall'].idxmax(), 'experiment']})")

            if 'dice_lt2' in completed.columns:
                print(f"Best Dice<2%: {completed['dice_lt2'].max():.4f} "
                      f"({completed.loc[completed['dice_lt2'].idxmax(), 'experiment']})")
    else:
        print("\nNo completed experiments found.")

    print("\n" + "="*80)
    print("Done!")
    print("="*80)


if __name__ == '__main__':
    main()
