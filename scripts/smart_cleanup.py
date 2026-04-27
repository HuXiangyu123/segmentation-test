#!/usr/bin/env python
"""
Smart cleanup for unused documents and model weights.

Identifies and removes:
1. Redundant/outdated documentation
2. Intermediate epoch checkpoints (keep only best + last 2)
3. Failed/incomplete experiment runs
4. Duplicate documentation
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
import argparse


class SmartCleaner:
    """Smart cleanup for documentation and weights."""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.to_remove = []
        self.to_keep = []

    def analyze_documentation(self):
        """Analyze documentation files for redundancy."""
        print(f"\n{'='*80}")
        print("Analyzing Documentation")
        print(f"{'='*80}\n")

        docs_dir = self.project_root / 'docs'
        if not docs_dir.exists():
            print("No docs directory found.")
            return

        docs = list(docs_dir.glob('*.md'))

        # Categorize documents
        essential_docs = [
            'README.md',
            'EXPERIMENTS_LOSS_SAMPLING.md',
            'EXPERIMENT_QUICKSTART.md',
            'TRAIN_PY_MODIFICATIONS.md',
            'PROJECT_REVIEW.md',
        ]

        outdated_docs = [
            'COMPLETION_REPORT.md',  # Superseded by PROJECT_REVIEW.md
            'VALIDATION_FIX.md',     # Historical, info in DIAGNOSIS_REPORT.md
            'ADAPTATION.md',         # Initial setup, no longer needed
            'ARCH.md',               # Basic info, covered in README
            'DATASET.md',            # Basic info, covered in README
        ]

        print("Essential Documentation (KEEP):")
        for doc_name in essential_docs:
            doc_path = docs_dir / doc_name
            if doc_path.exists():
                size = doc_path.stat().st_size / 1024
                print(f"  ✅ {doc_name} ({size:.1f} KB)")
                self.to_keep.append(doc_path)

        print("\nOutdated/Redundant Documentation (REMOVE):")
        for doc_name in outdated_docs:
            doc_path = docs_dir / doc_name
            if doc_path.exists():
                size = doc_path.stat().st_size / 1024
                print(f"  ❌ {doc_name} ({size:.1f} KB)")
                self.to_remove.append({
                    'path': doc_path,
                    'reason': 'Outdated/redundant documentation',
                    'size': doc_path.stat().st_size
                })

        print("\nOther Documentation:")
        for doc in docs:
            if doc.name not in essential_docs and doc.name not in outdated_docs:
                size = doc.stat().st_size / 1024
                print(f"  ℹ️  {doc.name} ({size:.1f} KB)")

    def analyze_model_weights(self):
        """Analyze model checkpoints and keep only necessary ones."""
        print(f"\n{'='*80}")
        print("Analyzing Model Weights")
        print(f"{'='*80}\n")

        out_dir = self.project_root / 'MulModSeg_2024' / 'out' / 'unet' / 'no_txt'
        if not out_dir.exists():
            print("No output directory found.")
            return

        for exp_dir in sorted(out_dir.iterdir()):
            if not exp_dir.is_dir():
                continue

            print(f"\n📁 {exp_dir.name}")

            # Find all checkpoints
            checkpoints = list(exp_dir.glob('epoch_*.pt'))
            best_model = exp_dir / 'best_model.pt'

            if not checkpoints and not best_model.exists():
                print("  ⚠️  No checkpoints found")
                continue

            # Sort by epoch number
            checkpoints.sort(key=lambda x: int(x.stem.split('_')[1]))

            total_size = sum(ckpt.stat().st_size for ckpt in checkpoints)
            if best_model.exists():
                total_size += best_model.stat().st_size

            print(f"  Total checkpoints: {len(checkpoints)}")
            print(f"  Total size: {total_size / (1024**3):.2f} GB")

            # Keep strategy: best_model + last 2 epochs
            if best_model.exists():
                print(f"  ✅ KEEP: best_model.pt ({best_model.stat().st_size / (1024**2):.1f} MB)")
                self.to_keep.append(best_model)

            if len(checkpoints) > 2:
                # Keep last 2 epochs
                keep_checkpoints = checkpoints[-2:]
                remove_checkpoints = checkpoints[:-2]

                print(f"  ✅ KEEP: Last 2 epochs")
                for ckpt in keep_checkpoints:
                    print(f"    - {ckpt.name} ({ckpt.stat().st_size / (1024**2):.1f} MB)")
                    self.to_keep.append(ckpt)

                print(f"  ❌ REMOVE: {len(remove_checkpoints)} intermediate checkpoints")
                for ckpt in remove_checkpoints:
                    size_mb = ckpt.stat().st_size / (1024**2)
                    print(f"    - {ckpt.name} ({size_mb:.1f} MB)")
                    self.to_remove.append({
                        'path': ckpt,
                        'reason': 'Intermediate checkpoint',
                        'size': ckpt.stat().st_size
                    })
            else:
                print(f"  ✅ KEEP: All {len(checkpoints)} checkpoints (≤2)")
                for ckpt in checkpoints:
                    self.to_keep.append(ckpt)

    def analyze_failed_experiments(self):
        """Identify failed or incomplete experiment runs."""
        print(f"\n{'='*80}")
        print("Analyzing Experiment Completeness")
        print(f"{'='*80}\n")

        out_dir = self.project_root / 'MulModSeg_2024' / 'out' / 'unet' / 'no_txt'
        if not out_dir.exists():
            return

        for exp_dir in sorted(out_dir.iterdir()):
            if not exp_dir.is_dir():
                continue

            best_model = exp_dir / 'best_model.pt'
            checkpoints = list(exp_dir.glob('epoch_*.pt'))

            # Check if experiment completed
            if not best_model.exists() and len(checkpoints) == 0:
                size = self._get_dir_size(exp_dir)
                print(f"❌ INCOMPLETE: {exp_dir.name}")
                print(f"   No checkpoints found, size: {size / (1024**2):.1f} MB")

                self.to_remove.append({
                    'path': exp_dir,
                    'reason': 'Incomplete experiment (no checkpoints)',
                    'size': size
                })

    def analyze_old_experiments(self):
        """Identify old experiment runs that can be archived."""
        print(f"\n{'='*80}")
        print("Analyzing Old Experiments")
        print(f"{'='*80}\n")

        out_dir = self.project_root / 'MulModSeg_2024' / 'out' / 'unet' / 'no_txt'
        if not out_dir.exists():
            return

        experiments = []
        for exp_dir in sorted(out_dir.iterdir()):
            if not exp_dir.is_dir():
                continue

            mtime = datetime.fromtimestamp(exp_dir.stat().st_mtime)
            age_days = (datetime.now() - mtime).days
            size = self._get_dir_size(exp_dir)

            experiments.append({
                'path': exp_dir,
                'name': exp_dir.name,
                'age_days': age_days,
                'size': size,
                'mtime': mtime
            })

        # Sort by age
        experiments.sort(key=lambda x: x['age_days'], reverse=True)

        # Identify experiments to keep
        # Keep: final runs, recent runs (<7 days)
        for exp in experiments:
            name = exp['name']
            age = exp['age_days']

            if 'final' in name.lower():
                print(f"✅ KEEP: {name} (final run)")
            elif age < 7:
                print(f"✅ KEEP: {name} (recent, {age} days old)")
            else:
                # Old non-final runs - suggest for review
                print(f"⚠️  OLD: {name} ({age} days old, {exp['size'] / (1024**2):.1f} MB)")
                print(f"   Consider archiving if not needed")

    def print_summary(self):
        """Print cleanup summary."""
        if not self.to_remove:
            print(f"\n{'='*80}")
            print("No items to remove.")
            print(f"{'='*80}\n")
            return

        total_size = sum(item['size'] for item in self.to_remove)

        print(f"\n{'='*80}")
        print("Cleanup Summary")
        print(f"{'='*80}\n")

        print(f"Items to remove: {len(self.to_remove)}")
        print(f"Space to free: {total_size / (1024**2):.2f} MB ({total_size / (1024**3):.2f} GB)\n")

        # Group by reason
        by_reason = {}
        for item in self.to_remove:
            reason = item['reason']
            if reason not in by_reason:
                by_reason[reason] = []
            by_reason[reason].append(item)

        for reason, items in by_reason.items():
            reason_size = sum(item['size'] for item in items)
            print(f"{reason}:")
            print(f"  Count: {len(items)}")
            print(f"  Size: {reason_size / (1024**2):.2f} MB")
            for item in items:
                print(f"    - {item['path'].relative_to(self.project_root)}")

    def execute_cleanup(self, dry_run=True, skip_confirm=False):
        """Execute cleanup."""
        if not self.to_remove:
            print("\nNothing to clean up.")
            return

        if dry_run:
            print(f"\n{'='*80}")
            print("DRY RUN - No files will be deleted")
            print(f"{'='*80}\n")
            print("Run with --execute to actually delete these items.")
            return

        print(f"\n{'='*80}")
        print("⚠️  EXECUTING CLEANUP")
        print(f"{'='*80}\n")

        if not skip_confirm:
            print("Items to be deleted:")
            for item in self.to_remove:
                print(f"  - {item['path'].relative_to(self.project_root)}")
            confirm = input("\nAre you sure? (yes/no): ")
            if confirm.lower() != 'yes':
                print("Cleanup cancelled.")
                return

        deleted_count = 0
        deleted_size = 0

        for item in self.to_remove:
            try:
                if item['path'].is_dir():
                    shutil.rmtree(item['path'])
                else:
                    item['path'].unlink()

                deleted_count += 1
                deleted_size += item['size']
                print(f"✓ Deleted: {item['path'].relative_to(self.project_root)}")

            except Exception as e:
                print(f"✗ Failed: {item['path'].relative_to(self.project_root)}: {e}")

        print(f"\n{'='*80}")
        print("Cleanup Complete!")
        print(f"{'='*80}")
        print(f"Deleted: {deleted_count} items")
        print(f"Freed: {deleted_size / (1024**2):.2f} MB ({deleted_size / (1024**3):.2f} GB)")
        print(f"{'='*80}\n")

    def _get_dir_size(self, path: Path) -> int:
        """Get total size of directory."""
        total = 0
        try:
            for entry in path.rglob('*'):
                if entry.is_file():
                    total += entry.stat().st_size
        except PermissionError:
            pass
        return total


def main():
    parser = argparse.ArgumentParser(description='Smart cleanup for docs and weights')
    parser.add_argument('--project_root', type=str,
                        default='/home/glcuser/projhighcv/bone_tumor',
                        help='Project root directory')
    parser.add_argument('--execute', action='store_true',
                        help='Actually delete files (default is dry-run)')
    parser.add_argument('--yes', action='store_true',
                        help='Skip confirmation prompt')
    parser.add_argument('--docs-only', action='store_true',
                        help='Only clean documentation')
    parser.add_argument('--weights-only', action='store_true',
                        help='Only clean model weights')
    args = parser.parse_args()

    cleaner = SmartCleaner(args.project_root)

    print(f"\n{'='*80}")
    print("Smart Cleanup - Documentation & Weights")
    print(f"{'='*80}")
    print(f"Project: {args.project_root}")
    print(f"Mode: {'EXECUTE' if args.execute else 'DRY RUN'}")
    print(f"{'='*80}\n")

    # Run analysis
    if not args.weights_only:
        cleaner.analyze_documentation()

    if not args.docs_only:
        cleaner.analyze_model_weights()
        cleaner.analyze_failed_experiments()
        cleaner.analyze_old_experiments()

    # Execute cleanup
    cleaner.print_summary()
    cleaner.execute_cleanup(dry_run=not args.execute, skip_confirm=args.yes)


if __name__ == '__main__':
    main()
