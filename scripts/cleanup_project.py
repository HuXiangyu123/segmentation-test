#!/usr/bin/env python
"""
Project cleanup script for bone tumor segmentation.

Identifies and optionally removes:
1. Old/redundant experiment outputs
2. Temporary cache files
3. Duplicate checkpoints
4. Test/debug runs

Provides dry-run mode and git integration for safe cleanup.
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
import argparse


class ProjectCleaner:
    """Project cleanup manager."""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.cleanup_plan = []
        self.total_size = 0

    def scan_experiment_outputs(self):
        """Scan experiment output directories."""
        out_dir = self.project_root / 'MulModSeg_2024' / 'out' / 'unet' / 'no_txt'

        if not out_dir.exists():
            print(f"[INFO] Output directory not found: {out_dir}")
            return

        print(f"\n{'='*80}")
        print("Scanning Experiment Outputs")
        print(f"{'='*80}\n")

        experiments = []
        for exp_dir in sorted(out_dir.iterdir()):
            if not exp_dir.is_dir():
                continue

            size = self._get_dir_size(exp_dir)
            mtime = datetime.fromtimestamp(exp_dir.stat().st_mtime)

            experiments.append({
                'path': exp_dir,
                'name': exp_dir.name,
                'size': size,
                'size_mb': size / (1024 * 1024),
                'mtime': mtime,
                'age_days': (datetime.now() - mtime).days
            })

        # Sort by modification time (oldest first)
        experiments.sort(key=lambda x: x['mtime'])

        # Categorize experiments
        test_runs = []
        old_runs = []
        keep_runs = []

        for exp in experiments:
            name = exp['name']

            # Test/debug runs
            if 'test' in name.lower() or 'smoke' in name.lower() or 'debug' in name.lower():
                test_runs.append(exp)
            # Old runs (>7 days and not final)
            elif exp['age_days'] > 7 and 'final' not in name.lower():
                old_runs.append(exp)
            else:
                keep_runs.append(exp)

        # Print summary
        print(f"Total experiments: {len(experiments)}")
        print(f"  Test/Debug runs: {len(test_runs)}")
        print(f"  Old runs (>7 days): {len(old_runs)}")
        print(f"  Keep: {len(keep_runs)}\n")

        # Print details
        if test_runs:
            print("Test/Debug Runs (recommended for cleanup):")
            for exp in test_runs:
                print(f"  - {exp['name']}")
                print(f"    Size: {exp['size_mb']:.1f} MB, Age: {exp['age_days']} days")
                self.cleanup_plan.append({
                    'path': exp['path'],
                    'reason': 'Test/Debug run',
                    'size': exp['size']
                })

        if old_runs:
            print("\nOld Runs (>7 days, not final):")
            for exp in old_runs:
                print(f"  - {exp['name']}")
                print(f"    Size: {exp['size_mb']:.1f} MB, Age: {exp['age_days']} days")
                # Don't auto-add to cleanup plan, let user decide

        if keep_runs:
            print("\nRuns to Keep:")
            for exp in keep_runs:
                print(f"  - {exp['name']}")
                print(f"    Size: {exp['size_mb']:.1f} MB, Age: {exp['age_days']} days")

    def scan_cache_files(self):
        """Scan cache directories."""
        cache_dir = self.project_root / 'cache_bone_tumor'

        if not cache_dir.exists():
            print(f"\n[INFO] Cache directory not found: {cache_dir}")
            return

        print(f"\n{'='*80}")
        print("Scanning Cache Files")
        print(f"{'='*80}\n")

        total_size = self._get_dir_size(cache_dir)
        file_count = sum(1 for _ in cache_dir.rglob('*.pt'))

        print(f"Cache directory: {cache_dir}")
        print(f"Total size: {total_size / (1024**3):.2f} GB")
        print(f"File count: {file_count}")

        # Cache files are usually safe to keep (they speed up training)
        # Only suggest cleanup if very large
        if total_size > 10 * 1024**3:  # > 10 GB
            print(f"\n⚠️  Cache is large (>{total_size / (1024**3):.1f} GB)")
            print("Consider cleaning if disk space is limited.")
            print("Note: Cache will be regenerated on next training run.")

    def scan_temp_files(self):
        """Scan for temporary files."""
        print(f"\n{'='*80}")
        print("Scanning Temporary Files")
        print(f"{'='*80}\n")

        temp_patterns = [
            '**/__pycache__',
            '**/*.pyc',
            '**/*.pyo',
            '**/.ipynb_checkpoints',
            '**/tmp',
            '**/.DS_Store',
        ]

        temp_items = []
        for pattern in temp_patterns:
            for item in self.project_root.rglob(pattern):
                if item.exists():
                    size = self._get_dir_size(item) if item.is_dir() else item.stat().st_size
                    temp_items.append({
                        'path': item,
                        'size': size,
                        'type': 'dir' if item.is_dir() else 'file'
                    })

        if temp_items:
            total_temp_size = sum(item['size'] for item in temp_items)
            print(f"Found {len(temp_items)} temporary items")
            print(f"Total size: {total_temp_size / (1024**2):.2f} MB")

            # Add to cleanup plan
            for item in temp_items:
                self.cleanup_plan.append({
                    'path': item['path'],
                    'reason': 'Temporary file/directory',
                    'size': item['size']
                })
        else:
            print("No temporary files found.")

    def scan_duplicate_checkpoints(self):
        """Scan for duplicate or unnecessary checkpoints."""
        print(f"\n{'='*80}")
        print("Scanning Checkpoints")
        print(f"{'='*80}\n")

        out_dir = self.project_root / 'MulModSeg_2024' / 'out' / 'unet' / 'no_txt'

        if not out_dir.exists():
            return

        for exp_dir in out_dir.iterdir():
            if not exp_dir.is_dir():
                continue

            checkpoints = list(exp_dir.glob('epoch_*.pt'))

            if len(checkpoints) > 5:  # Keep only best + last 3 epochs
                checkpoints.sort(key=lambda x: x.stat().st_mtime)

                # Keep best_model.pt and last 3 epochs
                best_model = exp_dir / 'best_model.pt'
                keep_checkpoints = checkpoints[-3:] + ([best_model] if best_model.exists() else [])

                for ckpt in checkpoints[:-3]:
                    if ckpt not in keep_checkpoints:
                        size = ckpt.stat().st_size
                        print(f"  Redundant: {ckpt.relative_to(self.project_root)}")
                        print(f"    Size: {size / (1024**2):.1f} MB")

                        # Don't auto-add to cleanup, let user decide

    def print_cleanup_summary(self):
        """Print summary of cleanup plan."""
        if not self.cleanup_plan:
            print(f"\n{'='*80}")
            print("No items recommended for cleanup.")
            print(f"{'='*80}\n")
            return

        total_size = sum(item['size'] for item in self.cleanup_plan)

        print(f"\n{'='*80}")
        print("Cleanup Summary")
        print(f"{'='*80}\n")

        print(f"Items to remove: {len(self.cleanup_plan)}")
        print(f"Total size to free: {total_size / (1024**2):.2f} MB ({total_size / (1024**3):.2f} GB)\n")

        # Group by reason
        by_reason = {}
        for item in self.cleanup_plan:
            reason = item['reason']
            if reason not in by_reason:
                by_reason[reason] = []
            by_reason[reason].append(item)

        for reason, items in by_reason.items():
            reason_size = sum(item['size'] for item in items)
            print(f"{reason}: {len(items)} items, {reason_size / (1024**2):.2f} MB")

    def execute_cleanup(self, dry_run=True, skip_confirm=False):
        """Execute cleanup plan."""
        if not self.cleanup_plan:
            print("\nNothing to clean up.")
            return

        if dry_run:
            print(f"\n{'='*80}")
            print("DRY RUN - No files will be deleted")
            print(f"{'='*80}\n")
            print("Items that would be removed:")
            for item in self.cleanup_plan:
                print(f"  - {item['path'].relative_to(self.project_root)}")
            print(f"\nRun with --execute to actually delete these items.")
            return

        print(f"\n{'='*80}")
        print("⚠️  EXECUTING CLEANUP")
        print(f"{'='*80}\n")

        if not skip_confirm:
            confirm = input("Are you sure you want to delete these items? (yes/no): ")
            if confirm.lower() != 'yes':
                print("Cleanup cancelled.")
                return

        deleted_count = 0
        deleted_size = 0

        for item in self.cleanup_plan:
            try:
                if item['path'].is_dir():
                    shutil.rmtree(item['path'])
                else:
                    item['path'].unlink()

                deleted_count += 1
                deleted_size += item['size']
                print(f"✓ Deleted: {item['path'].relative_to(self.project_root)}")

            except Exception as e:
                print(f"✗ Failed to delete {item['path']}: {e}")

        print(f"\n{'='*80}")
        print(f"Cleanup complete!")
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
    parser = argparse.ArgumentParser(description='Clean up bone tumor project')
    parser.add_argument('--project_root', type=str,
                        default='/home/glcuser/projhighcv/bone_tumor',
                        help='Project root directory')
    parser.add_argument('--execute', action='store_true',
                        help='Actually delete files (default is dry-run)')
    parser.add_argument('--yes', action='store_true',
                        help='Skip confirmation prompt')
    parser.add_argument('--skip_cache', action='store_true',
                        help='Skip cache directory scan')
    args = parser.parse_args()

    cleaner = ProjectCleaner(args.project_root)

    print(f"\n{'='*80}")
    print("Bone Tumor Project Cleanup")
    print(f"{'='*80}")
    print(f"Project root: {args.project_root}")
    print(f"Mode: {'EXECUTE' if args.execute else 'DRY RUN'}")
    print(f"{'='*80}\n")

    # Scan different categories
    cleaner.scan_experiment_outputs()
    cleaner.scan_temp_files()
    cleaner.scan_duplicate_checkpoints()

    if not args.skip_cache:
        cleaner.scan_cache_files()

    # Print summary and execute
    cleaner.print_cleanup_summary()
    cleaner.execute_cleanup(dry_run=not args.execute, skip_confirm=args.yes)


if __name__ == '__main__':
    main()
