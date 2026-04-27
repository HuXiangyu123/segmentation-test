#!/usr/bin/env python
"""
Environment verification script for Bone Tumor Segmentation Project.

Checks if all required packages are installed and working correctly.
"""

import sys
from typing import Tuple, List


def check_package(package_name: str, min_version: str = None) -> Tuple[bool, str]:
    """
    Check if a package is installed and optionally verify version.

    Args:
        package_name: Name of the package to check
        min_version: Minimum required version (optional)

    Returns:
        (success, message) tuple
    """
    try:
        module = __import__(package_name)
        version = getattr(module, '__version__', 'unknown')

        if min_version and version != 'unknown':
            from packaging import version as pkg_version
            if pkg_version.parse(version) < pkg_version.parse(min_version):
                return False, f"❌ {package_name}: {version} (requires >={min_version})"

        return True, f"✅ {package_name}: {version}"
    except ImportError:
        return False, f"❌ {package_name}: NOT INSTALLED"
    except Exception as e:
        return False, f"❌ {package_name}: ERROR ({str(e)})"


def check_cuda() -> Tuple[bool, str]:
    """Check CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            return True, f"✅ CUDA: Available (version {cuda_version}, device: {device_name})"
        else:
            return False, "⚠️  CUDA: Not available (CPU only mode)"
    except Exception as e:
        return False, f"❌ CUDA: ERROR ({str(e)})"


def main():
    """Run environment verification."""
    print("="*80)
    print("Bone Tumor Segmentation - Environment Verification")
    print("="*80)
    print()

    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"Python: {python_version}")
    if sys.version_info < (3, 10):
        print("⚠️  Warning: Python 3.10+ is recommended")
    print()

    # Required packages
    packages = [
        ("torch", "2.0.0"),
        ("monai", "1.3.0"),
        ("nibabel", "5.0.0"),
        ("numpy", "1.24.0"),
        ("scipy", "1.8.0"),
        ("pandas", "2.0.0"),
        ("matplotlib", "3.7.0"),
        ("tensorboardX", "2.6.0"),
        ("tqdm", "4.65.0"),
        ("yaml", "6.0.0"),
        ("h5py", "3.8.0"),
    ]

    print("Checking required packages:")
    print("-" * 80)

    results: List[Tuple[bool, str]] = []
    for package, min_ver in packages:
        success, message = check_package(package, min_ver)
        results.append((success, message))
        print(message)

    print()

    # Check CUDA
    print("Checking CUDA:")
    print("-" * 80)
    cuda_success, cuda_message = check_cuda()
    print(cuda_message)
    print()

    # Check MONAI components
    print("Checking MONAI components:")
    print("-" * 80)
    try:
        from monai.data import PersistentDataset, CacheDataset
        from monai.transforms import LoadImage, Compose
        from monai.networks.nets import UNet
        from monai.losses import DiceCELoss
        from monai.metrics import compute_dice
        print("✅ MONAI data loaders")
        print("✅ MONAI transforms")
        print("✅ MONAI networks")
        print("✅ MONAI losses")
        print("✅ MONAI metrics")
    except ImportError as e:
        print(f"❌ MONAI components: {str(e)}")
    print()

    # Summary
    print("="*80)
    print("Summary:")
    print("="*80)

    all_success = all(success for success, _ in results)

    if all_success and cuda_success:
        print("🎉 All checks passed! Environment is ready.")
        print()
        print("Next steps:")
        print("  1. cd /home/glcuser/projhighcv/bone_tumor/MulModSeg_2024")
        print("  2. python train.py --help")
        return 0
    elif all_success and not cuda_success:
        print("⚠️  All packages installed, but CUDA is not available.")
        print("   Training will be slow on CPU. Consider installing CUDA.")
        return 1
    else:
        print("❌ Some packages are missing or have incorrect versions.")
        print()
        print("To fix:")
        print("  pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
