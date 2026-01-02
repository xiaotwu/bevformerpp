#!/usr/bin/env python3
"""Verify that all dependencies are correctly installed."""

import sys
from pathlib import Path


def check_import(module_name, package_name=None):
    """Check if a module can be imported."""
    if package_name is None:
        package_name = module_name
    
    try:
        __import__(module_name)
        print(f"✓ {package_name}")
        return True
    except ImportError as e:
        print(f"✗ {package_name}: {e}")
        return False


def main():
    """Main verification function."""
    print("BEV Fusion System - Installation Verification")
    print("=" * 60)
    
    # Check Python version
    print(f"\nPython version: {sys.version}")
    if sys.version_info < (3, 8):
        print("✗ Python 3.8 or higher is required")
        return False
    else:
        print("✓ Python version OK")
    
    # Check core dependencies
    print("\nCore Dependencies:")
    print("-" * 60)
    
    all_ok = True
    dependencies = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("numpy", "NumPy"),
        ("matplotlib", "Matplotlib"),
        ("cv2", "OpenCV"),
        ("timm", "timm"),
        ("einops", "einops"),
        ("pyquaternion", "pyquaternion"),
        ("nuscenes", "nuScenes devkit"),
        ("yaml", "PyYAML"),
        ("tqdm", "tqdm"),
        ("tensorboard", "TensorBoard"),
        ("scipy", "SciPy"),
    ]
    
    for module, name in dependencies:
        if not check_import(module, name):
            all_ok = False
    
    # Check development dependencies
    print("\nDevelopment Dependencies:")
    print("-" * 60)
    
    dev_dependencies = [
        ("pytest", "pytest"),
        ("hypothesis", "Hypothesis"),
        ("black", "Black"),
        ("flake8", "Flake8"),
    ]
    
    for module, name in dev_dependencies:
        check_import(module, name)  # Don't fail on dev dependencies
    
    # Check PyTorch CUDA
    print("\nPyTorch Configuration:")
    print("-" * 60)
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    except Exception as e:
        print(f"Error checking PyTorch: {e}")
        all_ok = False
    
    # Check project structure
    print("\nProject Structure:")
    print("-" * 60)
    
    required_dirs = [
        "modules",
        "configs",
        "tests",
        "scripts",
        "data",
    ]
    
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"✓ {dir_name}/")
        else:
            print(f"✗ {dir_name}/ (missing)")
            all_ok = False
    
    # Check configuration files
    print("\nConfiguration Files:")
    print("-" * 60)
    
    config_files = [
        "configs/base_config.yaml",
        "configs/train_config.yaml",
        "configs/eval_config.yaml",
    ]
    
    for config_file in config_files:
        if Path(config_file).exists():
            print(f"✓ {config_file}")
        else:
            print(f"✗ {config_file} (missing)")
            all_ok = False
    
    # Check dataset
    print("\nDataset:")
    print("-" * 60)
    
    dataset_path = Path("data/v1.0-mini")
    if dataset_path.exists():
        print(f"✓ nuScenes v1.0-mini dataset found")
        
        # Check for key files
        key_files = [
            "sample.json",
            "sample_data.json",
            "scene.json",
            "sensor.json",
        ]
        
        for file_name in key_files:
            file_path = dataset_path / file_name
            if file_path.exists():
                print(f"  ✓ {file_name}")
            else:
                print(f"  ✗ {file_name} (missing)")
    else:
        print(f"✗ nuScenes v1.0-mini dataset not found")
        print(f"  Please extract v1.0-mini.tgz to data/")
    
    # Summary
    print("\n" + "=" * 60)
    if all_ok:
        print("✓ All core dependencies are installed correctly!")
        print("\nYou can now:")
        print("  - Run tests: pytest tests/")
        print("  - Start training: python train.py --config configs/train_config.yaml")
    else:
        print("✗ Some dependencies are missing or incorrectly installed")
        print("\nPlease run: python scripts/setup_env.py")
    print("=" * 60)
    
    return all_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
