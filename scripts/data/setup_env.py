#!/usr/bin/env python3
"""Setup script for BEV Fusion System environment."""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a shell command and handle errors."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("BEV Fusion System - Environment Setup")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"Python version: {sys.version}")
    
    # Install dependencies
    print("\nInstalling dependencies...")
    
    # Install core dependencies
    if not run_command(
        "pip install -e .",
        "Installing core dependencies from pyproject.toml"
    ):
        print("Warning: Failed to install core dependencies")
    
    # Install development dependencies
    if not run_command(
        "pip install -e .[dev]",
        "Installing development dependencies"
    ):
        print("Warning: Failed to install development dependencies")
    
    # Verify PyTorch installation
    print("\nVerifying PyTorch installation...")
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
    except ImportError:
        print("Warning: PyTorch not installed correctly")
    
    # Create necessary directories
    print("\nCreating necessary directories...")
    directories = [
        "checkpoints",
        "outputs",
        "outputs/evaluation",
        "outputs/visualization",
        "runs",
        "logs"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"Created: {dir_path}")
    
    # Extract dataset if needed
    print("\nChecking dataset...")
    data_dir = Path("data")
    if not (data_dir / "v1.0-mini").exists():
        if Path("v1.0-mini.tgz").exists():
            print("Extracting nuScenes v1.0-mini dataset...")
            if not run_command(
                "tar -xzf v1.0-mini.tgz -C data/",
                "Extracting dataset"
            ):
                print("Warning: Failed to extract dataset")
        else:
            print("Warning: v1.0-mini.tgz not found. Please download the dataset.")
    else:
        print("Dataset already extracted")
    
    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Verify dataset is in ./data/v1.0-mini/")
    print("2. Run tests: pytest tests/")
    print("3. Start training: python train.py --config configs/train_config.yaml")


if __name__ == "__main__":
    main()
