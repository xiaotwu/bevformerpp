# BEVFormer++ Training Pipeline

A modular Bird's-Eye View (BEV) perception system that integrates multi-view camera images for 3D object detection. The system uses an enhanced BEVFormer architecture with Memory Bank and Motion-Compensated ConvRNN for temporal aggregation.

## Features

- **BEVFormer Architecture**: Enhanced BEVFormer with spatial cross-attention
- **Temporal Aggregation**: Memory Bank + MC-ConvRNN for temporal feature fusion
- **Mixed Precision Training**: AMP support for efficient GPU memory usage
- **nuScenes Dataset**: Designed for nuScenes v1.0-mini dataset
- **Modular Design**: Clean separation of encoder, fusion, and detection components
- **Progress Tracking**: Real-time training visualization with tqdm

## Prerequisites

- **OS**: Windows / Linux / macOS
- **Python**: >= 3.10
- **GPU**: NVIDIA RTX 5070Ti with 16GB VRAM (testing)
- **CUDA**: 13.0

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/bevformerpp.git
cd bevformerpp
```

### 2. Install uv Package Manager

uv is a fast Python package installer and resolver. Install it first:

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Or via pip (all platforms):**
```bash
pip install uv
```

### 3. Create Virtual Environment

**Windows (PowerShell):**
```powershell
# Create virtual environment with Python 3.10+
uv venv --python 3.10

# Activate environment
.\.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
uv venv --python 3.10
.venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
# Create virtual environment
uv venv --python 3.10

# Activate environment
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
# Install project with all dependencies
uv pip install -e .

# Install with development dependencies (testing, linting)
uv pip install -e ".[dev]"
```

### 5. Verify Installation

```bash
python scripts/verify_installation.py
```

## Data Preparation

This project uses the **nuScenes** dataset (v1.0-mini split).

### Download nuScenes Mini

1. Register at [nuscenes.org](https://www.nuscenes.org/nuscenes#download)
2. Download the **"v1.0-mini"** split (~4GB)
3. Extract to the `data/` directory:

**Windows (PowerShell):**
```powershell
# If you have v1.0-mini.tgz in the project root
tar -xzf v1.0-mini.tgz -C data/
```

**macOS / Linux:**
```bash
tar -xzf v1.0-mini.tgz -C data/
```

### Expected Directory Structure

```
bevformerpp/
├── data/
│   ├── maps/
│   ├── samples/
│   │   ├── CAM_FRONT/
│   │   ├── CAM_FRONT_LEFT/
│   │   ├── CAM_FRONT_RIGHT/
│   │   ├── CAM_BACK/
│   │   ├── CAM_BACK_LEFT/
│   │   ├── CAM_BACK_RIGHT/
│   │   └── LIDAR_TOP/
│   ├── sweeps/
│   └── v1.0-mini/
│       ├── sample.json
│       ├── sample_data.json
│       ├── scene.json
│       └── ...
```

## Project Structure

```
bevformerpp/
├── modules/                    # Core model components
│   ├── bevformer.py           # Enhanced BEVFormer encoder
│   ├── camera_encoder.py      # Camera feature extractor
│   ├── lidar_encoder.py       # LiDAR PointPillars encoder
│   ├── fusion.py              # Spatial fusion module
│   ├── temporal_attention.py  # Temporal transformer
│   ├── mc_convrnn.py          # Motion-compensated ConvRNN
│   ├── memory_bank.py         # Temporal feature storage
│   ├── head.py                # Detection head
│   ├── nuscenes_dataset.py    # nuScenes data loader
│   ├── data_structures.py     # Data classes
│   └── utils.py               # Utility functions
├── configs/                    # Configuration files
│   ├── base_config.yaml       # Base configuration
│   ├── train_config.yaml      # Training configuration
│   └── eval_config.yaml       # Evaluation configuration
├── tests/                      # Test suite
│   ├── test_*.py              # Unit tests
│   └── property_tests/        # Property-based tests
├── scripts/                    # Utility scripts
│   └── verify_installation.py
├── docs/                       # Documentation
├── examples/                   # Demo scripts
├── main.ipynb                  # Training notebook
├── train.py                    # Training script
├── pyproject.toml              # Project configuration
└── README.md
```

## Usage

### Training (Jupyter Notebook)

Open and run `main.ipynb` in VS Code or JupyterLab:

```bash
# Using VS Code (recommended)
code main.ipynb

# Or JupyterLab
jupyter lab main.ipynb
```

### Training (Script)

```bash
python train.py --config configs/train_config.yaml
```

### Configuration

Key training parameters in `main.ipynb` or `configs/train_config.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 2 | Batch size (adjust based on VRAM) |
| `num_epochs` | 30 | Number of training epochs |
| `learning_rate` | 4e-5 | Initial learning rate |
| `use_amp` | True | Mixed precision training |
| `bev_h` / `bev_w` | 200 | BEV grid dimensions |
| `embed_dim` | 256 | Feature embedding dimension |

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=modules --cov-report=html

# Run property-based tests
pytest tests/ -m property
```

## GPU Memory Requirements

| Batch Size | Approx. VRAM | Recommended GPU |
|------------|--------------|-----------------|
| 1 | ~8GB | RTX 3070/4070 |
| 2 | ~12GB | RTX 3080/4080 |
| 4 | ~20GB | RTX 3090/4090 |

Enable AMP (`use_amp: True`) to reduce memory usage by ~30-40%.

## Troubleshooting

### CUDA Out of Memory

```python
# Reduce batch size in CONFIG
CONFIG['batch_size'] = 1

# Enable mixed precision
CONFIG['use_amp'] = True
```

### Module Import Errors

```bash
# Reinstall in development mode
uv pip install -e .
```

### uv Command Not Found

```bash
# Add to PATH or reinstall
pip install uv
```

## Model Architecture

```
Multi-View Images → Camera Encoder → BEV Features
                            ↓
                    Temporal Memory Bank
                            ↓
                    MC-ConvRNN Fusion
                            ↓
                    Detection Head → 3D Boxes
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [nuScenes](https://www.nuscenes.org/) dataset by Motional
- [BEVFormer](https://github.com/fundamentalvision/BEVFormer) architecture
- [PyTorch](https://pytorch.org/) and [timm](https://github.com/huggingface/pytorch-image-models)
