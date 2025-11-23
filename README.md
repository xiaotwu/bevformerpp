# BevFormer++

The model is designed to work with the **nuScenes** dataset.

## Prerequisites
-   **OS**: Windows / Linux
-   **Python**: >= 3.12
-   **Package Manager**: `uv` (Recommended for fast dependency management)
-   **Git**

## Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/xiaotwu/bevformerpp.git
    cd bevformerpp
    ```

2.  **Create a Virtual Environment**
    We recommend using `uv` for creating the environment and installing dependencies.
    ```bash
    # Install uv if you haven't already
    pip install uv

    # Create virtual environment
    uv venv
    ```

3.  **Install Dependencies**
    Activate the environment (if not using `uv run`) and install the required packages.
    ```bash
    # On Windows
    .venv\Scripts\activate

    # Install requirements
    uv pip install -r requirements.txt
    ```

## Data Preparation

This project uses the **nuScenes** dataset (v1.0-mini split).

1.  **Download Data**:
    -   Register at [nuscenes.org](https://www.nuscenes.org/nuscenes#download).
    -   Download the **"mini"** split.

2.  **Extract Data**:
    -   Extract the downloaded archive into the `data/` directory of this project.
    -   Ensure the directory structure looks like this:
        ```text
        bevformerpp/
        ├── data/
        │   └── nuscenes/
        │       ├── maps/
        │       ├── samples/
        │       ├── sweeps/
        │       └── v1.0-mini/
        ```
    -   **Note**: The training script expects the `v1.0-mini` folder to be found within `data/nuscenes` or `data/` depending on extraction. The script is configured to look in `data`.

## Training

To train the model, use the provided `train.py` script. This script handles:
-   Loading the NuScenes dataset.
-   Splitting data into **Train (70%)**, **Validation (15%)**, and **Test (15%)**.
-   Running the training loop with **Gaussian Focal Loss** (for classification) and **L1 Loss** (for bounding box regression).
-   Saving checkpoints to `checkpoints/`.

**Run Training:**
```bash
uv run python train.py
```

**Configuration:**
You can modify hyperparameters in `train.py`:
-   `BATCH_SIZE`: Default 1
-   `NUM_EPOCHS`: Default 10
-   `LR`: Learning rate (Default 2e-5)

## Evaluation & Testing

### Automated Evaluation
The `train.py` script automatically runs evaluation on the **Test Set** after training completes. It reports the average loss on the test set.

### Visualization
To visualize the model's predictions and internal BEV representations:

1.  **Open the Notebook**:
    ```bash
    uv run jupyter lab main.ipynb
    ```
2.  **Run Cells**:
    -   The notebook loads the trained model from `checkpoints/latest.pth`.
    -   It visualizes the **BEV Feature Map**.
    -   It projects BEV grid points onto camera images to demonstrate **Spatial Correspondence**.

## Project Structure
```text
bevformerpp/
├── modules/
│   ├── backbone.py       # ResNet50 Backbone
│   ├── neck.py           # FPN Neck
│   ├── attention.py      # Spatial & Temporal Attention
│   ├── convrnn.py        # ConvGRU for temporal fusion
│   ├── memory_bank.py    # History storage
│   ├── bevformer.py      # Main Model Architecture
│   ├── head.py           # Detection Head & Loss
│   └── dataset.py        # NuScenes & Dummy Data Loaders
├── data/                 # Dataset directory
├── checkpoints/          # Saved models
├── main.ipynb            # Visualization Notebook
├── train.py              # Training Script
└── requirements.txt      # Dependencies
```
