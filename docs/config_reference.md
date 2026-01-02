# Config Reference

This document lists configuration files under `configs/`, their purpose, and which modules read them.

## Files
- `configs/base_config.yaml`
  - Purpose: Default model/data/training hyperparameters.
  - Used by: `config_loader.py`, `train.py`, downstream modules via parsed args (BEV sizes, classes, learning rate).
- `configs/train_config.yaml`
  - Purpose: Training-specific overrides (epochs, batch size, optimizer).
  - Used by: `config_loader.py`, `train.py`.
- `configs/eval_config.yaml`
  - Purpose: Evaluation settings (checkpoint path, NMS thresholds).
  - Used by: `config_loader.py`, `train.py` during eval phases.
- `configs/__init__.py`
  - Purpose: Package marker; no runtime fields.
- `configs/config_loader.py`
  - Purpose: YAML loader/merger; exposes helpers to read configs into Python dict/namespace.
  - Used by: `train.py`, notebooks that proxy training.

## Notes
- Model and data parameters should be passed from configs into `train.py` and then into modules (BEV sizes, number of classes, curriculum settings).
- Avoid hard-coding paths or hyperparameters inside modules; prefer config-driven values.


