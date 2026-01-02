# Module: data_structures

## 1) Purpose
- Defines lightweight classes/structs (e.g., `Box3D`, BEV grid config) shared across modules.
- Central type definitions used by target generation, decoding, and post-processing.

## 2) Files
- `modules/data_structures.py` — Data classes for boxes, BEV grid config, and related helpers.

## 3) Public APIs
- `Box3D` (class): box fields, transforms.
- `BEVGridConfig` (class): grid size/range helpers.
- Used by: `target_generator.py`, `bev_fusion_model.py`, `head.py`, `metrics.py`, `train.py`.

## 4) Data Flow & Tensor Shapes
- Box tensors typically `[N, 7]` (x, y, z, w, l, h, yaw) or dict-based.
- BEV grid config provides `bev_h`, `bev_w`, and ranges for projection and decoding.

## 5) Configuration
- Grid parameters sourced from configs/train args and passed into constructors.

## 6) Training vs Inference Behavior
- Pure data classes; no mode differences.

## 7) Troubleshooting
- Incorrect coordinate frame: confirm use of same range in `target_generator` and decoders.
- Missing fields when constructing `Box3D`: check caller passes all required components.
- dtype/device: ensure tensors are float and on correct device when converting.

## 8) Alignment to Proposal
- Provides shared geometric primitives. Status: **Fully implemented** for current usage.

## 9) Minimal-change Extension Guide
- Add new fields/methods to `Box3D` in `data_structures.py`; keep backward-compatible defaults.***

