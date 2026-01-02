# Module: preprocessing

## 1) Purpose
- Image/LiDAR preprocessing utilities (resize/normalize/augment).
- Used by dataset/collate to prepare inputs for encoders.

## 2) Files
- `modules/preprocessing.py` — Preprocessing helpers.

## 3) Public APIs
- Functions for image transforms and point filtering (see file).
- Inputs: raw images or point arrays; Outputs: processed tensors/arrays ready for model.
- Called by: `nuscenes_dataset.py`, possibly visualization scripts.

## 4) Data Flow & Tensor Shapes
- Image outputs typically `[3, H, W]` normalized; stacked to `[B, T, N_cam, 3, H, W]` by dataset.
- LiDAR outputs are filtered point arrays before BEV encoding.

## 5) Configuration
- Resize/normalize parameters set within functions or passed from dataset; align with `img_size` in collate.

## 6) Training vs Inference Behavior
- Augmentations may be training-only; ensure flags are respected. Otherwise same preprocessing.

## 7) Troubleshooting
- Color/normalization issues: check mean/std used.
- Size mismatch: ensure `img_size` matches model/backbone expectations.
- Augmentations leaking into eval: verify train/eval flags when calling helpers.

## 8) Alignment to Proposal
- Provides input preprocessing. Status: **Partially implemented** (confirm augmentation set vs proposal).

## 9) Minimal-change Extension Guide
- Add optional augmentations as separate functions in `preprocessing.py`; call them conditionally from dataset based on config flags.***

