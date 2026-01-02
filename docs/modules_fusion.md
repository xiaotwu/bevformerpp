# Module: fusion

## 1) Purpose
- Fusion utilities for combining LiDAR and camera BEV features.
- Used inside higher-level fusion models to merge modality-specific BEV maps.

## 2) Files
- `modules/fusion.py` — Functions/layers for BEV fusion (concat/add/MLP/attention as implemented).

## 3) Public APIs
- Fusion helper functions/classes (see file) that take two BEV tensors and return a fused BEV.
- Inputs: BEV feature maps `[B, C_l, H, W]` and `[B, C_c, H, W]`.
- Outputs: fused BEV `[B, C_f, H, W]`.
- Called by: `bev_fusion_model.py`, possibly `neck.py`.

## 4) Data Flow & Tensor Shapes
- Receives aligned BEV features; performs channel-wise fusion; output feeds detection head via neck.

## 5) Configuration
- Fusion type/hidden channels set in constructor or function args; expose via config flags in `train.py` if needed.

## 6) Training vs Inference Behavior
- Same computation; dropout only if present. No special inference path.

## 7) Troubleshooting
- Channel mismatch: ensure LiDAR and camera BEV channels match fusion expectations.
- BEV size mismatch: align `bev_h/w` across encoders and `BEVGridConfig`.
- Device mismatch: fuse tensors on same device.

## 8) Alignment to Proposal
- Provides BEV-space fusion. Status: **Partially implemented** (confirm fusion type matches proposal; extend if cross-attention required).

## 9) Minimal-change Extension Guide
- Add new fusion op in `fusion.py` and select via a small switch in `bev_fusion_model.py`; avoid changing shapes seen by head.***

