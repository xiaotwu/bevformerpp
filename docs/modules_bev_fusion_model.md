# Module: bev_fusion_model

## 1) Purpose
- End-to-end multi-modal BEV fusion model combining LiDAR BEV and camera BEV streams.
- Provides unified forward for fusion training entry (`train.py` with `--use_fusion`).

## 2) Files
- `modules/bev_fusion_model.py` — Model wrapper wiring LiDAR encoder, camera encoder/BEVFormer, fusion neck, and detection head.

## 3) Public APIs
- `BEVFusionModel` (class): constructor wires components; `forward` / `predict` methods.
- Inputs: images `[B, T, N_cam, C, H, W]`, LiDAR points, calibration/extrinsics.
- Outputs: detection predictions and/or decoded boxes depending on call.
- Called by: `train.py` (fusion path).

## 4) Data Flow & Tensor Shapes
- Camera stream → BEV features `[B, C, H_bev, W_bev]`.
- LiDAR stream → BEV features `[B, C, H_bev, W_bev]` via `lidar_encoder`.
- Fusion (concat/add/neck) → fused BEV `[B, C_fused, H_bev, W_bev]`.
- Detection head consumes fused BEV and outputs class logits `[B, K, H, W]` and bbox `[B, 7, H, W]`.

## 5) Configuration
- BEV grid sizes, channels, and fusion method set in `train.py` args/configs; internal defaults reside in constructors.

## 6) Training vs Inference Behavior
- Same forward path; evaluation may run decoder/NMS in `predict`. No other difference.

## 7) Troubleshooting
- Shape mismatch between LiDAR and camera BEV: ensure `BEVGridConfig` matches both encoders.
- Missing keys (intrinsics/extrinsics) from dataset: check `nuscenes_dataset` collate.
- Device mismatch: move all submodules and inputs to same device (CUDA/CPU).

## 8) Alignment to Proposal
- Implements BEV-space multi-modal fusion. Status: **Partially implemented** (fusion mechanism present; verify aligns with proposal fusion variant if x-attn required).

## 9) Minimal-change Extension Guide
- Add fusion strategy switch inside `BEVFusionModel.forward` (e.g., concat vs add vs x-attn) with a new flag; keep existing paths untouched. Add config flag and route from `train.py`.***

