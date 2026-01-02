# Module: bevformer

## 1) Purpose
- Implements BEVFormer camera BEV encoder: projects multi-view images into BEV using spatial/temporal attention.
- Central to camera stream before fusion/detection.

## 2) Files
- `modules/bevformer.py` — BEVFormer encoder, integrates attention utilities and temporal modules.

## 3) Public APIs
- BEVFormer/EnhancedBEVFormer classes (see file for constructors).
- `forward_sequence(imgs, intrinsics, extrinsics, ego_pose, scene_tokens=None)` returns BEV feature sequences `[B, T, C, H, W]`.
- Called by: `train.py`, `tools/overfit_one_sample.py`, `bev_fusion_model.py`.

## 4) Data Flow & Tensor Shapes
- Inputs: images `[B, T, N_cam, 3, H, W]`, intrinsics/extrinsics `[B, T, N_cam, ...]`, ego poses `[B, T, 4, 4]`.
- Outputs: BEV feature maps `[B, T, C, H_bev, W_bev]`; often last timestep forwarded to head.

## 5) Configuration
- BEV size, embed dim, temporal method passed from `train.py` args/config.
- Uses attention parameters from `attention.py` and temporal settings from `temporal_attention.py` / `convrnn` / `mc_convrnn`.

## 6) Training vs Inference Behavior
- Same path; dropout/batchnorm follow mode. Temporal caches may be reused if exposed; otherwise no difference.

## 7) Troubleshooting
- Shape mismatch on camera dims: confirm dataset provides `[B, T, N_cam, ...]`.
- Pose/intrinsics misalignment: check `nuscenes_dataset` packing.
- BEV size mismatch with head: ensure `bev_h/w` consistent across encoders and head.

## 8) Alignment to Proposal
- Implements camera BEVFormer with temporal aggregation. Status: **Partially implemented** (core BEVFormer present; verify full temporal variants per proposal).

## 9) Minimal-change Extension Guide
- Add new temporal method switch inside `bevformer.py` constructor to choose different `temporal_attention` / `mc_convrnn`; keep interface unchanged.***

