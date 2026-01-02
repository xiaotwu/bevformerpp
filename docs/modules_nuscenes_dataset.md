# Module: nuscenes_dataset

## 1) Purpose
- Loads NuScenes samples, builds temporal sequences, packages camera/LiDAR tensors, and generates targets via `target_generator`.
- Provides collate functions for camera-only and fusion training.

## 2) Files
- `modules/nuscenes_dataset.py` — Dataset class, collate_fn, fusion collate, helper wrappers.

## 3) Public APIs
- `NuScenesDataset` (class): `__getitem__` returns dict with images, intrinsics/extrinsics, ego poses, annotations, optional LiDAR.
- `create_collate_fn(...)` / `create_fusion_collate_fn(...)`: return collate closures with `heatmap_mode`, `gaussian_overlap`, `min_radius`.
- Outputs (batch): `img`, `intrinsics`, `extrinsics`, `ego_pose`, `cls_targets`, `bbox_targets`, `reg_mask`, optional `lidar_points`, `lidar_mask`, `scene_tokens`.
- Called by: `train.py`, `tools/overfit_one_sample.py`.

## 4) Data Flow & Tensor Shapes
- Images: `[B, T, N_cam, 3, H, W]`; intrinsics/extrinsics: `[B, T, N_cam, ...]`; ego_pose: `[B, T, 4, 4]`.
- Targets: `cls_targets [B, K, H_bev, W_bev]`, `bbox_targets [B, 7, H_bev, W_bev]`, `reg_mask [B, 1, H_bev, W_bev]`.
- LiDAR (fusion): `lidar_points [B, max_points, 4]`, `lidar_mask [B, max_points]`.

## 5) Configuration
- BEV sizes, num_classes, heatmap_mode, gaussian_overlap, min_radius passed from `train.py` args/configs.
- max_points (fusion) set in collate creation.

## 6) Training vs Inference Behavior
- Same dataset/collate; generate_targets flag controls whether targets are produced. No inference-specific path inside dataset.

## 7) Troubleshooting
- `heatmap_mode` mismatch: check curriculum logs in `train.py` and `create_collate_fn` printout.
- Shape mismatches (camera dims): ensure `img_size` in collate matches model expectations.
- Missing annotations/keys: verify split and NuScenes root path.

## 8) Alignment to Proposal
- Implements data loading and target generation wrapper. Status: **Partially implemented** (covers NuScenes mini; extend for full dataset if required).

## 9) Minimal-change Extension Guide
- Add optional augmentations inside `__getitem__` or collate; guard with flags in `train.py`. Keep output dict keys stable.***

