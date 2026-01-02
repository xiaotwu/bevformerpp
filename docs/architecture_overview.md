# Architecture Overview

## End-to-End Data Flow
1. **Data loading**: `modules.nuscenes_dataset` reads NuScenes samples (images, poses, intrinsics/extrinsics, optional LiDAR), builds temporal sequences, and generates CenterNet-style targets via `modules.target_generator`.
2. **Preprocessing**: `modules.preprocessing` handles image resizing/normalization and any LiDAR pre-filtering.
3. **LiDAR BEV encoder**: `modules.lidar_encoder` (PointPillars-style) converts point clouds to BEV feature maps.
4. **Camera BEV encoder**: `modules.camera_encoder` + `modules.bevformer` project multi-view images into BEV using spatial/temporal attention utilities in `modules.attention` and `modules.temporal_attention`.
5. **BEV fusion**: `modules.fusion` / `modules.bev_fusion_model` merge LiDAR and camera BEV features; `modules.neck` refines fused BEV.
6. **Temporal aggregation**: `modules.memory_bank`, `modules.convrnn`, and `modules.mc_convrnn` provide recurrent/warping-based history; `modules.temporal_attention` provides attention-based history.
7. **Detection head**: `modules.head` predicts class heatmaps and box regression; `modules.losses` supplies auxiliary loss helpers; `modules.metrics` computes evaluation metrics.
8. **Training loop**: `train.py` orchestrates dataloaders, models, loss, optimizer, curriculum schedule (hard_center → gaussian), and evaluation.

## BEV Grid and Coordinates
- Default BEV grid: `bev_h=200`, `bev_w=200` (configurable).
- Coordinate range defined in `modules.target_generator.TargetConfig` (`x_range`, `y_range`).
- All BEV tensors follow `[B, C, H, W]` with H (y) and W (x) consistent between LiDAR and camera branches.

## Entrypoints
- **Training**: `train.py` (only training entry).
- **Visualization / debug**: `main.ipynb` must import and call `train.py` or reuse its utilities; no standalone training logic in the notebook.

## Shared Assumptions
- Classification targets can run a 2-stage curriculum: Stage 1 `hard_center`, Stage 2 `gaussian` with relaxed overlap/radius.
- Losses operate in logits space (BCE with logits) with stage-specific `pos_weight`.
- Temporal modules optionally use memory bank or motion-compensated ConvRNN for history.


