# Module: lidar_encoder

## 1) Purpose
- Encodes LiDAR point clouds into BEV feature maps (PointPillars-style).
- Provides LiDAR stream for fusion or LiDAR-only detection.

## 2) Files
- `modules/lidar_encoder.py` — Pillar/voxel encoding and scatter to BEV.

## 3) Public APIs
- Encoder classes/functions (see file) converting raw points to BEV features `[B, C, H, W]`.
- Inputs: point clouds (batched) with fields `[x, y, z, intensity]`.
- Called by: `bev_fusion_model.py`, potentially LiDAR-only pipelines.

## 4) Data Flow & Tensor Shapes
- Points → pillar/voxel features → BEV raster `[B, C, H_bev, W_bev]` aligned to grid from `BEVGridConfig`.

## 5) Configuration
- Pillar sizes, grid extents, channel dims set in constructors; should be driven by configs passed from `train.py`.

## 6) Training vs Inference Behavior
- Same computation; dropout/BN follow mode. No special inference-only branches.

## 7) Troubleshooting
- Empty BEV maps: check point filtering/grid ranges.
- Shape mismatch with fusion: ensure BEV grid matches camera stream.
- Performance issues: verify pillar limits/batch size; enable CUDA ops if available.

## 8) Alignment to Proposal
- Implements LiDAR BEV encoder. Status: **Partially implemented** (verify exact PointPillars variant vs proposal).

## 9) Minimal-change Extension Guide
- Add alternate pillar settings in `lidar_encoder.py` and select via config flag; avoid changing output shape without updating fusion/head.***

