# Module: target_generator

## 1) Purpose
- Generates CenterNet-style targets (class heatmaps, box regression, reg mask) for BEV detection.
- Supports `gaussian`, `hard_center`, and `hard_center_radius1` modes used in the curriculum.

## 2) Files
- `modules/target_generator.py` — TargetConfig, TargetGenerator, heatmap drawing helpers.

## 3) Public APIs
- `TargetConfig` (dataclass): num_classes, bev_h/w, ranges, heatmap_mode, gaussian_overlap, min_radius.
- `TargetGenerator.generate(annotations)` → dict with `cls_targets`, `bbox_targets`, `reg_mask` (torch tensors).
- Called by: `nuscenes_dataset.collate_fn`, `train.py` via collate creation.

## 4) Data Flow & Tensor Shapes
- Inputs: list of `Box3D` annotations.
- Outputs:
  - `cls_targets`: `[K, H, W]` (stacked later to `[B, K, H, W]`).
  - `bbox_targets`: `[7, H, W]` (dx, dy, z, log w, log l, log h, yaw).
  - `reg_mask`: `[1, H, W]`.

## 5) Configuration
- `heatmap_mode`, `gaussian_overlap`, `min_radius` set via `TargetConfig` from collate parameters (`train.py` curriculum stages).

## 6) Training vs Inference Behavior
- Generation is training-time; not used in inference. No mode differences otherwise.

## 7) Troubleshooting
- Non-binary targets in hard_center: ensure `heatmap_mode` passed correctly; check curriculum logs.
- Index clamping errors: verify annotations within BEV range (`x_range`, `y_range`).
- dtype/device mismatch: outputs are torch tensors; ensure caller moves to correct device.

## 8) Alignment to Proposal
- Implements ground-truth target creation per proposal with hard-center option. Status: **Fully implemented** (current modes).

## 9) Minimal-change Extension Guide
- Add new `heatmap_mode` branch inside `TargetGenerator.generate`; expose via `TargetConfig` and collate parameters without altering existing modes.***

