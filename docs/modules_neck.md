# Module: neck

## 1) Purpose
- Feature refinement neck operating on BEV feature maps before the detection head or fusion.
- May implement FPN-style or simple conv blocks.

## 2) Files
- `modules/neck.py` — Neck layers for BEV features.

## 3) Public APIs
- Neck classes/functions (see file) producing refined BEV `[B, C, H, W]`.
- Called by: `bev_fusion_model.py`, possibly camera/LiDAR pipelines before head.

## 4) Data Flow & Tensor Shapes
- Input: BEV feature `[B, C_in, H, W]`.
- Output: BEV feature `[B, C_out, H, W]` aligned with head expectations.

## 5) Configuration
- Channels/blocks defined in constructor; align with backbone output and head input; set via configs/train args if exposed.

## 6) Training vs Inference Behavior
- Same computation; dropout/BN follow mode. No special inference path.

## 7) Troubleshooting
- Channel mismatch with head: confirm neck output channels equal head input channels.
- BEV size mismatch: verify stride/padding keeps H/W consistent across fusion streams.
- Initialization issues: ensure weights initialized or loaded as intended.

## 8) Alignment to Proposal
- Provides BEV neck refinement. Status: **Partially implemented** (confirm neck design vs proposal).

## 9) Minimal-change Extension Guide
- Add alternate neck variant in `neck.py`; select via a small flag in `bev_fusion_model.py`/`train.py` without changing interfaces.***

