# Module: losses

## 1) Purpose
- Additional loss definitions (focal/SmoothL1/combined) beyond the primary `DetectionLoss` in `head.py`.
- Provides legacy/auxiliary loss components.

## 2) Files
- `modules/losses.py` — Focal loss, Smooth L1, combined detection loss (legacy alias).

## 3) Public APIs
- `FocalLoss`, `SmoothL1Loss`, `CombinedDetectionLoss` (legacy).
- Inputs/Outputs: tensors matching head predictions/targets; see signatures.
- Called by: Older training paths; current training prefers `modules.head.DetectionLoss`.

## 4) Data Flow & Tensor Shapes
- Operates on class logits/targets `[B, K, H, W]` and bbox `[B, 7, H, W]` similar to `head.py`.

## 5) Configuration
- Alpha/beta weights defined in constructors; adjust only via init arguments.

## 6) Training vs Inference Behavior
- Loss-only; not used in inference.

## 7) Troubleshooting
- Ambiguity with `DetectionLoss`: ensure `train.py` imports `modules.head.DetectionLoss`.
- Shape mismatch: confirm targets/outputs align with CenterNet shapes.
- Deprecated paths: remove or gate legacy calls if both losses present.

## 8) Alignment to Proposal
- Provides legacy focal/combined loss; proposal prefers main detection loss. Status: **Partially implemented** (auxiliary only).

## 9) Minimal-change Extension Guide
- If extending, keep API stable; add new loss class and select via flag in `train.py` without altering existing loss defaults.***

