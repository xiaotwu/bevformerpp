# Module: head

## 1) Purpose
- Detection head producing class logits and box regression from BEV features.
- Includes post-processing/decoder and primary `DetectionLoss`.

## 2) Files
- `modules/head.py` — Detection head, loss, post-processing utilities.

## 3) Public APIs
- `BEVHead` (class): returns `{'cls_score': logits, 'bbox_pred': boxes}`.
- `DetectionLoss` (class): BCE-with-logits classification + regression loss; accepts `pos_weight` (and neg weighting internally).
- Post-processing helpers for decoding boxes (see file).
- Called by: `train.py`, `bev_fusion_model.py`, `tools/overfit_one_sample.py`.

## 4) Data Flow & Tensor Shapes
- Input: BEV features `[B, C, H, W]`.
- Output: class logits `[B, K, H, W]`, bbox `[B, 7, H, W]`.
- Loss expects targets from `target_generator`: `cls_targets` `[B, K, H, W]`, `bbox_targets` `[B, 7, H, W]`, `reg_mask` `[B, 1, H, W]`.

## 5) Configuration
- `pos_weight` and curriculum-controlled weights set from `train.py` stage parameters.
- BEV sizes/num_classes passed from model constructors fed by `train.py`.

## 6) Training vs Inference Behavior
- Training: computes loss; inference/`predict` may run decoding/NMS.
- Otherwise same forward; logits always returned (sigmoid applied only for visualization).

## 7) Troubleshooting
- Non-binary targets: verify `target_generator` heatmap_mode matches stage (logged in `train.py`).
- Loss explosion: check `pos_weight` stage values; ensure logits not passed through sigmoid before loss.
- Shape mismatch: confirm `num_classes`, `bev_h/w` consistent between head and targets.

## 8) Alignment to Proposal
- Implements CenterNet-style BEV detection head. Status: **Partially implemented** (core head/loss present; confirm decoder/NMS vs proposal).

## 9) Minimal-change Extension Guide
- To adjust loss strength, add small parameters to `DetectionLoss` (already accepts `pos_weight`); wire additional knobs via `train.py` without altering head forward.***

