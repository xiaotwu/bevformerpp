# Module: metrics

## 1) Purpose
- Computes detection metrics (mAP/NDS components) and utility scoring functions.
- Used during validation/test phases.

## 2) Files
- `modules/metrics.py` — Metric computation helpers and constants (e.g., class lists).

## 3) Public APIs
- Metric classes/functions (see file) taking predictions/GT boxes and returning scalar scores.
- Inputs: decoded boxes, scores, class labels; GT boxes/labels.
- Called by: `train.py` eval loop, possibly scripts.

## 4) Data Flow & Tensor Shapes
- Operates on lists/tensors of boxes (often `[N, 7]`) and per-class scores; outputs floats/dicts of metrics.

## 5) Configuration
- Uses class lists (e.g., `NUSCENES_CLASSES`); thresholds may be configurable in configs/eval args.

## 6) Training vs Inference Behavior
- Used only in eval; no training-specific behavior.

## 7) Troubleshooting
- Mismatch between class ordering and head outputs: ensure class list aligns with targets.
- Zero metrics: check decoder/NMS thresholds and target generation alignment.
- Device/dtype issues: move tensors to CPU if metrics expect NumPy.

## 8) Alignment to Proposal
- Provides evaluation metrics per proposal. Status: **Partially implemented** (confirm full NDS components).

## 9) Minimal-change Extension Guide
- Add new metric functions in `metrics.py`; wire into `train.py` eval path via small hooks without altering training loop core.***

