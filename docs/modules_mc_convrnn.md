# Module: mc_convrnn

## 1) Purpose
- Motion-Compensated ConvRNN for temporal BEV aggregation with ego-motion warping and motion field modeling.
- Novel temporal module intended to improve stability across frames.

## 2) Files
- `modules/mc_convrnn.py` — MC-ConvRNN cell and utilities.

## 3) Public APIs
- MC-ConvRNN class (see file) with `forward(hidden, input, ego_pose, ...)`.
- Inputs: BEV feature `[B, C, H, W]`, hidden state `[B, C, H, W]`, ego-motion transforms/visibility.
- Outputs: updated hidden/BEV `[B, C, H, W]`.
- Called by: `bevformer.py` (temporal_method switch), `train.py`.

## 4) Data Flow & Tensor Shapes
- Applies warping of hidden state to current frame coordinates, then ConvRNN update; returns compensated hidden for next step.

## 5) Configuration
- Channel sizes and flags (disable warping/motion/visibility) set via args from `train.py`.

## 6) Training vs Inference Behavior
- Same recurrence; warping and gating applied in both modes. No special inference logic.

## 7) Troubleshooting
- Warping artifacts: verify ego poses passed from dataset; check device/dtype of transforms.
- Shape mismatch: ensure BEV grid and channel dims align with backbone output.
- Disabled flags: confirm mc_disable_* args propagate correctly from `train.py`.

## 8) Alignment to Proposal
- Implements proposed MC-ConvRNN temporal module. Status: **Partially implemented** (core cell present; validate full feature set vs proposal).

## 9) Minimal-change Extension Guide
- Add small switches in `mc_convrnn.py` for motion field variants; expose via `train.py` args without altering existing call signature.***

