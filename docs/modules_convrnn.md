# Module: convrnn

## 1) Purpose
- Provides convolutional RNN blocks for temporal aggregation of BEV features.
- Used as a baseline temporal module.

## 2) Files
- `modules/convrnn.py` — ConvRNN/ConvGRU utilities.

## 3) Public APIs
- ConvRNN/ConvGRU classes (see file) with `forward(hidden, input)` style signatures.
- Inputs/Outputs: BEV feature maps `[B, C, H, W]` with hidden state of same shape.
- Called by: `bevformer.py`, `train.py` (temporal_method switch).

## 4) Data Flow & Tensor Shapes
- Takes previous BEV hidden state and current BEV features; returns updated hidden for temporal smoothing.

## 5) Configuration
- Channels/hidden dims set in constructors; selected via temporal method args in `train.py`.

## 6) Training vs Inference Behavior
- Same recurrence; hidden state initialization may differ (zeros) but no separate inference logic.

## 7) Troubleshooting
- Hidden state device mismatch: ensure hidden is on same device as inputs.
- Shape mismatch: confirm C/H/W match model configuration.
- Exploding gradients: consider grad clipping (already in `train.py`).

## 8) Alignment to Proposal
- Covers baseline temporal RNN. Status: **Partially implemented** (novel MC-ConvRNN in `mc_convrnn.py`).

## 9) Minimal-change Extension Guide
- Add new cell variant inside `convrnn.py`; expose selection flag in `train.py` temporal_method without altering call signature.***

