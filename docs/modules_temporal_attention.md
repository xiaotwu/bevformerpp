# Module: temporal_attention

## 1) Purpose
- Implements temporal self-attention for aggregating BEV features across time.
- Provides baseline temporal alternative to recurrent modules.

## 2) Files
- `modules/temporal_attention.py` — Temporal attention layers/utilities.

## 3) Public APIs
- Temporal attention classes/functions (see file) that consume sequences of BEV tokens/features.
- Inputs: BEV history `[B, T, C, H, W]` (often flattened to tokens).
- Outputs: aggregated BEV features `[B, C, H, W]` or sequence.
- Called by: `bevformer.py`, `train.py` (temporal_method switch).

## 4) Data Flow & Tensor Shapes
- Receives history from `memory_bank`; performs attention over time; outputs fused BEV for current step.

## 5) Configuration
- Number of heads, embed dim, history length driven by constructors; `max_history` passed from `train.py`.

## 6) Training vs Inference Behavior
- Same attention; dropout active in training if enabled.

## 7) Troubleshooting
- Large memory use: reduce `max_history`.
- Shape mismatch: ensure history stacked as `[B, T, C, H, W]`.
- Masking errors: verify temporal masks align with valid frames.

## 8) Alignment to Proposal
- Implements baseline temporal self-attention. Status: **Partially implemented** (verify completeness vs proposal).

## 9) Minimal-change Extension Guide
- Add lightweight attention variant within `temporal_attention.py`; select via flag in `train.py` without changing input/output shapes.***

