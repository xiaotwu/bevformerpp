# Module: attention

## 1) Purpose
- Implements attention utilities (spatial/temporal) used by BEVFormer-style camera BEV encoding.
- Sits inside camera BEV pipeline and is called from `bevformer.py` / `temporal_attention.py`.

## 2) Files
- `modules/attention.py` — Attention layers and helpers for BEVFormer blocks.

## 3) Public APIs
- Core classes/functions (inspect file for signatures): multi-head attention layers and helper functions used by `BEVFormer` blocks.
- Inputs: BEV/query features `[B, Nq, C]` or `[B, C, H, W]` depending on caller.
- Outputs: Attended features with same spatial/query layout.
- Called by: `bevformer.py`, `temporal_attention.py`.

## 4) Data Flow & Tensor Shapes
- Receives query/key/value tensors from BEVFormer encoder blocks.
- Operates on flattened spatial tokens or BEV tokens; returns tensors reshaped back to BEV layout for downstream fusion/head.

## 5) Configuration
- Hyperparameters (heads, hidden dim, dropout) set inside model constructors in `bevformer.py`; pass-through from `train.py`/configs if exposed.

## 6) Training vs Inference Behavior
- Standard attention; dropout only active in training if present. No other differences.

## 7) Troubleshooting
- Shape mismatch `[B, heads, ...]`: check caller flattening in `bevformer.py`.
- Device/dtype mismatch: ensure inputs on same device as attention module.
- Masking errors: verify padding/mask tensors passed correctly from dataset/temporal stack.

## 8) Alignment to Proposal
- Implements BEVFormer attention blocks (camera BEV). Status: **Partially implemented** (covers core attention; confirm alignment with full proposal details if more variants required).

## 9) Minimal-change Extension Guide
- Add new attention variants inside `attention.py` and select via a small switch in `bevformer.py` constructor; avoid changing call sites.***

