# Module: memory_bank

## 1) Purpose
- Maintains historical BEV features for temporal aggregation (e.g., queue of BEV frames).
- Supports temporal attention or recurrent updates.

## 2) Files
- `modules/memory_bank.py` — Memory buffer utilities for BEV sequences.

## 3) Public APIs
- Memory bank class/functions (see file) to push/pop BEV tensors.
- Inputs: BEV feature `[B, C, H, W]`, metadata (scene tokens, timestamps).
- Outputs: retrieved history tensors for temporal modules.
- Called by: `bevformer.py`, `temporal_attention.py`, possibly `mc_convrnn`.

## 4) Data Flow & Tensor Shapes
- Stores BEV tensors in a list/queue; provides stacked history `[B, T_hist, C, H, W]` to temporal modules.

## 5) Configuration
- History length/max frames configured in constructors or passed from `train.py` args (max_history).

## 6) Training vs Inference Behavior
- Same logic; may limit history length; no special inference-only changes.

## 7) Troubleshooting
- History not advancing: ensure `scene_tokens`/reset logic triggered on new sequences.
- Memory growth: check max_history enforcement.
- Device mismatch: keep stored tensors on the correct device.

## 8) Alignment to Proposal
- Provides temporal buffering per proposal. Status: **Partially implemented** (verify advanced eviction/scene-handling needs).

## 9) Minimal-change Extension Guide
- Add reset/scene-boundary handling in `memory_bank.py`; guard with flags so existing behavior remains unchanged.***

