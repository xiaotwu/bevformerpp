# Module: backbone

## 1) Purpose
- Provides shared CNN backbone utilities for BEV encoders.
- Sits before neck/head, used by camera BEVFormer or fusion stacks.

## 2) Files
- `modules/backbone.py` — Backbone definitions (e.g., ResNet-style or custom CNN blocks).

## 3) Public APIs
- Backbone classes (see file for signatures), typically returning feature maps `[B, C, H, W]`.
- Called by: `bevformer.py`, `camera_encoder.py`, or other encoders.

## 4) Data Flow & Tensor Shapes
- Input: images or intermediate feature maps `[B, C_in, H, W]`.
- Output: downsampled feature pyramid or single-level feature `[B, C_out, H', W']` consumed by neck/attention.

## 5) Configuration
- Channel sizes/strides set in constructors; pass values from configs/train args where exposed.

## 6) Training vs Inference Behavior
- Standard CNN; dropout/batchnorm behave per mode. No special inference path.

## 7) Troubleshooting
- Shape mismatch into neck: confirm stride/out_channels match `neck.py` expectation.
- Pretrained weights path issues: ensure configs point to valid checkpoints.
- Device mismatch: move backbone to same device as caller.

## 8) Alignment to Proposal
- Implements shared vision backbone for BEVFormer camera encoder. Status: **Partially implemented** (confirm exact backbone choice vs proposal).

## 9) Minimal-change Extension Guide
- Add new backbone class in `backbone.py`; gate selection via small switch/arg in caller (`camera_encoder.py` / `bevformer.py`). Avoid altering downstream shapes without updating neck/head configs.***

