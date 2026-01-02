# Module: camera_encoder

## 1) Purpose
- Pre-BEV camera feature extraction before BEVFormer projection.
- Wraps image backbone and view-level encoding.

## 2) Files
- `modules/camera_encoder.py` — Camera feature encoder utilities/classes.

## 3) Public APIs
- Encoder classes/functions (see file) that output feature maps per camera view.
- Inputs: images `[B, T, N_cam, 3, H, W]`.
- Outputs: per-view features `[B, T, N_cam, C, H', W']`.
- Called by: `bevformer.py`, `bev_fusion_model.py`.

## 4) Data Flow & Tensor Shapes
- Receives normalized images from dataset; processes per camera; feeds BEVFormer which aggregates into BEV.

## 5) Configuration
- Channel counts/backbone selection set in constructor; pass via `train.py` args/configs if exposed.

## 6) Training vs Inference Behavior
- Standard CNN; dropout/BN follow mode. No special inference path.

## 7) Troubleshooting
- Shape mismatch on camera axis: ensure dataset stacks as `[B, T, N_cam, ...]`.
- Pretrained weight loading errors: check paths/config.
- Device mismatch: move encoder to same device as backbone/BEVFormer.

## 8) Alignment to Proposal
- Implements camera view encoder for BEVFormer pipeline. Status: **Partially implemented**.

## 9) Minimal-change Extension Guide
- Add optional backbone selection flag and instantiate inside `camera_encoder.py`; avoid touching BEVFormer interface.***

