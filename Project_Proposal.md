# Project Overview for Kiro

This document outlines the core concepts, architecture, modules, workflows, and implementation details of a multi-sensor Bird's-Eye View (BEV) perception system integrating LiDAR, multi-view cameras, and temporal modeling. It is structured to help Kiro understand the overall project and generate correct code and configurations.

---

## 1. System Goals

* Build a modular BEV perception system combining LiDAR (PointPillars), multi-view cameras (BEVFormer), and temporal modeling.
* Fuse geometric LiDAR features with semantic camera features in BEV space through lightweight cross-attention.
* Improve 3D detection robustness, reduce temporal instability, and maintain real-time performance.
* Implement both transformer-based and RNN-based temporal fusion (BEVFormer temporal encoder + Motion-Compensated ConvRNN).

---

## 2. High-Level Architecture

Components (also shown in proposal Figure 4):

1. **LiDAR BEV Encoder (PointPillars)**
2. **Camera BEV Encoder (BEVFormer)**
3. **Spatial Fusion Module (BEV-level cross-attention)**
4. **Temporal Module (Two branches)**

   * Temporal Self-Attention (enhanced BEVFormer-based)
   * Motion-Compensated ConvRNN (MC-ConvRNN)
5. **Detection Head** for 3D bounding box prediction

Pipeline:

```
Point Cloud → LiDAR BEV →
                     ↘
                      Fusion → Temporal Aggregation → Detection Head
                     ↗
Images → Camera BEV →
```

---

## 3. Module Specifications

### 3.1 LiDAR BEV Encoder (PointPillars)

**Input:** Raw point cloud `(x, y, z, r)`

**Steps:**

* Pillarization of 3D space
* PointNet-style per-pillar encoder
* Scatter to BEV grid (H × W × C)
* 2D CNN backbone

**Output:** BEV feature map `F_lidar ∈ R^{H×W×C1}`

**Key properties:** Efficient, real-time, geometric fidelity.

---

### 3.2 Camera BEV Encoder (BEVFormer)

**Input:** Six multi-view RGB images + camera intrinsics/extrinsics.

**Steps:**

* Image backbone (e.g., ResNet50)
* Spatial cross-attention (BEV queries ↔ image features)
* Temporal self-attention (optional at baseline)

**Output:** BEV feature map `F_cam ∈ R^{H×W×C2}`

**Key properties:** Semantic richness, long-range perception, transformer-based reasoning.

---

### 3.3 Spatial Fusion Module

**Inputs:** `F_lidar`, `F_cam`

**Mechanism:**

* Cross-attention where LiDAR BEV attends to camera BEV and/or vice versa
* Concatenation or additive merging

**Output:** Unified BEV feature `F_fused ∈ R^{H×W×C3}`

**Design goals:** Lightweight, spatially aligned, minimal latency.

---

## 4. Temporal Aggregation Modules

### 4.1 Enhanced BEVFormer Temporal Self-Attention

**Input:** A sequence of past fused BEV features aligned via ego-motion.

**Mechanisms:**

* BEV-frame alignment
* Temporal self-attention
* Temporal gating (adaptive confidence weighting)
* Residual update

**Output:** `F_temp_attn ∈ R^{H×W×C}`

**Use cases:** Short-term temporal consistency, handling occlusion.

---

### 4.2 Motion-Compensated ConvRNN (MC-ConvRNN)

**Inputs:**

* Previous BEV feature `F_{t-1}`
* Current BEV feature `F_t`
* Ego-motion transform
* Visibility mask

**Steps (matching proposal Fig. 5):**

1. Ego-motion warping of `F_{t-1}`
2. Dynamic residual motion field alignment
3. Visibility gating
4. ConvGRU recurrent fusion
5. Optional velocity refinement block

**Output:** `F'_t` (final temporally enriched BEV)

**Strength:** Stable long-range temporal reasoning, efficient, motion-aware.

---

## 5. Detection Head

* Dense 2D CNN prediction over BEV grid
* Classification head (object categories)
* Regression head (3D bounding boxes: center, size, yaw)
* NMS post-processing

**Output:** List of 3D bounding boxes with confidence.

---

## 6. Development Roadmap (Phase I)

As outlined in the proposal Table I:

### Month 1

* Implement BEVFormer and PointPillars baselines
* Reproduce baseline accuracy on nuScenes-mini

### Months 2–3

* Implement LiDAR BEV + Camera BEV generation
* Integrate spatial fusion module
* Add temporal self-attention with memory bank
* Implement MC-ConvRNN prototype

### Month 4

* Train full fusion pipeline on nuScenes
* Ablation studies on gating, warping, motion field
* Efficiency evaluation

### Month 5

* Optimization
* Visualization
* Thesis writing + defense preparation

---

## 7. Key Interfaces and Data Shapes

### BEV Grid

```
H = ~200 (depends on resolution)
W = ~200
C_lidar  ≈ 64
C_cam    ≈ 128–256
C_fused  ≈ 256
```

### Fusion Inputs

```
F_lidar: (B, C1, H, W)
F_cam:   (B, C2, H, W)
```

### Temporal Inputs

```
Sequence length: T = 3–5
Aligned frames: F_fused[t-k] → align → stack
```

### MC-ConvRNN Inputs

```
F_{t-1}, F_t ∈ (B, C, H, W)
M_t visibility mask ∈ (B, 1, H, W)
Ego-motion SE3 transform
```

---

## 8. Implementation Notes for Kiro

* Use MMDetection3D for baseline training and config structure.
* Modularize components:

  * `lidar_encoder/`
  * `camera_encoder/`
  * `fusion/`
  * `temporal/attn/`
  * `temporal/mc_convrnn/`
  * `detection/`
* Ensure all BEV resolutions and coordinate frames match.
* Validate calibration matrices before fusion.
* Implement visualization utilities for debugging alignment.

---

## 9. Expected Outputs for Kiro

Kiro should be able to:

* Generate PyTorch modules for each subsystem.
* Produce MMDetection3D-compatible configuration files.
* Implement warping functions, cross-attention blocks, ConvGRU, and detection heads.
* Provide training scripts, evaluation routines, and ablation experiment templates.

---

## 10. References

Use proposal references as implementation background.

---

This document serves as a condensed technical map for implementing the full BEV fusion system.
