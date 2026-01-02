# BEVFormer++ Architecture

## Proposal Alignment Statement

This document describes the architecture of BEVFormer++, a multi-modal BEV perception system. The implementation is **proposal-consistent** with the following key features:

1. **Cross-Attention BEV Fusion**: Alignment-aware bidirectional cross-attention (not concatenation-based)
2. **MC-ConvRNN**: Motion-Compensated ConvRNN with all three mechanisms:
   - Ego-motion warping
   - Dynamic residual motion refinement
   - Visibility-gated recurrent convolution
3. **Alignment-Aware BEV Processing**: Unified BEV grid with explicit coordinate chains
4. **Computational Efficiency**: Configurable for real-time use

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BEVFormer++ Pipeline                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐      ┌─────────────┐                                       │
│  │   LiDAR     │      │   Camera    │                                       │
│  │  Points     │      │   Images    │                                       │
│  └──────┬──────┘      └──────┬──────┘                                       │
│         │                    │                                              │
│         ▼                    ▼                                              │
│  ┌─────────────┐      ┌─────────────┐                                       │
│  │   LiDAR     │      │   Camera    │                                       │
│  │   BEV       │      │   BEV       │                                       │
│  │  Encoder    │      │  Encoder    │                                       │
│  └──────┬──────┘      └──────┬──────┘                                       │
│         │                    │                                              │
│         └────────┬───────────┘                                              │
│                  ▼                                                          │
│         ┌───────────────────┐                                               │
│         │   Cross-Attention │  ◄── Bidirectional, Gated, Pos-Encoded        │
│         │   BEV Fusion      │                                               │
│         └─────────┬─────────┘                                               │
│                   │                                                         │
│                   ▼                                                         │
│         ┌───────────────────┐                                               │
│         │    MC-ConvRNN     │  ◄── Warp + Residual + Visibility Gate        │
│         │    Temporal       │                                               │
│         └─────────┬─────────┘                                               │
│                   │                                                         │
│                   ▼                                                         │
│         ┌───────────────────┐                                               │
│         │   Detection       │                                               │
│         │   Head            │                                               │
│         └─────────┬─────────┘                                               │
│                   │                                                         │
│                   ▼                                                         │
│         ┌───────────────────┐                                               │
│         │   3D Detections   │                                               │
│         └───────────────────┘                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Module Descriptions

### 1. Spatial Fusion (Cross-Attention)

**File**: `modules/fusion/cross_attention_fusion.py`

**Implementation Status**: ✅ FULLY IMPLEMENTED

The spatial fusion module implements **bidirectional cross-attention** as specified in the proposal:

```
Camera BEV ──► Queries LiDAR ──► Semantic-enriched Camera features
LiDAR BEV ──► Queries Camera ──► Geometry-grounded LiDAR features
                    │
                    ▼
            Gated Residual Combination
                    │
                    ▼
            Fused BEV Features
```

**Key Features**:
- **2D Sinusoidal Positional Encoding**: Provides translation-equivariant spatial awareness
- **Bidirectional Attention**: Both modalities attend to each other
- **Gated Residual**: `fused = gate * cam_attended + (1-gate) * lidar_attended`
- **Debug Output**: Optional attention map output for visualization

**Configuration** (`configs/base_config.yaml`):
```yaml
fusion:
  type: "bidirectional_cross_attn"  # PROPOSAL DEFAULT
  dim: 256
  num_heads: 8
  use_bidirectional: true
  use_gate: true
  pos_encoding: "sinusoidal_2d"
```

### 2. Temporal Aggregation (MC-ConvRNN)

**File**: `modules/mc_convrnn.py`

**Supporting Files**:
- `modules/temporal/ego_motion_warp.py` - Mechanism 1
- `modules/temporal/residual_motion_refine.py` - Mechanism 2
- `modules/temporal/visibility_gate.py` - Mechanism 3

**Implementation Status**: ✅ FULLY IMPLEMENTED

The MC-ConvRNN implements all three proposal-mandated mechanisms:

#### Mechanism 1: Ego-Motion Warping

```
h_{t-1} ──► Apply T_{t-1→t} ──► Warped h'_{t-1}
```

- Uses SE(2) (x, y, yaw) extracted from SE(3) transform
- `grid_sample`-based BEV warping
- Handles boundary conditions with zero-padding

#### Mechanism 2: Dynamic Residual Motion Refinement

```
[Current BEV, Warped BEV] ──► CNN ──► Δ(u,v) flow ──► Refined warp
```

- Lightweight 3-layer CNN
- Predicts small offset field
- Zero-initialized for stable training

#### Mechanism 3: Visibility-Gated Recurrent Convolution

```
Visibility v = f(bounds, consistency)
h_new = v * update(current, warped) + (1-v) * h_prev
```

- **Bounds mask**: Identifies out-of-bounds regions
- **Consistency mask**: Identifies feature-inconsistent regions
- **Gated update**: Visibility explicitly modulates RNN update

**Configuration**:
```yaml
temporal:
  use_mc_convrnn: true
  mc_convrnn:
    enable_warp: true
    enable_residual: true
    enable_visibility_gate: true
```

**Ablation Support**: Each mechanism can be disabled via config for ablation studies.

### 3. BEV Grid Configuration

**Unified Definition** (`configs/base_config.yaml`):
```yaml
bev_grid:
  x_min: -51.2  # meters
  x_max: 51.2
  y_min: -51.2
  y_max: 51.2
  resolution: 0.2  # meters per pixel
```

All modules reference this single definition:
- Camera encoder
- LiDAR encoder
- Target generator
- Fusion
- Visualization

### 4. Coordinate Chain

```
Sensor Frame ──► Ego Frame ──► BEV Grid
     │               │              │
     │   extrinsics  │   bev_range  │
     └───────────────┴──────────────┘
```

**Documented Conventions**:
- X-axis: Forward
- Y-axis: Left
- Z-axis: Up
- BEV grid: X increases right, Y increases up in visualization

## File Structure

```
modules/
├── fusion/
│   ├── __init__.py
│   └── cross_attention_fusion.py    # Bidirectional cross-attention
├── temporal/
│   ├── __init__.py
│   ├── ego_motion_warp.py           # Mechanism 1
│   ├── residual_motion_refine.py    # Mechanism 2
│   └── visibility_gate.py           # Mechanism 3
├── utils/
│   ├── __init__.py
│   └── vis_bev.py                   # Visualization utilities
├── mc_convrnn.py                    # Integrated MC-ConvRNN
├── fusion.py                        # SpatialFusionModule wrapper
└── bev_fusion_model.py              # Complete model

tests/
├── test_fusion_cross_attn.py        # Fusion tests
├── test_mc_convrnn.py               # MC-ConvRNN tests
└── test_bev_alignment_sanity.py     # Alignment tests

tools/
├── overfit_one_sample.py            # Debug/validation tool
└── profile_runtime.py               # Runtime profiling

configs/
└── base_config.yaml                 # Unified configuration
```

## Running Tests

```bash
# All tests
pytest tests/ -v

# Specific module tests
pytest tests/test_fusion_cross_attn.py -v
pytest tests/test_mc_convrnn.py -v
pytest tests/test_bev_alignment_sanity.py -v
```

## Profiling

```bash
# Profile default configuration
python tools/profile_runtime.py

# Profile specific fusion type
python tools/profile_runtime.py --fusion_type bidirectional_cross_attn

# Compare temporal types
python tools/profile_runtime.py --temporal_type mc_convrnn
python tools/profile_runtime.py --temporal_type transformer
```

## Partial Implementation Notes

The following features are **fully implemented** as per the proposal:
- ✅ Bidirectional cross-attention fusion
- ✅ MC-ConvRNN with all three mechanisms
- ✅ Visibility gating
- ✅ Ablation support via config flags

The following are **standard implementations** (not proposal-specific):
- LiDAR encoder (PointPillars-style)
- Camera encoder (BEVFormer-style)
- Detection head (CenterPoint-style)

