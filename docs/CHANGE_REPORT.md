# Change Report: Proposal Alignment

## Summary

This report documents the changes made to align the BEVFormer++ codebase with the project proposal. The changes ensure the implementation is **design-consistent** while applying **minimal structural modifications**.

## Phase A: Cross-Attention Fusion (REWRITE)

### What Was Added

| File | Description |
|------|-------------|
| `modules/fusion/__init__.py` | Package exports |
| `modules/fusion/cross_attention_fusion.py` | **NEW**: Bidirectional cross-attention fusion |

### What Was Modified

| File | Change |
|------|--------|
| `modules/fusion.py` | Added import for `BidirectionalCrossAttentionFusion`; Updated `SpatialFusionModule` to support new fusion type |
| `modules/bev_fusion_model.py` | Changed default `fusion_type` from `"cross_attention"` to `"bidirectional_cross_attn"`; Added new fusion parameters |
| `configs/base_config.yaml` | Added complete fusion configuration section |

### Why It Matches the Proposal

The proposal requires **"Cross-attention BEV fusion (alignment-aware, not concatenation-based)"**:

- ✅ **Bidirectional attention**: Camera queries LiDAR AND LiDAR queries Camera
- ✅ **Gated residual**: `fused = gate * cam_attended + (1-gate) * lidar_attended`
- ✅ **2D sinusoidal positional encoding**: Translation-equivariant spatial awareness
- ✅ **Debug output**: Optional attention maps for visualization
- ✅ **Config-selectable**: Can switch to legacy modes for ablation

## Phase B: MC-ConvRNN (VERIFY + ORGANIZE)

### What Was Added

| File | Description |
|------|-------------|
| `modules/temporal/__init__.py` | Package exports |
| `modules/temporal/ego_motion_warp.py` | **NEW**: Isolated ego-motion warping module |
| `modules/temporal/residual_motion_refine.py` | **NEW**: Isolated residual motion module |
| `modules/temporal/visibility_gate.py` | **NEW**: Isolated visibility gating module |

### What Was Preserved

The existing `modules/mc_convrnn.py` already implemented all three mechanisms correctly. The new separate files provide:
1. Cleaner module isolation
2. Independent testability
3. Explicit documentation of each mechanism

### Why It Matches the Proposal

The proposal requires **"MC-ConvRNN with (1) ego-motion warping, (2) dynamic residual motion refinement, (3) visibility-gated recurrent convolution"**:

- ✅ **Ego-motion warping**: `EgoMotionWarp` class with SE(2) BEV warping
- ✅ **Residual motion refinement**: `ResidualMotionRefine` CNN for flow estimation
- ✅ **Visibility gating**: `VisibilityGate` with bounds + consistency masks
- ✅ **Ablation support**: Each mechanism can be disabled via config

## Phase C: Alignment & Debugging

### What Was Added

| File | Description |
|------|-------------|
| `modules/utils/__init__.py` | Package exports |
| `modules/utils/vis_bev.py` | **NEW**: BEV visualization utilities |
| `tests/test_bev_alignment_sanity.py` | **NEW**: Alignment correctness tests |
| `tests/test_fusion_cross_attn.py` | **NEW**: Fusion module tests |
| `tests/test_mc_convrnn.py` | **NEW**: MC-ConvRNN tests |

### Why It Matches the Proposal

The proposal requires **"Alignment-aware BEV processing"**:

- ✅ **Unified BEV grid**: Single definition in `configs/base_config.yaml`
- ✅ **Explicit coordinate chain**: Documented sensor → ego → BEV
- ✅ **Visualization utility**: `vis_bev.py` for feature/heatmap/box visualization
- ✅ **Alignment sanity test**: Verifies warp correctness within tolerance

## Phase D: Efficiency & Profiling

### What Was Added

| File | Description |
|------|-------------|
| `tools/profile_runtime.py` | **NEW**: Runtime profiling tool |

### Configuration Addition

```yaml
# configs/base_config.yaml
runtime:
  amp: true
  cudnn_benchmark: true
  torch_compile: false
  profile: false
```

### Why It Matches the Proposal

The proposal requires **"Computational efficiency suitable for real-time use"**:

- ✅ **Runtime config**: AMP, cudnn_benchmark, torch_compile options
- ✅ **Profiling tool**: Measures FPS, iteration time, CUDA memory
- ✅ **Train vs inference**: Both modes supported
- ✅ **Module comparison**: Can compare fusion types and temporal aggregation

## Tests Added

| Test File | Coverage |
|-----------|----------|
| `test_fusion_cross_attn.py` | Positional encoding, attention blocks, bidirectional fusion, gating, gradients |
| `test_mc_convrnn.py` | Ego warping, residual motion, visibility gating, ConvGRU, ablations |
| `test_bev_alignment_sanity.py` | Grid coordinates, transform chain, warp correctness |

## Configuration Changes

### Before (Default Fusion)
```yaml
fusion:
  fusion_type: "cross_attention"  # Unidirectional
```

### After (Proposal Default)
```yaml
fusion:
  type: "bidirectional_cross_attn"  # Bidirectional + gated
  use_bidirectional: true
  use_gate: true
  pos_encoding: "sinusoidal_2d"
```

### Before (Temporal)
```yaml
temporal:
  use_transformer: true
  use_convgru: true
```

### After (Proposal Default)
```yaml
temporal:
  use_mc_convrnn: true
  mc_convrnn:
    enable_warp: true
    enable_residual: true
    enable_visibility_gate: true
```

## Backward Compatibility

All changes maintain backward compatibility:

1. **Fusion**: Legacy `cross_attention` type still available
2. **Temporal**: Both transformer and MC-ConvRNN supported
3. **Config**: Default values match proposal specifications
4. **Tests**: Can run independently without modifications

## Files NOT Modified

The following were explicitly left unchanged per the minimal-change rule:

- Dataset loading (`nuscenes_dataset.py` structure preserved)
- Training loop (`train.py` entry point preserved)
- Detection head (`head.py` architecture preserved)
- Model forward pass flow (only wiring updated)

## Verification

All new Python files pass syntax verification:
```bash
python3 -m py_compile modules/fusion/*.py modules/temporal/*.py modules/utils/*.py tools/profile_runtime.py
# Exit code: 0
```

