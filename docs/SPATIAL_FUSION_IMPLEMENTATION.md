# Spatial Fusion Module Implementation

## Overview

This document describes the implementation of the Spatial Fusion Module for the BEV Fusion System, which combines LiDAR and camera BEV features using cross-attention mechanism.

## Implementation Summary

### Task 5: Implement Spatial Fusion Module ✓

All subtasks have been completed:

#### 5.1 Implement Cross-Attention Fusion ✓
- **File**: `modules/fusion.py`
- **Class**: `CrossAttentionFusion`
- **Features**:
  - Multi-head attention between LiDAR (query) and camera (key/value) features
  - Layer normalization after attention and FFN
  - Residual connections to preserve geometric information
  - Configurable number of attention heads (default: 8)
  - Feed-forward network with dropout for regularization

**Key Design Decisions**:
- LiDAR features serve as queries to attend to camera features
- Residual projection handles dimension mismatch between input and output
- Two-stage residual connections: after attention and after FFN
- Layer normalization applied in spatial format (B, H, W, C)

#### 5.2 Verify BEV Grid Alignment ✓
- **Method**: `SpatialFusionModule.verify_alignment()`
- **Checks**:
  - Batch sizes match between LiDAR and camera features
  - Spatial dimensions (H, W) match exactly
  - Channel dimensions match expected configuration
- **Returns**: Boolean indicating whether features are properly aligned

#### 5.3 Create SpatialFusionModule ✓
- **File**: `modules/fusion.py`
- **Class**: `SpatialFusionModule`
- **Features**:
  - Wraps `CrossAttentionFusion` with alignment verification
  - Ensures spatial correspondence before fusion
  - Produces unified BEV features with shape (B, C3, H, W)
  - Configurable channel dimensions for LiDAR, camera, and fused features

**Interface**:
```python
fusion_module = SpatialFusionModule(
    lidar_channels=64,      # C1
    camera_channels=256,    # C2
    fused_channels=256,     # C3
    num_heads=8
)

F_fused = fusion_module(F_lidar, F_cam)
```

#### 5.4 Write Property Test for Fusion Output Shape ✓
- **File**: `tests/property_tests/test_fusion_properties.py`
- **Test**: `test_property_10_fusion_output_shape_invariant`
- **Property 10**: Spatial fusion output shape invariant
- **Validates**: Requirements 3.1, 3.2
- **Checks**:
  - Output shape matches (B, C3, H, W)
  - All output values are finite (no NaN/Inf)
  - Output device matches input device
  - Output dtype matches input dtype
- **Configuration**: 100 test examples with random batch sizes and BEV dimensions

#### 5.5 Write Property Test for BEV Alignment ✓
- **File**: `tests/property_tests/test_fusion_properties.py`
- **Test**: `test_property_9_bev_grid_alignment`
- **Property 9**: BEV grid alignment between modalities
- **Validates**: Requirements 2.5, 3.3
- **Checks**:
  - Alignment verification passes for matching dimensions
  - Alignment verification fails for mismatched batch sizes
  - Alignment verification fails for mismatched spatial dimensions
  - Alignment verification fails for mismatched channel dimensions
- **Configuration**: 100 test examples with various dimension combinations

## Additional Tests Implemented

Beyond the required property tests, several additional tests were implemented to ensure robustness:

1. **Spatial Structure Preservation**: Verifies residual connections preserve spatial patterns
2. **Valid Attention Weights**: Ensures attention mechanism produces finite outputs
3. **Deterministic Evaluation**: Confirms reproducibility in eval mode
4. **Zero Features Handling**: Tests edge case of zero-valued inputs
5. **Batch Independence**: Verifies each batch element is processed independently
6. **Residual Information Preservation**: Confirms LiDAR geometric info is preserved
7. **Layer Normalization**: Validates proper normalization application
8. **Multi-head Configuration**: Checks attention head setup

## Architecture Details

### CrossAttentionFusion Architecture

```
Input: F_lidar (B, C1, H, W), F_cam (B, C2, H, W)

1. Projection:
   Q = Conv2d(F_lidar)  → (B, C3, H, W)
   K = Conv2d(F_cam)    → (B, C3, H, W)
   V = Conv2d(F_cam)    → (B, C3, H, W)

2. Reshape for Multi-Head Attention:
   Q, K, V → (B, num_heads, H*W, head_dim)

3. Attention:
   scores = Q @ K^T * scale
   weights = softmax(scores)
   attn_out = weights @ V

4. Reshape back:
   attn_out → (B, C3, H, W)

5. First Residual + LayerNorm:
   x = LayerNorm(residual_proj(F_lidar) + out_proj(attn_out))

6. FFN + Second Residual + LayerNorm:
   F_fused = LayerNorm(x + FFN(x))

Output: F_fused (B, C3, H, W)
```

### Memory Considerations

The attention mechanism computes full spatial attention with complexity O((H*W)²). For typical BEV grids (200x200), this results in 40,000 x 40,000 attention matrices per head, which can be memory-intensive.

**Optimization Strategies** (for future work):
- Implement deformable attention to reduce computational cost
- Use sparse attention patterns
- Apply window-based attention
- Implement gradient checkpointing for training

## Integration with Pipeline

The Spatial Fusion Module fits into the overall BEV Fusion pipeline as follows:

```
LiDAR Encoder → F_lidar (B, 64, 200, 200)
                    ↓
Camera Encoder → F_cam (B, 256, 200, 200)
                    ↓
         Spatial Fusion Module
                    ↓
              F_fused (B, 256, 200, 200)
                    ↓
         Temporal Aggregation
                    ↓
           Detection Head
```

## Configuration

Default configuration values:
- **LiDAR channels (C1)**: 64
- **Camera channels (C2)**: 256
- **Fused channels (C3)**: 256
- **Number of attention heads**: 8
- **Head dimension**: 32 (256 / 8)
- **FFN expansion ratio**: 4x
- **Dropout rate**: 0.1

## Requirements Validation

### Requirement 3.1 ✓
"WHEN the System receives LiDAR BEV features F_lidar and camera BEV features F_cam, THE System SHALL apply cross-attention fusion between them"

**Implementation**: `CrossAttentionFusion` class implements multi-head cross-attention where LiDAR features (queries) attend to camera features (keys/values).

### Requirement 3.2 ✓
"WHEN fusion is performed, THE System SHALL produce unified BEV features F_fused with shape (B, C3, H, W) where C3 ≈ 256"

**Implementation**: `SpatialFusionModule.forward()` produces output with configurable C3 dimension (default 256).

### Requirement 3.3 ✓
"THE System SHALL ensure spatial alignment between LiDAR and camera BEV grids before fusion"

**Implementation**: `SpatialFusionModule.verify_alignment()` checks spatial dimensions match before fusion.

### Requirement 2.5 ✓
"THE System SHALL maintain spatial alignment between camera BEV and LiDAR BEV representations"

**Implementation**: Alignment verification ensures both modalities share the same (H, W) dimensions.

## Testing Strategy

### Property-Based Testing
Using Hypothesis library with 100 examples per property:
- Random batch sizes (1-4)
- Random BEV dimensions (20x20 to 50x50 for memory efficiency)
- Random feature values in range [-10, 10]

### Unit Testing
Additional unit tests cover:
- Edge cases (zero features, single batch)
- Determinism in eval mode
- Batch independence
- Configuration validation

## Files Created/Modified

### New Files
1. `modules/fusion.py` - Spatial fusion implementation
2. `tests/property_tests/test_fusion_properties.py` - Property-based tests
3. `docs/SPATIAL_FUSION_IMPLEMENTATION.md` - This documentation

### Dependencies
- PyTorch >= 2.0
- Hypothesis >= 6.75 (for property-based testing)
- NumPy >= 1.24

## Usage Example

```python
import torch
from modules.fusion import SpatialFusionModule
from modules.lidar_encoder import LiDARBEVEncoder
from modules.camera_encoder import CameraBEVEncoder
from modules.data_structures import BEVGridConfig

# Initialize encoders
config = BEVGridConfig()
lidar_encoder = LiDARBEVEncoder(config, out_channels=64)
camera_encoder = CameraBEVEncoder(bev_h=200, bev_w=200, embed_dim=256)

# Initialize fusion module
fusion_module = SpatialFusionModule(
    lidar_channels=64,
    camera_channels=256,
    fused_channels=256,
    num_heads=8
)

# Forward pass
# ... (get point clouds and images)
F_lidar = lidar_encoder(points_batch)  # (B, 64, 200, 200)
F_cam = camera_encoder(images, intrinsics, extrinsics)  # (B, 256, 200, 200)

# Fuse features
F_fused = fusion_module(F_lidar, F_cam)  # (B, 256, 200, 200)
```

## Next Steps

The spatial fusion module is now complete and ready for integration with:
1. **Temporal Aggregation Module** (Task 6 & 7) - Aggregate features across time
2. **Detection Head** (Task 8) - Predict 3D bounding boxes
3. **Training Pipeline** (Task 12) - End-to-end training
4. **Evaluation** (Task 13) - Performance metrics

## Notes

- The implementation prioritizes correctness and clarity over optimization
- Full spatial attention is used; deformable attention can be added later for efficiency
- All property tests are designed to run without GPU (CPU-only testing)
- Memory-efficient testing uses smaller BEV dimensions (20-50) instead of full size (200)
