# Mask and Forward Fix Report

**Date**: January 2026
**Scope**: LiDAR Mask Shape Bug Fix and Forward Path Hardening

---

## Executive Summary

This report documents the fix for a critical runtime error in the BEVFormer++ fusion forward path:

```
IndexError: boolean index did not match indexed array along axis 1;
size of axis is 35000 but size of corresponding boolean axis is 4
```

The root cause was incorrect mask inference that produced shape `(B, 4)` instead of `(B, N)` where N=35000.

**Resolution**: Created a centralized `normalize_lidar_points_and_mask()` helper function with robust shape validation, updated all LiDAR processing paths to use it, and added comprehensive regression tests.

---

## 1. Root Cause Analysis

### 1.1 The Bug

When running the fusion "Case B" forward with T=1 sequence:
- `lidar_points` had shape `(B=1, T=1, N=35000, 4)`
- `camera_images` had shape `(B=1, T=1, N_cam=6, 3, 224, 400)`

The old `forward_padded()` method attempted to infer a mask when none was provided:

```python
# OLD CODE (WRONG)
mask = ~torch.all(points[:, :, :3] == 0, dim=2)
```

With 4D input `(B, T, N, 4)`, the indexing `points[:, :, :3]` produced shape `(B, T, 3)` instead of `(B, N, 3)`, causing the resulting mask to have shape `(B, T)` = `(1, 1)` or similar incorrect shapes.

### 1.2 Why This Happened

1. **4D temporal inputs not handled**: The `forward_padded()` method assumed 3D inputs `(B, N, 4)` but the notebook passed 4D temporal inputs `(B, T, N, 4)`.

2. **No shape validation**: There was no validation that the mask shape matched the expected `(B, N)` format.

3. **Dimension reduction bug potential**: Using `dim=1` instead of `dim=-1` for any reduction over features would produce wrong shapes.

---

## 2. Solution: Centralized Normalization Helper

### 2.1 New Helper Function

Created `normalize_lidar_points_and_mask()` in `modules/utils/core.py`:

```python
def normalize_lidar_points_and_mask(
    points: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    squeeze_temporal: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Normalize LiDAR points and mask to canonical shapes for LiDAR encoder.

    Handles:
    - 3D points (B, N, 4) -> pass through
    - 4D points (B, T, N, 4) -> squeeze if T==1 (when squeeze_temporal=True)
    - Mask inference when mask is None
    - Mask shape validation with helpful error messages

    CRITICAL: Mask inference uses reduction over the LAST dimension (dim=-1),
    NOT dim=1. Using dim=1 would produce a mask of shape (B, 4) which is WRONG.
    """
```

Key features:
- **Handles 4D inputs**: Automatically squeezes temporal dimension when T=1
- **Correct mask inference**: Uses `xyz.abs().sum(dim=-1) > 0` (note: `dim=-1`, NOT `dim=1`)
- **Shape validation**: Raises `ValueError` with clear message if mask shape is wrong
- **Guidance in errors**: Error messages suggest common fixes (e.g., "you likely used .any(dim=1) instead of .any(dim=-1)")

### 2.2 Validation Helper

Created `validate_lidar_mask_shape()` for explicit shape checks:

```python
def validate_lidar_mask_shape(
    mask: torch.Tensor,
    expected_batch: int,
    expected_points: int,
    context: str = ""
) -> None:
    """
    Validate that a LiDAR mask has the correct shape.
    Raises ValueError with clear message for common bugs like (B, 4) masks.
    """
```

---

## 3. Code Changes

### 3.1 `modules/utils/core.py`

Added:
- `normalize_lidar_points_and_mask()` - authoritative normalization function
- `validate_lidar_mask_shape()` - explicit shape validation

### 3.2 `modules/utils/__init__.py`

Updated exports to include new functions.

### 3.3 `modules/bev_fusion_model.py` - `forward()`

Updated to:
1. Detect temporal dimension in inputs
2. Delegate to `forward_sequence()` if T > 1
3. Squeeze temporal dimension if T == 1
4. Use `normalize_lidar_points_and_mask()` for LiDAR inputs

```python
def forward(self, lidar_points, camera_images, ...):
    # Detect temporal dimension
    has_temporal_camera = camera_images.dim() == 6

    # If T > 1, delegate to forward_sequence
    if has_temporal_camera and camera_images.shape[1] > 1:
        return self.forward_sequence(...)

    # Handle T=1 by squeezing
    if has_temporal_camera and camera_images.shape[1] == 1:
        camera_images = camera_images[:, 0]
        ...

    # Normalize LiDAR inputs using authoritative helper
    lidar_points_3d, lidar_mask_2d = normalize_lidar_points_and_mask(
        lidar_points, lidar_mask, squeeze_temporal=True
    )

    # Now proceed with validated shapes
    lidar_bev = self.lidar_encoder(lidar_points_3d, mask=lidar_mask_2d)
```

### 3.4 `modules/lidar_encoder.py` - `forward_padded()`

Updated to:
1. Use `normalize_lidar_points_and_mask()` for robust handling
2. Use `validate_lidar_mask_shape()` as belt-and-suspenders check

```python
def forward_padded(self, points, mask=None):
    from .utils.core import normalize_lidar_points_and_mask, validate_lidar_mask_shape

    # Use authoritative normalization function
    points_3d, mask_2d = normalize_lidar_points_and_mask(
        points, mask, squeeze_temporal=True
    )

    # Double-check mask shape (catches (B, 4) bug)
    validate_lidar_mask_shape(
        mask_2d, batch_size, num_points,
        context="LiDARBEVEncoder.forward_padded"
    )
```

---

## 4. New Tests

### 4.1 `tests/test_mask_shape_regression.py`

13 tests covering:
- Wrong mask shape `(B, 4)` raises `ValueError`
- Wrong mask shape `(B, N, 4)` raises `ValueError`
- Correct mask shape `(B, N)` works
- `validate_lidar_mask_shape()` catches bugs
- Error messages provide guidance about `dim=1` vs `dim=-1`
- Mask inferred with correct dimension produces `(B, N)`
- 4D input with T=1 squeezes correctly
- 4D input with 3D mask handled correctly
- Multi-frame input uses last timestep
- Encoder integration tests

### 4.2 `tests/test_forward_accepts_T1_sequence.py`

7 tests covering:
- `forward()` with T=1 temporal inputs doesn't crash
- `forward()` with T=1 and explicit `lidar_mask` works
- `forward()` with T=1 and `return_intermediate=True` produces fused_bev
- `forward()` with standard 3D inputs still works
- Various batch sizes and point counts

### 4.3 `tests/test_padding_invariance.py`

6 tests covering:
- Identical valid points with different padding produce same output
- Non-zero garbage padding doesn't leak into features
- All-padding input produces valid output (no NaN)
- Mask correctly filters trailing padding
- Mask correctly filters scattered padding
- `forward_list` equals `forward_padded` for same valid points

---

## 5. Test Results

### 5.1 New Tests

```
tests/test_mask_shape_regression.py ............. [13 passed]
tests/test_forward_accepts_T1_sequence.py ....... [7 passed]
tests/test_padding_invariance.py ................ [6 passed]

Total: 26 passed
```

### 5.2 Existing Tests (No Regressions)

```
tests/test_lidar_masking.py ..................... [11 passed]
tests/test_scene_boundary.py .................... [7 passed]

Total: 18 passed
```

---

## 6. How to Reproduce and Verify

### 6.1 Reproduce the Original Bug

The bug occurred when running Case B fusion forward in the notebook:

```python
# This would crash before the fix:
batch = get_one_batch(cfg_b, split='val', device='cuda')
# batch['lidar_points'].shape = (1, 1, 35000, 4)

output = model.forward(
    lidar_points=batch['lidar_points'],  # (B, T, N, 4)
    camera_images=batch['img'],          # (B, T, N_cam, 3, H, W)
    ...
)
```

### 6.2 Verify the Fix

```bash
# Run the new regression tests
pytest tests/test_mask_shape_regression.py -v

# Run the T=1 sequence tests
pytest tests/test_forward_accepts_T1_sequence.py -v

# Run padding invariance tests
pytest tests/test_padding_invariance.py -v

# Run all LiDAR-related tests
pytest tests/test_lidar_masking.py tests/test_scene_boundary.py -v
```

### 6.3 Verify in Notebook

Re-run the fusion Case B in the notebook - it should now work without the boolean index error.

---

## 7. Acceptance Checklist

| Requirement | Status |
|-------------|--------|
| Case B forward runs without crashing | ✅ |
| "boolean index...size 35000 vs 4" error eliminated | ✅ |
| Centralized `normalize_lidar_points_and_mask()` helper | ✅ |
| Shape validation with clear error messages | ✅ |
| Handles 4D temporal inputs (B, T, N, 4) | ✅ |
| Handles T=1 by squeezing temporal dimension | ✅ |
| Mask inference uses correct dim=-1 reduction | ✅ |
| `test_mask_shape_regression.py` tests pass | ✅ (13 tests) |
| `test_forward_accepts_T1_sequence.py` tests pass | ✅ (7 tests) |
| `test_padding_invariance.py` tests pass | ✅ (6 tests) |
| No regressions in existing tests | ✅ |

---

## 8. Files Modified

| File | Changes |
|------|---------|
| `modules/utils/core.py` | Added `normalize_lidar_points_and_mask()`, `validate_lidar_mask_shape()` |
| `modules/utils/__init__.py` | Updated exports |
| `modules/bev_fusion_model.py` | Updated `forward()` to handle temporal dims |
| `modules/lidar_encoder.py` | Updated `forward_padded()` to use helper |

## 9. Files Created

| File | Description |
|------|-------------|
| `tests/test_mask_shape_regression.py` | 13 tests for mask shape validation |
| `tests/test_forward_accepts_T1_sequence.py` | 7 tests for T=1 handling |
| `tests/test_padding_invariance.py` | 6 tests for padding invariance |
| `docs/MASK_AND_FORWARD_FIX_REPORT.md` | This report |

---

*Report generated as part of BEVFormer++ LiDAR Mask Bug Fix*
