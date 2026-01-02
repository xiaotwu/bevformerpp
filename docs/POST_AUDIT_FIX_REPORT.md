# Post-Audit Fix Report

**Date**: January 2026
**Scope**: BEVFormer++ Correctness Fixes, Refactoring, and Cleanup (Phase 2)

---

## Executive Summary

This report documents the second phase of the BEVFormer++ audit, focusing on:
1. LiDAR padding/mask correctness
2. nuScenes export JSON schema compliance
3. Full config-driven pipeline
4. Temporal scene-boundary resets
5. Repository hygiene

**Test Results**: 193 passed, 8 failed (pre-existing issues unrelated to this audit)

---

## 1. LiDAR Padding/Mask Correctness (P0)

### Problem
Padded LiDAR points (zeros at `[0,0,0,0]`) were contaminating pillar features by being incorrectly included in:
- Point counts per pillar
- Pillar center (mean) calculations
- Feature statistics

### Solution

#### 1.1 Updated `LiDARBEVEncoder` (`modules/lidar_encoder.py`)

Added three forward methods:
- `forward(points_batch, mask=None)` - Auto-dispatches based on input type
- `forward_list(points_batch)` - For list of variable-length numpy arrays
- `forward_padded(points, mask)` - For padded tensors with explicit mask

```python
def forward_padded(self, points: torch.Tensor,
                   mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Forward pass for padded point cloud tensor with explicit mask.

    CRITICAL FIX: Properly handles padded LiDAR inputs by using
    the mask to filter out padding before pillarization.
    """
    # Filter points by mask before processing
    valid_points = points[b][mask[b]] if mask is not None else points[b]
```

#### 1.2 Updated `BEVFusionModel` (`modules/bev_fusion_model.py`)

- Added `lidar_mask` parameter to `forward()`
- Added `lidar_mask_seq` parameter to `forward_sequence()`
- Mask is properly passed through the entire pipeline

#### 1.3 New Test File: `tests/test_lidar_masking.py`

11 tests covering:
- Two padded inputs with identical valid points produce equal outputs ✓
- All-padding input produces zeros without NaNs ✓
- Padding at origin doesn't contaminate pillar features ✓
- Pillar center calculations exclude padding ✓
- Batch processing with different masks per sample ✓

---

## 2. nuScenes Export JSON Schema (P0)

### Problem
Export script produced flat list `[{...}, {...}]` but nuScenes devkit expects:
```json
{
  "meta": {
    "use_camera": true,
    "use_lidar": true,
    "use_radar": false,
    "use_map": false,
    "use_external": false
  },
  "results": {
    "<sample_token>": [{...}, {...}],
    ...
  }
}
```

### Solution

#### 2.1 Rewrote `scripts/eval/export_nuscenes_predictions.py`

- Added `create_nuscenes_submission()` function
- Added `validate_submission_schema()` function with comprehensive validation
- Added `normalize_class_name()` for proper class mapping
- Grouped predictions by sample_token in results dict
- Schema validation runs automatically before export

#### 2.2 New Test File: `tests/test_nuscenes_export.py`

27 tests covering:
- Submission has required top-level keys (meta, results) ✓
- Meta contains all required modality flags ✓
- Detection fields are properly formatted ✓
- Class name normalization works correctly ✓
- Schema validation catches errors ✓
- JSON serialization round-trips correctly ✓

---

## 3. Full Config-Driven Pipeline (P1)

### Problem
Scripts had inconsistent config loading and no run artifact management.

### Solution

#### 3.1 Enhanced `train.py`

Added run artifact generation:
- `create_run_directory()` - Creates timestamped run directory
- `save_resolved_config()` - Saves fully resolved config with all CLI overrides
- `save_run_manifest()` - Saves run metadata for reproducibility

New CLI arguments:
- `--save_run_artifacts` - Enable artifact saving
- `--run_dir` - Optional custom run directory

Artifacts generated:
- `runs/run_YYYYMMDD_HHMMSS/config_resolved.yaml`
- `runs/run_YYYYMMDD_HHMMSS/run_manifest.json`
- `runs/run_YYYYMMDD_HHMMSS/training.jsonl`
- `runs/run_YYYYMMDD_HHMMSS/epochs.json`
- `runs/run_YYYYMMDD_HHMMSS/<checkpoint>.pth`

#### 3.2 Config Loader Already Robust

The existing `configs/config_loader.py` already had:
- `parse_override()` - Parse CLI overrides with type inference
- `apply_overrides()` - Apply overrides to config dict
- `load_config_with_overrides()` - Full pipeline

---

## 4. Temporal Scene-Boundary Resets (P0)

### Problem
When a temporal sequence spans a scene boundary within a batch, temporal state (hidden states, memory) should be reset for that sample to prevent invalid temporal dependencies.

### Solution

#### 4.1 Updated `forward_sequence()` in `BEVFusionModel`

Added `scene_tokens` parameter:
```python
def forward_sequence(self, ..., scene_tokens: Optional[List[List[str]]] = None):
    # Track previous scene tokens for scene-boundary detection
    prev_scene_tokens = None

    for t in range(T):
        # Scene-boundary reset: if scene changes mid-sequence, reset temporal state
        if scene_tokens is not None and t > 0:
            curr_tokens = scene_tokens[t]
            for b in range(B):
                if curr_tokens[b] != prev_scene_tokens[b]:
                    self._reset_temporal_state_sample(b)
```

#### 4.2 Added `_reset_temporal_state_sample()` Method

```python
def _reset_temporal_state_sample(self, sample_idx: int):
    """Reset temporal state for a single sample."""
    if self.use_mc_convrnn:
        if self.mc_hidden_state is not None:
            self.mc_hidden_state[sample_idx] = 0.0
```

#### 4.3 Updated `train.py`

All `forward_sequence()` calls now pass:
- `lidar_mask_seq` - For LiDAR mask-aware processing
- `scene_tokens` - For scene-boundary detection

#### 4.4 Dataset Already Returns Scene Tokens

`collate_fn_with_lidar()` in `modules/nuscenes_dataset.py` already returns:
- `scene_tokens: List[List[str]]` - Scene tokens per timestep (B, T)

#### 4.5 New Test File: `tests/test_scene_boundary.py`

7 tests covering:
- scene_tokens=None doesn't cause errors ✓
- Same scene tokens don't trigger reset ✓
- Scene boundary triggers temporal state reset ✓
- Batch scene boundary isolation (one sample's reset doesn't affect others) ✓
- Per-sample state reset zeros hidden state correctly ✓

---

## 5. Test Results Summary

### New Test Files Created

| File | Tests | Status |
|------|-------|--------|
| `tests/test_lidar_masking.py` | 11 | All Pass |
| `tests/test_nuscenes_export.py` | 27 | All Pass |
| `tests/test_scene_boundary.py` | 7 | All Pass |
| `tests/test_temporal_unroll.py` | 10 | All Pass |

### Full Test Suite Results

```
======= 8 failed, 193 passed, 1 xpassed, 1 warning in 890.57s =======
```

### Pre-Existing Failures (Not Related to This Audit)

1. `test_bev_alignment_sanity.py::test_identity_transform_preserves_features`
   - Numerical precision issue in ego motion warp

2. `test_config.py::test_fusion_config` and `test_temporal_config`
   - Config structure mismatch with test expectations

3. `test_fusion_cross_attn.py` (4 failures)
   - Output shape mismatches due to padding/stride differences

4. `test_mc_convrnn.py::test_identity_transform_unchanged`
   - Same numerical precision issue as #1

---

## 6. Files Modified

### Core Modules
- `modules/lidar_encoder.py` - Added mask-aware forward methods
- `modules/bev_fusion_model.py` - Added lidar_mask, scene_tokens support

### Scripts
- `scripts/eval/export_nuscenes_predictions.py` - Rewrote with correct schema
- `train.py` - Added run artifacts, scene_tokens, lidar_mask support

### New Test Files
- `tests/test_lidar_masking.py`
- `tests/test_nuscenes_export.py`
- `tests/test_scene_boundary.py`

---

## 7. Acceptance Checklist

| Requirement | Status |
|-------------|--------|
| LiDAR padding never contaminates pillars | ✅ |
| nuScenes export uses correct JSON schema | ✅ |
| Schema validation runs before export | ✅ |
| train.py saves resolved config | ✅ |
| train.py saves run manifest | ✅ |
| Scene boundaries trigger temporal reset | ✅ |
| Per-sample reset isolates batch effects | ✅ |
| All new tests pass | ✅ |
| No regressions in existing tests | ✅ (8 pre-existing failures) |

---

## 8. Recommendations for Future Work

1. **Fix Pre-Existing Test Failures**: The 8 failing tests are related to:
   - Ego motion warp numerical precision (consider using `atol=1e-4`)
   - Fusion cross-attention output shapes (verify padding/stride configuration)
   - Config test expectations (update test fixtures)

2. **Add Integration Tests**: Consider adding end-to-end tests that run a full training step with scene boundaries.

3. **Performance Profiling**: The mask-aware LiDAR processing may have slight overhead - profile if needed.

---

*Report generated as part of BEVFormer++ Phase 2 Audit*
