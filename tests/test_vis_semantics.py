"""
Tests for visualization semantic correctness.

Verifies that:
1. Camera-only mode does NOT produce fake fused/lidar features
2. Fusion mode properly distinguishes all three feature types
3. Peak extraction uses proper NMS (no clustering)
4. Normalization is applied correctly
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.vis.core import (
    normalize_for_display,
    compute_bev_energy,
    compute_shared_normalization,
    extract_nms_peaks,
    get_gt_centers,
    extract_visual_tensors,
    VisualTensors,
    Peak,
)


class TestModeSemantics:
    """Tests for strict camera_only vs fusion mode semantics."""
    
    def test_camera_only_no_lidar_bev(self):
        """Camera-only mode should NOT produce lidar_bev."""
        preds = {'cls_scores': torch.randn(1, 10, 50, 50)}
        tensors = {'bev_features': torch.randn(1, 64, 50, 50)}
        
        vis = extract_visual_tensors(preds, tensors, mode='camera_only')
        
        assert vis.mode == 'camera_only'
        assert vis.lidar_bev is None, "Camera-only should NOT have lidar_bev"
    
    def test_camera_only_no_fused_bev(self):
        """Camera-only mode should NOT produce fused_bev."""
        preds = {'cls_scores': torch.randn(1, 10, 50, 50)}
        tensors = {'bev_features': torch.randn(1, 64, 50, 50)}
        
        vis = extract_visual_tensors(preds, tensors, mode='camera_only')
        
        assert vis.fused_bev is None, "Camera-only should NOT have fused_bev"
    
    def test_camera_only_has_camera_bev(self):
        """Camera-only mode should have camera_bev."""
        preds = {'cls_scores': torch.randn(1, 10, 50, 50)}
        tensors = {'bev_features': torch.randn(1, 64, 50, 50)}
        
        vis = extract_visual_tensors(preds, tensors, mode='camera_only')
        
        assert vis.camera_bev is not None, "Camera-only should have camera_bev"
    
    def test_fusion_mode_can_have_all_features(self):
        """Fusion mode can have all three BEV features."""
        preds = {'cls_scores': torch.randn(1, 10, 50, 50)}
        tensors = {
            'camera_bev': torch.randn(1, 64, 50, 50),
            'lidar_bev': torch.randn(1, 64, 50, 50),
            'fused_bev': torch.randn(1, 64, 50, 50),
        }
        
        vis = extract_visual_tensors(preds, tensors, mode='fusion')
        
        assert vis.mode == 'fusion'
        assert vis.camera_bev is not None
        assert vis.lidar_bev is not None
        assert vis.fused_bev is not None
    
    def test_available_features_correct(self):
        """available_features() should list only non-None features."""
        preds = {'cls_scores': torch.randn(1, 10, 50, 50)}
        tensors = {'bev_features': torch.randn(1, 64, 50, 50)}
        
        vis = extract_visual_tensors(preds, tensors, mode='camera_only')
        available = vis.available_features()
        
        assert 'camera_bev' in available
        assert 'lidar_bev' not in available
        assert 'fused_bev' not in available


class TestNormalization:
    """Tests for normalization correctness."""
    
    def test_percentile_normalization_output_range(self):
        """Output should be in [0, 1]."""
        arr = np.random.randn(100, 100).astype(np.float32) * 100
        norm = normalize_for_display(torch.from_numpy(arr))
        
        assert norm.min() >= 0.0
        assert norm.max() <= 1.0
    
    def test_constant_tensor_handling(self):
        """Constant tensors should not cause division by zero."""
        arr = np.ones((50, 50), dtype=np.float32) * 5.0
        norm = normalize_for_display(torch.from_numpy(arr))
        
        assert not np.any(np.isnan(norm)), "Should not produce NaN"
        assert not np.any(np.isinf(norm)), "Should not produce Inf"
    
    def test_bev_energy_reduces_channels(self):
        """BEV energy should reduce (B, C, H, W) to (H, W)."""
        feat = torch.randn(2, 64, 100, 100)
        energy = compute_bev_energy(feat)
        
        assert energy.shape == (100, 100), f"Expected (100, 100), got {energy.shape}"
    
    def test_shared_normalization_excludes_none(self):
        """Shared normalization should skip None tensors."""
        tensors = [
            torch.randn(1, 64, 50, 50),
            None,
            torch.randn(1, 64, 50, 50),
        ]
        
        vmin, vmax = compute_shared_normalization(tensors)
        
        assert vmin < vmax, "Should produce valid range"
    
    def test_shared_normalization_empty_list(self):
        """Empty list should return default range."""
        vmin, vmax = compute_shared_normalization([])
        
        assert vmin == 0.0
        assert vmax == 1.0


class TestNMSPeaks:
    """Tests for NMS-based peak extraction."""
    
    def test_peaks_use_sigmoid(self):
        """Peaks should be extracted from sigmoid-transformed logits."""
        # Large negative logit -> near 0 probability
        heatmap = torch.full((1, 1, 50, 50), -10.0)
        heatmap[0, 0, 25, 25] = 5.0  # High logit -> high probability
        
        peaks = extract_nms_peaks(heatmap, k=10, radius=3, thresh=0.1)
        
        # Should find the single peak
        assert len(peaks) >= 1
        assert peaks[0].score > 0.9  # sigmoid(5) ≈ 0.993
    
    def test_nms_suppresses_nearby(self):
        """NMS should suppress nearby peaks."""
        # Use negative background so sigmoid gives low scores
        heatmap = torch.full((1, 1, 100, 100), -10.0)
        # Two peaks very close together
        heatmap[0, 0, 50, 50] = 5.0  # Higher
        heatmap[0, 0, 51, 51] = 4.0  # Lower, within radius
        
        peaks = extract_nms_peaks(heatmap, k=10, radius=3, thresh=0.5)
        
        # Only the higher peak should survive (the lower is within radius)
        assert len(peaks) == 1
        assert peaks[0].y == 50 and peaks[0].x == 50
    
    def test_nms_keeps_distant_peaks(self):
        """NMS should keep peaks that are far apart."""
        heatmap = torch.full((1, 1, 100, 100), -10.0)
        heatmap[0, 0, 20, 20] = 5.0
        heatmap[0, 0, 80, 80] = 4.0  # Far from first peak
        
        peaks = extract_nms_peaks(heatmap, k=10, radius=3, thresh=0.5)
        
        assert len(peaks) == 2, f"Expected 2 peaks, got {len(peaks)}"
    
    def test_threshold_filters_low_scores(self):
        """Threshold should filter out low-scoring peaks."""
        heatmap = torch.full((1, 1, 50, 50), -10.0)  # Low background
        heatmap[0, 0, 25, 25] = 2.0  # sigmoid(2) ≈ 0.88, above 0.5 threshold
        heatmap[0, 0, 30, 30] = -2.0  # sigmoid(-2) ≈ 0.12, below threshold
        
        peaks = extract_nms_peaks(heatmap, k=10, radius=3, thresh=0.5)
        
        # Only the higher peak should pass threshold
        assert len(peaks) == 1
        assert peaks[0].y == 25 and peaks[0].x == 25
    
    def test_multi_class_peaks(self):
        """Should extract peaks from multiple classes."""
        heatmap = torch.full((1, 3, 50, 50), -10.0)
        heatmap[0, 0, 10, 10] = 5.0  # Class 0
        heatmap[0, 1, 25, 25] = 4.0  # Class 1
        heatmap[0, 2, 40, 40] = 3.0  # Class 2
        
        peaks = extract_nms_peaks(heatmap, k=10, radius=3, thresh=0.5)
        
        assert len(peaks) == 3, f"Expected 3 peaks, got {len(peaks)}"
        classes = {p.class_id for p in peaks}
        assert classes == {0, 1, 2}


class TestGTCenters:
    """Tests for GT center extraction."""
    
    def test_hard_center_extraction(self):
        """Should extract centers from hard_center (binary) targets."""
        targets = torch.zeros(1, 3, 100, 100)
        targets[0, 0, 30, 30] = 1.0
        targets[0, 1, 50, 50] = 1.0
        targets[0, 2, 70, 70] = 1.0
        
        centers = get_gt_centers(targets)
        
        assert len(centers) == 3
        # Each center is (y, x, class_id)
        yx_set = {(c[0], c[1]) for c in centers}
        assert (30, 30) in yx_set
        assert (50, 50) in yx_set
        assert (70, 70) in yx_set


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

