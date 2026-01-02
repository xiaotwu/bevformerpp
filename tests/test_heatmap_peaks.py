"""
Tests for heatmap peak extraction utilities.

Verifies that:
1. Local maxima detection works correctly (NMS-like)
2. Top-K extraction does not cluster at single location
3. Multi-channel handling is correct
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.vis.heatmap_peaks import (
    extract_local_maxima,
    extract_topk_peaks_nms,
    summarize_heatmap,
)


class TestExtractLocalMaxima:
    """Tests for local maxima extraction."""
    
    def test_single_peak(self):
        """Test detection of a single peak."""
        # Create heatmap with single peak at center
        heatmap = torch.zeros(1, 1, 20, 20)
        heatmap[0, 0, 10, 10] = 1.0
        
        peaks = extract_local_maxima(heatmap, kernel_size=3)
        
        # Peak should be preserved
        assert peaks[0, 0, 10, 10] == 1.0
        # Other pixels should be zero
        assert peaks.sum() == 1.0
    
    def test_multiple_peaks(self):
        """Test detection of multiple well-separated peaks."""
        heatmap = torch.zeros(1, 1, 20, 20)
        heatmap[0, 0, 5, 5] = 0.8    # Peak 1
        heatmap[0, 0, 5, 15] = 0.6   # Peak 2
        heatmap[0, 0, 15, 5] = 0.9   # Peak 3
        heatmap[0, 0, 15, 15] = 0.7  # Peak 4
        
        peaks = extract_local_maxima(heatmap, kernel_size=3)
        
        # All 4 peaks should be detected
        expected_locations = [(5, 5), (5, 15), (15, 5), (15, 15)]
        for y, x in expected_locations:
            assert peaks[0, 0, y, x] > 0, f"Peak at ({y}, {x}) not detected"
    
    def test_plateau_handling(self):
        """Test handling of flat regions (plateaus)."""
        heatmap = torch.zeros(1, 1, 10, 10)
        # Create a 3x3 plateau
        heatmap[0, 0, 3:6, 3:6] = 1.0
        
        peaks = extract_local_maxima(heatmap, kernel_size=3)
        
        # Plateaus: all values equal local max, so all should be preserved
        # (This is expected behavior - we don't do strict > comparison)
        assert (peaks[0, 0, 3:6, 3:6] > 0).sum() == 9
    
    def test_suppresses_neighbors(self):
        """Test that non-maxima neighbors are suppressed."""
        heatmap = torch.zeros(1, 1, 10, 10)
        heatmap[0, 0, 5, 5] = 1.0  # Peak
        heatmap[0, 0, 5, 4] = 0.5  # Neighbor (should be suppressed)
        heatmap[0, 0, 4, 5] = 0.5  # Neighbor (should be suppressed)
        
        peaks = extract_local_maxima(heatmap, kernel_size=3)
        
        # Only the center peak should remain
        assert peaks[0, 0, 5, 5] == 1.0
        assert peaks[0, 0, 5, 4] == 0.0, "Non-maximum neighbor should be suppressed"
        assert peaks[0, 0, 4, 5] == 0.0, "Non-maximum neighbor should be suppressed"
    
    def test_2d_input(self):
        """Test with 2D input (H, W)."""
        heatmap = torch.zeros(20, 20)
        heatmap[10, 10] = 1.0
        
        peaks = extract_local_maxima(heatmap, kernel_size=3)
        
        assert peaks.shape == (20, 20)
        assert peaks[10, 10] == 1.0


class TestExtractTopkPeaksNMS:
    """Tests for top-K peak extraction with NMS."""
    
    def test_known_peaks(self):
        """Test extraction of known peak locations."""
        heatmap = torch.zeros(1, 2, 20, 20)  # 2 classes
        
        # Class 0: peaks at (5,5) and (15,15)
        heatmap[0, 0, 5, 5] = 0.9
        heatmap[0, 0, 15, 15] = 0.8
        
        # Class 1: peaks at (5,15) and (15,5)
        heatmap[0, 1, 5, 15] = 0.7
        heatmap[0, 1, 15, 5] = 0.6
        
        coords, scores, _, counts = extract_topk_peaks_nms(
            heatmap, k=10, kernel_size=3, threshold=0.1
        )
        
        # Should find 4 peaks total
        assert len(coords) == 4, f"Expected 4 peaks, got {len(coords)}"
        assert len(scores) == 4
        
        # Verify counts per class
        assert counts[0, 0] == 2, f"Expected 2 peaks in class 0"
        assert counts[0, 1] == 2, f"Expected 2 peaks in class 1"
    
    def test_no_clustering(self):
        """Test that peaks don't cluster at single high-activation region."""
        heatmap = torch.zeros(1, 1, 100, 100)
        
        # Create a high-activation region
        heatmap[0, 0, 40:60, 40:60] = torch.randn(20, 20).abs() * 0.5
        heatmap[0, 0, 50, 50] = 1.0  # Single peak in the region
        
        # Also add well-separated peaks
        heatmap[0, 0, 10, 10] = 0.8
        heatmap[0, 0, 10, 90] = 0.7
        heatmap[0, 0, 90, 10] = 0.6
        heatmap[0, 0, 90, 90] = 0.5
        
        coords, scores, _, _ = extract_topk_peaks_nms(
            heatmap, k=50, kernel_size=5, threshold=0.3
        )
        
        # Extract y, x coordinates
        if len(coords) > 0:
            y_coords = coords[:, 2].numpy()
            x_coords = coords[:, 3].numpy()
            
            # Peaks should be spread out, not all in center region
            # At least some peaks should be far from (50, 50)
            distances = np.sqrt((y_coords - 50)**2 + (x_coords - 50)**2)
            far_peaks = (distances > 30).sum()
            
            assert far_peaks >= 4, \
                f"Expected at least 4 peaks far from center, got {far_peaks}. " \
                f"Peaks may be clustering."
    
    def test_threshold_filtering(self):
        """Test that threshold correctly filters peaks."""
        heatmap = torch.zeros(1, 1, 20, 20)
        heatmap[0, 0, 5, 5] = 0.9    # Above threshold
        heatmap[0, 0, 10, 10] = 0.05  # Below threshold
        heatmap[0, 0, 15, 15] = 0.3   # Above threshold
        
        coords, scores, _, _ = extract_topk_peaks_nms(
            heatmap, k=10, threshold=0.1
        )
        
        # Only 2 peaks should be found (above threshold)
        assert len(coords) == 2, f"Expected 2 peaks above threshold, got {len(coords)}"
        assert all(s > 0.1 for s in scores), "All scores should be above threshold"


class TestSummarizeHeatmap:
    """Tests for heatmap summary statistics."""
    
    def test_hard_center_targets(self):
        """Test summary of hard_center (binary) targets."""
        heatmap = torch.zeros(1, 10, 200, 200)
        # Set a few center pixels to 1
        heatmap[0, 0, 50, 50] = 1.0
        heatmap[0, 1, 100, 100] = 1.0
        heatmap[0, 2, 150, 150] = 1.0
        
        stats = summarize_heatmap(heatmap)
        
        # Should have only {0, 1} as unique values
        assert set(stats.get('unique_values', [])) == {0.0, 1.0}, \
            f"Hard center should have only {{0, 1}}, got {stats.get('unique_values')}"
    
    def test_gaussian_targets(self):
        """Test summary of Gaussian targets."""
        heatmap = torch.zeros(1, 1, 20, 20)
        
        # Create Gaussian-like peak
        y, x = torch.meshgrid(torch.arange(20), torch.arange(20), indexing='ij')
        gaussian = torch.exp(-((x-10)**2 + (y-10)**2) / (2 * 3**2))
        heatmap[0, 0] = gaussian
        
        stats = summarize_heatmap(heatmap)
        
        # Should have many unique values (not just {0, 1})
        assert stats.get('num_unique', len(stats.get('unique_values', []))) > 10, \
            "Gaussian targets should have many unique values"
    
    def test_sigmoid_application(self):
        """Test sigmoid application to logits."""
        logits = torch.tensor([[[[0.0]]]])  # Single pixel
        
        stats_no_sig = summarize_heatmap(logits, apply_sigmoid=False)
        stats_with_sig = summarize_heatmap(logits, apply_sigmoid=True)
        
        assert stats_no_sig['mean'] == 0.0
        assert abs(stats_with_sig['mean'] - 0.5) < 1e-5  # sigmoid(0) = 0.5


class TestMultiChannelHandling:
    """Tests for proper multi-channel/multi-batch handling."""
    
    def test_batch_dimension(self):
        """Test handling of batch dimension."""
        heatmap = torch.zeros(4, 2, 20, 20)  # Batch of 4
        
        # Different peaks in different batches
        heatmap[0, 0, 5, 5] = 1.0
        heatmap[1, 0, 10, 10] = 1.0
        heatmap[2, 0, 15, 15] = 1.0
        heatmap[3, 0, 5, 15] = 1.0
        
        coords, scores, _, counts = extract_topk_peaks_nms(
            heatmap, k=10, threshold=0.1
        )
        
        # Each batch should have 1 peak
        for b in range(4):
            batch_peaks = (coords[:, 0] == b).sum()
            assert batch_peaks == 1, f"Batch {b} should have 1 peak, got {batch_peaks}"
    
    def test_no_axis_mixing(self):
        """Ensure batch/channel axes are not mixed during flattening."""
        heatmap = torch.zeros(2, 3, 10, 10)
        
        # Batch 0, Class 0
        heatmap[0, 0, 2, 2] = 0.9
        # Batch 0, Class 1
        heatmap[0, 1, 5, 5] = 0.8
        # Batch 1, Class 2
        heatmap[1, 2, 8, 8] = 0.7
        
        coords, scores, _, _ = extract_topk_peaks_nms(heatmap, k=10, threshold=0.1)
        
        # Verify correct batch/class assignment
        for i in range(len(coords)):
            b, c, y, x = coords[i].tolist()
            expected_score = heatmap[b, c, y, x].item()
            assert abs(scores[i].item() - expected_score) < 1e-5, \
                f"Score mismatch at ({b}, {c}, {y}, {x})"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])



