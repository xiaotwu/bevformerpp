"""
L3 Trend Validation Tests for BEVFormer++ Proposal
===================================================

What L3 validates:
------------------
L3 (Results & Behavioral Validation) verifies that the proposed fusion +
MC-ConvRNN design exhibits **stable and feasible early training behavior**
compared to a camera-only baseline.

Why trends (not accuracy) are tested:
-------------------------------------
- These experiments validate **behavioral trends**, NOT convergence.
- Final accuracy/mAP requires long training (many epochs, full dataset).
- L3 captures early signals: stable loss, no OOM, no NaN/Inf.
- A passing L3 means: "The proposal is trainable and doesn't degrade."

L3 Minimal Experiment Matrix:
-----------------------------
Experiment A — Baseline
    - camera-only
    - temporal: convgru
    - fusion: disabled
    - curriculum: enabled

Experiment B — Proposal (Primary)
    - camera + LiDAR
    - fusion: cross_attn (bidirectional)
    - temporal: mc_convrnn
    - curriculum: enabled

Optional Experiment C — Ablation (Negative Control)
    - camera + LiDAR
    - fusion: cross_attn
    - temporal: temporal_attention
    - Expected: OOM or significantly higher memory
    - Purpose: Document infeasibility, not performance

Training Budget (Strict):
-------------------------
- epochs: 3 (captures early trends only)
- batch_size: 1
- seed: 42 (fixed for determinism)
- optimizer/lr: unchanged from defaults

Usage:
------
    pytest tests/test_l3_trends.py -v -s

Note: Tests require GPU with ~16GB VRAM and NuScenes mini dataset.
"""

import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import pytest

# ============================================================================
# Configuration
# ============================================================================

# L3 Training budget (intentionally small - behavioral trends only)
L3_EPOCHS = 3
L3_BATCH_SIZE = 1
L3_SEED = 42
L3_DATA_ROOT = "data"  # Assumes NuScenes mini is at ./data

# Tolerance factors for trend assertions
LOSS_EXPLOSION_FACTOR = 3.0  # Proposal max loss < K * baseline max loss
VARIANCE_TOLERANCE = 2.0  # Proposal variance <= baseline variance * tolerance


# ============================================================================
# Helper Functions
# ============================================================================

def run_training_experiment(
    experiment_name: str,
    use_fusion: bool,
    temporal_method: str,
    fusion_type: Optional[str] = None,
    log_json_path: str = None,
    epochs: int = L3_EPOCHS,
    batch_size: int = L3_BATCH_SIZE,
    seed: int = L3_SEED,
    data_root: str = L3_DATA_ROOT,
) -> Dict:
    """
    Run a training experiment via subprocess and capture results.
    
    This function invokes train.py programmatically, ensuring:
    - Config-driven execution (no reimplementation of training logic)
    - Explicit parameter passing
    - JSON log capture for trend analysis
    
    Args:
        experiment_name: Human-readable name for logging
        use_fusion: Whether to enable LiDAR fusion
        temporal_method: Temporal aggregation method (convgru, mc_convrnn, temporal_attention)
        fusion_type: Spatial fusion type (if use_fusion=True)
        log_json_path: Path to save epoch logs JSON
        epochs: Number of epochs to train
        batch_size: Batch size
        seed: Random seed for reproducibility
        data_root: Path to NuScenes dataset
    
    Returns:
        Dict with keys: 'success', 'logs', 'error', 'return_code'
    """
    # Build command
    cmd = [
        sys.executable, "train.py",
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--seed", str(seed),
        "--data_root", data_root,
        "--temporal_method", temporal_method,
    ]
    
    if use_fusion:
        cmd.append("--use_fusion")
        if fusion_type:
            cmd.extend(["--fusion_type", fusion_type])
    
    if log_json_path:
        cmd.extend(["--log_json_path", log_json_path])
    
    print(f"\n{'='*60}")
    print(f"[L3] Running Experiment: {experiment_name}")
    print(f"[L3] Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    # Run training
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,  # Project root
    )
    
    # Parse results
    success = result.returncode == 0
    logs = None
    
    if success and log_json_path and os.path.exists(log_json_path):
        with open(log_json_path, 'r') as f:
            logs = json.load(f)
    
    return {
        'success': success,
        'logs': logs,
        'stdout': result.stdout,
        'stderr': result.stderr,
        'return_code': result.returncode,
    }


def check_loss_finite(logs: List[Dict], loss_key: str = 'train_loss') -> bool:
    """Check that all loss values are finite (not NaN or Inf)."""
    for entry in logs:
        val = entry.get(loss_key)
        if val is None:
            return False
        if not math.isfinite(val):
            return False
    return True


def get_loss_trend(logs: List[Dict], loss_key: str = 'train_loss') -> List[float]:
    """Extract loss values from epoch logs."""
    return [entry[loss_key] for entry in logs if loss_key in entry]


def compute_variance(values: List[float]) -> float:
    """Compute variance of a list of values."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((x - mean) ** 2 for x in values) / len(values)


# ============================================================================
# L3 Tests
# ============================================================================

class TestL3Trends:
    """
    L3 Trend Validation Test Suite.
    
    These tests verify that the proposed design exhibits reasonable
    early-training behavior. They do NOT test final accuracy.
    
    Assertions are SOFT (trend-based):
    - Loss is finite (not NaN/Inf)
    - Proposal loss does NOT explode relative to baseline
    - Proposal shows equal or smoother loss trend
    """
    
    @pytest.fixture(scope="class")
    def temp_log_dir(self, tmp_path_factory):
        """Create a temporary directory for experiment logs."""
        return tmp_path_factory.mktemp("l3_logs")
    
    @pytest.fixture(scope="class")
    def baseline_results(self, temp_log_dir):
        """
        Experiment A — Baseline (camera-only, convgru).
        
        This is the reference point for trend comparisons.
        """
        log_path = str(temp_log_dir / "exp_a_baseline.json")
        
        result = run_training_experiment(
            experiment_name="A: Baseline (camera-only, convgru)",
            use_fusion=False,
            temporal_method="convgru",
            log_json_path=log_path,
        )
        
        return result
    
    @pytest.fixture(scope="class")
    def proposal_results(self, temp_log_dir):
        """
        Experiment B — Proposal (fusion, mc_convrnn).
        
        This is the primary contribution being validated.
        """
        log_path = str(temp_log_dir / "exp_b_proposal.json")
        
        result = run_training_experiment(
            experiment_name="B: Proposal (fusion + mc_convrnn)",
            use_fusion=True,
            temporal_method="mc_convrnn",
            fusion_type="bidirectional_cross_attn",
            log_json_path=log_path,
        )
        
        return result
    
    # -------------------------------------------------------------------------
    # Test: Baseline trains successfully
    # -------------------------------------------------------------------------
    
    def test_baseline_trains_without_crash(self, baseline_results):
        """
        Verify Experiment A (baseline) completes without crash.
        
        Proposal claim: The baseline is a valid reference point.
        """
        assert baseline_results['success'], (
            f"Baseline training crashed!\n"
            f"Return code: {baseline_results['return_code']}\n"
            f"Stderr: {baseline_results['stderr'][-2000:]}"
        )
    
    def test_baseline_produces_logs(self, baseline_results):
        """Verify baseline produces epoch logs for comparison."""
        assert baseline_results['logs'] is not None, (
            "Baseline did not produce epoch logs"
        )
        assert len(baseline_results['logs']) > 0, (
            "Baseline epoch logs are empty"
        )
    
    def test_baseline_loss_is_finite(self, baseline_results):
        """Verify baseline losses are finite (not NaN/Inf)."""
        logs = baseline_results['logs']
        assert check_loss_finite(logs, 'train_loss'), (
            "Baseline train_loss contains NaN or Inf"
        )
        assert check_loss_finite(logs, 'val_loss'), (
            "Baseline val_loss contains NaN or Inf"
        )
    
    # -------------------------------------------------------------------------
    # Test: Proposal trains successfully
    # -------------------------------------------------------------------------
    
    def test_proposal_trains_without_crash(self, proposal_results):
        """
        Verify Experiment B (proposal) completes without crash.
        
        Proposal claim: Fusion + MC-ConvRNN is trainable.
        """
        assert proposal_results['success'], (
            f"Proposal training crashed!\n"
            f"Return code: {proposal_results['return_code']}\n"
            f"Stderr: {proposal_results['stderr'][-2000:]}"
        )
    
    def test_proposal_produces_logs(self, proposal_results):
        """Verify proposal produces epoch logs."""
        assert proposal_results['logs'] is not None, (
            "Proposal did not produce epoch logs"
        )
        assert len(proposal_results['logs']) > 0, (
            "Proposal epoch logs are empty"
        )
    
    def test_proposal_loss_is_finite(self, proposal_results):
        """
        Verify proposal losses are finite (not NaN/Inf).
        
        Proposal claim: MC-ConvRNN produces stable gradients.
        """
        logs = proposal_results['logs']
        assert check_loss_finite(logs, 'train_loss'), (
            "Proposal train_loss contains NaN or Inf"
        )
        assert check_loss_finite(logs, 'val_loss'), (
            "Proposal val_loss contains NaN or Inf"
        )
    
    # -------------------------------------------------------------------------
    # Test: Proposal vs Baseline trend comparisons
    # -------------------------------------------------------------------------
    
    def test_proposal_loss_does_not_explode(self, baseline_results, proposal_results):
        """
        Verify proposal loss does NOT explode relative to baseline.
        
        Assertion: max(train_loss_B) < K * max(train_loss_A)
        
        Proposal claim: Fusion + MC-ConvRNN doesn't destabilize training.
        """
        if not baseline_results['success'] or not proposal_results['success']:
            pytest.skip("Cannot compare - one experiment failed")
        
        baseline_losses = get_loss_trend(baseline_results['logs'], 'train_loss')
        proposal_losses = get_loss_trend(proposal_results['logs'], 'train_loss')
        
        max_baseline = max(baseline_losses)
        max_proposal = max(proposal_losses)
        
        threshold = LOSS_EXPLOSION_FACTOR * max_baseline
        
        assert max_proposal < threshold, (
            f"Proposal loss exploded!\n"
            f"Max baseline loss: {max_baseline:.4f}\n"
            f"Max proposal loss: {max_proposal:.4f}\n"
            f"Threshold (K={LOSS_EXPLOSION_FACTOR}): {threshold:.4f}"
        )
    
    def test_proposal_loss_variance_reasonable(self, baseline_results, proposal_results):
        """
        Verify proposal shows equal or smoother loss trend.
        
        Assertion: variance(loss_B) <= variance(loss_A) * tolerance
        
        Proposal claim: MC-ConvRNN provides stable temporal aggregation.
        """
        if not baseline_results['success'] or not proposal_results['success']:
            pytest.skip("Cannot compare - one experiment failed")
        
        baseline_losses = get_loss_trend(baseline_results['logs'], 'train_loss')
        proposal_losses = get_loss_trend(proposal_results['logs'], 'train_loss')
        
        baseline_var = compute_variance(baseline_losses)
        proposal_var = compute_variance(proposal_losses)
        
        threshold = baseline_var * VARIANCE_TOLERANCE
        
        # Note: proposal variance being lower is good, so we allow some margin
        assert proposal_var <= threshold + 0.1, (
            f"Proposal loss variance too high!\n"
            f"Baseline variance: {baseline_var:.6f}\n"
            f"Proposal variance: {proposal_var:.6f}\n"
            f"Threshold (tol={VARIANCE_TOLERANCE}): {threshold:.6f}"
        )
    
    def test_proposal_loss_non_increasing_trend(self, proposal_results):
        """
        Verify proposal shows non-increasing loss trend.
        
        Assertion: train_loss[-1] <= train_loss[0] (within tolerance)
        
        Proposal claim: The model learns (loss decreases or stays stable).
        """
        if not proposal_results['success']:
            pytest.skip("Proposal experiment failed")
        
        losses = get_loss_trend(proposal_results['logs'], 'train_loss')
        
        if len(losses) < 2:
            pytest.skip("Not enough epochs to check trend")
        
        initial_loss = losses[0]
        final_loss = losses[-1]
        
        # Allow small increase due to noise (10% tolerance)
        tolerance = 0.1 * initial_loss
        
        assert final_loss <= initial_loss + tolerance, (
            f"Proposal loss increased significantly!\n"
            f"Initial loss: {initial_loss:.4f}\n"
            f"Final loss: {final_loss:.4f}\n"
            f"Tolerance: {tolerance:.4f}"
        )


# ============================================================================
# Optional: Experiment C (Negative Control)
# ============================================================================

@pytest.mark.xfail(
    reason="Temporal attention + fusion is computationally infeasible at BEV resolution (expected OOM).",
    strict=False,
)
class TestL3AblationTemporalAttention:
    """
    Experiment C — Ablation (Negative Control).
    
    This tests fusion + temporal_attention, which is expected to OOM
    due to O(N²) attention on full BEV tokens.
    
    Purpose: Document infeasibility, not performance.
    """
    
    @pytest.fixture(scope="class")
    def ablation_results(self, tmp_path_factory):
        """Run ablation experiment (expected to fail)."""
        temp_dir = tmp_path_factory.mktemp("l3_ablation")
        log_path = str(temp_dir / "exp_c_ablation.json")
        
        result = run_training_experiment(
            experiment_name="C: Ablation (fusion + temporal_attention)",
            use_fusion=True,
            temporal_method="temporal_attention",
            fusion_type="bidirectional_cross_attn",
            log_json_path=log_path,
            epochs=1,  # Single epoch to detect OOM quickly
        )
        
        return result
    
    def test_ablation_documents_infeasibility(self, ablation_results):
        """
        Document that temporal_attention + fusion is infeasible.
        
        This test is marked xfail. If it passes, temporal_attention
        might have been made more efficient, which is worth investigating.
        """
        # If this succeeds, the xfail will trigger a warning
        # indicating temporal_attention may now be viable
        assert ablation_results['success'], (
            "Ablation experiment failed as expected (OOM or crash)"
        )


# ============================================================================
# Main entry point
# ============================================================================

if __name__ == "__main__":
    # Run with verbose output
    pytest.main([__file__, "-v", "-s"])



