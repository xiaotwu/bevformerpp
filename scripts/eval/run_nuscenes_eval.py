#!/usr/bin/env python3
"""
Run official nuScenes detection evaluation.

This script runs the official nuScenes devkit evaluation on exported predictions.
It requires the nuScenes devkit to be installed:
    pip install nuscenes-devkit

Usage:
    python scripts/eval/run_nuscenes_eval.py \
        --predictions outputs/predictions.json \
        --data_root data \
        --version v1.0-mini \
        --output outputs/metrics_official.json

If the devkit is not installed, this script will gracefully exit with instructions.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def check_nuscenes_devkit():
    """Check if nuScenes devkit is installed."""
    try:
        from nuscenes import NuScenes
        from nuscenes.eval.detection.config import config_factory
        from nuscenes.eval.detection.evaluate import NuScenesEval
        return True
    except ImportError:
        return False


def run_official_eval(
    predictions_path: str,
    data_root: str,
    version: str,
    eval_set: str,
    output_path: str,
    verbose: bool = True
) -> dict:
    """Run official nuScenes detection evaluation.

    Args:
        predictions_path: Path to predictions JSON
        data_root: Path to nuScenes data
        version: nuScenes version (e.g., 'v1.0-mini')
        eval_set: Evaluation set ('val' or 'test')
        output_path: Path to save evaluation results
        verbose: Whether to print verbose output

    Returns:
        Dictionary of evaluation metrics
    """
    from nuscenes import NuScenes
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.detection.evaluate import NuScenesEval

    # Load nuScenes
    if verbose:
        print(f"Loading nuScenes {version} from {data_root}...")
    nusc = NuScenes(version=version, dataroot=data_root, verbose=verbose)

    # Get evaluation config
    cfg = config_factory('detection_cvpr_2019')

    # Create output directory
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run evaluation
    if verbose:
        print(f"Running evaluation on {eval_set} set...")
    nusc_eval = NuScenesEval(
        nusc,
        config=cfg,
        result_path=predictions_path,
        eval_set=eval_set,
        output_dir=str(output_dir),
        verbose=verbose
    )

    # Compute metrics
    metrics = nusc_eval.main(render_curves=False)

    # Extract key metrics
    results = {
        'mAP': metrics['mean_ap'],
        'NDS': metrics['nd_score'],
        'mATE': metrics['mean_trans_err'],
        'mASE': metrics['mean_scale_err'],
        'mAOE': metrics['mean_orient_err'],
        'mAVE': metrics['mean_vel_err'],
        'mAAE': metrics['mean_attr_err'],
        'per_class_ap': {},
        'per_class_tp': {}
    }

    # Per-class results
    for class_name in metrics['label_aps']:
        results['per_class_ap'][class_name] = metrics['label_aps'][class_name]
        results['per_class_tp'][class_name] = metrics['label_tp_errors'][class_name]

    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    if verbose:
        print(f"\n{'='*50}")
        print("Official nuScenes Detection Metrics")
        print(f"{'='*50}")
        print(f"mAP:  {results['mAP']:.4f}")
        print(f"NDS:  {results['NDS']:.4f}")
        print(f"mATE: {results['mATE']:.4f}")
        print(f"mASE: {results['mASE']:.4f}")
        print(f"mAOE: {results['mAOE']:.4f}")
        print(f"mAVE: {results['mAVE']:.4f}")
        print(f"mAAE: {results['mAAE']:.4f}")
        print(f"{'='*50}")
        print(f"Results saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run official nuScenes evaluation")
    parser.add_argument("--predictions", type=str, required=True,
                        help="Path to predictions JSON file")
    parser.add_argument("--data_root", type=str, default="data",
                        help="Path to nuScenes data")
    parser.add_argument("--version", type=str, default="v1.0-mini",
                        help="nuScenes version")
    parser.add_argument("--eval_set", type=str, default="val",
                        choices=["train", "val", "test"],
                        help="Evaluation set")
    parser.add_argument("--output", type=str, default="outputs/metrics_official.json",
                        help="Output path for metrics JSON")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress verbose output")

    args = parser.parse_args()

    # Check if devkit is installed
    if not check_nuscenes_devkit():
        print("=" * 60)
        print("ERROR: nuScenes devkit is not installed.")
        print("=" * 60)
        print("\nTo install the devkit, run:")
        print("    pip install nuscenes-devkit")
        print("\nAlternatively, you can use the exported predictions JSON")
        print("for custom evaluation or visualization.")
        print(f"\nPredictions file: {args.predictions}")
        print("=" * 60)
        sys.exit(1)

    # Check predictions file exists
    if not Path(args.predictions).exists():
        print(f"ERROR: Predictions file not found: {args.predictions}")
        print("\nRun export_nuscenes_predictions.py first to generate predictions.")
        sys.exit(1)

    # Run evaluation
    run_official_eval(
        predictions_path=args.predictions,
        data_root=args.data_root,
        version=args.version,
        eval_set=args.eval_set,
        output_path=args.output,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
