#!/usr/bin/env python3
"""
Export model predictions to nuScenes detection JSON format.

This script loads a trained model checkpoint and runs inference on a dataset split,
exporting predictions in the OFFICIAL nuScenes detection format for evaluation
using nuscenes-devkit.

Usage:
    python scripts/eval/export_nuscenes_predictions.py \
        --checkpoint checkpoints/model.pt \
        --config configs/eval_config.yaml \
        --output outputs/predictions.json \
        --split val

Official nuScenes Detection Format:
    {
        "meta": {
            "use_camera": true,
            "use_lidar": true,
            "use_radar": false,
            "use_map": false,
            "use_external": false
        },
        "results": {
            "<sample_token>": [
                {
                    "sample_token": "...",
                    "translation": [x, y, z],
                    "size": [w, l, h],
                    "rotation": [w, x, y, z],
                    "velocity": [vx, vy],
                    "detection_name": "car",
                    "detection_score": 0.95,
                    "attribute_name": ""
                },
                ...
            ],
            ...
        }
    }

Reference:
    https://www.nuscenes.org/object-detection?externalData=all&mapData=all&modalities=Any
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modules.bev_fusion_model import BEVFusionModel
from modules.data_structures import BEVGridConfig, Box3D
from modules.nuscenes_dataset import NuScenesDataset, create_fusion_collate_fn
from torch.utils.data import DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Official nuScenes detection class names (10 classes)
NUSCENES_DETECTION_CLASSES = [
    'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
    'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier'
]


def yaw_to_quaternion(yaw: float) -> List[float]:
    """Convert yaw angle to quaternion [w, x, y, z].

    Args:
        yaw: Yaw angle in radians

    Returns:
        Quaternion as [w, x, y, z]
    """
    try:
        from pyquaternion import Quaternion
        q = Quaternion(axis=[0, 0, 1], angle=yaw)
        return [q.w, q.x, q.y, q.z]
    except ImportError:
        # Fallback: manual quaternion computation for rotation around z-axis
        half_yaw = yaw / 2
        return [np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)]


def transform_box_to_global(
    center: np.ndarray,
    yaw: float,
    ego_pose: np.ndarray
) -> tuple:
    """Transform box center and orientation from ego to global frame.

    Args:
        center: Box center in ego frame [x, y, z]
        yaw: Yaw angle in ego frame
        ego_pose: 4x4 ego pose matrix (ego to global)

    Returns:
        (center_global, quaternion_global) in global frame
    """
    try:
        from pyquaternion import Quaternion

        # Transform center to global frame
        center_homo = np.append(center, 1.0)
        center_global = (ego_pose @ center_homo)[:3]

        # Transform rotation to global frame
        ego_rotation = Quaternion(matrix=ego_pose[:3, :3])
        yaw_quat = Quaternion(axis=[0, 0, 1], angle=yaw)
        global_rotation = ego_rotation * yaw_quat

        return center_global, [global_rotation.w, global_rotation.x,
                               global_rotation.y, global_rotation.z]

    except ImportError:
        # Fallback without pyquaternion
        center_homo = np.append(center, 1.0)
        center_global = (ego_pose @ center_homo)[:3]

        # Extract yaw from ego pose rotation matrix and add box yaw
        ego_yaw = np.arctan2(ego_pose[1, 0], ego_pose[0, 0])
        global_yaw = ego_yaw + yaw
        quat = yaw_to_quaternion(global_yaw)

        return center_global, quat


def normalize_class_name(label: str) -> Optional[str]:
    """Normalize label to official nuScenes detection class name.

    Args:
        label: Raw label string (e.g., 'vehicle.car', 'Car', 'CAR')

    Returns:
        Normalized nuScenes class name or None if not mappable
    """
    label_lower = label.lower()

    # Direct match
    if label_lower in NUSCENES_DETECTION_CLASSES:
        return label_lower

    # Handle hierarchical labels (e.g., 'vehicle.car' -> 'car')
    if '.' in label_lower:
        suffix = label_lower.split('.')[-1]
        if suffix in NUSCENES_DETECTION_CLASSES:
            return suffix

    # Partial match
    for cls_name in NUSCENES_DETECTION_CLASSES:
        if cls_name in label_lower or label_lower in cls_name:
            return cls_name

    logger.warning(f"Could not map label '{label}' to nuScenes class")
    return None


def box3d_to_nuscenes_detection(
    box: Box3D,
    sample_token: str,
    ego_pose: np.ndarray
) -> Optional[Dict]:
    """Convert Box3D to nuScenes detection format.

    Args:
        box: Box3D prediction in ego frame
        sample_token: nuScenes sample token
        ego_pose: 4x4 ego pose matrix (ego to global)

    Returns:
        Dictionary in nuScenes detection format, or None if class invalid
    """
    # Normalize class name
    detection_name = normalize_class_name(box.label)
    if detection_name is None:
        return None

    # Transform to global frame
    center_global, rotation_quat = transform_box_to_global(
        box.center, box.yaw, ego_pose
    )

    return {
        "sample_token": sample_token,
        "translation": center_global.tolist(),
        "size": box.size.tolist(),  # [w, l, h]
        "rotation": rotation_quat,  # [w, x, y, z]
        "velocity": [0.0, 0.0],  # Placeholder (requires tracking)
        "detection_name": detection_name,
        "detection_score": float(box.score),
        "attribute_name": ""  # Placeholder (can be predicted separately)
    }


def create_nuscenes_submission(
    results_by_token: Dict[str, List[Dict]],
    use_camera: bool = True,
    use_lidar: bool = True,
    use_radar: bool = False,
    use_map: bool = False,
    use_external: bool = False
) -> Dict:
    """Create nuScenes submission JSON with proper schema.

    Args:
        results_by_token: Dict mapping sample_token to list of detections
        use_camera: Whether camera data was used
        use_lidar: Whether lidar data was used
        use_radar: Whether radar data was used
        use_map: Whether map data was used
        use_external: Whether external data was used

    Returns:
        nuScenes submission dict with 'meta' and 'results' keys
    """
    return {
        "meta": {
            "use_camera": use_camera,
            "use_lidar": use_lidar,
            "use_radar": use_radar,
            "use_map": use_map,
            "use_external": use_external
        },
        "results": results_by_token
    }


def validate_submission_schema(submission: Dict) -> bool:
    """Validate that submission matches nuScenes expected schema.

    Args:
        submission: Submission dictionary

    Returns:
        True if valid, raises ValueError otherwise
    """
    # Check top-level keys
    if "meta" not in submission:
        raise ValueError("Submission missing 'meta' key")
    if "results" not in submission:
        raise ValueError("Submission missing 'results' key")

    # Check meta keys
    required_meta_keys = ["use_camera", "use_lidar", "use_radar", "use_map", "use_external"]
    for key in required_meta_keys:
        if key not in submission["meta"]:
            raise ValueError(f"Submission meta missing '{key}' key")
        if not isinstance(submission["meta"][key], bool):
            raise ValueError(f"Submission meta['{key}'] must be bool")

    # Check results structure
    if not isinstance(submission["results"], dict):
        raise ValueError("Submission 'results' must be a dict")

    # Check detection format (sample a few)
    required_detection_keys = [
        "sample_token", "translation", "size", "rotation",
        "velocity", "detection_name", "detection_score", "attribute_name"
    ]

    for sample_token, detections in list(submission["results"].items())[:5]:
        if not isinstance(detections, list):
            raise ValueError(f"results['{sample_token}'] must be a list")
        for det in detections:
            for key in required_detection_keys:
                if key not in det:
                    raise ValueError(f"Detection missing '{key}' key")

            # Validate detection_name
            if det["detection_name"] not in NUSCENES_DETECTION_CLASSES:
                raise ValueError(
                    f"Invalid detection_name: {det['detection_name']}. "
                    f"Must be one of {NUSCENES_DETECTION_CLASSES}"
                )

            # Validate array lengths
            if len(det["translation"]) != 3:
                raise ValueError("translation must have 3 elements")
            if len(det["size"]) != 3:
                raise ValueError("size must have 3 elements")
            if len(det["rotation"]) != 4:
                raise ValueError("rotation must have 4 elements (quaternion)")
            if len(det["velocity"]) != 2:
                raise ValueError("velocity must have 2 elements")

    return True


def export_predictions(
    model: BEVFusionModel,
    dataloader: DataLoader,
    dataset: NuScenesDataset,
    output_path: str,
    device: torch.device,
    score_threshold: float = 0.1,
    max_predictions_per_sample: int = 500
) -> Dict:
    """Run inference and export predictions to nuScenes format.

    Args:
        model: Trained BEVFusionModel
        dataloader: DataLoader for the dataset split
        dataset: NuScenesDataset instance
        output_path: Path to save predictions JSON
        device: Torch device
        score_threshold: Minimum score for predictions
        max_predictions_per_sample: Max detections per sample

    Returns:
        nuScenes submission dictionary
    """
    model.eval()

    # Group predictions by sample token
    results_by_token: Dict[str, List[Dict]] = defaultdict(list)
    total_predictions = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Exporting predictions")):
            # Move to device
            imgs = batch['img'].to(device)
            intrinsics = batch['intrinsics'].to(device)
            extrinsics = batch['extrinsics'].to(device)
            ego_pose = batch['ego_pose'].to(device)
            lidar_points = batch['lidar_points'].to(device)

            # Get mask if available
            lidar_mask = batch.get('lidar_mask')
            if lidar_mask is not None:
                lidar_mask = lidar_mask.to(device)

            batch_size = imgs.shape[0]

            # Run inference
            # Use last timestep for single-frame prediction
            detections_batch = model.predict(
                lidar_points=lidar_points[:, -1] if lidar_points.dim() == 4 else lidar_points,
                camera_images=imgs[:, -1] if imgs.dim() == 6 else imgs,
                camera_intrinsics=intrinsics[:, -1] if intrinsics.dim() == 5 else intrinsics,
                camera_extrinsics=extrinsics[:, -1] if extrinsics.dim() == 5 else extrinsics,
                lidar_mask=lidar_mask[:, -1] if lidar_mask is not None and lidar_mask.dim() == 3 else lidar_mask,
                ego_transform=None
            )

            # Process each sample in batch
            for b in range(batch_size):
                # Get sample token
                sample_idx = batch_idx * dataloader.batch_size + b
                if sample_idx >= len(dataset.samples):
                    continue

                # Handle different sample storage formats
                sample_data = dataset.samples[sample_idx]
                if isinstance(sample_data, str):
                    sample_token = sample_data
                elif hasattr(sample_data, 'sample_token'):
                    sample_token = sample_data.sample_token
                elif isinstance(sample_data, dict):
                    sample_token = sample_data.get('sample_token', sample_data.get('token', str(sample_idx)))
                else:
                    sample_token = str(sample_idx)

                # Get ego pose for this sample
                if ego_pose.dim() == 4:  # (B, T, 4, 4)
                    ego_pose_np = ego_pose[b, -1].cpu().numpy()
                else:  # (B, 4, 4)
                    ego_pose_np = ego_pose[b].cpu().numpy()

                # Convert detections to nuScenes format
                sample_detections = []
                detections = detections_batch[b] if isinstance(detections_batch, list) else []

                for box in detections:
                    if box.score < score_threshold:
                        continue

                    det_dict = box3d_to_nuscenes_detection(
                        box, sample_token, ego_pose_np
                    )
                    if det_dict is not None:
                        sample_detections.append(det_dict)
                        total_predictions += 1

                # Limit predictions per sample (sort by score, keep top-k)
                if len(sample_detections) > max_predictions_per_sample:
                    sample_detections.sort(key=lambda x: x['detection_score'], reverse=True)
                    sample_detections = sample_detections[:max_predictions_per_sample]

                results_by_token[sample_token] = sample_detections

    # Create submission with proper schema
    submission = create_nuscenes_submission(
        results_by_token=dict(results_by_token),
        use_camera=True,
        use_lidar=True
    )

    # Validate schema
    try:
        validate_submission_schema(submission)
        logger.info("Submission schema validated successfully")
    except ValueError as e:
        logger.error(f"Schema validation failed: {e}")
        raise

    # Save predictions
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(submission, f, indent=2)

    logger.info(f"Exported {total_predictions} predictions across "
                f"{len(results_by_token)} samples to {output_path}")

    return submission


def main():
    parser = argparse.ArgumentParser(
        description="Export predictions to nuScenes format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/eval_config.yaml",
                        help="Path to config file")
    parser.add_argument("--output", type=str, default="outputs/predictions.json",
                        help="Output path for predictions JSON")
    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "val", "test"],
                        help="Dataset split to evaluate")
    parser.add_argument("--data_root", type=str, default="data",
                        help="Path to nuScenes data")
    parser.add_argument("--version", type=str, default="v1.0-mini",
                        help="nuScenes version")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for inference")
    parser.add_argument("--score_threshold", type=float, default=0.1,
                        help="Minimum score threshold for predictions")
    parser.add_argument("--max_predictions", type=int, default=500,
                        help="Maximum predictions per sample")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run inference on")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Create BEV config
    bev_config = BEVGridConfig.from_grid_size(bev_h=200, bev_w=200)

    # Create model
    model = BEVFusionModel(
        bev_config=bev_config,
        lidar_channels=64,
        camera_channels=256,
        fused_channels=256,
        use_temporal_attention=False,
        use_mc_convrnn=False,
        num_classes=10
    )

    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'backbone_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['backbone_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model = model.to(device)
    model.eval()

    # Create dataset
    logger.info(f"Loading {args.split} split from {args.data_root}/{args.version}")
    dataset = NuScenesDataset(
        dataroot=args.data_root,
        version=args.version,
        split=args.split,
        sequence_length=1,
        load_lidar=True,
        load_images=True,
        load_annotations=False,
        bev_config=bev_config
    )

    # Create dataloader
    collate = create_fusion_collate_fn(
        bev_h=200, bev_w=200,
        num_classes=10,
        generate_targets=False
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate
    )

    # Export predictions
    submission = export_predictions(
        model=model,
        dataloader=dataloader,
        dataset=dataset,
        output_path=args.output,
        device=device,
        score_threshold=args.score_threshold,
        max_predictions_per_sample=args.max_predictions
    )

    logger.info(f"Done! Created nuScenes submission at {args.output}")
    logger.info(f"Submission contains {len(submission['results'])} samples")


if __name__ == "__main__":
    main()
