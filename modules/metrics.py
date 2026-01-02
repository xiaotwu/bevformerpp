"""
Detection Metrics for 3D Object Detection Evaluation.

This module implements standard evaluation metrics for 3D object detection
following the nuScenes detection benchmark methodology:
- Average Precision (AP) per class and mean AP (mAP)
- True Positive metrics: ATE, ASE, AOE, AVE, AAE
- nuScenes Detection Score (NDS)

References:
    - nuScenes: A multimodal dataset for autonomous driving (Caesar et al., 2020)
    - KITTI 3D Object Detection Benchmark
    - COCO Detection Evaluation
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict
import warnings

from .data_structures import Box3D


# nuScenes class mapping
NUSCENES_CLASSES = [
    'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
    'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier'
]

# Distance thresholds for AP calculation (meters)
# Following nuScenes: match predictions to GT within these thresholds
DISTANCE_THRESHOLDS = [0.5, 1.0, 2.0, 4.0]


@dataclass
class DetectionMetrics:
    """
    Container for detection evaluation metrics.

    Attributes:
        ap_per_class: AP for each class at each distance threshold
        mAP: Mean AP across all classes and thresholds
        ate: Average Translation Error (meters)
        ase: Average Scale Error (1 - IOU after alignment)
        aoe: Average Orientation Error (radians)
        ave: Average Velocity Error (m/s) - if velocity available
        aae: Average Attribute Error - if attributes available
        nds: nuScenes Detection Score
        precision_recall: Per-class precision-recall curves
    """
    ap_per_class: Dict[str, Dict[float, float]] = field(default_factory=dict)
    mAP: float = 0.0
    ate: float = 0.0
    ase: float = 0.0
    aoe: float = 0.0
    ave: float = 0.0
    aae: float = 0.0
    nds: float = 0.0
    precision_recall: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    num_gt: int = 0
    num_pred: int = 0
    num_tp: int = 0

    def summary(self) -> str:
        """Generate a summary string of the metrics."""
        lines = [
            "=" * 60,
            "Detection Evaluation Results",
            "=" * 60,
            f"mAP: {self.mAP:.4f}",
            f"NDS: {self.nds:.4f}",
            "-" * 60,
            "True Positive Metrics:",
            f"  ATE (Translation Error): {self.ate:.4f} m",
            f"  ASE (Scale Error): {self.ase:.4f}",
            f"  AOE (Orientation Error): {self.aoe:.4f} rad",
            "-" * 60,
            "Per-Class AP (@ 2.0m):",
        ]

        for cls_name in NUSCENES_CLASSES:
            if cls_name in self.ap_per_class:
                ap_2m = self.ap_per_class[cls_name].get(2.0, 0.0)
                lines.append(f"  {cls_name:20s}: {ap_2m:.4f}")

        lines.append("-" * 60)
        lines.append(f"Total GT boxes: {self.num_gt}")
        lines.append(f"Total predictions: {self.num_pred}")
        lines.append(f"True positives: {self.num_tp}")
        lines.append("=" * 60)

        return "\n".join(lines)


def compute_center_distance(box1: Box3D, box2: Box3D) -> float:
    """
    Compute Euclidean distance between box centers in BEV (x, y).

    Args:
        box1: First bounding box
        box2: Second bounding box

    Returns:
        Distance in meters
    """
    dx = box1.center[0] - box2.center[0]
    dy = box1.center[1] - box2.center[1]
    return np.sqrt(dx ** 2 + dy ** 2)


def compute_3d_iou(box1: Box3D, box2: Box3D) -> float:
    """
    Compute 3D IoU between two bounding boxes.

    Uses axis-aligned approximation for efficiency.
    For more accurate results with rotated boxes, use Shapely or
    specialized 3D IoU libraries.

    Args:
        box1: First bounding box
        box2: Second bounding box

    Returns:
        IoU value in [0, 1]
    """
    # Extract parameters
    c1, s1 = box1.center, box1.size
    c2, s2 = box2.center, box2.size

    # Axis-aligned bounds
    min1 = c1 - s1 / 2
    max1 = c1 + s1 / 2
    min2 = c2 - s2 / 2
    max2 = c2 + s2 / 2

    # Intersection
    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)
    inter_size = np.maximum(inter_max - inter_min, 0)
    inter_vol = np.prod(inter_size)

    # Union
    vol1 = np.prod(s1)
    vol2 = np.prod(s2)
    union_vol = vol1 + vol2 - inter_vol

    if union_vol < 1e-6:
        return 0.0

    return float(inter_vol / union_vol)


def compute_orientation_error(box1: Box3D, box2: Box3D) -> float:
    """
    Compute orientation error between two boxes.

    Uses the minimum angle difference accounting for symmetry
    (180-degree ambiguity for symmetric objects).

    Args:
        box1: Predicted box
        box2: Ground truth box

    Returns:
        Orientation error in radians [0, pi]
    """
    diff = np.abs(box1.yaw - box2.yaw)
    # Handle angle wraparound
    diff = np.minimum(diff, 2 * np.pi - diff)
    # Handle 180-degree symmetry for some objects
    diff = np.minimum(diff, np.pi - diff)
    return float(diff)


def compute_scale_error(box1: Box3D, box2: Box3D) -> float:
    """
    Compute scale error between two boxes.

    Defined as 1 - IoU of the boxes when perfectly aligned
    (translation and rotation removed).

    Args:
        box1: Predicted box
        box2: Ground truth box

    Returns:
        Scale error in [0, 1]
    """
    # Compute IoU assuming perfect alignment
    s1, s2 = box1.size, box2.size

    # Intersection (assuming centered and axis-aligned)
    inter_size = np.minimum(s1, s2)
    inter_vol = np.prod(inter_size)

    # Union
    vol1 = np.prod(s1)
    vol2 = np.prod(s2)
    union_vol = vol1 + vol2 - inter_vol

    if union_vol < 1e-6:
        return 1.0

    iou = inter_vol / union_vol
    return float(1.0 - iou)


def match_boxes(
    predictions: List[Box3D],
    ground_truths: List[Box3D],
    distance_threshold: float
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Match predictions to ground truths using center distance.

    Uses greedy matching: for each prediction (sorted by confidence),
    match to the nearest unmatched GT within the distance threshold.

    Args:
        predictions: List of predicted boxes (should be sorted by confidence)
        ground_truths: List of ground truth boxes
        distance_threshold: Maximum distance for a valid match (meters)

    Returns:
        Tuple of:
            - matches: List of (pred_idx, gt_idx) pairs
            - unmatched_preds: List of unmatched prediction indices
            - unmatched_gts: List of unmatched ground truth indices
    """
    matches = []
    matched_gt_indices = set()
    unmatched_preds = []

    # Sort predictions by confidence (descending)
    pred_indices = sorted(
        range(len(predictions)),
        key=lambda i: predictions[i].score,
        reverse=True
    )

    for pred_idx in pred_indices:
        pred = predictions[pred_idx]
        best_gt_idx = -1
        best_dist = float('inf')

        for gt_idx, gt in enumerate(ground_truths):
            if gt_idx in matched_gt_indices:
                continue

            # Check class match
            if not _class_matches(pred.label, gt.label):
                continue

            dist = compute_center_distance(pred, gt)
            if dist < distance_threshold and dist < best_dist:
                best_dist = dist
                best_gt_idx = gt_idx

        if best_gt_idx >= 0:
            matches.append((pred_idx, best_gt_idx))
            matched_gt_indices.add(best_gt_idx)
        else:
            unmatched_preds.append(pred_idx)

    unmatched_gts = [i for i in range(len(ground_truths)) if i not in matched_gt_indices]

    return matches, unmatched_preds, unmatched_gts


def _class_matches(pred_label: str, gt_label: str) -> bool:
    """
    Check if prediction and ground truth class labels match.

    Handles nuScenes hierarchical labels (e.g., 'vehicle.car' matches 'car').

    Args:
        pred_label: Predicted class label
        gt_label: Ground truth class label

    Returns:
        True if classes match
    """
    pred_lower = pred_label.lower()
    gt_lower = gt_label.lower()

    # Direct match
    if pred_lower == gt_lower:
        return True

    # Check if either contains the other (for hierarchical labels)
    if pred_lower in gt_lower or gt_lower in pred_lower:
        return True

    # Check class name suffixes
    for cls_name in NUSCENES_CLASSES:
        if pred_lower.endswith(cls_name) and gt_lower.endswith(cls_name):
            return True
        if cls_name in pred_lower and cls_name in gt_lower:
            return True

    return False


def compute_ap(
    predictions: List[Box3D],
    ground_truths: List[Box3D],
    distance_threshold: float,
    recall_thresholds: np.ndarray = None
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute Average Precision for a single class at a given distance threshold.

    Uses the 11-point interpolation method (PASCAL VOC style) or
    all-point interpolation (COCO style).

    Args:
        predictions: List of predicted boxes for this class
        ground_truths: List of ground truth boxes for this class
        distance_threshold: Distance threshold for matching
        recall_thresholds: Recall points for AP interpolation

    Returns:
        Tuple of (AP, precision_array, recall_array)
    """
    if recall_thresholds is None:
        recall_thresholds = np.linspace(0, 1, 101)  # COCO style

    if len(ground_truths) == 0:
        if len(predictions) == 0:
            return 1.0, np.ones(len(recall_thresholds)), recall_thresholds
        else:
            return 0.0, np.zeros(len(recall_thresholds)), recall_thresholds

    if len(predictions) == 0:
        return 0.0, np.zeros(len(recall_thresholds)), recall_thresholds

    # Sort predictions by confidence
    predictions = sorted(predictions, key=lambda x: x.score, reverse=True)

    # Match predictions to ground truths
    matches, unmatched_preds, unmatched_gts = match_boxes(
        predictions, ground_truths, distance_threshold
    )

    # Build TP/FP arrays
    n_pred = len(predictions)
    n_gt = len(ground_truths)

    tp = np.zeros(n_pred)
    fp = np.zeros(n_pred)

    matched_gt_set = set()
    for pred_idx, gt_idx in matches:
        if gt_idx not in matched_gt_set:
            tp[pred_idx] = 1
            matched_gt_set.add(gt_idx)
        else:
            fp[pred_idx] = 1

    for pred_idx in unmatched_preds:
        fp[pred_idx] = 1

    # Cumulative sums
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    # Precision and recall
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    recall = tp_cumsum / (n_gt + 1e-6)

    # Interpolate precision at recall thresholds
    precision_interp = np.zeros(len(recall_thresholds))
    for i, r_thresh in enumerate(recall_thresholds):
        precisions_above = precision[recall >= r_thresh]
        if len(precisions_above) > 0:
            precision_interp[i] = np.max(precisions_above)

    # AP is mean of interpolated precisions
    ap = np.mean(precision_interp)

    return float(ap), precision_interp, recall_thresholds


def compute_tp_metrics(
    matches: List[Tuple[Box3D, Box3D]]
) -> Dict[str, float]:
    """
    Compute True Positive metrics (ATE, ASE, AOE) from matched box pairs.

    Args:
        matches: List of (prediction, ground_truth) box pairs

    Returns:
        Dictionary with ATE, ASE, AOE values
    """
    if len(matches) == 0:
        return {'ate': 0.0, 'ase': 0.0, 'aoe': 0.0}

    translation_errors = []
    scale_errors = []
    orientation_errors = []

    for pred, gt in matches:
        # Translation error (Euclidean distance)
        te = compute_center_distance(pred, gt)
        translation_errors.append(te)

        # Scale error
        se = compute_scale_error(pred, gt)
        scale_errors.append(se)

        # Orientation error
        oe = compute_orientation_error(pred, gt)
        orientation_errors.append(oe)

    return {
        'ate': float(np.mean(translation_errors)),
        'ase': float(np.mean(scale_errors)),
        'aoe': float(np.mean(orientation_errors))
    }


def compute_nds(mAP: float, ate: float, ase: float, aoe: float,
                ave: float = 0.0, aae: float = 0.0) -> float:
    """
    Compute nuScenes Detection Score (NDS).

    NDS = (5 * mAP + sum(1 - min(1, TP_metric))) / 10

    Where TP_metrics are: ATE, ASE, AOE, AVE, AAE
    Each TP metric is clamped to [0, 1] before computing (1 - metric).

    Args:
        mAP: Mean Average Precision
        ate: Average Translation Error (meters)
        ase: Average Scale Error
        aoe: Average Orientation Error (radians, normalized by pi)
        ave: Average Velocity Error (m/s, normalized)
        aae: Average Attribute Error

    Returns:
        NDS score in [0, 1]
    """
    # Normalize TP metrics to [0, 1] range
    # ATE: cap at 2 meters
    ate_score = max(0, 1 - min(ate / 2.0, 1.0))
    # ASE: already in [0, 1]
    ase_score = max(0, 1 - min(ase, 1.0))
    # AOE: normalize by pi
    aoe_score = max(0, 1 - min(aoe / np.pi, 1.0))
    # AVE: cap at 2 m/s (if used)
    ave_score = max(0, 1 - min(ave / 2.0, 1.0))
    # AAE: already in [0, 1] (if used)
    aae_score = max(0, 1 - min(aae, 1.0))

    # NDS formula
    tp_score = ate_score + ase_score + aoe_score + ave_score + aae_score
    nds = (5 * mAP + tp_score) / 10.0

    return float(nds)


class DetectionEvaluator:
    """
    Evaluator for 3D object detection following nuScenes methodology.

    This class accumulates predictions and ground truths across multiple
    samples and computes comprehensive evaluation metrics.

    Example:
        >>> evaluator = DetectionEvaluator()
        >>> for batch in dataloader:
        ...     predictions = model(batch)
        ...     evaluator.add_batch(predictions, ground_truths)
        >>> metrics = evaluator.compute_metrics()
        >>> print(metrics.summary())
    """

    def __init__(
        self,
        class_names: List[str] = None,
        distance_thresholds: List[float] = None,
        iou_thresholds: List[float] = None
    ):
        """
        Initialize evaluator.

        Args:
            class_names: List of class names to evaluate
            distance_thresholds: Distance thresholds for AP calculation
            iou_thresholds: IoU thresholds (alternative to distance)
        """
        self.class_names = class_names or NUSCENES_CLASSES
        self.distance_thresholds = distance_thresholds or DISTANCE_THRESHOLDS

        # Accumulate predictions and GTs per class
        self.predictions_per_class: Dict[str, List[Box3D]] = defaultdict(list)
        self.ground_truths_per_class: Dict[str, List[Box3D]] = defaultdict(list)

        # Store matched pairs for TP metrics
        self.matched_pairs: List[Tuple[Box3D, Box3D]] = []

    def reset(self):
        """Reset accumulated data."""
        self.predictions_per_class = defaultdict(list)
        self.ground_truths_per_class = defaultdict(list)
        self.matched_pairs = []

    def add_batch(
        self,
        predictions: List[Box3D],
        ground_truths: List[Box3D],
        sample_id: Optional[str] = None
    ):
        """
        Add predictions and ground truths from a single sample.

        Args:
            predictions: List of predicted boxes
            ground_truths: List of ground truth boxes
            sample_id: Optional sample identifier for debugging
        """
        # Organize by class
        for pred in predictions:
            cls_name = self._normalize_class(pred.label)
            if cls_name:
                self.predictions_per_class[cls_name].append(pred)

        for gt in ground_truths:
            cls_name = self._normalize_class(gt.label)
            if cls_name:
                self.ground_truths_per_class[cls_name].append(gt)

    def _normalize_class(self, label: str) -> Optional[str]:
        """Normalize class label to standard name."""
        label_lower = label.lower()

        # Direct match
        if label_lower in self.class_names:
            return label_lower

        # Try suffix matching
        for cls_name in self.class_names:
            if label_lower.endswith(cls_name) or cls_name in label_lower:
                return cls_name

        return None

    def compute_metrics(self) -> DetectionMetrics:
        """
        Compute all evaluation metrics.

        Returns:
            DetectionMetrics object with all computed metrics
        """
        metrics = DetectionMetrics()

        # Count totals
        metrics.num_pred = sum(len(preds) for preds in self.predictions_per_class.values())
        metrics.num_gt = sum(len(gts) for gts in self.ground_truths_per_class.values())

        # Compute AP per class per threshold
        all_aps = []
        all_matched_pairs = []

        for cls_name in self.class_names:
            preds = self.predictions_per_class.get(cls_name, [])
            gts = self.ground_truths_per_class.get(cls_name, [])

            metrics.ap_per_class[cls_name] = {}

            for dist_thresh in self.distance_thresholds:
                ap, precision, recall = compute_ap(preds, gts, dist_thresh)
                metrics.ap_per_class[cls_name][dist_thresh] = ap
                all_aps.append(ap)

                # Store precision-recall for visualization
                if dist_thresh == 2.0:  # Standard threshold
                    metrics.precision_recall[cls_name] = {
                        'precision': precision,
                        'recall': recall
                    }

            # Collect matched pairs for TP metrics (at 2.0m threshold)
            if len(preds) > 0 and len(gts) > 0:
                matches, _, _ = match_boxes(preds, gts, distance_threshold=2.0)
                for pred_idx, gt_idx in matches:
                    all_matched_pairs.append((preds[pred_idx], gts[gt_idx]))

        # Mean AP
        metrics.mAP = float(np.mean(all_aps)) if all_aps else 0.0

        # True Positive metrics
        metrics.num_tp = len(all_matched_pairs)
        tp_metrics = compute_tp_metrics(all_matched_pairs)
        metrics.ate = tp_metrics['ate']
        metrics.ase = tp_metrics['ase']
        metrics.aoe = tp_metrics['aoe']

        # NDS
        metrics.nds = compute_nds(
            metrics.mAP, metrics.ate, metrics.ase, metrics.aoe
        )

        return metrics


def evaluate_detections(
    predictions: List[List[Box3D]],
    ground_truths: List[List[Box3D]],
    class_names: List[str] = None
) -> DetectionMetrics:
    """
    Convenience function to evaluate detections from multiple samples.

    Args:
        predictions: List of prediction lists, one per sample
        ground_truths: List of ground truth lists, one per sample
        class_names: List of class names to evaluate

    Returns:
        DetectionMetrics object
    """
    evaluator = DetectionEvaluator(class_names=class_names)

    for preds, gts in zip(predictions, ground_truths):
        evaluator.add_batch(preds, gts)

    return evaluator.compute_metrics()
