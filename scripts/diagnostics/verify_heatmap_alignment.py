#!/usr/bin/env python
"""
Verification script for BEV heatmap alignment.

This script:
1. Loads one training sample
2. Runs forward pass through the model
3. Prints diagnostic statistics (logits, probs, positive counts)
4. Saves visualization overlaying GT centers on predicted heatmap
5. Optionally runs overfit-one-batch experiment

Usage:
    python scripts/verify_heatmap_alignment.py
    python scripts/verify_heatmap_alignment.py --overfit --steps 500
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.nuscenes_dataset import NuScenesDataset, create_collate_fn
from modules.bevformer import EnhancedBEVFormer
from modules.head import BEVHead
from modules.head import DetectionLoss  # Explicit import to avoid ambiguity


def print_diagnostics(cls_scores, cls_targets, reg_mask):
    """Print diagnostic statistics for heatmap predictions."""
    print("\n" + "=" * 60)
    print("HEATMAP DIAGNOSTICS")
    print("=" * 60)

    # Pre-sigmoid logits (we need to invert sigmoid)
    # cls_scores are already after sigmoid, so we compute logits
    eps = 1e-6
    cls_probs = cls_scores.clamp(eps, 1 - eps)
    cls_logits = torch.log(cls_probs / (1 - cls_probs))

    print(f"\nClassification Logits (pre-sigmoid):")
    print(f"  Min:  {cls_logits.min().item():.4f}")
    print(f"  Mean: {cls_logits.mean().item():.4f}")
    print(f"  Max:  {cls_logits.max().item():.4f}")

    print(f"\nClassification Probabilities (post-sigmoid):")
    print(f"  Min:  {cls_probs.min().item():.4f}")
    print(f"  Mean: {cls_probs.mean().item():.4f}")
    print(f"  Max:  {cls_probs.max().item():.4f}")

    # Per-class statistics
    print(f"\nPer-Class Statistics:")
    print(f"{'Class':<6} {'GT Pos':<8} {'GT Max':<8} {'Pred Max':<10} {'Pred@GTMax':<12}")
    print("-" * 50)

    B, K, H, W = cls_targets.shape
    class_names = ['car', 'truck', 'bus', 'trailer', 'constr', 'ped', 'moto', 'bike', 'cone', 'barrier']

    for k in range(K):
        gt_heatmap = cls_targets[0, k].cpu().numpy()
        pred_heatmap = cls_probs[0, k].cpu().numpy()

        # Count positive cells (where GT == 1.0)
        num_pos = (gt_heatmap == 1.0).sum()
        gt_max = gt_heatmap.max()
        pred_max = pred_heatmap.max()

        # Find GT peak location and check prediction there
        if gt_max > 0:
            gt_peak_idx = np.unravel_index(gt_heatmap.argmax(), gt_heatmap.shape)
            pred_at_gt = pred_heatmap[gt_peak_idx]
        else:
            pred_at_gt = 0.0

        name = class_names[k] if k < len(class_names) else f"cls_{k}"
        print(f"{name:<6} {num_pos:<8} {gt_max:<8.3f} {pred_max:<10.4f} {pred_at_gt:<12.4f}")

    # Regression mask stats
    num_reg_pos = reg_mask.sum().item()
    print(f"\nRegression mask positive pixels: {int(num_reg_pos)}")
    print("=" * 60)


def save_visualization(cls_scores, cls_targets, reg_mask, save_path):
    """Save visualization overlaying GT centers on predicted heatmap."""
    # Take first sample, max over classes
    pred_heatmap = cls_scores[0].max(dim=0)[0].cpu().numpy()  # (H, W)
    gt_heatmap = cls_targets[0].max(dim=0)[0].cpu().numpy()  # (H, W)
    mask = reg_mask[0, 0].cpu().numpy()  # (H, W)

    # Find GT center locations (where mask == 1)
    gt_centers_y, gt_centers_x = np.where(mask > 0.5)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Predicted heatmap with GT centers overlaid
    im0 = axes[0].imshow(pred_heatmap, cmap='hot', vmin=0, vmax=1)
    axes[0].scatter(gt_centers_x, gt_centers_y, c='cyan', s=50, marker='x', linewidths=2)
    axes[0].set_title(f'Predicted Heatmap (max={pred_heatmap.max():.3f})\nCyan X = GT centers')
    plt.colorbar(im0, ax=axes[0])

    # GT heatmap
    im1 = axes[1].imshow(gt_heatmap, cmap='hot', vmin=0, vmax=1)
    axes[1].scatter(gt_centers_x, gt_centers_y, c='cyan', s=50, marker='x', linewidths=2)
    axes[1].set_title('GT Heatmap')
    plt.colorbar(im1, ax=axes[1])

    # Difference
    diff = np.abs(pred_heatmap - gt_heatmap)
    im2 = axes[2].imshow(diff, cmap='coolwarm', vmin=0, vmax=1)
    axes[2].set_title(f'|Pred - GT| (mean={diff.mean():.4f})')
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nVisualization saved to: {save_path}")


def overfit_one_batch(model_backbone, model_head, batch, criterion, device,
                      num_steps=500, lr=1e-4, save_dir='checkpoints'):
    """
    Overfit on a single batch to verify the model can learn sharp peaks.

    If the model cannot overfit one sample, there's a fundamental issue
    with the architecture or loss function.
    """
    print("\n" + "=" * 60)
    print("OVERFIT-ONE-BATCH EXPERIMENT")
    print("=" * 60)
    print(f"Training on 1 fixed sample for {num_steps} steps...")

    model_backbone.train()
    model_head.train()

    optimizer = optim.Adam(
        list(model_backbone.parameters()) + list(model_head.parameters()),
        lr=lr
    )

    # Prepare batch
    imgs = batch['img'].to(device)
    intrinsics = batch['intrinsics'].to(device)
    extrinsics = batch['extrinsics'].to(device)
    ego_pose = batch['ego_pose'].to(device)
    targets = {
        'cls_targets': batch['cls_targets'].to(device),
        'bbox_targets': batch['bbox_targets'].to(device),
        'reg_mask': batch['reg_mask'].to(device)
    }

    losses = []
    cls_maxes = []

    pbar = tqdm(range(num_steps), desc='Overfitting')
    for step in pbar:
        optimizer.zero_grad()

        # Forward
        bev_seq = model_backbone.forward_sequence(imgs, intrinsics, extrinsics, ego_pose)
        preds = model_head(bev_seq[:, -1])

        # Loss
        loss_dict = criterion(preds, targets)
        loss = loss_dict['loss_total']

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_backbone.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(model_head.parameters(), 1.0)
        optimizer.step()

        # Track
        losses.append(loss.item())
        cls_max = preds['cls_score'].max().item()
        cls_maxes.append(cls_max)

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'cls_max': f'{cls_max:.3f}'
        })

    # Final evaluation
    model_backbone.eval()
    model_head.eval()
    with torch.no_grad():
        bev_seq = model_backbone.forward_sequence(imgs, intrinsics, extrinsics, ego_pose)
        preds = model_head(bev_seq[:, -1])

    print(f"\nFinal Results after {num_steps} steps:")
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss:   {losses[-1]:.4f}")
    print(f"  Loss reduction: {(1 - losses[-1]/losses[0])*100:.1f}%")
    print(f"  Initial cls max: {cls_maxes[0]:.4f}")
    print(f"  Final cls max:   {cls_maxes[-1]:.4f}")

    # Check if peaks formed at GT locations
    cls_scores = preds['cls_score']
    cls_targets = targets['cls_targets']
    reg_mask = targets['reg_mask']

    # Find GT centers and check prediction values there
    mask = reg_mask[0, 0].cpu().numpy()
    gt_y, gt_x = np.where(mask > 0.5)

    if len(gt_y) > 0:
        pred_at_gt = []
        for y, x in zip(gt_y, gt_x):
            val = cls_scores[0, :, y, x].max().item()
            pred_at_gt.append(val)
        mean_pred_at_gt = np.mean(pred_at_gt)
        print(f"  Mean pred at GT centers: {mean_pred_at_gt:.4f}")

        if mean_pred_at_gt > 0.5:
            print("  ✓ SUCCESS: Sharp peaks formed at GT centers!")
        elif mean_pred_at_gt > 0.2:
            print("  ~ PARTIAL: Peaks forming but not sharp yet")
        else:
            print("  ✗ FAILURE: No peaks at GT centers - check architecture/loss")

    # Save visualization
    save_path = os.path.join(save_dir, 'overfit_result.png')
    save_visualization(cls_scores, cls_targets, reg_mask, save_path)

    # Plot loss curve
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(losses)
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True)

    axes[1].plot(cls_maxes)
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Max Classification Score')
    axes[1].set_title('Classification Confidence')
    axes[1].grid(True)

    plt.tight_layout()
    curve_path = os.path.join(save_dir, 'overfit_curves.png')
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print(f"  Loss curves saved to: {curve_path}")


def main():
    parser = argparse.ArgumentParser(description='Verify BEV heatmap alignment')
    parser.add_argument('--data-root', type=str, default='data',
                        help='Path to nuScenes data')
    parser.add_argument('--overfit', action='store_true',
                        help='Run overfit-one-batch experiment')
    parser.add_argument('--steps', type=int, default=500,
                        help='Number of overfit steps')
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                        help='Directory to save outputs')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # Load dataset
    print(f"\nLoading dataset from {args.data_root}...")
    dataset = NuScenesDataset(dataroot=args.data_root, version='v1.0-mini')
    collate_fn = create_collate_fn(bev_h=200, bev_w=200, num_classes=10)

    # Get one sample
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    batch = next(iter(loader))

    print(f"Sample loaded: {batch['img'].shape}")
    print(f"GT targets: cls={batch['cls_targets'].shape}, reg_mask sum={batch['reg_mask'].sum().item()}")

    # Create model
    print("\nCreating model...")
    model_backbone = EnhancedBEVFormer(
        bev_h=200, bev_w=200, embed_dim=256,
        temporal_method='mc_convrnn'
    ).to(device)
    model_head = BEVHead(embed_dim=256, num_classes=10).to(device)
    criterion = DetectionLoss()
    # Safety check: fail fast if wrong loss is imported
    assert criterion.__class__.__module__.endswith("modules.head"), \
        f"Wrong DetectionLoss imported! Got {criterion.__class__.__module__}.{criterion.__class__.__name__}. " \
        f"Must use modules.head.DetectionLoss"
    print(f"✓ Using correct DetectionLoss: {criterion.__class__.__module__}.{criterion.__class__.__name__}")

    # Verify coordinate alignment
    print("\n" + "=" * 60)
    print("COORDINATE ALIGNMENT CHECK")
    print("=" * 60)
    bev_coords = model_backbone.camera_encoder.bev_coords
    print(f"BEV grid shape: {bev_coords.shape}")
    print(f"Pixel [0, 0]: x={bev_coords[0, 0, 0]:.4f}, y={bev_coords[0, 0, 1]:.4f}")
    print(f"Pixel [100, 100]: x={bev_coords[100, 100, 0]:.4f}, y={bev_coords[100, 100, 1]:.4f}")
    print(f"Pixel [199, 199]: x={bev_coords[199, 199, 0]:.4f}, y={bev_coords[199, 199, 1]:.4f}")

    # With pixel-center semantics:
    # - Pixel i covers [x_min + i*res, x_min + (i+1)*res)
    # - Pixel 100 covers [0, 0.512) meters, center at 0.256m
    # - This is CORRECT: objects at x=0 map to pixel 100, and the network
    #   samples at pixel centers (0.256m). The regression head learns dx/dy
    #   to recover sub-pixel offsets.
    x_at_100 = bev_coords[100, 100, 0].item()
    y_at_100 = bev_coords[100, 100, 1].item()
    expected_center = 0.256  # Pixel 100 center = x_min + (100 + 0.5) * 0.512 = 0.256m
    if abs(x_at_100 - expected_center) < 0.01 and abs(y_at_100 - expected_center) < 0.01:
        print(f"✓ Pixel-center alignment correct (pixel 100,100 ≈ {expected_center},{expected_center} meters)")
    else:
        print(f"✗ Alignment issue: pixel 100,100 = ({x_at_100:.4f}, {y_at_100:.4f}) meters")
        print(f"  Expected: ({expected_center:.4f}, {expected_center:.4f}) meters")

    # Forward pass
    print("\nRunning forward pass...")
    model_backbone.eval()
    model_head.eval()

    with torch.no_grad():
        imgs = batch['img'].to(device)
        intrinsics = batch['intrinsics'].to(device)
        extrinsics = batch['extrinsics'].to(device)
        ego_pose = batch['ego_pose'].to(device)

        bev_seq = model_backbone.forward_sequence(imgs, intrinsics, extrinsics, ego_pose)
        preds = model_head(bev_seq[:, -1])

    cls_scores = preds['cls_score']
    cls_targets = batch['cls_targets'].to(device)
    reg_mask = batch['reg_mask'].to(device)

    # Print diagnostics
    print_diagnostics(cls_scores, cls_targets, reg_mask)

    # Save visualization
    save_path = os.path.join(args.save_dir, 'heatmap_alignment.png')
    save_visualization(cls_scores, cls_targets, reg_mask, save_path)

    # Overfit experiment
    if args.overfit:
        # Re-create models for clean overfit
        model_backbone = EnhancedBEVFormer(
            bev_h=200, bev_w=200, embed_dim=256,
            temporal_method='mc_convrnn'
        ).to(device)
        model_head = BEVHead(embed_dim=256, num_classes=10).to(device)

        overfit_one_batch(
            model_backbone, model_head, batch, criterion, device,
            num_steps=args.steps, save_dir=args.save_dir
        )

    print("\nDone!")


if __name__ == '__main__':
    main()
