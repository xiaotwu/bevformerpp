#!/usr/bin/env python
"""
Overfit a single sample to validate hard-center BEV heatmap supervision and
detect camera-ray shortcut artifacts.

Interpretation:
- If wiring is correct, within ~300–500 steps the predicted BEV heatmap should
  collapse to point-like peaks exactly at GT centers (hard-center targets are
  binary {0,1}).
- If heatmaps still diffuse along camera rays, the training wiring or targets
  are still incorrect.
"""
import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from modules.nuscenes_dataset import NuScenesDataset, create_collate_fn
import modules.target_generator as tg
print("TARGET_GEN_FILE =", tg.__file__)
print("TARGET_GEN_VERSION =", getattr(tg, "_TARGET_GEN_VERSION", "MISSING"))
print("TargetGenerator class =", tg.TargetGenerator)

from modules.bevformer import EnhancedBEVFormer
from modules.head import BEVHead, DetectionLoss


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SingleSampleDataset(torch.utils.data.Dataset):
    """Wrap a base dataset and always return the same indexed sample."""

    def __init__(self, base_dataset, sample_idx: int = 0):
        self.base = base_dataset
        self.sample_idx = sample_idx

    def __len__(self):
        # Minimal length; DataLoader will iterate repeatedly per epoch
        return 1

    def __getitem__(self, idx):
        return self.base[self.sample_idx]


def visualize(step, cls_pred, cls_targets, out_dir):
    """Save simple heatmap comparison PNG."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # cls_pred expected logits; apply sigmoid
    pred_sig = torch.sigmoid(cls_pred)
    pred_max = pred_sig.max(dim=1)[0].detach().cpu().numpy()
    gt_max = cls_targets.max(dim=1)[0].detach().cpu().numpy()
    diff = np.abs(pred_max - gt_max)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, img, title in zip(
        axes,
        [pred_max[0], gt_max[0], diff[0]],
        ["Pred heatmap (max over classes)", "GT heatmap", "Abs diff"],
    ):
        im = ax.imshow(img, cmap="hot", origin="lower")
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_dir / f"step_{step:04d}.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Overfit one sample to validate hard-center supervision.")
    parser.add_argument("--data_root", type=str, default="data", help="NuScenes data root")
    parser.add_argument("--version", type=str, default="v1.0-mini", help="NuScenes version")
    parser.add_argument("--sample_idx", type=int, default=0, help="Sample index to overfit")
    parser.add_argument("--steps", type=int, default=300, help="Optimization steps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--save_vis", action="store_true", help="Save visualizations every vis_interval steps")
    parser.add_argument("--vis_interval", type=int, default=50, help="Visualization interval")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for determinism")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cuda" else "cpu")

    # Dataset and collate with hard-center targets
    base_dataset = NuScenesDataset(dataroot=args.data_root, version=args.version)
    single_ds = SingleSampleDataset(base_dataset, sample_idx=args.sample_idx)
    collate_fn = create_collate_fn(
        bev_h=200,  # match defaults
        bev_w=200,
        num_classes=10,
        generate_targets=True,
        img_size=(224, 400),
        heatmap_mode="hard_center",
    )

    loader = DataLoader(single_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # Model + head (camera-only pipeline to minimize dependencies)
    embed_dim = 256
    bev_h, bev_w = 200, 200
    num_classes = 10
    model_backbone = EnhancedBEVFormer(
        bev_h=bev_h,
        bev_w=bev_w,
        embed_dim=embed_dim,
        temporal_method="convgru",
        max_history=1,
        enable_bptt=False,
        mc_disable_warping=False,
        mc_disable_motion_field=False,
        mc_disable_visibility=False,
    ).to(device)
    model_head = BEVHead(embed_dim=embed_dim, num_classes=num_classes).to(device)

    criterion = DetectionLoss()
    print("LOSS =", criterion.__class__.__name__, criterion.__module__)
    print("heatmap_mode = hard_center (collate_fn)")

    optimizer = torch.optim.AdamW(
        list(model_backbone.parameters()) + list(model_head.parameters()),
        lr=args.lr,
        weight_decay=1e-2,
    )

    vis_dir = Path("outputs/overfit_debug")
    cls_pos_mask = None

    data_iter = iter(loader)
    for step in range(1, args.steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        imgs = batch["img"].to(device)
        intrinsics = batch["intrinsics"].to(device)
        extrinsics = batch["extrinsics"].to(device)
        ego_pose = batch["ego_pose"].to(device)

        targets = {
            "cls_targets": batch["cls_targets"].to(device),
            "bbox_targets": batch["bbox_targets"].to(device),
            "reg_mask": batch["reg_mask"].to(device),
        }

        # Save GT mask for overlap checking
        if cls_pos_mask is None:
            cls_pos_mask = (targets["cls_targets"] > 0.5)

        optimizer.zero_grad()

        bev_seq = model_backbone.forward_sequence(imgs, intrinsics, extrinsics, ego_pose)
        bev_features = bev_seq[:, -1]
        preds = model_head(bev_features)

        loss_dict = criterion(preds, targets)
        loss = loss_dict["loss_total"]
        loss.backward()
        optimizer.step()

        cls_logits = preds["cls_score"]  # (B, K, H, W)
        cls_sigmoid = torch.sigmoid(cls_logits)
        cls_target = targets["cls_targets"]

        if step % 10 == 0 or step == 1:
            cls_min = cls_sigmoid.min().item()
            cls_mean = cls_sigmoid.mean().item()
            cls_max = cls_sigmoid.max().item()
            unique_vals = torch.unique(cls_target).detach().cpu().numpy()
            print(
                f"Step {step:04d} | "
                f"loss={loss.item():.4f} "
                f"cls={loss_dict['loss_cls'].item():.4f} "
                f"bbox={loss_dict['loss_bbox'].item():.4f} | "
                f"cls_pred[min/mean/max]={cls_min:.3f}/{cls_mean:.3f}/{cls_max:.3f} | "
                f"cls_targets unique={unique_vals}"
            )

        if args.save_vis and (step % args.vis_interval == 0 or step == args.steps):
            visualize(step, cls_logits.detach().cpu(), cls_target.detach().cpu(), vis_dir)

    # Success metrics: GT-center focused evaluation
    cls_sigmoid = torch.sigmoid(cls_logits)  # (B, K, H, W)
    cls_target = targets["cls_targets"]  # (B, K, H, W)
    
    # 1. Compute GT center set: positions where cls_targets == 1.0 for each class
    gt_centers = (cls_target == 1.0)  # (B, K, H, W) boolean mask
    
    # 2. For each GT center, read predicted prob at SAME class channel and position
    gt_center_probs = cls_sigmoid[gt_centers]  # 1D tensor of probs at GT centers
    
    # 3. Background mean prob (where cls_targets==0)
    bg_mask = (cls_target == 0.0)
    bg_probs = cls_sigmoid[bg_mask]
    
    # 4. Precision proxy: For each class with >=1 GT center, take top-N peaks and count GT center hits
    B, K, H, W = cls_sigmoid.shape
    top_n = 10  # Check top-10 peaks per class
    precision_hits = []
    classes_with_gt = []
    
    for cls_idx in range(K):
        class_gt_centers = gt_centers[0, cls_idx]  # (H, W) boolean
        
        # Only evaluate classes that have at least one GT center
        if class_gt_centers.sum().item() == 0:
            continue
        
        classes_with_gt.append(cls_idx)
        class_channel = cls_sigmoid[0, cls_idx]  # (H, W)
        
        # Get top-N peaks in this class channel
        flat_channel = class_channel.view(-1)
        topk_vals, topk_idx = torch.topk(flat_channel, k=min(top_n, flat_channel.numel()))
        
        # Convert flat indices to (y, x) coordinates
        topk_y = topk_idx // W
        topk_x = topk_idx % W
        
        # Check how many hit GT centers
        hits = class_gt_centers[topk_y, topk_x].sum().item()
        precision_hits.append((hits, len(topk_vals)))
    
    # Report metrics
    print("\n=== Overfit Summary ===")
    if len(gt_center_probs) > 0:
        print(f"GT-center probabilities: mean={gt_center_probs.mean().item():.4f}, "
              f"min={gt_center_probs.min().item():.4f}, "
              f"n_centers={len(gt_center_probs)}")
    else:
        print("GT-center probabilities: NO GT CENTERS FOUND")
    
    if len(bg_probs) > 0:
        print(f"Background mean prob (where cls_targets==0): {bg_probs.mean().item():.4f}")
    
    # Precision proxy: only over classes with GT centers
    num_classes_with_gt = len(classes_with_gt)
    if num_classes_with_gt > 0:
        total_hits = sum(h for h, _ in precision_hits)
        total_peaks = sum(n for _, n in precision_hits)
        precision_rate = total_hits / total_peaks if total_peaks > 0 else 0.0
        print(f"Precision proxy (top-{top_n} peaks per class hitting GT centers): "
              f"{total_hits}/{total_peaks} = {precision_rate:.2%}")
        print(f"  Evaluated over {num_classes_with_gt}/{K} classes with >=1 GT center "
              f"(classes: {classes_with_gt})")
    else:
        print(f"Precision proxy: No classes with GT centers found (all {K} classes have zero GT)")
    
    print("Expectation: GT-center probs should approach 1.0; background should stay low; "
          "precision proxy should increase as model learns to localize at GT centers.")


if __name__ == "__main__":
    main()

