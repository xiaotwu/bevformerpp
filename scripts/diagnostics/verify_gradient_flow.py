#!/usr/bin/env python3
"""
Gradient Flow Verification Script

This script verifies that gradients properly flow from the detection head
all the way back to the backbone (ResNet) through the spatial cross-attention.

Run this after applying the gradient flow fixes to attention.py.
"""

import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.bevformer import EnhancedBEVFormer
from modules.head import BEVHead


def verify_gradient_flow():
    """Verify gradient flow through the entire model."""
    print("=" * 60)
    print("Gradient Flow Verification")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Create model
    BEV_H, BEV_W = 50, 50  # Smaller for testing
    EMBED_DIM = 256
    NUM_CLASSES = 10

    backbone = EnhancedBEVFormer(
        bev_h=BEV_H, bev_w=BEV_W, embed_dim=EMBED_DIM
    ).to(device)
    head = BEVHead(
        embed_dim=EMBED_DIM, num_classes=NUM_CLASSES, reg_channels=7
    ).to(device)

    # Create dummy input
    B = 1
    N_cam = 6
    H_img, W_img = 224, 400

    imgs = torch.randn(B, 1, N_cam, 3, H_img, W_img, device=device, requires_grad=True)

    # Create realistic intrinsics (scaled for image size)
    intrinsics = torch.zeros(B, 1, N_cam, 3, 3, device=device)
    fx = 400.0  # Focal length (scaled)
    fy = 300.0
    cx = W_img / 2
    cy = H_img / 2
    for i in range(N_cam):
        intrinsics[:, :, i, 0, 0] = fx
        intrinsics[:, :, i, 1, 1] = fy
        intrinsics[:, :, i, 0, 2] = cx
        intrinsics[:, :, i, 1, 2] = cy
        intrinsics[:, :, i, 2, 2] = 1.0

    # Create realistic extrinsics (cameras looking outward from ego vehicle)
    extrinsics = torch.zeros(B, 1, N_cam, 4, 4, device=device)
    for i in range(N_cam):
        angle = i * (2 * 3.14159 / N_cam)  # Evenly spaced around vehicle
        R = torch.tensor([
            [torch.cos(torch.tensor(angle)), -torch.sin(torch.tensor(angle)), 0],
            [torch.sin(torch.tensor(angle)), torch.cos(torch.tensor(angle)), 0],
            [0, 0, 1]
        ], dtype=torch.float32, device=device)
        extrinsics[:, :, i, :3, :3] = R.unsqueeze(0).unsqueeze(0)
        extrinsics[:, :, i, 3, 3] = 1.0

    ego_pose = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)

    # Forward pass
    print("Running forward pass...")
    backbone.train()
    head.train()

    bev_seq = backbone.forward_sequence(imgs, intrinsics, extrinsics, ego_pose)
    bev_features = bev_seq[:, -1]
    preds = head(bev_features)

    # Create dummy loss (BEVHead returns 'cls_score' and 'bbox_pred')
    loss = preds['cls_score'].mean() + preds['bbox_pred'].mean()

    # Backward pass
    print("Running backward pass...")
    loss.backward()

    # Check gradients at each component
    print("\n" + "=" * 60)
    print("Gradient Check Results")
    print("=" * 60)

    results = {}

    # 1. Check BEV Head
    head_grad = sum(p.grad.abs().sum().item() for p in head.parameters() if p.grad is not None)
    results['BEV Head'] = head_grad
    print(f"1. BEV Head gradient magnitude: {head_grad:.6f}")

    # 2. Check Attention output_proj
    attn_proj_grad = backbone.spatial_cross_attention.output_proj.weight.grad
    if attn_proj_grad is not None:
        results['Attention output_proj'] = attn_proj_grad.abs().sum().item()
        print(f"2. Attention output_proj gradient: {results['Attention output_proj']:.6f}")
    else:
        results['Attention output_proj'] = 0.0
        print("2. Attention output_proj gradient: None")

    # 3. Check ConvGRU
    convgru_grad = sum(p.grad.abs().sum().item() for p in backbone.conv_gru.parameters() if p.grad is not None)
    results['ConvGRU'] = convgru_grad
    print(f"3. ConvGRU gradient magnitude: {convgru_grad:.6f}")

    # 4. Check FPN (Neck)
    fpn_grad = sum(p.grad.abs().sum().item() for p in backbone.neck.parameters() if p.grad is not None)
    results['FPN Neck'] = fpn_grad
    print(f"4. FPN Neck gradient magnitude: {fpn_grad:.6f}")

    # 5. Check Backbone (ResNet)
    backbone_grad = sum(p.grad.abs().sum().item() for p in backbone.backbone.parameters() if p.grad is not None)
    results['ResNet Backbone'] = backbone_grad
    print(f"5. ResNet Backbone gradient magnitude: {backbone_grad:.6f}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    all_pass = True
    for name, grad_mag in results.items():
        status = "✓ PASS" if grad_mag > 0 else "✗ FAIL"
        if grad_mag == 0:
            all_pass = False
        print(f"  {name}: {status} (grad={grad_mag:.6f})")

    print("\n" + "=" * 60)
    if all_pass:
        print("✓ ALL GRADIENTS FLOWING CORRECTLY!")
    else:
        print("✗ GRADIENT FLOW ISSUE DETECTED")
        print("\nDebug info:")
        print("- Check if reference points are within [-1, 1]")
        print("- Check if bev_mask has any valid (1.0) values")
        print("- Verify padding_mode='border' in grid_sample")
    print("=" * 60)

    return all_pass


if __name__ == '__main__':
    success = verify_gradient_flow()
    sys.exit(0 if success else 1)
