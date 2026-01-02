"""
Comparison script for evaluating all three temporal methods.

This script trains and evaluates:
1. ConvGRU (baseline)
2. Temporal Attention with Memory Bank
3. MC-ConvRNN (proposed)

Usage:
    python scripts/compare_temporal_methods.py
    # Or from notebook:
    %run scripts/compare_temporal_methods.py
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import json
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modules.nuscenes_dataset import NuScenesDataset, create_collate_fn
from modules.bevformer import EnhancedBEVFormer
from modules.head import BEVHead
from modules.head import DetectionLoss  # Explicit import to avoid ambiguity
from modules.metrics import DetectionEvaluator, NUSCENES_CLASSES
from modules.head import DetectionPostProcessor


def train_and_evaluate(
    temporal_method: str,
    config: dict,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int = 5  # Reduced for comparison
) -> dict:
    """
    Train and evaluate a model with a specific temporal method.
    
    Returns:
        Dictionary with training and evaluation metrics
    """
    print(f"\n{'='*80}")
    print(f"Training with temporal method: {temporal_method}")
    print(f"{'='*80}\n")
    
    # Model
    model_backbone = EnhancedBEVFormer(
        bev_h=config['bev_h'],
        bev_w=config['bev_w'],
        embed_dim=config['embed_dim'],
        temporal_method=temporal_method,
        max_history=config.get('max_history', 5),
        enable_bptt=config.get('enable_bptt', False),
        mc_disable_warping=config.get('mc_disable_warping', False),
        mc_disable_motion_field=config.get('mc_disable_motion_field', False),
        mc_disable_visibility=config.get('mc_disable_visibility', False)
    ).to(device)
    
    model_head = BEVHead(
        embed_dim=config['embed_dim'],
        num_classes=config['num_classes']
    ).to(device)
    
    criterion = DetectionLoss()
    # Safety check: fail fast if wrong loss is imported
    assert criterion.__class__.__module__.endswith("modules.head"), \
        f"Wrong DetectionLoss imported! Got {criterion.__class__.__module__}.{criterion.__class__.__name__}. " \
        f"Must use modules.head.DetectionLoss"
    print(f"✓ Using correct DetectionLoss: {criterion.__class__.__module__}.{criterion.__class__.__name__}")
    optimizer = optim.AdamW(
        list(model_backbone.parameters()) + list(model_head.parameters()),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Train
        model_backbone.train()
        model_head.train()
        epoch_train_loss = 0
        
        for batch in train_loader:
            imgs = batch['img'].to(device)
            intrinsics = batch['intrinsics'].to(device)
            extrinsics = batch['extrinsics'].to(device)
            ego_pose = batch['ego_pose'].to(device)
            scene_tokens = batch.get('scene_tokens', None)
            
            targets = {
                'cls_targets': batch['cls_targets'].to(device),
                'bbox_targets': batch['bbox_targets'].to(device),
                'reg_mask': batch['reg_mask'].to(device)
            }
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=config.get('use_amp', False)):
                bev_seq = model_backbone.forward_sequence(
                    imgs, intrinsics, extrinsics, ego_pose,
                    scene_tokens=scene_tokens
                )
                preds = model_head(bev_seq[:, -1])
                loss_dict = criterion(preds, targets)
                loss = loss_dict['loss_total']
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_backbone.parameters(), config['grad_clip'])
            torch.nn.utils.clip_grad_norm_(model_head.parameters(), config['grad_clip'])
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validate
        model_backbone.eval()
        model_head.eval()
        epoch_val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch['img'].to(device)
                intrinsics = batch['intrinsics'].to(device)
                extrinsics = batch['extrinsics'].to(device)
                ego_pose = batch['ego_pose'].to(device)
                scene_tokens = batch.get('scene_tokens', None)
                
                targets = {
                    'cls_targets': batch['cls_targets'].to(device),
                    'bbox_targets': batch['bbox_targets'].to(device),
                    'reg_mask': batch['reg_mask'].to(device)
                }
                
                with torch.amp.autocast('cuda', enabled=config.get('use_amp', False)):
                    bev_seq = model_backbone.forward_sequence(
                        imgs, intrinsics, extrinsics, ego_pose,
                        scene_tokens=scene_tokens
                    )
                    preds = model_head(bev_seq[:, -1])
                    loss_dict = criterion(preds, targets)
                    epoch_val_loss += loss_dict['loss_total'].item()
        
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        
        print(f"Epoch {epoch+1}/{num_epochs} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")
    
    # Evaluate on test set
    print(f"\nEvaluating on test set...")
    evaluator = DetectionEvaluator(class_names=NUSCENES_CLASSES)
    post_processor = DetectionPostProcessor(
        score_threshold=0.1,
        nms_threshold=0.5,
        max_detections=100,
        class_names=NUSCENES_CLASSES,
        use_fast_decode=True
    )
    
    test_loss = 0
    model_backbone.eval()
    model_head.eval()
    
    with torch.no_grad():
        for batch in test_loader:
            imgs = batch['img'].to(device)
            intrinsics = batch['intrinsics'].to(device)
            extrinsics = batch['extrinsics'].to(device)
            ego_pose = batch['ego_pose'].to(device)
            scene_tokens = batch.get('scene_tokens', None)
            
            targets = {
                'cls_targets': batch['cls_targets'].to(device),
                'bbox_targets': batch['bbox_targets'].to(device),
                'reg_mask': batch['reg_mask'].to(device)
            }
            
            with torch.amp.autocast('cuda', enabled=config.get('use_amp', False)):
                bev_seq = model_backbone.forward_sequence(
                    imgs, intrinsics, extrinsics, ego_pose,
                    scene_tokens=scene_tokens
                )
                preds = model_head(bev_seq[:, -1])
                loss_dict = criterion(preds, targets)
                test_loss += loss_dict['loss_total'].item()
            
            # Decode predictions
            cls_scores = preds['cls_score']
            bbox_preds = preds['bbox_pred']
            predicted_boxes_batch = post_processor(cls_scores, bbox_preds)  # List[List[Box3D]]
            
            # Extract GT boxes
            batch_gt_boxes = batch.get('annotations', [])
            for sample_idx in range(len(batch_gt_boxes)):
                sample_gt_boxes = batch_gt_boxes[sample_idx]
                sample_pred_boxes = predicted_boxes_batch[sample_idx] if sample_idx < len(predicted_boxes_batch) else []
                evaluator.add_batch(sample_pred_boxes, sample_gt_boxes)
    
    avg_test_loss = test_loss / len(test_loader)
    metrics = evaluator.compute_metrics()
    
    results = {
        'temporal_method': temporal_method,
        'best_val_loss': best_val_loss,
        'test_loss': avg_test_loss,
        'mAP': metrics.mAP,
        'NDS': metrics.nds,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'num_params': sum(p.numel() for p in model_backbone.parameters()) + 
                     sum(p.numel() for p in model_head.parameters())
    }
    
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"mAP: {metrics.mAP:.4f} | NDS: {metrics.nds:.4f}")
    
    return results


def main():
    """Main comparison function."""
    # Configuration
    config = {
        'data_root': 'data',
        'bev_h': 200,
        'bev_w': 200,
        'embed_dim': 256,
        'num_classes': 10,
        'batch_size': 1,  # Smaller for comparison
        'learning_rate': 4e-5,
        'weight_decay': 1e-2,
        'grad_clip': 1.0,
        'use_amp': True,
        'max_history': 5,
        'enable_bptt': False
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = NuScenesDataset(dataroot=config['data_root'], version='v1.0-mini', split='train')
    val_dataset = NuScenesDataset(dataroot=config['data_root'], version='v1.0-mini', split='val')
    test_dataset = NuScenesDataset(dataroot=config['data_root'], version='v1.0-mini', split='test')
    
    collate_fn = create_collate_fn(
        bev_h=config['bev_h'],
        bev_w=config['bev_w'],
        num_classes=config['num_classes'],
        generate_targets=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                              num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False,
                            num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False,
                             num_workers=0, collate_fn=collate_fn)
    
    # Compare all temporal methods
    temporal_methods = ['convgru', 'temporal_attention', 'mc_convrnn']
    all_results = []
    
    for method in temporal_methods:
        try:
            results = train_and_evaluate(
                temporal_method=method,
                config=config,
                device=device,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                num_epochs=5  # Quick comparison
            )
            all_results.append(results)
        except Exception as e:
            print(f"Error training {method}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print comparison summary
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}\n")
    print(f"{'Method':<20} {'Val Loss':<12} {'Test Loss':<12} {'mAP':<8} {'NDS':<8} {'Params':<12}")
    print("-" * 80)
    
    for r in all_results:
        print(f"{r['temporal_method']:<20} {r['best_val_loss']:<12.4f} "
              f"{r['test_loss']:<12.4f} {r['mAP']:<8.4f} {r['NDS']:<8.4f} "
              f"{r['num_params']:<12,}")
    
    # Save results
    output_dir = Path('checkpoints')
    output_dir.mkdir(exist_ok=True)
    results_file = output_dir / 'temporal_comparison.json'
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    return all_results


if __name__ == '__main__':
    main()

