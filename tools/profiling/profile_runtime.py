#!/usr/bin/env python
"""
Runtime Profiling Tool for BEV Fusion Model.

Measures:
- Mean iteration time (forward + backward)
- Forward-only inference time
- FPS (frames per second)
- Peak CUDA memory usage
- Optional module-level timing breakdown

Outputs results to JSON and console.

Usage:
    python tools/profile_runtime.py --config configs/base_config.yaml
    python tools/profile_runtime.py --mode train --batch_size 2
    python tools/profile_runtime.py --mode inference --profile_modules
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.bev_fusion_model import BEVFusionModel, create_bev_fusion_model
from modules.data_structures import BEVGridConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Profile BEV Fusion Model Runtime')
    
    # Config
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML file')
    
    # Mode
    parser.add_argument('--mode', type=str, default='both',
                        choices=['train', 'inference', 'both'],
                        help='Profiling mode: train, inference, or both')
    
    # Model params (override config)
    parser.add_argument('--fusion_type', type=str, default='bidirectional_cross_attn',
                        choices=['bidirectional_cross_attn', 'cross_attention', 
                                 'local_attention', 'convolutional'],
                        help='Fusion type to profile')
    parser.add_argument('--temporal_type', type=str, default='mc_convrnn',
                        choices=['transformer', 'mc_convrnn', 'none'],
                        help='Temporal aggregation type')
    
    # Runtime params
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for profiling')
    parser.add_argument('--bev_size', type=int, default=200,
                        help='BEV grid size (H=W)')
    parser.add_argument('--num_cameras', type=int, default=6,
                        help='Number of camera views')
    parser.add_argument('--img_h', type=int, default=224,
                        help='Image height')
    parser.add_argument('--img_w', type=int, default=400,
                        help='Image width')
    parser.add_argument('--num_points', type=int, default=10000,
                        help='Number of LiDAR points')
    
    # Profiling params
    parser.add_argument('--warmup_iters', type=int, default=10,
                        help='Number of warmup iterations')
    parser.add_argument('--profile_iters', type=int, default=50,
                        help='Number of profiling iterations')
    parser.add_argument('--profile_modules', action='store_true',
                        help='Enable per-module timing breakdown')
    
    # Hardware
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to profile on')
    parser.add_argument('--amp', action='store_true',
                        help='Use automatic mixed precision')
    parser.add_argument('--torch_compile', action='store_true',
                        help='Use torch.compile (PyTorch 2.0+)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='outputs/profile',
                        help='Output directory for results')
    
    return parser.parse_args()


def create_dummy_inputs(
    batch_size: int,
    num_cameras: int,
    img_h: int,
    img_w: int,
    num_points: int,
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """Create dummy inputs for profiling."""
    return {
        'lidar_points': torch.randn(batch_size, num_points, 4, device=device),
        'camera_images': torch.randn(batch_size, num_cameras, 3, img_h, img_w, device=device),
        'camera_intrinsics': torch.eye(3, device=device).unsqueeze(0).unsqueeze(0)
                            .expand(batch_size, num_cameras, -1, -1).clone(),
        'camera_extrinsics': torch.eye(4, device=device).unsqueeze(0).unsqueeze(0)
                            .expand(batch_size, num_cameras, -1, -1).clone(),
        'ego_transform': torch.eye(4, device=device).unsqueeze(0).expand(batch_size, -1, -1).clone()
    }


class ModuleTimer:
    """Context manager for timing modules."""
    
    def __init__(self, name: str, timings: Dict[str, List[float]], device: torch.device):
        self.name = name
        self.timings = timings
        self.device = device
        self.start_time = None
        
    def __enter__(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - self.start_time) * 1000  # ms
        
        if self.name not in self.timings:
            self.timings[self.name] = []
        self.timings[self.name].append(elapsed)


def profile_forward(
    model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    num_iters: int,
    device: torch.device,
    use_amp: bool = False,
    profile_modules: bool = False
) -> Dict[str, any]:
    """Profile forward pass."""
    model.eval()
    
    timings: Dict[str, List[float]] = {'total': []}
    
    with torch.no_grad():
        for _ in range(num_iters):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            
            if use_amp:
                with torch.cuda.amp.autocast():
                    _ = model(**inputs)
            else:
                _ = model(**inputs)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            elapsed = (time.perf_counter() - start) * 1000  # ms
            timings['total'].append(elapsed)
    
    # Compute statistics
    total_times = timings['total']
    return {
        'mean_ms': sum(total_times) / len(total_times),
        'std_ms': (sum((t - sum(total_times)/len(total_times))**2 for t in total_times) / len(total_times)) ** 0.5,
        'min_ms': min(total_times),
        'max_ms': max(total_times),
        'fps': 1000.0 / (sum(total_times) / len(total_times)),
    }


def profile_train_step(
    model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    num_iters: int,
    device: torch.device,
    use_amp: bool = False
) -> Dict[str, any]:
    """Profile training step (forward + backward)."""
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    timings = []
    
    for _ in range(num_iters):
        optimizer.zero_grad()
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(**inputs)
                loss = outputs['cls_scores'].sum() + outputs['bbox_preds'].sum()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(**inputs)
            loss = outputs['cls_scores'].sum() + outputs['bbox_preds'].sum()
            loss.backward()
            optimizer.step()
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = (time.perf_counter() - start) * 1000  # ms
        timings.append(elapsed)
    
    return {
        'mean_ms': sum(timings) / len(timings),
        'std_ms': (sum((t - sum(timings)/len(timings))**2 for t in timings) / len(timings)) ** 0.5,
        'min_ms': min(timings),
        'max_ms': max(timings),
        'iters_per_sec': 1000.0 / (sum(timings) / len(timings)),
    }


def get_memory_stats(device: torch.device) -> Dict[str, float]:
    """Get CUDA memory statistics."""
    if device.type != 'cuda':
        return {}
    
    return {
        'allocated_mb': torch.cuda.memory_allocated(device) / (1024 ** 2),
        'reserved_mb': torch.cuda.memory_reserved(device) / (1024 ** 2),
        'max_allocated_mb': torch.cuda.max_memory_allocated(device) / (1024 ** 2),
        'max_reserved_mb': torch.cuda.max_memory_reserved(device) / (1024 ** 2),
    }


def main():
    args = parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    print(f"Profiling on device: {device}")
    
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats()
    
    # Create model config
    config = {
        'bev_x_min': -51.2,
        'bev_x_max': 51.2,
        'bev_y_min': -51.2,
        'bev_y_max': 51.2,
        'bev_resolution': 102.4 / args.bev_size,
        'lidar_channels': 64,
        'camera_channels': 256,
        'fused_channels': 256,
        'fusion_type': args.fusion_type,
        'use_temporal_attention': args.temporal_type == 'transformer',
        'use_mc_convrnn': args.temporal_type == 'mc_convrnn',
        'num_classes': 10,
    }
    
    print(f"\nModel Configuration:")
    print(f"  Fusion type: {args.fusion_type}")
    print(f"  Temporal type: {args.temporal_type}")
    print(f"  BEV size: {args.bev_size}x{args.bev_size}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  AMP: {args.amp}")
    
    # Create model
    print("\nCreating model...")
    model = create_bev_fusion_model(config)
    model = model.to(device)
    
    if args.torch_compile and hasattr(torch, 'compile'):
        print("Applying torch.compile...")
        model = torch.compile(model)
    
    # Model stats
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"Model size: {model.get_model_size_mb():.2f} MB")
    
    # Create inputs
    inputs = create_dummy_inputs(
        batch_size=args.batch_size,
        num_cameras=args.num_cameras,
        img_h=args.img_h,
        img_w=args.img_w,
        num_points=args.num_points,
        device=device
    )
    
    results = {
        'config': {
            'fusion_type': args.fusion_type,
            'temporal_type': args.temporal_type,
            'batch_size': args.batch_size,
            'bev_size': args.bev_size,
            'device': str(device),
            'amp': args.amp,
            'torch_compile': args.torch_compile,
        },
        'model': {
            'num_params': num_params,
            'size_mb': model.get_model_size_mb(),
        },
        'timestamp': datetime.now().isoformat(),
    }
    
    # Warmup
    print(f"\nWarming up ({args.warmup_iters} iterations)...")
    with torch.no_grad():
        for _ in range(args.warmup_iters):
            _ = model(**inputs)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    
    # Profile inference
    if args.mode in ['inference', 'both']:
        print(f"\nProfiling inference ({args.profile_iters} iterations)...")
        inference_results = profile_forward(
            model, inputs, args.profile_iters, device, args.amp, args.profile_modules
        )
        results['inference'] = inference_results
        
        print(f"  Mean: {inference_results['mean_ms']:.2f} ms")
        print(f"  Std:  {inference_results['std_ms']:.2f} ms")
        print(f"  FPS:  {inference_results['fps']:.1f}")
    
    # Profile training
    if args.mode in ['train', 'both']:
        print(f"\nProfiling training ({args.profile_iters} iterations)...")
        train_results = profile_train_step(
            model, inputs, args.profile_iters, device, args.amp
        )
        results['train'] = train_results
        
        print(f"  Mean: {train_results['mean_ms']:.2f} ms")
        print(f"  Std:  {train_results['std_ms']:.2f} ms")
        print(f"  Iters/sec: {train_results['iters_per_sec']:.1f}")
    
    # Memory stats
    if device.type == 'cuda':
        memory_stats = get_memory_stats(device)
        results['memory'] = memory_stats
        print(f"\nMemory Usage:")
        print(f"  Peak allocated: {memory_stats['max_allocated_mb']:.1f} MB")
        print(f"  Peak reserved:  {memory_stats['max_reserved_mb']:.1f} MB")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(args.output_dir, f'profile_{args.fusion_type}_{timestamp}.json')
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Summary comparison
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Fusion: {args.fusion_type}")
    print(f"Temporal: {args.temporal_type}")
    if 'inference' in results:
        print(f"Inference FPS: {results['inference']['fps']:.1f}")
    if 'train' in results:
        print(f"Training iters/sec: {results['train']['iters_per_sec']:.1f}")
    if 'memory' in results:
        print(f"Peak memory: {results['memory']['max_allocated_mb']:.1f} MB")


if __name__ == '__main__':
    main()

