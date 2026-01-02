"""BEVFormer++ Training Script

This script supports multiple training configurations:

1. Camera-only with different temporal methods:
   - convgru: Simple ConvGRU baseline (original)
   - temporal_attention: Transformer attention with MemoryBank
   - mc_convrnn: Motion-Compensated ConvRNN (proposed contribution)

2. Multi-modal fusion (LiDAR + Camera):
   - Uses BEVFusionModel with configurable fusion and temporal methods

Usage:
    python train.py --temporal_method mc_convrnn
    python train.py --temporal_method temporal_attention
    python train.py --use_fusion --temporal_method mc_convrnn
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import random
import numpy as np
import shutil
from pathlib import Path
from modules.nuscenes_dataset import NuScenesDataset, create_collate_fn, create_fusion_collate_fn
from modules.bevformer import EnhancedBEVFormer
from modules.head import BEVHead
from modules.head import DetectionLoss  # Explicit import to avoid ambiguity
from modules.metrics import DetectionEvaluator, NUSCENES_CLASSES
import json
import yaml
from datetime import datetime


# ============================================================================
# JSONL Logger for visualization / notebook integration
# ============================================================================

class JSONLLogger:
    """Structured event logger that writes JSONL (one JSON object per line).

    Events include: timestamp, epoch, global_step, split, loss components, lr.
    Used by main.ipynb to visualize training without parsing stdout.
    """

    def __init__(self, filepath: str = None, log_every_n: int = 1):
        self.filepath = filepath
        self.log_every_n = log_every_n
        self._file = None
        if filepath:
            os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
            self._file = open(filepath, 'w')  # Overwrite existing

    def log(self, event: dict, force: bool = False):
        """Log an event dict to JSONL file."""
        if self._file is None:
            return
        # Add timestamp
        event['time'] = datetime.now().isoformat()
        self._file.write(json.dumps(event) + '\n')
        self._file.flush()

    def close(self):
        if self._file:
            self._file.close()
            self._file = None


# ============================================================================
# Run Artifact Generation
# ============================================================================

def create_run_directory(base_dir: str = "runs") -> Path:
    """Create a timestamped run directory for artifacts.

    Args:
        base_dir: Base directory for all runs

    Returns:
        Path to the created run directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_resolved_config(args, run_dir: Path, config_path: str = None) -> Path:
    """Save the fully resolved configuration to run directory.

    This includes all CLI overrides applied to base config.

    Args:
        args: Parsed arguments
        run_dir: Run directory path
        config_path: Original config file path (if any)

    Returns:
        Path to saved config file
    """
    # Build resolved config dict from args
    resolved = {
        'training': {
            'batch_size': args.batch_size,
            'num_epochs': args.epochs,
            'learning_rate': args.lr,
            'max_steps': args.max_steps,
            'seed': args.seed,
        },
        'dataset': {
            'data_root': args.data_root,
            'version': 'v1.0-mini',
        },
        'model': {
            'use_fusion': args.use_fusion,
            'temporal': {
                'type': args.temporal_method,
                'max_history': args.max_history,
                'enable_bptt': args.enable_bptt,
            },
            'fusion': {
                'type': args.fusion_type,
            },
            'mc_ablations': {
                'disable_warping': args.mc_disable_warping,
                'disable_motion_field': args.mc_disable_motion_field,
                'disable_visibility': args.mc_disable_visibility,
            }
        },
        'curriculum': {
            'stage1_epochs': args.stage1_epochs,
            'stage1': {
                'heatmap_mode': 'hard_center',
                'gaussian_overlap': args.stage1_gaussian_overlap,
                'min_radius': args.stage1_min_radius,
                'pos_weight': args.stage1_pos_weight,
            },
            'stage2': {
                'heatmap_mode': 'gaussian',
                'gaussian_overlap': args.stage2_gaussian_overlap,
                'min_radius': args.stage2_min_radius,
                'pos_weight': args.stage2_pos_weight,
            }
        },
        'logging': {
            'log_json_path': args.log_json_path,
            'log_jsonl_path': args.log_jsonl_path,
            'log_every_n_steps': args.log_every_n_steps,
        },
        '_meta': {
            'original_config': config_path,
            'resolved_at': datetime.now().isoformat(),
        }
    }

    config_file = run_dir / "config_resolved.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(resolved, f, default_flow_style=False, sort_keys=False)

    return config_file


def save_run_manifest(run_dir: Path, args, checkpoint_path: str = None) -> Path:
    """Save run manifest with metadata for reproducibility.

    Args:
        run_dir: Run directory path
        args: Parsed arguments
        checkpoint_path: Path to saved checkpoint (if any)

    Returns:
        Path to manifest file
    """
    manifest = {
        'run_id': run_dir.name,
        'created_at': datetime.now().isoformat(),
        'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': args.seed,
        'mode': 'fusion' if args.use_fusion else 'camera_only',
        'temporal_method': args.temporal_method,
        'fusion_type': args.fusion_type if args.use_fusion else None,
        'files': {
            'config_resolved': 'config_resolved.yaml',
            'training_log': 'training.jsonl',
            'epoch_log': 'epochs.json',
            'checkpoint': checkpoint_path,
        }
    }

    manifest_file = run_dir / "run_manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)

    return manifest_file


def setup_run_artifacts(args) -> tuple:
    """Setup run directory and artifacts for training.

    Args:
        args: Parsed arguments

    Returns:
        Tuple of (run_dir, jsonl_path, epoch_log_path)
    """
    run_dir = create_run_directory()

    # Update logging paths to use run directory if not explicitly set
    jsonl_path = args.log_jsonl_path
    if jsonl_path is None:
        jsonl_path = str(run_dir / "training.jsonl")

    epoch_log_path = args.log_json_path
    if epoch_log_path is None:
        epoch_log_path = str(run_dir / "epochs.json")

    # Save resolved config
    save_resolved_config(args, run_dir, args.config)

    print(f"Run artifacts will be saved to: {run_dir}")

    return run_dir, jsonl_path, epoch_log_path


def parse_args():
    parser = argparse.ArgumentParser(description='BEVFormer++ Training')
    
    # Config file (optional, CLI args override config)
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file (optional)')
    
    # Model configuration
    temporal_choices = ['convgru', 'temporal_attention', 'mc_convrnn']
    parser.add_argument('--temporal_method', type=str, default=None,
                        choices=temporal_choices,
                        help='Temporal fusion method (default: convgru for camera-only, mc_convrnn for fusion)')
    parser.add_argument('--temporal.type', dest='temporal_type_dot', type=str, default=None,
                        choices=temporal_choices,
                        help='Alias for --temporal_method')
    parser.add_argument('--use_fusion', action='store_true',
                        help='Use multi-modal fusion with LiDAR')
    parser.add_argument('--use_lidar', action='store_true',
                        help='Alias for --use_fusion')
    fusion_choices = ['bidirectional_cross_attn', 'cross_attention', 'cross_attn', 'local_attention', 'convolutional']
    parser.add_argument('--fusion_type', type=str, default=None,
                        choices=fusion_choices,
                        help='Spatial fusion type for BEVFusionModel (defaults to bidirectional_cross_attn)')
    # Alias to match dot-notation style
    parser.add_argument('--fusion.type', dest='fusion_type_dot', type=str, default=None,
                        choices=fusion_choices,
                        help='Alias for --fusion_type')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=None,
                        help='Max training steps (for debugging, overrides epochs)')
    parser.add_argument('--data_root', type=str, default='data', help='Dataset root')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--log_json_path', type=str, default=None,
                        help='Optional path to save per-epoch loss trends as JSON (train/val)')
    parser.add_argument('--log_jsonl_path', type=str, default=None,
                        help='Optional path for detailed per-step JSONL logging (for visualization)')
    parser.add_argument('--log_every_n_steps', type=int, default=1,
                        help='Log every N training steps to JSONL (default: 1)')
    parser.add_argument('--run_dir', type=str, default=None,
                        help='Directory for run artifacts (auto-generated if not set)')
    parser.add_argument('--save_run_artifacts', action='store_true',
                        help='Save run artifacts (config_resolved.yaml, run_manifest.json)')
    
    # Temporal configuration
    parser.add_argument('--max_history', type=int, default=5, help='Max temporal history')
    parser.add_argument('--enable_bptt', action='store_true', help='Enable BPTT for temporal modules')
    
    # Curriculum schedule for heatmap targets
    parser.add_argument('--stage1_epochs', type=int, default=2,
                        help='Number of epochs to use hard_center targets (Stage 1), then switch to gaussian (Stage 2)')
    
    # Stage 1 parameters (shortcut breaking)
    parser.add_argument('--stage1_gaussian_overlap', type=float, default=0.01,
                        help='Gaussian overlap for Stage 1 (unused if hard_center, but kept for consistency)')
    parser.add_argument('--stage1_min_radius', type=int, default=1,
                        help='Min radius for Stage 1 (unused if hard_center, but kept for consistency)')
    parser.add_argument('--stage1_pos_weight', type=float, default=50.0,
                        help='Classification loss pos_weight for Stage 1 (strong negative suppression)')
    
    # Stage 2 parameters (recall recovery)
    parser.add_argument('--stage2_gaussian_overlap', type=float, default=0.1,
                        help='Gaussian overlap for Stage 2 (relaxed for better recall)')
    parser.add_argument('--stage2_min_radius', type=int, default=2,
                        help='Min radius for Stage 2 (relaxed for better recall)')
    parser.add_argument('--stage2_pos_weight', type=float, default=10.0,
                        help='Classification loss pos_weight for Stage 2 (softened for better recall)')
    
    # MC-ConvRNN ablation study options
    parser.add_argument('--mc_disable_warping', action='store_true',
                        help='Ablation: Disable ego-motion warping in MC-ConvRNN')
    parser.add_argument('--mc_disable_motion_field', action='store_true',
                        help='Ablation: Disable dynamic motion field estimation')
    parser.add_argument('--mc_disable_visibility', action='store_true',
                        help='Ablation: Disable visibility gating')
    
    args = parser.parse_args()
    
    # Map config keys to args (only if arg not explicitly set via CLI)
    config_mapping = {
        'training.batch_size': 'batch_size',
        'training.num_epochs': 'epochs',
        'training.learning_rate': 'lr',
        'dataset.data_root': 'data_root',
        'model.temporal.sequence_length': 'max_history',
        'model.fusion.type': 'fusion_type',
    }

    # Load config file and merge with CLI args (CLI takes precedence)
    if args.config is not None:
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        for config_path, arg_name in config_mapping.items():
            keys = config_path.split('.')
            val = config
            try:
                for k in keys:
                    val = val[k]
                # Only set if arg is still at default
                if getattr(args, arg_name) == parser.get_default(arg_name):
                    setattr(args, arg_name, val)
            except (KeyError, TypeError):
                pass

    # Handle alias: fusion.type
    if args.fusion_type is None and args.fusion_type_dot is not None:
        args.fusion_type = args.fusion_type_dot
    if args.fusion_type is None:
        args.fusion_type = 'bidirectional_cross_attn'

    # Normalize shorthand aliases
    fusion_alias_map = {
        'cross_attn': 'cross_attention',
        'cross-attn': 'cross_attention',
        'bidir': 'bidirectional_cross_attn',
        'bidir_cross_attn': 'bidirectional_cross_attn',
    }
    if args.fusion_type in fusion_alias_map:
        args.fusion_type = fusion_alias_map[args.fusion_type]

    # Handle alias: use_lidar -> use_fusion
    if args.use_lidar:
        args.use_fusion = True

    # Handle alias: temporal.type -> temporal_method
    if args.temporal_method is None and args.temporal_type_dot is not None:
        args.temporal_method = args.temporal_type_dot
    
    # FIX 2: DEFAULT TEMPORAL IN FUSION MUST BE MC-CONVRNN
    # Set appropriate default based on training mode
    if args.temporal_method is None:
        if args.use_fusion:
            args.temporal_method = 'mc_convrnn'  # Proposal default for fusion
        else:
            args.temporal_method = 'convgru'  # Legacy default for camera-only

    return args


def get_stage_params(epoch: int, stage1_epochs: int, args) -> dict:
    """Get all stage-specific parameters based on curriculum schedule.
    
    Stage 1 (epochs 0 to stage1_epochs-1): hard_center with strong loss
    Stage 2 (epoch >= stage1_epochs): gaussian with relaxed loss
    
    Returns:
        Dictionary with keys: heatmap_mode, gaussian_overlap, min_radius, pos_weight
    """
    if epoch < stage1_epochs:
        return {
            'heatmap_mode': 'hard_center',
            'gaussian_overlap': args.stage1_gaussian_overlap,
            'min_radius': args.stage1_min_radius,
            'pos_weight': args.stage1_pos_weight
        }
    else:
        return {
            'heatmap_mode': 'gaussian',
            'gaussian_overlap': args.stage2_gaussian_overlap,
            'min_radius': args.stage2_min_radius,
            'pos_weight': args.stage2_pos_weight
        }


def train_camera_only(args):
    """Training loop for camera-only EnhancedBEVFormer with configurable temporal method."""
    # Seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Setup run artifacts if enabled
    run_dir = None
    jsonl_path = args.log_jsonl_path
    epoch_log_path = args.log_json_path
    if args.save_run_artifacts:
        run_dir, jsonl_path, epoch_log_path = setup_run_artifacts(args)

    # 1. Config
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs
    LR = args.lr
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    STAGE1_EPOCHS = args.stage1_epochs

    # Dataset Paths
    NUSCENES_ROOT = args.data_root

    # BEV configuration
    BEV_H, BEV_W = 200, 200
    EMBED_DIM = 256
    NUM_CLASSES = 10

    print(f"Starting camera-only training on {DEVICE}...")
    print(f"Temporal method: {args.temporal_method}")
    print(f"Curriculum schedule: Stage 1 (hard_center) for epochs 0-{STAGE1_EPOCHS-1}, "
          f"Stage 2 (gaussian) for epochs {STAGE1_EPOCHS}+")
    print(f"Stage 1 params: pos_weight={args.stage1_pos_weight}, "
          f"gaussian_overlap={args.stage1_gaussian_overlap}, min_radius={args.stage1_min_radius}")
    print(f"Stage 2 params: pos_weight={args.stage2_pos_weight}, "
          f"gaussian_overlap={args.stage2_gaussian_overlap}, min_radius={args.stage2_min_radius}")

    # 2. Data - use scene-level splits to prevent data leakage
    print(f"Loading NuScenes Dataset from {NUSCENES_ROOT}...")
    train_dataset = NuScenesDataset(dataroot=NUSCENES_ROOT, version='v1.0-mini',
                                     split='train', load_lidar=False)
    val_dataset = NuScenesDataset(dataroot=NUSCENES_ROOT, version='v1.0-mini',
                                   split='val', load_lidar=False)
    test_dataset = NuScenesDataset(dataroot=NUSCENES_ROOT, version='v1.0-mini',
                                    split='test', load_lidar=False)

    print(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    # Initialize loaders (will be recreated per epoch if mode changes)
    train_loader = None
    val_loader = None
    test_loader = None
    current_mode = None

    # 3. Model - with configurable temporal method
    model_backbone = EnhancedBEVFormer(
        bev_h=BEV_H, 
        bev_w=BEV_W, 
        embed_dim=EMBED_DIM,
        temporal_method=args.temporal_method,
        max_history=args.max_history,
        enable_bptt=args.enable_bptt,
        # MC-ConvRNN ablation flags
        mc_disable_warping=args.mc_disable_warping,
        mc_disable_motion_field=args.mc_disable_motion_field,
        mc_disable_visibility=args.mc_disable_visibility
    ).to(DEVICE)
    model_head = BEVHead(
        embed_dim=EMBED_DIM, num_classes=NUM_CLASSES, reg_channels=7
    ).to(DEVICE)
    print(f"Using EnhancedBEVFormer (camera-only) with {args.temporal_method}")

    # Loss will be created per-stage (initialized in training loop)
    criterion = None
    
    # Guardrail: Verify target generation configuration
    initial_params = get_stage_params(0, STAGE1_EPOCHS, args)
    print(f"✓ Target generation will use curriculum schedule (starts with heatmap_mode='{initial_params['heatmap_mode']}')")

    # Optimizer
    optimizer = optim.AdamW(list(model_backbone.parameters()) + list(model_head.parameters()), lr=LR, weight_decay=1e-2)

    # 4. Training Loop
    global_step = 0
    max_steps = args.max_steps  # None means no limit
    stop_training = False
    epoch_logs = []  # L3: Track per-epoch metrics for trend validation

    # JSONL logger for detailed step-by-step logging (used by notebook visualization)
    jsonl_logger = JSONLLogger(jsonl_path, args.log_every_n_steps)
    jsonl_logger.log({
        'event': 'config',
        'epochs': NUM_EPOCHS,
        'batch_size': BATCH_SIZE,
        'lr': LR,
        'temporal_method': args.temporal_method,
        'use_fusion': False,
        'seed': args.seed,
        'run_dir': str(run_dir) if run_dir else None,
    })
    
    for epoch in range(NUM_EPOCHS):
        if stop_training:
            break
            
        # Get stage-specific parameters
        stage_params = get_stage_params(epoch, STAGE1_EPOCHS, args)
        heatmap_mode = stage_params['heatmap_mode']
        
        # Check if we need to switch stage (recreate collate_fn and criterion)
        if current_mode != heatmap_mode:
            if current_mode is None:
                print(f"\n[CURRICULUM] Initializing Stage 1 at epoch {epoch+1}")
            else:
                print(f"\n[CURRICULUM] Switching stages at epoch {epoch+1}")
            
            # Log all active parameters
            print(f"[CURRICULUM]")
            print(f"  epoch={epoch+1}")
            print(f"  heatmap_mode={stage_params['heatmap_mode']}")
            print(f"  gaussian_overlap={stage_params['gaussian_overlap']}")
            print(f"  min_radius={stage_params['min_radius']}")
            print(f"  cls_loss:")
            print(f"    pos_weight={stage_params['pos_weight']}")
            
            current_mode = heatmap_mode
            
            # Recreate collate_fn and DataLoaders with stage-specific parameters
            collate_fn = create_collate_fn(
                bev_h=BEV_H,
                bev_w=BEV_W,
                num_classes=NUM_CLASSES,
                generate_targets=True,
                heatmap_mode=stage_params['heatmap_mode'],
                gaussian_overlap=stage_params['gaussian_overlap'],
                min_radius=stage_params['min_radius']
            )
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                      num_workers=0, collate_fn=collate_fn)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                    num_workers=0, collate_fn=collate_fn)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                     num_workers=0, collate_fn=collate_fn)
            
            # Recreate DetectionLoss with stage-specific pos_weight
            criterion = DetectionLoss(pos_weight=stage_params['pos_weight'])
            # Safety check: fail fast if wrong loss is imported
            assert criterion.__class__.__module__.endswith("modules.head"), \
                f"Wrong DetectionLoss imported! Got {criterion.__class__.__module__}.{criterion.__class__.__name__}. " \
                f"Must use modules.head.DetectionLoss"
            print(f"  ✓ Recreated DetectionLoss with pos_weight={stage_params['pos_weight']}")
        
        # --- TRAIN ---
        model_backbone.train()
        model_head.train()
        epoch_loss = 0
        epoch_cls_loss = 0
        epoch_bbox_loss = 0
        start_time = time.time()

        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}] Training... (heatmap_mode={heatmap_mode}, pos_weight={stage_params['pos_weight']})")
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            imgs = batch['img'].to(DEVICE)  # (B, Seq, 6, 3, H, W)
            intrinsics = batch['intrinsics'].to(DEVICE)
            extrinsics = batch['extrinsics'].to(DEVICE)
            ego_pose = batch['ego_pose'].to(DEVICE)

            # Ground truth targets from collate_fn (CenterNet-style heatmaps)
            targets = {
                'cls_targets': batch['cls_targets'].to(DEVICE),
                'bbox_targets': batch['bbox_targets'].to(DEVICE),
                'reg_mask': batch['reg_mask'].to(DEVICE)
            }

            # Forward
            bev_seq_features = model_backbone.forward_sequence(imgs, intrinsics, extrinsics, ego_pose)
            bev_features = bev_seq_features[:, -1]
            preds = model_head(bev_features)

            # Loss
            loss_dict = criterion(preds, targets)
            loss = loss_dict['loss_total']

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_backbone.parameters(), 5.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_cls_loss += loss_dict.get('loss_cls', torch.tensor(0.0)).item()
            epoch_bbox_loss += loss_dict.get('loss_bbox', torch.tensor(0.0)).item()
            global_step += 1

            # JSONL step logging
            if global_step % args.log_every_n_steps == 0:
                jsonl_logger.log({
                    'event': 'step',
                    'split': 'train',
                    'epoch': epoch + 1,
                    'global_step': global_step,
                    'batch_idx': batch_idx,
                    'loss_total': loss.item(),
                    'loss_cls': loss_dict.get('loss_cls', torch.tensor(0.0)).item(),
                    'loss_bbox': loss_dict.get('loss_bbox', torch.tensor(0.0)).item(),
                    'lr': optimizer.param_groups[0]['lr'],
                })

            if batch_idx % 10 == 0:
                print(f"  Batch [{batch_idx}/{len(train_loader)}] Train Loss: {loss.item():.4f}")
            
            # Check max_steps limit
            if max_steps is not None and global_step >= max_steps:
                print(f"Reached max_steps={max_steps}, stopping training.")
                stop_training = True
                break

        if stop_training:
            break
            
        n_train = max(len(train_loader), 1)
        avg_train_loss = epoch_loss / n_train
        avg_train_cls = epoch_cls_loss / n_train
        avg_train_bbox = epoch_bbox_loss / n_train

        # --- VALIDATION ---
        model_backbone.eval()
        model_head.eval()
        val_loss = 0
        val_cls_loss = 0
        val_bbox_loss = 0

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Validating...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                imgs = batch['img'].to(DEVICE)
                intrinsics = batch['intrinsics'].to(DEVICE)
                extrinsics = batch['extrinsics'].to(DEVICE)
                ego_pose = batch['ego_pose'].to(DEVICE)

                # Ground truth targets from collate_fn
                targets = {
                    'cls_targets': batch['cls_targets'].to(DEVICE),
                    'bbox_targets': batch['bbox_targets'].to(DEVICE),
                    'reg_mask': batch['reg_mask'].to(DEVICE)
                }

                bev_seq_features = model_backbone.forward_sequence(imgs, intrinsics, extrinsics, ego_pose)
                bev_features = bev_seq_features[:, -1]
                preds = model_head(bev_features)

                loss_dict = criterion(preds, targets)
                val_loss += loss_dict['loss_total'].item()
                val_cls_loss += loss_dict.get('loss_cls', torch.tensor(0.0)).item()
                val_bbox_loss += loss_dict.get('loss_bbox', torch.tensor(0.0)).item()

        n_val = max(len(val_loader), 1)
        avg_val_loss = val_loss / n_val
        avg_val_cls = val_cls_loss / n_val
        avg_val_bbox = val_bbox_loss / n_val

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {time.time() - start_time:.2f}s")

        # JSONL epoch summary logging
        jsonl_logger.log({
            'event': 'epoch',
            'split': 'train',
            'epoch': epoch + 1,
            'global_step': global_step,
            'loss_total': avg_train_loss,
            'loss_cls': avg_train_cls,
            'loss_bbox': avg_train_bbox,
            'lr': optimizer.param_groups[0]['lr'],
        })
        jsonl_logger.log({
            'event': 'epoch',
            'split': 'val',
            'epoch': epoch + 1,
            'global_step': global_step,
            'loss_total': avg_val_loss,
            'loss_cls': avg_val_cls,
            'loss_bbox': avg_val_bbox,
        })

        # L3: Record epoch metrics for trend validation
        epoch_logs.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'train_cls_loss': avg_train_cls,
            'train_bbox_loss': avg_train_bbox,
            'val_cls_loss': avg_val_cls,
            'val_bbox_loss': avg_val_bbox,
        })

    # L3: Optionally dump epoch logs to JSON for automated trend tests
    if epoch_log_path:
        os.makedirs(os.path.dirname(epoch_log_path) or '.', exist_ok=True)
        with open(epoch_log_path, 'w') as f:
            json.dump(epoch_logs, f, indent=2)
        print(f"Epoch logs saved to {epoch_log_path}")

    jsonl_logger.close()
    print("Training Complete.")

    # --- TEST ---
    print("\nRunning Evaluation on Test Set...")
    model_backbone.eval()
    model_head.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            imgs = batch['img'].to(DEVICE)
            intrinsics = batch['intrinsics'].to(DEVICE)
            extrinsics = batch['extrinsics'].to(DEVICE)
            ego_pose = batch['ego_pose'].to(DEVICE)

            # Ground truth targets from collate_fn
            targets = {
                'cls_targets': batch['cls_targets'].to(DEVICE),
                'bbox_targets': batch['bbox_targets'].to(DEVICE),
                'reg_mask': batch['reg_mask'].to(DEVICE)
            }

            bev_seq_features = model_backbone.forward_sequence(imgs, intrinsics, extrinsics, ego_pose)
            bev_features = bev_seq_features[:, -1]
            preds = model_head(bev_features)

            loss_dict = criterion(preds, targets)
            test_loss += loss_dict['loss_total'].item()

    print(f"Test Set Average Loss: {test_loss / len(test_loader):.4f}")

    # Save checkpoint
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint_name = f'camera_{args.temporal_method}_latest.pth'
    checkpoint_path = f'checkpoints/{checkpoint_name}'
    torch.save({
        'backbone_state_dict': model_backbone.state_dict(),
        'head_state_dict': model_head.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'temporal_method': args.temporal_method,
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

    # Save run manifest if artifacts enabled
    if run_dir is not None:
        save_run_manifest(run_dir, args, checkpoint_path)
        # Copy checkpoint to run directory
        shutil.copy(checkpoint_path, run_dir / checkpoint_name)
        print(f"Run manifest saved to {run_dir / 'run_manifest.json'}")


def train_fusion(args):
    """Training loop for multi-modal BEVFusionModel with LiDAR and Camera."""
    from modules.bev_fusion_model import BEVFusionModel
    from modules.data_structures import BEVGridConfig

    # Setup run artifacts if enabled
    run_dir = None
    jsonl_path = args.log_jsonl_path
    epoch_log_path = args.log_json_path
    if args.save_run_artifacts:
        run_dir, jsonl_path, epoch_log_path = setup_run_artifacts(args)

    # 1. Config
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs
    LR = args.lr
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    STAGE1_EPOCHS = args.stage1_epochs

    # Dataset Paths
    NUSCENES_ROOT = args.data_root

    # BEV configuration - use BEVGridConfig for consistency
    BEV_H, BEV_W = 200, 200
    EMBED_DIM = 256
    NUM_CLASSES = 10
    LIDAR_CHANNELS = 64

    # Create BEVGridConfig with proper resolution for 200×200 grid
    bev_config = BEVGridConfig.from_grid_size(bev_h=BEV_H, bev_w=BEV_W)

    print(f"Starting fusion training on {DEVICE}...")
    print(f"BEV grid: {bev_config.grid_size}, range: {bev_config.bev_range}")
    print(f"Temporal method: {args.temporal_method}")
    print(f"Curriculum schedule: Stage 1 (hard_center) for epochs 0-{STAGE1_EPOCHS-1}, "
          f"Stage 2 (gaussian) for epochs {STAGE1_EPOCHS}+")
    print(f"Stage 1 params: pos_weight={args.stage1_pos_weight}, "
          f"gaussian_overlap={args.stage1_gaussian_overlap}, min_radius={args.stage1_min_radius}")
    print(f"Stage 2 params: pos_weight={args.stage2_pos_weight}, "
          f"gaussian_overlap={args.stage2_gaussian_overlap}, min_radius={args.stage2_min_radius}")

    # 2. Data - use scene-level splits with LiDAR loading enabled
    print(f"Loading NuScenes Dataset from {NUSCENES_ROOT} with LiDAR...")
    train_dataset = NuScenesDataset(dataroot=NUSCENES_ROOT, version='v1.0-mini',
                                     split='train', load_lidar=True)
    val_dataset = NuScenesDataset(dataroot=NUSCENES_ROOT, version='v1.0-mini',
                                   split='val', load_lidar=True)
    test_dataset = NuScenesDataset(dataroot=NUSCENES_ROOT, version='v1.0-mini',
                                    split='test', load_lidar=True)

    print(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    # Initialize loaders (will be recreated per epoch if mode changes)
    train_loader = None
    val_loader = None
    test_loader = None
    current_mode = None

    # 3. Model - BEVFusionModel with LiDAR
    # FIX 1: FORBID IMPLICIT TEMPORAL FALLBACK IN FUSION
    # convgru is NOT supported in BEVFusionModel - must fail fast with clear error
    if args.temporal_method == 'convgru':
        raise ValueError(
            "convgru is not supported in BEVFusionModel. "
            "Use --temporal_method mc_convrnn (proposal default) or temporal_attention explicitly."
        )
    
    # FIX 2: DEFAULT TEMPORAL IN FUSION MUST BE MC-CONVRNN
    # mc_convrnn is the proposal-intended default for fusion mode
    use_temporal_attention = args.temporal_method == 'temporal_attention'
    use_mc_convrnn = args.temporal_method == 'mc_convrnn'
    
    print(f"[FUSION] Temporal method: {args.temporal_method} (use_mc_convrnn={use_mc_convrnn}, use_temporal_attention={use_temporal_attention})")

    model = BEVFusionModel(
        bev_config=bev_config,
        lidar_channels=LIDAR_CHANNELS,
        camera_channels=EMBED_DIM,
        fused_channels=EMBED_DIM,
        num_classes=NUM_CLASSES,
        use_temporal_attention=use_temporal_attention,
        use_mc_convrnn=use_mc_convrnn,
        fusion_type=args.fusion_type
    ).to(DEVICE)

    temporal_desc = "mc_convrnn" if use_mc_convrnn else "temporal_attention"
    print(f"Using BEVFusionModel (LiDAR + Camera) with {temporal_desc}")

    # Loss will be created per-stage (initialized in training loop)
    criterion = None
    
    # Guardrail: Verify target generation configuration
    initial_params = get_stage_params(0, STAGE1_EPOCHS, args)
    print(f"✓ Target generation will use curriculum schedule (starts with heatmap_mode='{initial_params['heatmap_mode']}')")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)

    # 4. Training Loop
    global_step = 0
    max_steps = args.max_steps  # None means no limit
    stop_training = False
    epoch_logs = []  # L3: Track per-epoch metrics for trend validation

    # JSONL logger for detailed step-by-step logging (used by notebook visualization)
    jsonl_logger = JSONLLogger(jsonl_path, args.log_every_n_steps)
    jsonl_logger.log({
        'event': 'config',
        'epochs': NUM_EPOCHS,
        'batch_size': BATCH_SIZE,
        'lr': LR,
        'temporal_method': args.temporal_method,
        'fusion_type': args.fusion_type,
        'use_fusion': True,
        'seed': args.seed,
        'run_dir': str(run_dir) if run_dir else None,
    })

    for epoch in range(NUM_EPOCHS):
        if stop_training:
            break
            
        # Get stage-specific parameters
        stage_params = get_stage_params(epoch, STAGE1_EPOCHS, args)
        heatmap_mode = stage_params['heatmap_mode']
        
        # Check if we need to switch stage (recreate collate_fn and criterion)
        if current_mode != heatmap_mode:
            if current_mode is None:
                print(f"\n[CURRICULUM] Initializing Stage 1 at epoch {epoch+1}")
            else:
                print(f"\n[CURRICULUM] Switching stages at epoch {epoch+1}")
            
            # Log all active parameters
            print(f"[CURRICULUM]")
            print(f"  epoch={epoch+1}")
            print(f"  heatmap_mode={stage_params['heatmap_mode']}")
            print(f"  gaussian_overlap={stage_params['gaussian_overlap']}")
            print(f"  min_radius={stage_params['min_radius']}")
            print(f"  cls_loss:")
            print(f"    pos_weight={stage_params['pos_weight']}")
            
            current_mode = heatmap_mode
            
            # Recreate collate_fn and DataLoaders with stage-specific parameters
            collate_fn = create_fusion_collate_fn(
                bev_h=BEV_H,
                bev_w=BEV_W,
                num_classes=NUM_CLASSES,
                generate_targets=True,
                max_points=35000,
                heatmap_mode=stage_params['heatmap_mode'],
                gaussian_overlap=stage_params['gaussian_overlap'],
                min_radius=stage_params['min_radius']
            )
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                      num_workers=0, collate_fn=collate_fn)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                    num_workers=0, collate_fn=collate_fn)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                     num_workers=0, collate_fn=collate_fn)
            
            # Recreate DetectionLoss with stage-specific pos_weight
            criterion = DetectionLoss(pos_weight=stage_params['pos_weight'])
            # Safety check: fail fast if wrong loss is imported
            assert criterion.__class__.__module__.endswith("modules.head"), \
                f"Wrong DetectionLoss imported! Got {criterion.__class__.__module__}.{criterion.__class__.__name__}. " \
                f"Must use modules.head.DetectionLoss"
            print(f"  ✓ Recreated DetectionLoss with pos_weight={stage_params['pos_weight']}")
        
        # --- TRAIN ---
        model.train()
        epoch_loss = 0
        epoch_cls_loss = 0
        epoch_bbox_loss = 0
        start_time = time.time()

        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}] Training... (heatmap_mode={heatmap_mode}, pos_weight={stage_params['pos_weight']})")
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            # P0-1 FIX: Use full sequences for in-sample temporal unrolling
            imgs = batch['img'].to(DEVICE)  # (B, T, N_cams, 3, H, W)
            intrinsics = batch['intrinsics'].to(DEVICE)  # (B, T, N_cams, 3, 3)
            extrinsics = batch['extrinsics'].to(DEVICE)  # (B, T, N_cams, 4, 4)
            ego_pose = batch['ego_pose'].to(DEVICE)  # (B, T, 4, 4)
            lidar_points = batch['lidar_points'].to(DEVICE)  # (B, T, max_points, 4)
            lidar_mask = batch.get('lidar_mask')  # (B, T, max_points) or None
            if lidar_mask is not None:
                lidar_mask = lidar_mask.to(DEVICE)
            scene_tokens = batch.get('scene_tokens')  # List[List[str]] (B, T)

            # Ground truth targets (last frame only, as per nuScenes annotation)
            targets = {
                'cls_targets': batch['cls_targets'].to(DEVICE),
                'bbox_targets': batch['bbox_targets'].to(DEVICE),
                'reg_mask': batch['reg_mask'].to(DEVICE)
            }

            # P0-1 FIX: Use forward_sequence for proper in-sample temporal unrolling
            # This prevents cross-batch state leakage by resetting temporal state
            # at the start of each batch and unrolling through the full sequence
            # Also detects scene boundaries and resets temporal state per sample
            preds = model.forward_sequence(
                lidar_points_seq=lidar_points,
                camera_images_seq=imgs,
                camera_intrinsics_seq=intrinsics,
                camera_extrinsics_seq=extrinsics,
                ego_pose_seq=ego_pose,
                lidar_mask_seq=lidar_mask,
                scene_tokens=scene_tokens
            )

            # Loss
            loss_dict = criterion(preds, targets)
            loss = loss_dict['loss_total']

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_cls_loss += loss_dict.get('loss_cls', torch.tensor(0.0)).item()
            epoch_bbox_loss += loss_dict.get('loss_bbox', torch.tensor(0.0)).item()
            global_step += 1

            # JSONL step logging
            if global_step % args.log_every_n_steps == 0:
                jsonl_logger.log({
                    'event': 'step',
                    'split': 'train',
                    'epoch': epoch + 1,
                    'global_step': global_step,
                    'batch_idx': batch_idx,
                    'loss_total': loss.item(),
                    'loss_cls': loss_dict.get('loss_cls', torch.tensor(0.0)).item(),
                    'loss_bbox': loss_dict.get('loss_bbox', torch.tensor(0.0)).item(),
                    'lr': optimizer.param_groups[0]['lr'],
                })

            if batch_idx % 10 == 0:
                print(f"  Batch [{batch_idx}/{len(train_loader)}] Train Loss: {loss.item():.4f}")
            
            # Check max_steps limit
            if max_steps is not None and global_step >= max_steps:
                print(f"Reached max_steps={max_steps}, stopping training.")
                stop_training = True
                break

        if stop_training:
            break
            
        n_train = max(len(train_loader), 1)
        avg_train_loss = epoch_loss / n_train
        avg_train_cls = epoch_cls_loss / n_train
        avg_train_bbox = epoch_bbox_loss / n_train

        # --- VALIDATION ---
        model.eval()
        val_loss = 0
        val_cls_loss = 0
        val_bbox_loss = 0

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Validating...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                # P0-1 FIX: Use full sequences for in-sample temporal unrolling
                imgs = batch['img'].to(DEVICE)
                intrinsics = batch['intrinsics'].to(DEVICE)
                extrinsics = batch['extrinsics'].to(DEVICE)
                ego_pose = batch['ego_pose'].to(DEVICE)
                lidar_points = batch['lidar_points'].to(DEVICE)
                lidar_mask = batch.get('lidar_mask')
                if lidar_mask is not None:
                    lidar_mask = lidar_mask.to(DEVICE)
                scene_tokens = batch.get('scene_tokens')

                targets = {
                    'cls_targets': batch['cls_targets'].to(DEVICE),
                    'bbox_targets': batch['bbox_targets'].to(DEVICE),
                    'reg_mask': batch['reg_mask'].to(DEVICE)
                }

                # P0-1 FIX: Use forward_sequence for validation too
                preds = model.forward_sequence(
                    lidar_points_seq=lidar_points,
                    camera_images_seq=imgs,
                    camera_intrinsics_seq=intrinsics,
                    camera_extrinsics_seq=extrinsics,
                    ego_pose_seq=ego_pose,
                    lidar_mask_seq=lidar_mask,
                    scene_tokens=scene_tokens
                )

                loss_dict = criterion(preds, targets)
                val_loss += loss_dict['loss_total'].item()
                val_cls_loss += loss_dict.get('loss_cls', torch.tensor(0.0)).item()
                val_bbox_loss += loss_dict.get('loss_bbox', torch.tensor(0.0)).item()

        n_val = max(len(val_loader), 1)
        avg_val_loss = val_loss / n_val
        avg_val_cls = val_cls_loss / n_val
        avg_val_bbox = val_bbox_loss / n_val

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {time.time() - start_time:.2f}s")

        # JSONL epoch summary logging
        jsonl_logger.log({
            'event': 'epoch',
            'split': 'train',
            'epoch': epoch + 1,
            'global_step': global_step,
            'loss_total': avg_train_loss,
            'loss_cls': avg_train_cls,
            'loss_bbox': avg_train_bbox,
            'lr': optimizer.param_groups[0]['lr'],
        })
        jsonl_logger.log({
            'event': 'epoch',
            'split': 'val',
            'epoch': epoch + 1,
            'global_step': global_step,
            'loss_total': avg_val_loss,
            'loss_cls': avg_val_cls,
            'loss_bbox': avg_val_bbox,
        })

        # L3: Record epoch metrics for trend validation
        epoch_logs.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'train_cls_loss': avg_train_cls,
            'train_bbox_loss': avg_train_bbox,
            'val_cls_loss': avg_val_cls,
            'val_bbox_loss': avg_val_bbox,
        })

    # L3: Optionally dump epoch logs to JSON for automated trend tests
    if epoch_log_path:
        os.makedirs(os.path.dirname(epoch_log_path) or '.', exist_ok=True)
        with open(epoch_log_path, 'w') as f:
            json.dump(epoch_logs, f, indent=2)
        print(f"Epoch logs saved to {epoch_log_path}")

    jsonl_logger.close()
    print("Training Complete.")

    # --- TEST ---
    print("\nRunning Evaluation on Test Set...")
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # P0-1 FIX: Use full sequences for in-sample temporal unrolling
            imgs = batch['img'].to(DEVICE)
            intrinsics = batch['intrinsics'].to(DEVICE)
            extrinsics = batch['extrinsics'].to(DEVICE)
            ego_pose = batch['ego_pose'].to(DEVICE)
            lidar_points = batch['lidar_points'].to(DEVICE)
            lidar_mask = batch.get('lidar_mask')
            if lidar_mask is not None:
                lidar_mask = lidar_mask.to(DEVICE)
            scene_tokens = batch.get('scene_tokens')

            targets = {
                'cls_targets': batch['cls_targets'].to(DEVICE),
                'bbox_targets': batch['bbox_targets'].to(DEVICE),
                'reg_mask': batch['reg_mask'].to(DEVICE)
            }

            # P0-1 FIX: Use forward_sequence for test too
            preds = model.forward_sequence(
                lidar_points_seq=lidar_points,
                camera_images_seq=imgs,
                camera_intrinsics_seq=intrinsics,
                camera_extrinsics_seq=extrinsics,
                ego_pose_seq=ego_pose,
                lidar_mask_seq=lidar_mask,
                scene_tokens=scene_tokens
            )

            loss_dict = criterion(preds, targets)
            test_loss += loss_dict['loss_total'].item()

    print(f"Test Set Average Loss: {test_loss / len(test_loader):.4f}")

    # Save checkpoint
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint_name = f'fusion_{args.temporal_method}_latest.pth'
    checkpoint_path = f'checkpoints/{checkpoint_name}'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'temporal_method': args.temporal_method,
        'fusion_type': args.fusion_type,
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

    # Save run manifest if artifacts enabled
    if run_dir is not None:
        save_run_manifest(run_dir, args, checkpoint_path)
        # Copy checkpoint to run directory
        shutil.copy(checkpoint_path, run_dir / checkpoint_name)
        print(f"Run manifest saved to {run_dir / 'run_manifest.json'}")


def train():
    """Main training function that dispatches to the appropriate training loop."""
    args = parse_args()
    
    if args.use_fusion:
        train_fusion(args)
    else:
        train_camera_only(args)


def main():
    """Main entry point - can be called from notebook or command line."""
    args = parse_args()
    if args.use_fusion:
        train_fusion(args)
    else:
        train_camera_only(args)


if __name__ == '__main__':
    main()
