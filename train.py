import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
from modules.nuscenes_dataset import NuScenesDataset
from modules.bevformer import EnhancedBEVFormer
from modules.head import BEVHead, DetectionLoss

def train():
    # 1. Config
    BATCH_SIZE = 1
    NUM_EPOCHS = 10
    LR = 2e-5 # Lower LR for stability
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dataset Paths
    NUSCENES_ROOT = 'data'
    
    print(f"Starting training on {DEVICE}...")
    
    # 2. Data
    # Load Full NuScenes Dataset
    print(f"Loading NuScenes Dataset from {NUSCENES_ROOT}...")
    full_dataset = NuScenesDataset(dataroot=NUSCENES_ROOT, version='v1.0-mini')
    
    # Split: 70% Train, 15% Val, 15% Test
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    print(f"Splitting dataset: Train={train_size}, Val={val_size}, Test={test_size}")
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size], 
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 3. Model
    # Backbone + Transformer
    model_backbone = EnhancedBEVFormer(bev_h=200, bev_w=200, embed_dim=256).to(DEVICE)
    # Detection Head
    model_head = BEVHead(embed_dim=256, num_classes=10, reg_channels=8).to(DEVICE)
    
    # Loss
    criterion = DetectionLoss()
    
    # Optimizer
    optimizer = optim.AdamW(list(model_backbone.parameters()) + list(model_head.parameters()), lr=LR, weight_decay=1e-2)
    
    # 4. Training Loop
    
    for epoch in range(NUM_EPOCHS):
        # --- TRAIN ---
        model_backbone.train()
        model_head.train()
        epoch_loss = 0
        start_time = time.time()
        
        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}] Training...")
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            imgs = batch['img'].to(DEVICE) # (B, Seq, 6, 3, H, W)
            intrinsics = batch['intrinsics'].to(DEVICE)
            extrinsics = batch['extrinsics'].to(DEVICE)
            ego_pose = batch['ego_pose'].to(DEVICE)
            
            # Targets
            B = imgs.shape[0]
            H, W = 200, 200
            
            if 'gt_cls' in batch:
                targets = {
                    'gt_cls': batch['gt_cls'].to(DEVICE),
                    'gt_bbox': batch['gt_bbox'].to(DEVICE),
                    'mask': batch['mask'].to(DEVICE)
                }
            else:
                targets = {
                    'gt_cls': torch.zeros(B, 10, H, W).to(DEVICE),
                    'gt_bbox': torch.zeros(B, 8, H, W).to(DEVICE),
                    'mask': torch.zeros(B, 1, H, W).to(DEVICE)
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
            
            if batch_idx % 10 == 0:
                print(f"  Batch [{batch_idx}/{len(train_loader)}] Train Loss: {loss.item():.4f}")
        
        avg_train_loss = epoch_loss / len(train_loader)
        
        # --- VALIDATION ---
        model_backbone.eval()
        model_head.eval()
        val_loss = 0
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Validating...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                imgs = batch['img'].to(DEVICE)
                intrinsics = batch['intrinsics'].to(DEVICE)
                extrinsics = batch['extrinsics'].to(DEVICE)
                ego_pose = batch['ego_pose'].to(DEVICE)
                
                if 'gt_cls' in batch:
                    targets = {
                        'gt_cls': batch['gt_cls'].to(DEVICE),
                        'gt_bbox': batch['gt_bbox'].to(DEVICE),
                        'mask': batch['mask'].to(DEVICE)
                    }
                else:
                    B = imgs.shape[0]
                    targets = {
                        'gt_cls': torch.zeros(B, 10, H, W).to(DEVICE),
                        'gt_bbox': torch.zeros(B, 8, H, W).to(DEVICE),
                        'mask': torch.zeros(B, 1, H, W).to(DEVICE)
                    }
                
                bev_seq_features = model_backbone.forward_sequence(imgs, intrinsics, extrinsics, ego_pose)
                bev_features = bev_seq_features[:, -1]
                preds = model_head(bev_features)
                
                loss_dict = criterion(preds, targets)
                val_loss += loss_dict['loss_total'].item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {time.time() - start_time:.2f}s")
        
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
            
            if 'gt_cls' in batch:
                targets = {
                    'gt_cls': batch['gt_cls'].to(DEVICE),
                    'gt_bbox': batch['gt_bbox'].to(DEVICE),
                    'mask': batch['mask'].to(DEVICE)
                }
            else:
                B = imgs.shape[0]
                targets = {
                    'gt_cls': torch.zeros(B, 10, H, W).to(DEVICE),
                    'gt_bbox': torch.zeros(B, 8, H, W).to(DEVICE),
                    'mask': torch.zeros(B, 1, H, W).to(DEVICE)
                }
            
            bev_seq_features = model_backbone.forward_sequence(imgs, intrinsics, extrinsics, ego_pose)
            bev_features = bev_seq_features[:, -1]
            preds = model_head(bev_features)
            
            loss_dict = criterion(preds, targets)
            test_loss += loss_dict['loss_total'].item()
            
    print(f"Test Set Average Loss: {test_loss / len(test_loader):.4f}")
        
    print("Training Complete.")
    
    # Save checkpoint
    os.makedirs('checkpoints', exist_ok=True)
    torch.save({
        'backbone_state_dict': model_backbone.state_dict(),
        'head_state_dict': model_head.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'checkpoints/latest.pth')
    print("Checkpoint saved to checkpoints/latest.pth")

if __name__ == '__main__':
    train()
