import torch
import torch.nn as nn
from .backbone import ResNetBackbone
from .neck import FPN
from .attention import SpatialCrossAttention, TemporalSelfAttention
from .convrnn import ConvGRU
from .memory_bank import MemoryBank
from .utils import warp_bev

class EnhancedBEVFormer(nn.Module):
    def __init__(self, bev_h=200, bev_w=200, embed_dim=256):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.embed_dim = embed_dim
        
        # Components
        self.backbone = ResNetBackbone()
        self.neck = FPN(in_channels=self.backbone.out_channels, out_channels=embed_dim)
        
        self.spatial_cross_attention = SpatialCrossAttention(embed_dim=embed_dim)
        self.temporal_self_attention = TemporalSelfAttention(embed_dim=embed_dim)
        
        self.conv_gru = ConvGRU(input_dim=embed_dim, hidden_dim=embed_dim)
        self.memory_bank = MemoryBank()
        
        # BEV Queries (Learnable)
        self.bev_queries = nn.Parameter(torch.randn(bev_h * bev_w, embed_dim))
        self.bev_pos = nn.Parameter(torch.randn(bev_h, bev_w, 3)) # 3D position of grid points

    def forward(self, imgs, intrinsics, extrinsics, ego_pose, prev_bev=None, ego_motion=None):
        """
        imgs: (B, N_cam, 3, H, W)
        intrinsics: (B, N_cam, 3, 3)
        extrinsics: (B, N_cam, 4, 4)
        ego_pose: (B, 4, 4) - Current ego pose
        prev_bev: (B, C, H_bev, W_bev) - Previous BEV state
        ego_motion: (B, 4, 4) - Transform from prev to current
        """
        B, N_cam, C, H, W = imgs.shape
        
        # 1. Backbone + Neck
        imgs_flat = imgs.view(-1, C, H, W)
        feats = self.backbone(imgs_flat)
        # Use the last feature map from FPN for simplicity
        fpn_feats = self.neck(feats)[-1] # (B*N_cam, embed_dim, H_feat, W_feat)
        
        # 2. Temporal Alignment (Warping)
        if prev_bev is not None and ego_motion is not None:
            prev_bev_warped = warp_bev(prev_bev, ego_motion)
        else:
            prev_bev_warped = torch.zeros(B, self.embed_dim, self.bev_h, self.bev_w, device=imgs.device)
            
        # 3. Spatial Cross Attention (Image -> BEV)
        # Project BEV queries to images and sample
        # We need to compute reference points
        
        # Mocking reference points calculation for now
        # In real impl, we project self.bev_pos using extrinsics/intrinsics
        # (B, N_cam, H_bev*W_bev, 2)
        
        # For the simplified attention module we wrote:
        # It expects (B, N_cam, H_bev*W_bev, 2)
        
        # Let's generate a dummy grid for now or implement projection
        # To keep it running, we'll pass a placeholder that matches shape
        ref_points = torch.zeros(B, N_cam, self.bev_h * self.bev_w, 2, device=imgs.device)
        bev_mask = torch.ones(B, N_cam, self.bev_h * self.bev_w, device=imgs.device)
        
        # Query: (B, L, C)
        query = self.bev_queries.unsqueeze(0).repeat(B, 1, 1)
        
        bev_embed = self.spatial_cross_attention(
            query, 
            value=fpn_feats, 
            key=None, 
            query_pos=None, 
            reference_points_cam=ref_points, 
            bev_mask=bev_mask,
            spatial_shapes=None,
            level_start_index=None
        )
        
        # bev_embed: (B, L, C) -> (B, C, H, W)
        bev_embed = bev_embed.permute(0, 2, 1).view(B, self.embed_dim, self.bev_h, self.bev_w)
        
        # 4. Temporal Fusion (ConvGRU)
        # Input: Current BEV features from images
        # Hidden: Warped previous BEV
        
        curr_bev = self.conv_gru(bev_embed, prev_bev_warped)
        
        return curr_bev

    def forward_sequence(self, seq_imgs, seq_intrinsics, seq_extrinsics, seq_ego_pose):
        """
        Process a sequence of frames.
        seq_imgs: (B, Seq, N_cam, 3, H, W)
        """
        B, Seq, N_cam, C, H, W = seq_imgs.shape
        
        bev_outputs = []
        prev_bev = None
        
        for t in range(Seq):
            imgs = seq_imgs[:, t]
            intrinsics = seq_intrinsics[:, t]
            extrinsics = seq_extrinsics[:, t]
            ego_pose = seq_ego_pose[:, t]
            
            ego_motion = None
            if t > 0:
                # Compute ego motion from t-1 to t
                # Pose is global to ego. Motion = inv(Pose_t) * Pose_{t-1} ? 
                # Or Pose_{t-1} -> Pose_t
                # Let's assume ego_pose is Global -> Ego? Or Ego -> Global?
                # Usually Ego -> Global.
                # T_{t-1->t} = inv(T_{t}) * T_{t-1}
                pose_t = ego_pose
                pose_prev = seq_ego_pose[:, t-1]
                ego_motion = torch.matmul(torch.inverse(pose_t), pose_prev)
            
            curr_bev = self.forward(imgs, intrinsics, extrinsics, ego_pose, prev_bev, ego_motion)
            bev_outputs.append(curr_bev)
            prev_bev = curr_bev
            
        return torch.stack(bev_outputs, dim=1) # (B, Seq, C, H, W)
