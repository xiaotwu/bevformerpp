import torch
import torch.nn.functional as F

def warp_bev(prev_bev, ego_motion):
    """
    Warp previous BEV to current frame using ego motion.
    prev_bev: (B, C, H, W)
    ego_motion: (B, 4, 4) - Transformation from prev to current
    """
    B, C, H, W = prev_bev.shape
    device = prev_bev.device
    
    # Create grid
    # ... (Implementation of grid generation and sampling)
    # For now, return prev_bev (Identity warp) for simplicity or if motion is small
    # Real implementation needs affine_grid
    
    # theta = ego_motion[:, :2, :] # (B, 2, 3) ? No, 4x4
    # We need to invert ego_motion to get sampling grid? 
    # Grid sample uses T_dest_to_src
    
    # Let's assume ego_motion is T_{t-1 -> t}
    # We want to sample at x_t. x_{t-1} = T_{t->t-1} * x_t
    # T_{t->t-1} = inv(ego_motion)
    
    grid = F.affine_grid(torch.inverse(ego_motion)[:, :2, :3], [B, C, H, W], align_corners=False)
    warped_bev = F.grid_sample(prev_bev, grid, align_corners=False)
    
    return warped_bev
