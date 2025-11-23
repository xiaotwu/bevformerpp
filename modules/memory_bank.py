import torch
import torch.nn as nn

class MemoryBank(nn.Module):
    def __init__(self, max_history=3):
        super().__init__()
        self.max_history = max_history
        self.history = [] # List of (bev_features, ego_pose, timestamp)
        
    def update(self, bev_features, ego_pose, timestamp):
        """
        bev_features: (B, C, H, W)
        ego_pose: (B, 4, 4)
        timestamp: (B,)
        """
        # Detach to stop gradients flowing back too far if needed, 
        # but for BPTT we might want to keep them attached within a sequence.
        # For this implementation, we assume BPTT is handled by the training loop 
        # passing the state explicitly. This memory bank is for long-term storage logic.
        
        self.history.append({
            'bev': bev_features,
            'pose': ego_pose,
            'ts': timestamp
        })
        
        if len(self.history) > self.max_history:
            self.history.pop(0)
            
    def get_history(self):
        return self.history
    
    def clear(self):
        self.history = []
