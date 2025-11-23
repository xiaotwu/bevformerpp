import torch
import torch.nn as nn

class ConvGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        
        self.conv_z = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.conv_r = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.conv_h = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)
        
    def forward(self, x, h):
        """
        x: (B, C, H, W) - Current BEV features
        h: (B, C, H, W) - Previous hidden state (warped)
        """
        if h is None:
            h = torch.zeros_like(x)
            
        combined = torch.cat([x, h], dim=1)
        
        z = torch.sigmoid(self.conv_z(combined))
        r = torch.sigmoid(self.conv_r(combined))
        
        combined_r = torch.cat([x, r * h], dim=1)
        h_tilde = torch.tanh(self.conv_h(combined_r))
        
        h_next = (1 - z) * h + z * h_tilde
        return h_next
