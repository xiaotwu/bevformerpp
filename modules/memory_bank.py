"""
MemoryBank for storing past BEV features and ego-motion transforms.
Implements a FIFO queue for temporal aggregation.
"""

import torch
import torch.nn as nn
from collections import deque
from typing import List, Tuple, Optional


class MemoryBank(nn.Module):
    """
    Stores past BEV features for temporal aggregation.
    Implements a FIFO queue with configurable maximum length.

    Attributes:
        max_length: Maximum number of past frames to store
        features: Deque of past BEV feature tensors
        transforms: Deque of ego-motion transforms between consecutive frames
        enable_bptt: Whether to allow gradients to flow through stored features

    BPTT (Backpropagation Through Time) Design Choices:
    ---------------------------------------------------
    By default, features are detached when pushed to the memory bank (enable_bptt=False).
    This is the recommended setting for most use cases because:

    1. Memory efficiency: Prevents storing computation graphs for old frames
    2. Training stability: Avoids exploding/vanishing gradients over long sequences
    3. Speed: Reduces backward pass computation

    Set enable_bptt=True only if you need gradients to flow through temporal
    connections (e.g., for end-to-end training of prediction tasks). In this case:
    - Use gradient checkpointing to manage memory
    - Consider gradient clipping for stability
    - Limit max_length to prevent memory issues
    """

    def __init__(self, max_length: int = 5, enable_bptt: bool = False):
        """
        Initialize MemoryBank.

        Args:
            max_length: Maximum number of past frames to store (default: 5)
            enable_bptt: Whether to preserve gradients through stored features.
                        Default False for memory efficiency and training stability.
                        Set True for full backpropagation through time.
        """
        super().__init__()
        self.max_length = max_length
        self.enable_bptt = enable_bptt
        self.features = deque(maxlen=max_length)
        self.transforms = deque(maxlen=max_length - 1)

    def push(self, feature: torch.Tensor, transform: Optional[torch.Tensor] = None):
        """
        Add a new BEV feature and optional ego-motion transform to the memory bank.

        Args:
            feature: BEV feature tensor of shape (B, C, H, W)
            transform: Optional ego-motion transform of shape (B, 4, 4)
                      Transform from previous frame to current frame
        """
        if self.enable_bptt:
            # Keep gradients for full BPTT
            # WARNING: This can cause high memory usage for long sequences
            self.features.append(feature)
            if transform is not None:
                self.transforms.append(transform)
        else:
            # Detach features to prevent gradient flow through long sequences
            # This is the default for memory efficiency during training
            self.features.append(feature.detach())
            if transform is not None:
                self.transforms.append(transform.detach())

    def get_sequence(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Get the stored sequence of features and transforms.

        Returns:
            Tuple of (features_list, transforms_list)
            - features_list: List of BEV features, oldest to newest
            - transforms_list: List of ego-motion transforms
        """
        return list(self.features), list(self.transforms)

    def clear(self):
        """Clear all stored features and transforms."""
        self.features.clear()
        self.transforms.clear()

    def __len__(self) -> int:
        """Return the number of stored features."""
        return len(self.features)

    def is_empty(self) -> bool:
        """Check if the memory bank is empty."""
        return len(self.features) == 0

    def is_full(self) -> bool:
        """Check if the memory bank has reached maximum capacity."""
        return len(self.features) == self.max_length
