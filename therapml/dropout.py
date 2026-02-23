from typing import List
import torch
import torch.nn as nn
from torch import Tensor


class Dropout(nn.Module):
    def __init__(self, prob: float):
        if prob < 0 or prob > 1:
            raise ValueError("Dropout probability must be between 0 and 1.")
        super().__init__()
        self.prob = prob

    def forward(self, input):
        if self.prob == 0.0:
            return input
        elif self.prob == 1.0:
            return torch.zeros_like(input)
        
        mask = (torch.rand_like(input) >= self.prob).float()
        output = mask * input / (1.0 - self.prob)
        return output