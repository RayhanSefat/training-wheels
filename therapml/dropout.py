from typing import List
from jaxtyping import Float
import torch
from torch import Tensor


def dropout(input: Float[Tensor, "..."], prob: float) -> Float[Tensor, "..."]:
    if prob < 0 or prob > 1:
        raise ValueError("Dropout probability must be between 0 and 1.")
    
    if prob == 0.0:
        return input
    elif prob == 1.0:
        return torch.zeros_like(input)
    
    mask = (torch.rand_like(input) >= prob).float()
    
    output = mask * input / (1.0 - prob)
    
    return output