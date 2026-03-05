import torch
import torch.nn as nn
from .nn_blocks import softmax

class CrossEntropyLoss(nn.Module):
    def __init__(self, eps=1e-12):
        super(CrossEntropyLoss, self).__init__()
        self.eps = eps

    def forward(self, logits, ground_truth):
        loss = 0.0

        shifted_logits = logits - logits.max(dim=1, keepdim=True)[0]
        softmax_probs = softmax(shifted_logits, dim=1)
        loss = -torch.sum(ground_truth * torch.log(softmax_probs + self.eps))
        
        return loss / logits.shape[0]