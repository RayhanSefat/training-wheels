import torch
import torch.nn as nn
from .nn_blocks import softmax

class CrossEntropyLoss(nn.Module):
    def __init__(self, ground_truth,eps=1e-12):
        super(CrossEntropyLoss, self).__init__()
        self.ground_truth = ground_truth
        self.eps = eps

    def forward(self, logits):
        loss = 0.0

        for i in range(logits.shape[0]):
            shifted_logits = logits[i] - logits[i].max()
            
            softmax_probs = softmax(shifted_logits, dim=0)
            
            for j in range(logits.shape[1]):
                if self.ground_truth[i][j] == 1:
                    loss -= torch.log(softmax_probs[j] + self.eps)
        
        return loss / logits.shape[0]