import numpy as np
import torch

def cross_entropy_loss(logits, ground_truth, eps=1e-12):
    if torch.is_tensor(logits):
        logits = logits.detach().cpu().numpy()
    if torch.is_tensor(ground_truth):
        ground_truth = ground_truth.detach().cpu().numpy()

    loss = 0.0

    for i in range(logits.shape[0]):
        shifted_logits = logits[i] - np.max(logits[i])
        
        exps = np.exp(shifted_logits)
        softmax_probs = exps / np.sum(exps)
        
        for j in range(logits.shape[1]):
            if ground_truth[i][j] == 1:
                loss -= np.log(softmax_probs[j] + eps)
    
    return loss / logits.shape[0]