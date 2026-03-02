import math
import torch
from torch.optim.lr_scheduler import LambdaLR

def clip_grad_value(value, minimum_threshold=-1e5, maximum_threshold=1e5):
    if value < minimum_threshold:
        return minimum_threshold
    
    if value > maximum_threshold:
        return maximum_threshold
    
    return value

def clip_grad_norm(parameters, max_norm, norm_type=2.0):
    if isinstance(parameters, torch.tensor):
        parameters = [parameters]

    grads = [p.grad for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)

    if len(grads) == 0:
        return torch.tensor(0.0)
    
    device = grads[0].device
    total_norm = torch.norm(
        torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]), 
        norm_type
    )

    clip_coeff = max_norm / (total_norm + 1e-6)

    if clip_coeff < 1:
        for g in grads:
            g.detach().mul_(clip_coeff)
            
    return total_norm

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 1.0 - progress)

    return LambdaLR(optimizer, lr_lambda)

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.0):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        progress = min(1.0, progress)
        
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        return max(min_lr_ratio, (1.0 - min_lr_ratio) * cosine_decay + min_lr_ratio)

    return LambdaLR(optimizer, lr_lambda)