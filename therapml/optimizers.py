import torch

class SGD:
    def __init__(self, params, lr, weight_decay):
        self.params = list(params)
        self.lr = lr
        self.weight_decay = weight_decay

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

    @torch.no_grad()
    def step(self):
        for param in self.params:
            if param.grad is None:
                continue
            
            grad = param.grad
            
            if self.weight_decay != 0:
                grad = grad.add(param, alpha=self.weight_decay)
            
            param.add_(grad, alpha=-self.lr)