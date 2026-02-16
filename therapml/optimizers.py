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

class Adam:
    def __init__(self, params, lr, weight_decay, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas
        self.eps = eps
        self.state = {}

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
            
            state = self.state.setdefault(param, {
                'step': 0,
                'exp_avg': torch.zeros_like(param),
                'exp_avg_sq': torch.zeros_like(param)
            })
            
            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            beta1, beta2 = self.betas
            
            state['step'] += 1
            
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            
            bias_correction1 = 1 - beta1 ** state['step']
            bias_correction2 = 1 - beta2 ** state['step']
            
            step_size = self.lr * (bias_correction2 ** 0.5) / bias_correction1
            
            denom = exp_avg_sq.sqrt().add_(self.eps)
            
            param.addcdiv_(exp_avg, denom, value=-step_size)