import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, gamma, beta, eps=5e-6):
        super(LayerNorm, self).__init__()
        self.gamma = gamma
        self.beta = beta
        self.eps = eps

    def forward(self, input):
        shape = input.shape
        result = torch.empty_like(input)
        
        for i in range(shape[0]):
            for j in range(shape[1]):
                mean = torch.mean(input[i][j])
                variance = torch.var(input[i][j], unbiased=False)
                result[i][j] = ((input[i][j] - mean) / torch.sqrt(variance + self.eps)) * self.gamma[i][j] + self.beta[i][j]

        return result

class RMSNorm(nn.Module):
    def __init__(self, gamma, eps=1e-5):
        super(RMSNorm, self).__init__()
        self.gamma = nn.Parameter(torch.zeros_like(gamma))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.gamma