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
    def __init__(self, gamma, eps=5e-6):
        super(RMSNorm, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input):
        shape = input.shape
        result = torch.empty_like(input)
        
        for i in range(shape[0]):
            for j in range(shape[1]):
                rms = torch.sqrt(torch.mean(input[i][j] ** 2) + self.eps)
                result[i][j] = (input[i][j] / rms) * self.gamma[i][j]

        return result