import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, gamma, beta, eps=1e-5):
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
                for k in range(shape[2]):
                    result[i][j][k] = ((input[i][j][k] - mean) / torch.sqrt(variance + self.eps)) * self.gamma[i][j][k] + self.beta[i][j][k]

        return result

class RMSNorm(nn.Module):
    def __init__(self, gamma, eps=1e-5):
        super(RMSNorm, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input):
        shape = input.shape
        result = torch.empty_like(input)
        
        for i in range(shape[0]):
            for j in range(shape[1]):
                rms = torch.sqrt(torch.mean(input[i][j] ** 2))
                for k in range(shape[2]):
                    result[i][j][k] = (input[i][j][k] / (rms + self.eps)) * self.gamma[i][j][k]

        return result