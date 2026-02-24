import numpy as np
import torch
import torch.nn as nn
from scipy.special import erf

def ReLU(in_features):
    data = torch.as_tensor(in_features)

    return torch.maximum(data, torch.tensor(0.0))

def GELU(in_features):
    data = torch.as_tensor(in_features)
    
    return 0.5 * data * (1 + erf(data / torch.sqrt(torch.tensor(2.0))))

def softmax(in_features, dim):
    data = torch.as_tensor(in_features)
    
    exp_data = torch.exp(data - torch.max(data, dim=dim, keepdim=True)[0])
    return exp_data / torch.sum(exp_data, dim=dim, keepdim=True)

class Linear(nn.Module):
    def __init__(self, d_in, d_out, weight):
        super(Linear, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.weight = weight

    def forward(self, in_features):
        weights_transposed = self.weight.T
        return in_features @ weights_transposed

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, w1_weight, w2_weight, w3_weight):
        super(SwiGLU, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1_weight = w1_weight
        self.w2_weight = w2_weight
        self.w3_weight = w3_weight

    def SiLU(self, in_features):
        data = torch.as_tensor(in_features)
        return data * torch.sigmoid(data)

    def forward(self, in_features):
        gate = torch.matmul(in_features, self.w1_weight.T)
        gate_activated = self.SiLU(gate)

        value = torch.matmul(in_features, self.w3_weight.T)

        output = gate_activated * value
        return torch.matmul(output, self.w2_weight.T)