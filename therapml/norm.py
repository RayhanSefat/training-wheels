import torch

def layer_norm(input, gamma, beta):
    shape = input.shape
    result = torch.empty_like(input)
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            mean = torch.mean(input[i][j])
            variance = torch.var(input[i][j], unbiased=False)
            for k in range(shape[2]):
                result[i][j][k] = ((input[i][j][k] - mean) / torch.sqrt(variance + 1e-5)) * gamma[i][j][k] + beta[i][j][k]

    return result

def rms_norm(input, gamma):
    shape = input.shape
    result = torch.empty_like(input)
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            rms = torch.sqrt(torch.mean(input[i][j] ** 2))
            for k in range(shape[2]):
                result[i][j][k] = (input[i][j][k] / (rms + 1e-5)) * gamma[i][j][k]

    return result