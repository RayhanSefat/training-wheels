import numpy as np
import torch
from scipy.special import erf

def ReLU(in_features):
    data = np.asanyarray(in_features)

    return np.maximum(data, 0)

def GELU(in_features):
    data = np.asanyarray(in_features)
    
    return 0.5 * data * (1 + erf(data / np.sqrt(2)))

def softmax(in_features, dim):
    data = np.asanyarray(in_features)
    
    exp_data = np.exp(data - np.max(data, axis=dim, keepdims=True))
    return exp_data / np.sum(exp_data, axis=dim, keepdims=True)

def linear(d_in, d_out, weights, in_features):
    weights_transposed = weights.T
    return np.dot(in_features, weights_transposed)

def swiglu(d_model, d_ff, w1_weight, w2_weight, w3_weight, in_features):
    return in_features