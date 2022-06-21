# -*- coding: utf-8 -*-
import numpy as np

def sigmoid(x):
    return 1 / (1+ np.exp(-x))

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)
