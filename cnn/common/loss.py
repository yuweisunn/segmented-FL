# coding: utf-8
import numpy as np

def mean_squared_error(y, t):
    """
    y : 出力値
    t : 正解値
    """    
    if y.ndim==1:
        t = t.reshape(1, -1)
        y = y.reshape(1, -1)
        
    batch_size = y.shape[0]    
    return 0.5 * np.sum((y - t)**2) / batch_size

def cross_entropy_error(y, t):
    """
    y : 出力値(通常は、0-1の確率)  
    t : 正解値(通常は、0or1)  
    """
    if y.ndim==1:
        t = t.reshape(1, -1)
        y = y.reshape(1, -1)
        
    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum( t * np.log(y + delta)) / batch_size
