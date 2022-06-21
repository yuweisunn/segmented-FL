# -*- coding: utf-8 -*-
import numpy as np


class SGD:
    """
    Stochastic Gradient Descent
    """
    def __init__(self, lr=0.01):
        """
        lr : 学習係数 learning rate
        """
        self.lr = lr
        
    def update(self, params, grads):
        """
        重みの更新
        """
        for key in params.keys():
            params[key] -= self.lr * grads[key]
            
class NesterovAG:
    """
    Nesterov Accelerated Gradient
    """
    def __init__(self, lr=0.01, momentum=0.9):
        """
        lr : 学習係数 learning rate
        momentm : モーメンタム係数
        """
        self.lr = lr
        self.momentum = momentum
        self.v = None
        
    def update(self, params, grads):
        """
        重みの更新
        """
        if self.v is None:
            """
            初回のみ
            """
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
                
        # 重みを更新
        for key in params.keys():
            v_pre = self.v[key].copy()
            self.v[key] = v_pre * self.momentum - self.lr * grads[key]
            params[key] += -self.momentum* v_pre + (self.momentum+1) * self.v[key]


class RMSProp:
    """
    RMSProp
    """
    def __init__(self, lr=0.01, rho=0.9):
        """
        lr : 学習係数 learning rate
        rho : 減衰率
        """
        self.lr = lr
        self.h = None
        self.rho = rho
        self.epsilon = 1e-6
        
    def update(self, params, grads):
        """
        重みの更新
        """
        if self.h is None:
            """
            初回のみ
            """
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
                
        for key in params.keys():
            self.h[key] = self.rho * self.h[key] + (1 - self.rho) * grads[key] * grads[key]          
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key] + self.epsilon) ) # 原論文に合わせてepsilonをルートの中に入れる


