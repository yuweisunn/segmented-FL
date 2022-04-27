# -*- coding: utf-8 -*-
from collections import OrderedDict
from cnn.common.layers import Convolution, MaxPooling, ReLU, Affine, SoftmaxWithLoss, Dropout, BatchNormalization
import numpy as np
from cnn.common.activations import*

def he(n1): 
    return np.sqrt(2/n1)


class ConvNet:
    """
    Parameters
    ----------
    activation : 'relu' or 'sigmoid'
    weight_init_std : 重みの標準偏差を指定（e.g. 0.01）
        'relu'または'he'を指定した場合は「Heの初期値」を設定
        'sigmoid'または'xavier'を指定した場合は「Xavierの初期値」を設定
    """
    def __init__(self, input_dim=(1, 48, 48), 
        conv_param={'filter1_num':1,'filter1_size':3,'filter2_num':1, 'filter2_size':3, 'pad':0, 'stride':3},
        hidden_size= 90, output_size= 5 ,use_dropout = False, dropout_ration = 0.5, use_batchnorm = False):
        filter1_num = conv_param['filter1_num']
        filter2_num = conv_param['filter2_num']
        filter1_size = conv_param['filter1_size']
        filter2_size = conv_param['filter2_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        self.filter_pad = conv_param['pad']
        self.filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv1_output_size = int((input_size - filter1_size + 2*filter_pad) / filter_stride + 1)
        #pool1_output_size = int(filter1_num * (conv1_output_size/2) * (conv1_output_size/2))
        conv2_output_size = int((conv1_output_size/2 - filter2_size + 2*filter_pad) / filter_stride + 1)
        pool2_output_size = int(filter2_num * (conv2_output_size/2) * (conv2_output_size/2))#int(filter2_num * int(conv2_output_size/2) * int(conv2_output_size/2))
        # 重みの初期化
        self.params = {}
        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm
        self.dropout_ration = dropout_ration
        self.params['W1'] = np.random.randn(filter1_num,input_dim[0], filter1_size, filter1_size)* he(24*24)
        self.params['b1'] = np.zeros(filter1_num)
        self.params['W2'] = np.random.randn(filter2_num, filter1_num, filter2_size, filter2_size)* he(conv1_output_size * conv1_output_size) # W1は畳み込みフィルターの重みになる
        self.params['b2'] = np.zeros(filter2_num)
        self.params['W3'] = np.random.randn(pool2_output_size, hidden_size)*he(pool2_output_size)
        self.params['b3'] = np.zeros(hidden_size)
        self.params['W4'] = np.random.randn(hidden_size, output_size)*he(hidden_size)
        self.params['b4'] = np.zeros(output_size)
        if self.use_batchnorm:
            self.params['gamma1'] = np.ones(filter1_num * conv1_output_size * conv1_output_size)
            self.params['beta1'] = np.ones(filter1_num * conv1_output_size * conv1_output_size)
            self.params['gamma2'] = np.ones(filter2_num * conv2_output_size * conv2_output_size)
            self.params['beta2'] = np.ones(filter2_num * conv2_output_size * conv2_output_size)
            self.params['gamma3'] = np.ones(hidden_size)
            self.params['beta3'] = np.ones(hidden_size)
            self.params['gamma4'] = np.ones(output_size)
            self.params['beta4'] = np.ones(output_size)
        self.flag = False
        


    def layer(self):
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], self.filter_stride, self.filter_pad) # W1が畳み込みフィルターの重み, b1が畳み込みフィルターのバイアスになる
        self.layers['ReLU1'] = ReLU()
        if self.use_batchnorm:
            self.layers['BatchNormalization1'] = BatchNormalization(self.params['gamma1'], self.params['beta1'])
        
        self.layers['MaxPool1'] = MaxPooling(pool_h=2, pool_w=2, stride=2)
        if self.use_dropout:
            self.layers['Dropout1'] = Dropout(self.dropout_ration)
        self.layers['Conv2'] = Convolution(self.params['W2'], self.params['b2'], self.filter_stride, self.filter_pad) # W1が畳み込みフィルターの重み, b1が畳み込みフィルターのバイアスになる
        if self.use_batchnorm:    
            self.layers['BatchNormalization2'] = BatchNormalization(self.params['gamma2'], self.params['beta2'])
        self.layers['ReLU2'] = ReLU()
        self.layers['MaxPool2'] = MaxPooling(pool_h=2, pool_w=2, stride=2)
        #if self.use_dropout:
        #    self.layers['Dropout2'] = Dropout(self.dropout_ration)
        self.layers['Affine1'] = Affine(self.params['W3'], self.params['b3'])
        self.layers['ReLU3'] = ReLU()
        if self.use_batchnorm:
            self.layers['BatchNormalization3'] = BatchNormalization(self.params['gamma3'], self.params['beta3'])
        
        #if self.use_dropout:
        #    self.layers['Dropout3'] = Dropout(self.dropout_ration)
        self.layers['Affine2'] = Affine(self.params['W4'], self.params['b4'])
        if self.use_batchnorm:
            self.layers['BatchNormalization4'] = BatchNormalization(self.params['gamma4'], self.params['beta4'])
        self.last_layer = SoftmaxWithLoss()
    

    def predict(self, x, train_flg=False):
        for key,layer in self.layers.items():
            if self.flag:
                print(key, x.shape)
                
            if "Dropout" in key or "BatchNorm" in key:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
            
            """
            if len(x.shape) == 4 :
                max_x = np.max(x[0][0], axis = 1)
                cv2.imwrite("../%s.jpg" %key,255 - 255*(x[0][0]/max_x))
            else:
                max_x = np.max(x[0])
                cv2.imwrite("../%s.jpg" %key,255 - 255*(x[0]/max_x))
            """
            
        return x

    def loss(self, x, t, train_flg=False):

        y = self.predict(x, train_flg)
        last_loss = self.last_layer.forward(y, t)
        
        if self.flag:
            print("Softmax", y.shape)
            print("=============================")
        self.flag = False  
        
        return last_loss

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        acc = 0.0
        
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt) 
        
        return acc / x.shape[0]
    

    def gradient(self, x, t):
        """勾配を求める（誤差逆伝播法）
        Parameters
        ----------
        x : 入力データ
        t : 教師データ
        Returns
        -------
        """
        # forward
        self.loss(x, t, train_flg=True)
        
        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
            
        # 設定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Conv2'].dW, self.layers['Conv2'].db
        grads['W3'], grads['b3'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W4'], grads['b4'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        if self.use_batchnorm:
            grads['gamma1'], grads['beta1'] = self.layers['BatchNormalization1'].gamma, self.layers['BatchNormalization1'].beta
            grads['gamma2'], grads['beta2'] = self.layers['BatchNormalization2'].gamma, self.layers['BatchNormalization2'].beta
            grads['gamma3'], grads['beta3'] = self.layers['BatchNormalization3'].gamma, self.layers['BatchNormalization3'].beta
            grads['gamma4'], grads['beta4'] = self.layers['BatchNormalization4'].gamma, self.layers['BatchNormalization4'].beta
        
        return grads
    
    
    
 
    
    
    
    
    
    
