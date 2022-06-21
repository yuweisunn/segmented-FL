import numpy as np 
import cv2 
from cnn.ConvNet import *
from module.builder import *
import random
from cnn.common.optimizer import RMSProp



def train(node_name, pcap, local_params_init, pattern):
    data, label = makedataset("%s_%s/dataset" %(pcap, pattern)) 
    train_accuracy = []
    
    epochs = 1
    batch_size = 100
    xsize = data.shape[0]
    iter_num = np.ceil(xsize / batch_size).astype(np.int)
    
    snet =ConvNet(input_dim=(1, 48, 48), 
                 conv_param={'filter1_num':10, 'filter1_size':3, 'filter2_num':10, 'filter2_size':1, 'pad':1, 'stride':1},
                 hidden_size= 200, output_size= 2 ,use_dropout = False, dropout_ration = 0.1, use_batchnorm = False)
  
    snet.params = local_params_init
    snet.layer()
    optimizer = RMSProp(lr=0.00001, rho=0.9)
    
    epoch_list = []

    for epoch in range(epochs):
        epoch_list.append("epoch %s" %(epoch+1))

        idx = np.arange(xsize)
        np.random.shuffle(idx)
    
        for it in range(iter_num):
            mask = idx[batch_size*it : batch_size*(it+1)]
  
            x_train = data[mask]
            t_train = label[mask]
        
            grads = snet.gradient(x_train, t_train)
            optimizer.update(snet.params, grads)
    
        train_accuracy.append(snet.accuracy(data, label ,batch_size))
        
        if (epoch+1)%1 == 0:
            print("Retraining the local model: %s/%s" %((epoch+1), epochs))
            
      
    model_name = "%s/local.npy" %node_name
    np.save(model_name, snet.params) 
    
