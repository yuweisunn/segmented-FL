import numpy as np
import math 
from cnn.ConvNet import *
import os
import pickle

def getListValues(dict): 
    list = [] 
    for value in dict.values(): 
        list.append(value) 
          
    return list

def getListKeys(dict): 
    list = [] 
    for key in dict.keys(): 
        list.append(key) 
          
    return list

def sigmoid(a):
    e = math.e
    s = 1 / (1 + e**-a)
    return s
    
def dropout(acc_list, server_list, fineness, acc_dict):
    flag = []
    acc_mean = np.mean(acc_list)
    server = "00%s" %(len(server_list))
    
    for i in range(len(acc_list)):
        loss = sigmoid(acc_list[i] - acc_mean)
        if loss < fineness:
            flag.append(i)
    if len(flag) > 0:        
        segment(acc_dict, server, flag)
            
    return flag, server
            
            
def segment(acc_dict, server, flag):
    reborn = []
    global_params = np.load("init/init.npy", allow_pickle = True).item()
    for p in range(len(flag)):
        node_name = getListKeys(acc_dict)[flag[p]]
        reborn.append(np.load("%s/local.npy" %node_name, allow_pickle = True).item())
    
    for k in global_params.keys(): 
        temp = [] 
        for i in range(len(reborn)):
            temp.append(reborn[i][k])
        global_params[k] = (np.mean(temp, axis=0))

    reNet =ConvNet(input_dim=(1, 48, 48), 
                 conv_param={'filter1_num':10, 'filter1_size':3, 'filter2_num':10, 'filter2_size':1, 'pad':1, 'stride':1},
                 hidden_size= 200, output_size= 2 ,use_dropout = True, dropout_ration = 0.1, use_batchnorm = False)
    
    reNet.params = global_params
    reNet.layer()
    
    try:
        os.mkdir('%s' %server)
    except:
        pass
    
    model_name = '%s/global' %server
    np.save(model_name, reNet.params) 
    

def server_update(server_list, node_name, server):
    node_stack = getListValues(server_list)
    for r in range(len(node_stack)):
        for b in range(len(node_stack[r])):
            if node_stack[r][b] == node_name:
                node_stack[r].pop(b)
                break
                
    try:        
        server_list[server].append(node_name)
    except:
        server_list[server] = [node_name]
    
    counter = 0
    for r in range(len(getListKeys(server_list))):
        value_list = getListValues(server_list)[counter]
        key = getListKeys(server_list)[counter]
        counter = counter + 1
        
        if len(value_list) == 0:
           server_list.pop(key) 
           counter = counter-1

    with open("server_dict.txt", "wb") as myFile:
         pickle.dump(server_list, myFile)
       
    print(f"The current client nodes segmentation: {server_list}")
