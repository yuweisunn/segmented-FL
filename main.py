import numpy as np 
import cv2 
import pandas as pd
import random
from cnn.ConvNet import *
from cnn.common.optimizer import RMSProp
from module.train import *
from module.segment import *
from module.builder import *
import pickle
import matplotlib.pyplot as plt
import csv
import os
from PIL import Image
import glob



# Retrieve the global model
def download(server):
    global_params = np.load("%s/global.npy" %server, allow_pickle = True).item()

    return global_params 
    

# Retrieve the local model
def upload(node_name):  
    local_params = np.load("%s/local.npy" %node_name, allow_pickle = True).item() 
    
    return local_params


# Evaluation
def precision(local_params, train_data, label):
    label = np.array(label)
    
    snet = ConvNet(input_dim=(1, 48, 48), 
                 conv_param={'filter1_num':10, 'filter1_size':3, 'filter2_num':10, 'filter2_size':1, 'pad':1, 'stride':1},
                 hidden_size= 200, output_size= 2 ,use_dropout = False, dropout_ration = 0.1, use_batchnorm = False)
  
    snet.params = local_params
    snet.layer()
    acc = snet.accuracy(train_data, label, 125)
    
    return acc
    

# Update local models
def Execute(node_name, pcap, server, pattern):
    local_params_init = download(server)
    train(node_name, pcap, local_params_init, pattern)
    local_params = upload(node_name)
    
    return local_params


# Segmented Model Aggregation
def Aggregate(local_parameters_list, global_parameters, server_list, server):
    global_params = local_parameters_list[0]
    global_list = []
    
    for i in range(len(server_list)):
        model_name = '%s/global.npy' %server_list[i]
        global_list.append(np.load(model_name, allow_pickle = True).item())
    
    # To update the global model, we consider model parameters from three different sources with different aggregation weights      
    w_local = 0.1
    w_server = 0.01
    w_global = 0.9 - w_server*len(global_list)
   
    for k in local_parameters_list[0].keys():
         temp = []   
         server_params = []
         for i in range(len(global_list)):
             server_params.append(global_list[i][k])
                 
         if len(local_parameters_list) == 1:
             global_params[k] = local_parameters_list[0][k]*w_local+global_parameters[k]*w_global+w_server*(np.sum(server_params, axis=0))
       
         else:     
             for i in range(len(local_parameters_list)):
                 temp.append(local_parameters_list[i][k])
                
             global_params[k] = (np.mean(temp, axis=0))*w_local+global_parameters[k]*w_global+w_server*(np.sum(server_params, axis=0))
    
    model_name = "%s/global" %server
    np.save(model_name, global_params)    
    

# Retrieve image paths based on the node name and the anomaly type 
def readList(node, pattern):
    dirpath ="%s" %node
    li_fpath = sorted(glob.glob(os.path.join(dirpath, "pcap", "*_%s" %pattern)))
    
    return li_fpath






def main():
    # The dataset we used for training
    node_list = ["n005", "n008",  "n036", "n047", "n006",  "n034", "n038", "n045", "n048",  "n031", "n035", "n041", "n046", "n053", "n056"]
    
    Knowledge_type = input("Enter an anomaly type for training (select from  TypeA,  TypeB): ")
    Knowledge_types = {"TypeA": 1, "TypeB": 3}
    pattern = Knowledge_types[Knowledge_type]


    # We define hyperparameters and initialize the global model
    max_rounds = 20
    global_params = {}
    print("Systems initialize...")
    global_params = np.load("init/init.npy", allow_pickle = True).item()
    
    osNet =ConvNet(input_dim=(1, 48, 48),
                 conv_param={'filter1_num':10, 'filter1_size':3, 'filter2_num':10, 'filter2_size':1, 'pad':1, 'stride':1},
                 hidden_size= 200, output_size= 2 ,use_dropout = True, dropout_ration = 0.1, use_batchnorm = False)

    osNet.params = global_params
    osNet.layer()
    
    model_name = '000/global'
    np.save(model_name, osNet.params) 
    
    for i in range(len(node_list)):
        model_name = "%s/local.npy" %node_list[i]
        np.save(model_name, osNet.params) 
        
   

    # The server list contains the segmentation info.
    server_list = {"000": node_list}

    with open("server_dict.txt", "wb") as myFile:
        pickle.dump(server_list, myFile)
    
    acc_dict = {}
    for i in range(len(node_list)):
        acc_dict[node_list[i]] = []
    full_acc_dict = {}
    for i in range(len(node_list)):
        full_acc_dict[node_list[i]] = []
    acc_list = []



    # Start training
    for i in range(max_rounds):

        # Read current segmentation info. from the txt file. 
        with open("server_dict.txt", "rb") as myFile:
            server_list = pickle.load(myFile)
        
        # For each existing server, we perform FL.    
        for s in range(len(getListKeys(server_list))):
            server_name = getListKeys(server_list)[s] # Server name
            node_num = len(getListValues(server_list)[s]) # Node numbers
            node_list = getListValues(server_list)[s] # Nodes assigned to the server
            

            # Randomly select a group of train_num clients for updating every round 
            train_num = (node_num+1)//2
            select_client = [] 
            clients = []
            remaining_list = node_list
            
            for j in range(train_num):
                select_client.append((i*train_num+j)%node_num)
    
            for j in range(train_num):
                with open("server_dict.txt", "rb") as myFile:
                     server_list = pickle.load(myFile)
                node_list = getListValues(server_list)[s]
                clients.append(node_list[select_client[j]])
                remaining_list.remove(node_list[select_client[j]])
            
            print("[Round:%s][Server:%s] Clients %s will update local models..." %(i+1, server_name, clients))
            
            
            # Retrieve the latest global model
            global_parameters = download(server_name) 
            

            # Broadcast the global model to clients for evaluation  
            # Selected clients every round
            for j in range(len(clients)):

                # Retrieve the client local data
                data_dir = readList(clients[j], pattern)[i]
                data, label = makedataset("%s/dataset" %(data_dir))
                # Local model
                local_params = upload(clients[j])
                
                # Evaluation on the client model with acc
                acc = precision(local_params, data, label)

                acc_dict[clients[j]].append(acc)
                full_acc_dict[clients[j]].append(acc)
                
                print("[Round:%s][Server:%s][%s] ML detection precision: %s" %(i+1, server_name, clients[j], acc))
                
                
            # The remaining clients
            for j in range(len(remaining_list)):

                # Retrieve the latest global model
                model_name = "%s/local.npy" %remaining_list[j]
                np.save(model_name, global_parameters) 
                
                # Retrieve the local client data 
                data_dir = readList(remaining_list[j], pattern)[i]
                data, label = makedataset("%s/dataset" %(data_dir))
                # Local model
                local_params = upload(remaining_list[j])
                
                # Evaluation on the client model with acc
                acc = precision(local_params, data, label)

                acc_dict[remaining_list[j]].append(acc)
                full_acc_dict[remaining_list[j]].append(acc)
                
                print("[Round:%s][Server:%s][%s] ML detection precision: %s" %(i+1, server_name, remaining_list[j], acc))
               
            
            # Perform local model training         
            local_parameters_list = []
            for j in range(len(clients)):
                print("[Round:%s][Server:%s][%s] Downloading global parameters and retraining in the local..." % (i+1, server_name, clients[j]))
                
                data_dir = readList(clients[j], pattern)[i]
                exp_file = data_dir.replace("_%s" %pattern, "")

                # Update the local model
                local_params = Execute(clients[j], exp_file, server_name, pattern)
                local_parameters_list.append(local_params)
               
            # Update the list of neighboring servers
            other_server_list = getListKeys(server_list)
            other_server_list.pop(s)
            
            
            # Model aggregation to update the global model based on the updated local models
            Aggregate(local_parameters_list, global_parameters, other_server_list, server_name)
            
            # One round of the current server is completed
            print("[Round:%s][Server:%s] Aggregating successfully" % (i+1, server_name))  
            

            
        # For every six rounds, we evaluate the performance of nodes under each server, and depending on their performance update the segmentation list  
        if (i+1)%6 == 0:
            
            fineness = 2
            
            # Compute the average acc of the nodes in each server    
            for k in range(len(getListKeys(acc_dict))):
                acc_dict[getListKeys(acc_dict)[k]] = np.mean(acc_dict[getListKeys(acc_dict)[k]])    
            
            for k in range(len(getListValues(server_list)[s])):
                acc_list.append(acc_dict[getListValues(server_list)[s][k]])
            
            print("[Round:%s][Server:%s] Average accuarcy:%s" %(i+1, server_name, acc_list))

            
            # Depending on the average acc, we set the threshold and update the segmentation list 
            print("[Round:%s] Start rearranging nodes" %(i+1))

            # Find out the nodes with acc scores under the threshold in each server     
            index, server = dropout(acc_list, getListKeys(server_list), 0.5-fineness/100, acc_dict)
            # If such nodes exist, update the segmentation list 
            if len(index) > 0:
                for p in range(len(index)):
                    server_update(server_list, getListKeys(acc_dict)[index[p]], server)

            # Reset the acc metric list
            for k in range(len(getListKeys(acc_dict))):
                    acc_dict[getListKeys(acc_dict)[k]] = []             
            acc_list = []  

        print("===============================================")



if __name__ == '__main__':
    main()
