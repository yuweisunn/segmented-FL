import numpy as np 
import cv2 
import pandas as pd
import random
from cnn.ConvNet import *
from cnn.common.optimizer import RMSProp
from module.train import *
from module.builder import *
from module.segment import *
import pickle
import matplotlib.pyplot as plt
import csv



def download(server):
    global_params = np.load("%s/global.npy" %server, allow_pickle = True).item()

    return global_params 
    
def upload(node_name):  
    local_params = np.load("%s/local.npy" %node_name, allow_pickle = True).item() 
    
    return local_params


def precision(local_params, train_data, label):
    label = np.array(label)
    
    snet = ConvNet(input_dim=(1, 48, 48), 
                 conv_param={'filter1_num':10, 'filter1_size':3, 'filter2_num':10, 'filter2_size':1, 'pad':1, 'stride':1},
                 hidden_size= 200, output_size= 2 ,use_dropout = False, dropout_ration = 0.1, use_batchnorm = False)
  
    snet.params = local_params
    snet.layer()
    acc = snet.accuracy(train_data, label, 125)
    
    return acc
    

def Execute(node_name, pcap, server, pattern):
    local_params_init = download(server)
    train(node_name, pcap, local_params_init, pattern)
    local_params = upload(node_name)
    
    return local_params


def Aggregate(local_parameters_list, global_parameters, server_list, server):
    global_params = local_parameters_list[0]
    global_list = []
    
    for i in range(len(server_list)):
        model_name = '%s/global.npy' %server_list[i]
        global_list.append(np.load(model_name, allow_pickle = True).item())
    
    
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
    

def readList(node, pattern):
    dirpath ="%s" %node
    li_fpath = sorted(glob.glob(os.path.join(dirpath, "pcap", "*_%s" %pattern)))
    
    return li_fpath
    


def main():
    # The dataset we used for training

    node_list = ["n005", "n008",  "n036", "n047", "n006",  "n034", "n038", "n045", "n048",  "n031", "n035", "n041", "n046", "n053", "n056"]
    Knowledge_type = input("Enter an anomaly type for training (select from  Type A,  Type B,  Type C): ")
    Knowledge_types = {"Type A": 1, "Type B": 2, "Type C": 3}
    pattern = Knowledge_types[Knowledge_type]

    # We define hyperparameters and initialize the global model

    max_rounds = 20

    global_params = {}
    server_list = {"000": node_list}

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
        
    # Save segmentation info. in a txt file.

    with open("server_dict.txt", "wb") as myFile:
        pickle.dump(server_list, myFile)
    
    
    acc_dict = {}
    for i in range(len(node_list)):
        	acc_dict[node_list[i]] = []
    full_acc_dict = {}
    for i in range(len(node_list)):
        	full_acc_dict[node_list[i]] = []
    node_server_dict = {}
    for i in range(len(node_list)):
        node_server_dict[node_list[i]] = ["000"]    
    acc_list = []
    segmentation = []


    # Start training
    for i in range(max_rounds):
        # Read current segmentation info. from the txt file. 
        with open("server_dict.txt", "rb") as myFile:
            server_list = pickle.load(myFile)
        
        # For each existing server, we perform FL.    
        for s in range(len(getListKeys(server_list))):
            server_name = getListKeys(server_list)[s]
            node_num = len(getListValues(server_list)[s])
            node_list = getListValues(server_list)[s]
            select_client = []
            remaining_list = node_list
            clients = []
            train_num = (node_num+1)//2
            
            for j in range(train_num):
                select_client.append((i*train_num+j)%node_num)
    
            for j in range(train_num):
                
                with open("server_dict.txt", "rb") as myFile:
                     server_list = pickle.load(myFile)
                
                node_list = getListValues(server_list)[s]
                
                clients.append(node_list[select_client[j]])
                remaining_list.remove(node_list[select_client[j]])
            print("[Round:%s][Server:%s] Clients %s will update local models..." %(i+1, server_name, clients))
            
            local_parameters_list = []
            global_parameters = download(server_name)
            
            
                
            for j in range(len(clients)):
                pcap_file = readList(clients[j], pattern)[i]
                local_params = upload(clients[j])
                data, label = makedataset("%s/dataset" %(pcap_file), 4) 
                
                acc = precision(local_params, data, label)
                print("[Round:%s][Server:%s][%s] ML detection precision: %s" %(i+1, server_name, clients[j], acc))
                if not i == 0:    
                    acc_dict[clients[j]].append(acc)
                
                full_acc_dict[clients[j]].append(acc)
                
            for j in range(len(remaining_list)):
                model_name = "%s/local.npy" %remaining_list[j]
                np.save(model_name, global_parameters) 
                
                pcap_file = readList(remaining_list[j], pattern)[i]
                #pcap_file = pcap_file.replace(".pcap", "")
                local_params = upload(remaining_list[j])
                data, label = makedataset("%s/dataset" %(pcap_file), 4)
                
                if data.shape[0] == 0:
                    build_data(pcap_file, local_params, pattern, node_mac[remaining_list[j]])
                  
                
                data, label = makedataset("%s/dataset" %(pcap_file), 4)
                acc = precision(local_params, data, label)
                print("[Round:%s][Server:%s][%s] ML detection precision: %s" %(i+1, server_name, remaining_list[j], acc))
                if not i == 0: 
                    acc_dict[remaining_list[j]].append(acc)
                full_acc_dict[remaining_list[j]].append(acc)
            
            
            for j in range(len(clients)):
                print("[Round:%s][Server:%s][%s] Downloading global parameters and retraining in the local..." % (i+1, server_name, clients[j]))
                pcap_file = readList(clients[j], pattern)[i]
                exp_file = pcap_file.replace("_%s" %pattern, "")
                local_params = Execute(clients[j], exp_file, server_name, pattern)
                local_parameters_list.append(local_params)
               
            other_server_list = getListKeys(server_list)
            other_server_list.pop(s)
            
            Aggregate(local_parameters_list, global_parameters, other_server_list, server_name)
            print("[Round:%s][Server:%s] Aggregating successfully" % (i+1, server_name))  
            
            
        # Evalution and segmentation update.    
        if (i+1)%6 == 0:
            node_list = ["n005", "n008", "n029", "n032", "n036", "n043", "n047", "n006", "n009", "n034", "n038", "n045", "n048", "n007", "n031", "n035", "n041", "n046", "n053", "n056"]
            
            node_server_dict = {}
            for i in range(len(node_list)):
                node_server_dict[node_list[i]] = []
            
            
            with open("server_dict.txt", "rb") as myFile:
                server_list = pickle.load(myFile)
            
            segmentation.append(server_list)
            print(server_list)
            
            for server, node in server_list.items():
                for n in range(len(node)):
                    node_server_dict[node[n]].append(server)      
                    
        print("===============================================")
        
main()

