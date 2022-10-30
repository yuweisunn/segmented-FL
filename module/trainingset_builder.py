# -*- coding: utf-8 -*-

import dpkt
import numpy as np
import cv2
import os
import shutil
import socket
import pandas as pd
from PIL import Image
import matplotlib.pyplot as pltdat
import glob
from module.hilbert_formc import *
import math
from cnn.ConvNet import *
from dpkt.compat import compat_ord
from module.segment import *

def tcpFlags(tcp):
    """Returns a list of the set flags in this TCP packet."""
    ret = list()

    if tcp.flags & dpkt.tcp.TH_FIN != 0:
        ret.append('FIN')
    if tcp.flags & dpkt.tcp.TH_SYN  != 0:
        ret.append('SYN')
    if tcp.flags & dpkt.tcp.TH_RST  != 0:
        ret.append('RST')
    if tcp.flags & dpkt.tcp.TH_PUSH != 0:
        ret.append('PSH')
    if tcp.flags & dpkt.tcp.TH_ACK  != 0:
        ret.append('ACK')
    if tcp.flags & dpkt.tcp.TH_URG  != 0:
        ret.append('URG')
    if tcp.flags & dpkt.tcp.TH_ECE  != 0:
        ret.append('ECE')
    if tcp.flags & dpkt.tcp.TH_CWR  != 0:
        ret.append('CWR')

    # print ret
    return ret

def inet_to_str(inet):
    """Convert inet object to a string

        Args:
            inet (inet struct): inet network address
        Returns:
            str: Printable/readable IP address
    """
    # First try ipv4 and then ipv6
    try:
        return socket.inet_ntop(socket.AF_INET, inet)
    except ValueError:
        return socket.inet_ntop(socket.AF_INET6, inet)
    
    
def mac_addr(address):
    return ':'.join('%02x' % compat_ord(b) for b in address)


def build_data(pcap, parameters, pattern, monitor):
    snet =ConvNet(input_dim=(1, 48, 48), 
                 conv_param={'filter1_num':5, 'filter1_size':3, 'filter2_num':5, 'filter2_size':1, 'pad':1, 'stride':1},
                 hidden_size= 40, output_size= 2 ,use_dropout = False, dropout_ration = 0.5, use_batchnorm = False)
  
    snet.params = parameters
    snet.layer()
    
    type_dic = {1:"malicious", 0:"benign"}
    cluster_list = [0, 0, 0, 0]
    size = 16
    span = 0.5
    
    if os.path.exists("%s_%s" %(pcap, pattern)):
        shutil.rmtree("%s_%s" %(pcap, pattern))

    os.mkdir("%s_%s" %(pcap, pattern))
    os.mkdir("%s_%s/dataset" %(pcap, pattern))
    os.mkdir("%s_%s/dataset/benign" %(pcap, pattern))
    os.mkdir("%s_%s/dataset/malicious" %(pcap, pattern))
    
    img_counter = 0
    ini_flag = 0
    
    
    pcr = dpkt.pcap.Reader(open("%s.pcap" %pcap,'rb'))

    
    time_counter = 0

    ip_counter_list = []
    arp_counter_list = []
    http_counter_list = []
    https_counter_list = []
    mDNS_counter_list = []
    tcp_counter_list = []
    udp_counter_list = []
    DHCP_counter_list = []
    other_counter_list = []
    ip_counter =0
    arp_counter = 0
    http_counter = 0
    https_counter = 0
    mDNS_counter = 0
    tcp_counter = 0
    udp_counter = 0
    DHCP_counter =0
    other_counter = 0
    
    cluster_list = [0, 0, 0, 0]
    ip_dict = {}
    syn_dict = {}
    uniUDP = 0
    
    try:
    	    for ts,buf in pcr:
    	        if ini_flag == 0:
    	            ts_b = ts
    	            ini_flag = 1
    	      
    	        if ts_b+2*span > ts and ts > ts_b+span:
    	            ts_b = ts_b+span
    	            ip_counter_list.append(ip_counter)
    	            arp_counter_list.append(arp_counter)
    	            http_counter_list.append(http_counter)
    	            https_counter_list.append(https_counter)
    	            mDNS_counter_list.append(mDNS_counter)
    	            tcp_counter_list.append(tcp_counter)
    	            udp_counter_list.append(udp_counter)
    	            DHCP_counter_list.append(DHCP_counter)
    	            other_counter_list.append(other_counter)
    	            
    	            ip_counter =0
    	            arp_counter = 0
    	            http_counter = 0
    	            https_counter = 0
    	            mDNS_counter = 0
    	            tcp_counter = 0
    	            udp_counter = 0
    	            DHCP_counter =0
    	            other_counter = 0
    	            
    	            time_counter = time_counter+1
    	            
    	        elif ts > ts_b+2*span:
    	            for i in range(int((ts-ts_b-span)//0.5)):
    	                ip_counter_list.append(0)
    	                arp_counter_list.append(0)
    	                http_counter_list.append(0)
    	                https_counter_list.append(0)
    	                mDNS_counter_list.append(0)
    	                tcp_counter_list.append(0)
    	                udp_counter_list.append(0)
    	                DHCP_counter_list.append(0)
    	                other_counter_list.append(0)
    	                
    	                time_counter = time_counter+1
    	                ts_b = ts_b+span
    	                
    	                
    	                if time_counter == 256:
    	                    ip_num = []
    	                    ip_num = getListValues(ip_dict)
    	                    for p in range(len(ip_num)):
    	                        single_ip = list(dict.fromkeys(ip_num[p]))
    	                        if len(single_ip)>127:
    	                            cluster_list[1] = 1
    	                            
    	                    syn_num = []
    	                    syn_num = getListValues(syn_dict)
    	                    for p in range(len(syn_num)):
    	                        if syn_num[p]>2:
    	                            cluster_list[2] = 1
    	                        
    	                    time_counter = 0
    	                    img_counter = img_counter + 1
    	                
    	                    flow = np.array([ip_counter_list, arp_counter_list, tcp_counter_list,
    	                                         http_counter_list, https_counter_list, udp_counter_list,
    	                                         mDNS_counter_list, DHCP_counter_list, other_counter_list
    	                                         ])
    	                        
    	                    max_arp = np.max(flow,axis = 1)
    	                    
    	                    flow_temp = np.zeros([len(flow),size*size])
    	                    for i in range(len(flow)):
    	                        if max_arp[i] == 0:
    	                            flow_temp[i] = 0    
    	                        else:
    	                            flow_temp[i]= flow[i]/max_arp[i]
    	                
    	                    feature_map_arp = np.array(255 - flow_temp*255,dtype=np.uint8)
    	                    feature_map_arp = feature_map_arp.reshape(len(flow), size*size)
    	                    feature_map = np.zeros([len(flow), size, size])
    	                    
    	                    point_list = hilbert_formc(int(math.sqrt(size)))
    	                    for t in range(len(flow)):
    	                        for i in range(size*size):
    	                            feature_map[t, point_list[i][0], point_list[i][1]] = feature_map_arp[t, i]
    	                    
    	                    train_data = np.zeros([3*size,3*size])
    	                    for p in range(3):
    	                        for q in range(3):        
    	                            train_data[p*size:(p+1)*size, q*size:(q+1)*size] = feature_map[3*p+q]
    	                    
    	                    if pattern == 5:
    	                        cluster = type_dic[0]
    	                        for pa in range(len(cluster_list)):
    	                            if cluster_list[pa] == 1:
    	                                cluster = type_dic[1]
    	                    else:
    	                        cluster = type_dic[cluster_list[pattern-1]]    
    	                    
    	                    cv2.imwrite("%s_%s/dataset/%s/%s.jpg" %(pcap, pattern, cluster, img_counter), train_data)               
    	                    if img_counter%225 == 0:
    	                        print("                     Analyzing...%s/3" %(img_counter//225))
    	                    
    	                    cluster_list = [0, 0, 0, 0]
    	                    ip_dict = {}
    	                    syn_dict = {}
    	                    uniUDP = 0
    	            
    	                    
    	                    ip_counter_list = []
    	                    arp_counter_list = []
    	                    http_counter_list = []
    	                    https_counter_list = []
    	                    mDNS_counter_list = []
    	                    tcp_counter_list = []
    	                    udp_counter_list = []
    	                    DHCP_counter_list = []
    	                    other_counter_list = []
    	                    
    	
    	            
    	        if time_counter == 256: 
    	            ip_num = []
    	            ip_num = getListValues(ip_dict)
    	            for p in range(len(ip_num)):
    	                single_ip = list(dict.fromkeys(ip_num[p]))
    	                if len(single_ip)>127:
    	                    cluster_list[1] = 1
    	                    
    	                    
    	            syn_num = []
    	            syn_num = getListValues(syn_dict)
    	            for p in range(len(syn_num)):
    	                
    	                if syn_num[p]>2:
    	                    cluster_list[2] = 1
    	                
    	                
    	            time_counter = 0
    	            img_counter = img_counter + 1
    	        
    	            flow = np.array([ip_counter_list, arp_counter_list, tcp_counter_list,
    	                                 http_counter_list, https_counter_list, udp_counter_list,
    	                                 mDNS_counter_list, DHCP_counter_list, other_counter_list
    	                                 ])
    	                
    	            max_arp = np.max(flow,axis = 1)
    	            
    	            flow_temp = np.zeros([len(flow),size*size])
    	            for i in range(len(flow)):
    	                if max_arp[i] == 0:
    	                    flow_temp[i] = 0    
    	                else:
    	                    flow_temp[i]= flow[i]/max_arp[i]
    	              
    	            feature_map_arp = np.array(255 - flow_temp*255,dtype=np.uint8)
    	            feature_map_arp = feature_map_arp.reshape(len(flow), size*size)
    	            feature_map = np.zeros([len(flow), size, size])
    	            
    	            point_list = hilbert_formc(int(math.sqrt(size)))
    	            for t in range(len(flow)):
    	                for i in range(size*size):
    	                    feature_map[t, point_list[i][0], point_list[i][1]] = feature_map_arp[t, i]
    	            
    	            train_data = np.zeros([3*size,3*size])
    	            for p in range(3):
    	                for q in range(3):        
    	                    train_data[p*size:(p+1)*size, q*size:(q+1)*size] = feature_map[3*p+q]
    	            
    	            if pattern == 5:
    	                cluster = type_dic[0]
    	                for pa in range(len(cluster_list)):
    	                    if cluster_list[pa] == 1:
    	                        cluster = type_dic[1]
    	            else:
    	                cluster = type_dic[cluster_list[pattern-1]]  
    	                
    	            cv2.imwrite("%s_%s/dataset/%s/%s.jpg" %(pcap, pattern, cluster, img_counter), train_data)
    	            if img_counter%225 == 0:
    	                print("                     Analyzing...%s/3" %(img_counter//225))
    	                        
    	            
    	            
    	            cluster_list = [0, 0, 0, 0]
    	            ip_dict = {}
    	            syn_dict = {}
    	            uniUDP = 0
    	            
    	            ip_counter_list = []
    	            arp_counter_list = []
    	            http_counter_list = []
    	            https_counter_list = []
    	            mDNS_counter_list = []
    	            tcp_counter_list = []
    	            udp_counter_list = []
    	            DHCP_counter_list = []
    	            other_counter_list = []
    	            
    	           
    	                
    	        try:    
    	            eth = dpkt.ethernet.Ethernet(buf)
    	            dstMac = mac_addr(eth.dst)
    	            srcMac = mac_addr(eth.src)
    	            
    	            
    	            if type(eth.data) == dpkt.arp.ARP:
    	                try: 
    	                    ip_dict[srcMac].append(eth.dst)
    	                except:
    	                    ip_dict[srcMac] = [eth.dst]
    	                    
    	                arp_counter = arp_counter+1
    	           
    	            
    	            elif type(eth.data) == dpkt.ip.IP:
    	                ip = eth.data
    	                ip_counter = ip_counter+1
    	                dstIP = inet_to_str(ip.dst)
    	                
    	                if type(ip.data) == dpkt.tcp.TCP:
    	                    tcp_counter = tcp_counter+1
    	                    tcp = ip.data
    	                    
    	                    
    	                    """
    	                    Pattern Recognition
    	                    """
    	                    if tcp.dport == 445 and dstMac == monitor: 
    	                        cluster_list[0] = 1
    	                      
    	                    tcpFlag = tcpFlags(tcp)
    	                    if {'SYN'} == set(tcpFlag):     
    	                        try: 
    	                            syn_dict[eth.src] = syn_dict[eth.src] + 1
    	                        except:
    	                            syn_dict[eth.src] = 1
    	                    
    	                    
    	                    if tcp.dport == 80 or tcp.sport == 80:
    	                        http_counter = http_counter+1
    	                    if tcp.dport == 443 or tcp.sport == 443:
    	                        https_counter = https_counter+1
    	                    
    	                elif type(ip.data) == dpkt.udp.UDP:
    	                    udp_counter = udp_counter + 1
    	                    udp = ip.data
    	                    if  udp.dport == 5353 or udp.sport == 5353:
    	                        mDNS_counter = mDNS_counter + 1
    	                    if  udp.dport == 67 or udp.sport == 67:
    	                        DHCP_counter = DHCP_counter + 1
    	                    
    	                    dstEnding = dstIP.split('.')[3]
    	                    
    	                    if  dstEnding != "255" and dstMac == monitor and udp.sport != 123 and udp.sport != 53 :
    	                        cluster_list[3] = 1
    	  
    	                    
    	        except:
    	               pass 
    	                         
    	           
    	            
    	    ip_counter =0
    	    arp_counter = 0
    	    http_counter = 0
    	    https_counter = 0
    	    mDNS_counter = 0
    	    tcp_counter = 0
    	    udp_counter = 0
    	    DHCP_counter =0
    	    other_counter = 0
    	        
    	    record = len(ip_counter_list)
    	    for i in range(256-record):
    	        ip_counter_list.append(ip_counter)
    	        arp_counter_list.append(arp_counter)
    	        http_counter_list.append(http_counter)
    	        https_counter_list.append(https_counter)
    	        mDNS_counter_list.append(mDNS_counter)
    	        tcp_counter_list.append(tcp_counter)
    	        udp_counter_list.append(udp_counter)
    	        DHCP_counter_list.append(DHCP_counter)
    	        other_counter_list.append(other_counter)
    	        
    	     
    	        
    	    ip_num = []
    	    ip_num = getListValues(ip_dict)
    	    for p in range(len(ip_num)):
    	        single_ip = list(dict.fromkeys(ip_num[p]))
    	        if len(single_ip)>127:
    	           cluster_list[1] = 1
    	                    
    	    syn_num = []
    	    syn_num = getListValues(syn_dict)
    	    for p in range(len(syn_num)):
    	        
    	        if syn_num[p]>2:
    	            cluster_list[2] = 1
    	                
    	              
    	    img_counter = img_counter + 1     
    	    flow = np.array([ip_counter_list, arp_counter_list, tcp_counter_list,
    	                         http_counter_list, https_counter_list, udp_counter_list,
    	                         mDNS_counter_list, DHCP_counter_list, other_counter_list
    	                         ])
    	        
    	    max_arp = np.max(flow,axis = 1)
    	    flow_temp = np.zeros([len(flow),size*size])
    	        
    	    for i in range(len(flow)):
    	        if max_arp[i] == 0:
    	            flow_temp[i] = 0    
    	        else:
    	            flow_temp[i]= flow[i]/max_arp[i]
    	                
    	    feature_map_arp = np.array(255 - flow_temp*255,dtype=np.uint8)
    	    feature_map_arp = feature_map_arp.reshape(len(flow), size*size)
    	    feature_map = np.zeros([len(flow), size, size])
    	        
    	    point_list = hilbert_formc(int(math.sqrt(size)))
    	    for t in range(len(flow)):
    	        for i in range(size*size):
    	            feature_map[t, point_list[i][0], point_list[i][1]] = feature_map_arp[t, i]
    	                
    	    train_data = np.zeros([3*size,3*size])
    	    for p in range(3):
    	        for q in range(3):        
    	            train_data[p*size:(p+1)*size, q*size:(q+1)*size] = feature_map[3*p+q]
    	                
    	    if pattern == 5:
    	        cluster = type_dic[0]
    	        for pa in range(len(cluster_list)):
    	            if cluster_list[pa] == 1:
    	                cluster = type_dic[1]
    	    else:
    	        cluster = type_dic[cluster_list[pattern-1]]   
    	                
    	    cv2.imwrite("%s_%s/dataset/%s/%s.jpg" %(pcap, pattern, cluster, img_counter), train_data)               
    	    if img_counter%225 == 0:
    	        print("                     Analyzing...%s/3" %(img_counter//225))
    	                
    except:
        pass

