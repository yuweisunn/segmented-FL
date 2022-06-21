import numpy as np
import cv2
import os
from PIL import Image
import glob
import random


# Preprocess the SIDD dataset
def makedataset(dirpath):
    li_fpath = glob.glob(os.path.join(dirpath, "*","*"))

    dic_katakana = {"benign":0, "malicious":1}
    
    num_image = len(li_fpath)
    data = np.empty((num_image, 48*48))
    label = []
    np.random.shuffle(li_fpath)
    
    for i, fpath in enumerate(li_fpath):
        label_int = [0,0]    
        label_str = os.path.split(fpath)[0].split("/")[4]
        label_num = dic_katakana[label_str]
        label_int[label_num]=1
        label.append(label_int)

        img_ = Image.open(fpath)
        img_ = np.array(img_).astype(np.float32)
        data[i, :] = img_.flatten()

    label = np.array(label)
    data = data.astype('float32')
    data = data.reshape(-1, 1, 48, 48)
    
    return data, label
