# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 15:13:05 2020

@author: admin
"""

import torch
import torch.nn as nn
# import torchvision
# import torch.nn.functional as F
from torch import optim
# import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


import os,fnmatch
import numpy as np


class NB_Dataset(Dataset):

    def __init__(self, data_path, batchsize = 512, time_steps = 192, shuffle=True):
        self.data_path = data_path
        self.time_steps = time_steps
        self.batchsize = batchsize
        self.shuffle = shuffle
        self.on_epoch_end()
        

    def __getitem__(self, index):
        
        batchname = fnmatch.filter(os.listdir(self.data_path),'batch{}*'.format(self.indexes[index]))[0]
        sample = np.load(self.data_path+ batchname)
        X = sample['X'][:self.batchsize,:self.time_steps,:].astype('float32')
        y = sample['mrm'][:self.batchsize,:self.time_steps].reshape(self.batchsize,self.time_steps,1).astype('float32')
        

        X = torch.from_numpy(X)
        y = torch.from_numpy(y)
        
        return X, y

    def __len__(self):
        
        return len(fnmatch.filter(os.listdir(self.data_path),'batch*.npz'))
    
    def on_epoch_end(self):
    #    'Updates indexes after each epoch'
        self.indexes = np.arange(self.__len__())
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
            
"""           
if __name__ == "__main__":
    
    train_path = '../Array_position/train_val_batch/train_batch/'
    val_path = '../Array_position/train_val_batch/validation_batch/'
    
    # wavFiles = fnmatch.filter(os.listdir(train_path),'batch10606*.npz')
    
    
    btz = 1
    
    train_NBDataset = NBDataset(data_path = train_path, time_steps=192, shuffle = True)
    val_NBDataset = NBDataset(data_path = val_path,  time_steps=192, shuffle = True)
    

    train_DataLoader = DataLoader(
        dataset=train_NBDataset,      # torch TensorDataset format
        batch_size=btz,      # mini batch size
        shuffle=True,       # random shuffle for training
        drop_last=True,
        num_workers=0,      # subprocesses for loading data
    )

    val_DataLoader = DataLoader(
        dataset=val_NBDataset,      # torch TensorDataset format
        batch_size=btz,      # mini batch size
        shuffle=True,       # random shuffle for training
        drop_last=True,
        num_workers=0,      # subprocesses for loading data
    )
    
    n,m = train_NBDataset[2]
    print(n.shape,m.shape)
    
    for a,b in train_DataLoader:
        print(a.shape,b.shape)
""" 