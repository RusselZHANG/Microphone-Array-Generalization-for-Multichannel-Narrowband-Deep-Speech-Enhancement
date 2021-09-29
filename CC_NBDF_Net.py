# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 15:23:55 2020

@author: admin
"""
import torch
import torch.nn as nn


class CC_NBDF(nn.Module):
    def __init__(feature_map_num = 64, hidden1_dim= 256, hidden2_dim = 128, num_direction = 2, num_layers = 1 ,biFlag = True):

        super(CRNN_NBDFNet,self).__init__()
        
        self.feature_map_num = feature_map_num
        
        self.hidden1_dim=hidden1_dim
        self.hidden2_dim=hidden2_dim
        
        self.num_layers= num_layers
        self.num_direction = num_direction
        
        self.output1_dim=self.hidden1_dim*self.num_direction
        self.output2_dim=self.hidden2_dim*self.num_direction
        
        

        self.biFlag=biFlag
        
        self.BN = nn.BatchNorm2d(self.feature_map_num)
        
        self.relu = torch.nn.ReLU()
        
        self.cnn1 = nn.Conv2d(2,self.feature_map_num,(2,1))
        
        self.cnn2 = nn.Conv2d(self.feature_map_num,self.feature_map_num,(2,1))

        self.rnn1 = nn.LSTM(input_size=self.feature_map_num, hidden_size = self.hidden1_dim, \
                        num_layers=self.num_layers,batch_first=True, \
                        bidirectional=biFlag)
        
        self.rnn2 = nn.LSTM(input_size=self.output1_dim,hidden_size = self.hidden2_dim, \
                        num_layers=self.num_layers,batch_first=True, \
                        bidirectional=biFlag)
        
        
        self.linearTimeDistributed = nn.Linear(self.output2_dim, 1)
        
    def forward(self,inputsignal):
        
        
        cnn1out = self.relu(self.cnn1(inputsignal))           # (512,2,4,192) -> (512,64,3,192)

        
        while cnn1out.shape[2] != 1:                          # recursive convolution (512,64,3,192) -> (512,64,192)
            cnn1out = self.relu(self.cnn2(cnn1out))
            
        cnn2out = torch.squeeze(cnn1out)
        cnn2out = torch.transpose(cnn2out,1,2)
        
        rnn1out,_ = self.rnn1(cnn2out)
        
        rnn2out,_ = self.rnn2(rnn1out)
            
        outsignal = torch.sigmoid(self.linearTimeDistributed(rnn2out))  # linear1out.dim =  (1024, 192, 8)  -> (1024, 192, 1)
            
        
        
        return outsignal