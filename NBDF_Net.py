# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 15:23:55 2020

@author: admin
"""
import torch
import torch.nn as nn





class NBDF_Net(nn.Module):
    def __init__(self, n_chanels, hidden1_dim= 256, hidden2_dim = 128, num_direction = 2, num_layers = 1 ,biFlag = True):

        super(NBDFNet,self).__init__()
        
        self.n_chanels=n_chanels
        
        self.input_dim= 2*self.n_chanels
        
        self.hidden1_dim=hidden1_dim
        self.hidden2_dim=hidden2_dim
        
        self.output1_dim=self.hidden1_dim*num_direction
        self.output2_dim=self.hidden2_dim*num_direction
        
        self.num_layers= num_layers

        self.target = target
        self.biFlag=biFlag

        
        self.rnn1 = nn.LSTM(input_size=self.input_dim, hidden_size = self.hidden1_dim, \
                        num_layers=self.num_layers,batch_first=True, \
                        bidirectional=biFlag)
        
        self.rnn2 = nn.LSTM(input_size=self.output1_dim,hidden_size = self.hidden2_dim, \
                        num_layers=self.num_layers,batch_first=True, \
                        bidirectional=biFlag)

        
        self.linearTimeDistributed = nn.Linear(self.output2_dim, 1)
        

    def forward(self,inputsignal):
        
        
        rnn1out,_ = self.rnn1(inputsignal)
        
        rnn2out,_ = self.rnn2(rnn1out)

        outsignal = torch.sigmoid(self.linearTimeDistributed(rnn2out))  # linear1out.dim =  (1024, 192, 512)  -> (1024, 192, 1)
            
        
        
        return outsignal
