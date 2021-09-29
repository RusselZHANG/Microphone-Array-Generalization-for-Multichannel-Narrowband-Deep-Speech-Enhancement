# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 15:23:55 2020

@author: admin
"""
import torch
import torch.nn as nn
from tqdm import tqdm




class PW_NBDF(nn.Module):
    def __init__(self, input_dim = 4, hidden1_dim = 256, hidden2_dim = 128, num_direction = 2, device = 'cuda', num_layers = 1, biFlag = True):

        super(PW_NBDF,self).__init__()
        
        self.input_dim=input_dim
        
        self.hidden1_dim=hidden1_dim
        self.hidden2_dim=hidden2_dim
        
        self.num_direction = num_direction
        
        self.output1_dim=self.hidden1_dim*num_direction
        self.output2_dim=self.hidden2_dim*num_direction
        
        self.num_layers= num_layers
        self.device = device


        self.biFlag=biFlag
        
        
        self.rnn1 = nn.LSTM(input_size=self.input_dim, hidden_size = self.hidden1_dim, \
                        num_layers=self.num_layers,batch_first=True, \
                        bidirectional=self.biFlag)
        
        self.rnn2 = nn.LSTM(input_size=self.output1_dim,hidden_size = self.hidden2_dim, \
                        num_layers=self.num_layers,batch_first=True, \
                        bidirectional=self.biFlag)
        
        self.linearTimeDistributed = nn.Linear(self.output2_dim, 1)
        
        


    def forward(self,inputsignal):
        
        B,T,C = inputsignal.shape # (B,T,C)
        
        n_pairs = C//2-1  # number of channel of pairs
        x = torch.zeros(B, n_pairs, T, 4, device = self.device)  # (B,N,T,4)
        
        for i in range(n_pairs):
            x[:, i, :, :2] = inputsignal[:,:,:2]
            x[:, i, :, 2:] = inputsignal[:,:,(i+1)*2:(i+2)*2]
        x = x.view(B*n_pairs, T, 4)                             # (B*N , T,  4)
        rnn1out, _ = self.rnn1(x)
        rnn1out = rnn1out.view(B, n_pairs, T, self.output1_dim) # (B, N, T, Dim1*2)
        rnn1out_combined = torch.mean(rnn1out,dim = 1)          # (B, T, Dim1*2)

        rnn2out,_ = self.rnn2(rnn1out_combined)                 # (B, T, Dim2*2)
        outsignal = torch.sigmoid(self.linearTimeDistributed(rnn2out)).squeeze()  # (B, T) 
            

        
        return outsignal       # 1D mask output
