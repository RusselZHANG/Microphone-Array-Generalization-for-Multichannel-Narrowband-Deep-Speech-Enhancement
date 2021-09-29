# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 15:26:54 2020

@author: admin
"""


import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from NB_Dataset import NB_Dataset
from PW_NBDF_Net import PW_NBDF
import matplotlib.pyplot as plt


import os
import numpy as np
import time
import argparse
from tqdm import tqdm 


SEED = 123
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
np.random.seed(SEED)


parser = argparse.ArgumentParser( "PW-NBDF base")
parser.add_argument('--datapath', type=str, default='../FaSNet/NF_WHITE_random_ref_order_train_data/PW_NBDF_adjusted_batch', help='path to tr_val_data')
parser.add_argument('--gpuid', type=int, default=7, help='Using which gpu')
parser.add_argument('--MA', type=int, default=1, help='Whether use magnitude augmentation')
parser.add_argument('--num_epoch', type=int, default=15, help='Number of Epoch for training')
parser.add_argument('--n_workers', type=int, default=4, help='Num_workers: if on PC, set 0')
parser.add_argument('--time_steps', type=int, default=192, help='The number of frames in each batch')
parser.add_argument('--lr', type=float, default=1e-3, help='Fine tuning learning rate')
parser.add_argument('--batch_size', type=int, default=512, help='Batch size')

args = parser.parse_args()



os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpuid)
print(torch.cuda.is_available())
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())
print(torch.cuda.get_device_name(0))



# CUDA for PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device is :', device)
torch.cuda.empty_cache()

NowTime = time.localtime() 

if __name__ == "__main__":
       
    train_path = '{}/train_batch/'.format(args.datapath)
    val_path = '{}/validation_batch/'.format(args.datapath)
    
    iter_count = 0
    time_step = args.time_steps 
    batch_size = args.batch_size
    
    
    def count_parameters(model):
        return sum(p.numel() for p in network.parameters() if p.requires_grad)
   
    print("##################### Trainning  model ###########################")
    
    
    
    network = PW_NBDF()
    print(f'The model has {count_parameters(network):,} trainable parameters')
    network.to(device)

    optimizer = optim.Adam(network.parameters(), lr=args.lr)

    loss_function = nn.MSELoss()

    writer = SummaryWriter( 'runs/Fine_tuning_{}/'.format(time.strftime("%Y-%m-%d-%H-%M-%S", NowTime)))
    
    if args.MA:
        modelpath = 'PW_NBDF_MA_models/'
    else:
        modelpath = 'PW_NBDF_models/'

    if not os.path.isdir(modelpath):
        os.makedirs(modelpath)

    loss_train_epoch = []
    loss_val_epoch = []

    loss_train_sequence = []
    loss_val_sequence = []
    

    for epoch in range(args.num_epoch):
        
        train_NBDataset = NB_Dataset(data_path = train_path, batchsize = batch_size, time_steps=time_step, shuffle = True)
        val_NBDataset = NB_Dataset(data_path = val_path, batchsize = batch_size, time_steps=time_step, shuffle = True)
        
    
        train_loader = DataLoader(
            dataset=train_NBDataset,      # torch TensorDataset format
            batch_size=1,      # mini batch size
            shuffle=True,       # random shuffle for training
            drop_last=True,
            num_workers=args.n_workers,      # subprocesses for loading data
        )
    
        val_loader = DataLoader(
            dataset=val_NBDataset,      # torch TensorDataset format
            batch_size=1,      # mini batch size
            shuffle=True,       # random shuffle for training
            drop_last=True,
            num_workers=args.n_workers,      # subprocesses for loading data
        )
        
        print("############################ Epoch {} ################################".format(epoch+1))
        ############# Train ############################################################################################################

        network.train()  # set the network in train mode
        
        for inputs, targets in tqdm(train_loader):

            inputs = inputs.squeeze().to(device)
            targets  = targets.squeeze().to(device)

            optimizer.zero_grad()

            outputs = network(inputs)     
            # compute loss   
            loss = loss_function(outputs,targets)
            loss_train_sequence.append(loss.detach().cpu().numpy())

            loss.backward()
            optimizer.step()
            
            writer.add_scalars('Loss', {"Train": loss.item()},iter_count)
            iter_count += 1

        loss_train_epoch.append(np.mean(loss_train_sequence[epoch*len(train_loader):(epoch+1)*len(train_loader)]))

        ############## Validation ######################################################################################################
        network.eval()

        for inputs, targets in val_loader:  

            inputs = inputs.squeeze().to(device)
            targets  = targets.squeeze().to(device)

            outputs = network(inputs)
  
            loss = loss_function(outputs,targets)

            loss_val_sequence.append(loss.detach().cpu().numpy())        


        loss_val_epoch.append(np.mean(loss_val_sequence[epoch*len(val_loader):(epoch+1)*len(val_loader)]))   
    
        ############## Save Model ######################################################################################################
        torch.save(network.state_dict(), modelpath + 'network_epoch{}.pth'.format(epoch+1))
        
        ############## Loss evaluation ######################################################################################################
        np.save(modelpath + 'loss_val_epoch.npy',loss_val_epoch)
        np.save(modelpath + 'loss_train_epoch.npy',loss_train_epoch)
        
    
        curves = [loss_train_epoch, loss_val_epoch]
        labels = ['train_loss', 'val_loss'] 
            
        f1 = plt.figure(epoch+1)        
        plt.title("MSELoss of general model")
        plt.xlabel('Epoch')
        plt.ylabel('MSEloss')
        # plt.ylim([0.04,0.15])
        for i, curve in enumerate(curves):
            plt.plot(curve, label = labels[i])
        plt.legend()
        f1.savefig(modelpath+'Network_loss.png')
        
        writer.add_scalars('Loss', {"Validation": loss_val_epoch[epoch] },iter_count)