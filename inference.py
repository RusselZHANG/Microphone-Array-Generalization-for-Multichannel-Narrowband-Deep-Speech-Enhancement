# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 15:43:18 2020

@author: admin
"""
import torch
import torch.nn as nn
from PW_NBDF_Net import PW_NBDF

import numpy as np

import scipy.signal as signal
import scipy.io.wavfile 
import os,fnmatch 
from scipy import io



print(torch.cuda.is_available())
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())
print(torch.cuda.get_device_name(0))


os.environ["CUDA_VISIBLE_DEVICES"] = '7'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device is :', device)
torch.cuda.empty_cache()


    
    
def wav_generator(noise):
    
    print("Processing {}, {}".format(array,noise))
    wavPath = testPath + noise +'/'
    wavFiles = fnmatch.filter(os.listdir(wavPath),'*_ms.wav')
    outDir = outPath + noise +'/'

    if not os.path.isdir(outDir):
        os.makedirs(outDir)
    

    for wavIndx in range(len(wavFiles)):

        rate,s = scipy.io.wavfile.read(wavPath+wavFiles[wavIndx])        
        if len(s.shape) == 2:
            if s.shape[0] > s.shape[1]:
                s = np.transpose(s)                
        
        f, t, S = signal.stft(s,window=win,nperseg=ft_len,noverlap=ft_overlap)

        Sref = S[0,:,:]
 
        mu = np.abs(Sref).mean(axis=1)
        fra_num = S.shape[2]
        X = np.empty((fre_num,fra_num,n_channels*2))
        for ch in range(n_channels):
            X[:,:,2*ch] = np.real(S[ch,:,:])
            X[:,:,2*ch+1] = np.imag(S[ch,:,:]) 
        X = torch.from_numpy(X/mu.reshape(fre_num,1,1)).to(device)

        # prediction: directly input the whole utterance to the network 

        load_network.eval()
        y = load_network(X.float())
        y = y.cpu().detach().numpy()
        y = y.reshape(fre_num,fra_num)

        Y = Sref*y
        # istft
        t,enhanced = signal.istft(Y,window=win,nperseg=ft_len,noverlap=ft_overlap,input_onesided=True)      
        enhanced = np.int16(amp*enhanced/np.max(np.abs(enhanced)))


        outname = outDir + wavFiles[wavIndx][:-7]+'.wav'
        scipy.io.wavfile.write(outname,rate,enhanced) 

                
if __name__ == "__main__":
    
    
    modelpath = 'PW_NBDF_adjusted_models/'
    modelindex = 10
    test_array = ['2ch-line-D8cm','2ch-cir-D14cm','4ch-cir-D20cm','4ch-line-D16cm','6ch-cir-D20cm','6ch-line-D24cm']        
    noise_type = ['babble','white','wind']
    
    for dataPath  in ['../SIMU_array_specfic_baseline_data_new/']:         
        for array in test_array:
            
            n_channels = int(array[0])
            testPath = dataPath + 'test_mixed_wav/' + array + '/'
            outPath = dataPath  + 'predictions/prediction_wav_{}/'.format(modelpath[:-8]) + array + '/'
            modelname = modelpath + 'network_epoch{}.pth'.format(modelindex) 
    
            print("Processing {}  {} test ...".format(dataPath[3:-1] , array))
        
            # STFT parameters, should be identical to the ones used for train
            ft_len = 512
            ft_overlap = ft_len//2
            fre_num = ft_len//2+1
            win = 'hann'
        
            amp = np.iinfo(np.int16).max
            
            
            load_network = PW_NBDF()
            
            load_network.load_state_dict(torch.load(modelname))
            load_network = load_network.to(device)
     
    
            for noise in noise_type:
                
                wav_generator(noise)
