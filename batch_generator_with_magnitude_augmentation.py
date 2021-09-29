##################################################################
#Extract and shuffle STFT frequency-wise sequences with various num_channels, then store as .npz files in disk. 
#Do this, to read in one (mini)batch during training, only one .npz file (with shuffled order) is loaded, which makes training faster. 
#One problem of this setup is that the sequences in one batch are fixed for all epochs, which however is not a critical issue.
##################################################################

import numpy as np
import scipy.signal as signal
import scipy.io.wavfile 
import os,fnmatch
import argparse
import random
from tqdm import tqdm


parser = argparse.ArgumentParser( "NBDF base")



parser.add_argument('--dataPath', type=str, default='../FaSNet/NF_WHITE_random_ref_order_train_data/', help='path to tr_val_data')
parser.add_argument('--time_steps', type=int, default=192, help='Time length of each batch')
parser.add_argument('--MA', type=bool, default=True, help='Magnitude augmentation flag')
parser.add_argument('--batch_size', type=int, default=512, help='batch_size')


# please set datapath to the directory of mixed wav data
args = parser.parse_args()
dataPath = args.dataPath
datasets = ['train','validation']


ref_channel = 0           

# default parameters, adjustable
ft_len = 512
ft_overlap = 256
time_steps = args.time_steps
batch_size = args.batch_size
fre_num = ft_len//2+1
step_inc = time_steps//2	

# One needs to first collect the training sequences, and then shuffle them. 
# However, due to the memory limit, it is difficult to collect all the training sequences at a time, thence they are processed block by block.
block_size =1*1e6   


if __name__ == "__main__": 

        
    if args.MA:
        MA_lower = 0.75*0.75
        MA_upper = 1.33*1.33
    

    for dataset in datasets:  
      print("Processing {} data ...".format(dataset))
    
      wavPath = dataPath + dataset+'_mixed_wav/'
      if args.MA:
          batchPath = dataPath + 'PW_NBDF_augmented_batch/' + dataset+'_batch/'
      else:
          batchPath = dataPath + 'PW_NBDF_batch/' + dataset+'_batch/'
          
      if not os.path.isdir(batchPath):
        os.makedirs(batchPath)
        
      batchIndx = 0
      for n_channels in range(2,9):
          wavFiles = fnmatch.filter(os.listdir(wavPath),'{}ch*.wav'.format(n_channels))
          shuWavIndx = list(range(len(wavFiles)))
          print('{}ch_wavfile_nums: {}'.format(n_channels,len(wavFiles)))
          np.random.shuffle(shuWavIndx)
       
          wavIndx = 0    
          
          while wavIndx<len(wavFiles):       
              nb_sequence = np.empty((int(block_size),time_steps,(n_channels+1)*2))	
              seqIndx = 0
        
              # Collect sequences of one block  
              while wavIndx<len(wavFiles):
                rate,s = scipy.io.wavfile.read(wavPath+wavFiles[shuWavIndx[wavIndx]])         
                if len(s.shape) == 2:
                  if s.shape[0] > s.shape[1]:
                    s = np.transpose(s)
                f, t, S = signal.stft(s,nperseg=ft_len,noverlap=ft_overlap) 

                
                S = np.transpose(S,(1,2,0))
                fra_num = S.shape[1]
                if seqIndx+len(range(0,fra_num-time_steps,step_inc))*fre_num>block_size:
                  break
                for fra in range(0,fra_num-time_steps,step_inc):
                  nb_sequence[seqIndx:seqIndx+fre_num,:,0:(n_channels+1)*2:2] = np.real(S[:,fra:fra+time_steps,:])
                  nb_sequence[seqIndx:seqIndx+fre_num,:,1:(n_channels+1)*2:2] = np.imag(S[:,fra:fra+time_steps,:])				
                  seqIndx += fre_num
                wavIndx += 1
        
              # Shuffle sequences and extract batch
              shuSeqIndx = list(range(seqIndx))
              np.random.shuffle(shuSeqIndx)
              for i in range(0,seqIndx-batch_size,batch_size): 
                batch = np.empty((batch_size,time_steps,(n_channels+1)*2))
                for j in range(batch_size):
                    if args.MA:
                        M_factor = np.sqrt(np.random.uniform(MA_lower, MA_upper, n_channels))
                        M_factor = np.append(M_factor,M_factor[ref_channel])
                        for k in range(n_channels+1):
                            batch[j,:,2*k:2*(k+1)] = nb_sequence[shuSeqIndx[i+j],:,2*k:2*(k+1)]*M_factor[k]
                    else:
                        batch[j,] = nb_sequence[shuSeqIndx[i+j],]	
        
                Xr = batch[:,:,ref_channel*2:(ref_channel+1)*2] + 1e-8
                cln = batch[:,:,-2:]
        
                ##### compute training targets, i.e. mrm, cirm and cc #### and normalize sequence #########	
                mrm = np.sqrt(np.square(cln).sum(axis=2))/np.sqrt(np.square(Xr).sum(axis=2))
                mrm = (mrm>=1)+((mrm<1)*mrm)
        
        
                mu = np.sqrt(np.square(Xr).sum(axis=2)).mean(axis=1)
                X = batch[:,:,:n_channels*2]/mu.reshape(batch_size,1,1)      
        		
                # save one batch        
                np.savez(batchPath+'batch{}_{}ch.npz'.format(batchIndx,n_channels),X=np.float32(X),mrm=np.float32(mrm))
                batchIndx += 1
              del nb_sequence
              print("{}/{} wav files have been processed".format(wavIndx,len(wavFiles)))
        
          print("Number of batchs for {}: {}".format(dataset,batchIndx))