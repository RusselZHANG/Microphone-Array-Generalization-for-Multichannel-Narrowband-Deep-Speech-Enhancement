# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 20:09:25 2020

@author: admin
"""


import numpy as np
from pesq import pesq
from pystoi.stoi import stoi
from mir_eval.separation import bss_eval_sources


import scipy.signal as signal
import scipy.io.wavfile 
import os,fnmatch
import pandas as pd


import scipy.io as io
import argparse

parser = argparse.ArgumentParser( "NBDF base")
parser.add_argument('--prediction_path', type=str, default='prediction_wav_original_PW_NBDF_256_128_normal_models_2021-07-21_18_59_02', help='path to tr_val_data')

args = parser.parse_args()

def SDR(reference, estimation, sr=16000):
    sdr, _, _, _ = bss_eval_sources(reference[None, :], estimation[None, :])
    return sdr


def SI_SDR(reference, estimation, sr=16000):
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)
    Args:
        reference: numpy.ndarray, [..., T]
        estimation: numpy.ndarray, [..., T]
    Returns:
        SI-SDR
    [1] SDR– Half- Baked or Well Done?
    http://www.merl.com/publications/docs/TR2019-013.pdf
    """
    
    
    estimation, reference = np.broadcast_arrays(estimation, reference)
    reference_energy = np.sum(reference ** 2, axis=-1, keepdims=True)

    # # This is $\alpha$ after Equation (3) in [1].
    optimal_scaling = np.sum(reference * estimation, axis=-1, keepdims=True) / reference_energy

    # # This is $e_{\text{target}}$ in Equation (4) in [1].
    projection = optimal_scaling * reference

    # # This is $e_{\text{res}}$ in Equation (4) in [1].
    noise = estimation - projection

    ratio = np.sum(projection ** 2, axis=-1) / np.sum(noise ** 2, axis=-1)
    return 10 * np.log10(ratio)


def STOI(ref, est, sr=16000):
    return stoi(ref, est, sr, extended=False)


def WB_PESQ(ref, est, sr=16000):
    return pesq(sr, ref, est, "wb")


def NB_PESQ(ref, est, sr=16000):
    # return nb_pesq(ref, est, sr)
    return pesq(sr, ref, est, "nb")




if __name__ == "__main__":
    
    folders = ['prediction_wav_BeamformIt',]
    
    array_shapes = ['2ch-cir-D14cm','2ch-line-D8cm','4ch-cir-D20cm','4ch-line-D16cm','6ch-cir-D20cm','6ch-line-D24cm']
    
    noise_type = ['babble','white','wind']
        
    ref_channel = 0
    
    
    a = noise_type.copy()
    a.append("mean")
    
    
    
    for Dataset in  ['../SIMU_test_baseline_data_new']:
        
        datapath = "{}/".format(Dataset)
        
        folder = args.prediction_path
            
        PESQ_total = np.zeros((2*len(array_shapes),))
        STOI_total = np.zeros((2*len(array_shapes),))
        SDR_total = np.zeros((2*len(array_shapes),))
        SI_SDR_total = np.zeros((2*len(array_shapes),))
        
        predpath = "predictions/{}/".format(folder)
        
        respath = predpath + '/results/' 
        if not os.path.isdir(respath):
            os.makedirs(respath)
        
        for array_id, array in enumerate(array_shapes):
            
            
            
            PESQ_mat   = np.zeros((len(noise_type),1+1))
            STOI_mat   = np.zeros((len(noise_type),1+1))
            SDR_mat    = np.zeros((len(noise_type),1+1))
            SI_SDR_mat = np.zeros((len(noise_type),1+1))
            


        
            unprocessed_dir = datapath + "test_mixed_wav/"
            
            for i,noise in enumerate(noise_type):    
                mixedpath = datapath + "test_mixed_wav/{}/{}/".format(array,noise)  

                files = fnmatch.filter(os.listdir(mixedpath),'*_cln.wav')
                nfiles = len(files)
                        
                print("Processing\t" + noise + "\t0 " + str(nfiles) + "\t utterences ... \n")  
                        
                for file in files:
                    
                    spath = mixedpath + file
                    ypath = mixedpath + file[:-8] + "_ms.wav"
                    xpath = predpath + array +'/' + noise +'/' + file[:-8] + ".wav"
                            
                    _,s = scipy.io.wavfile.read(spath)
                    
                    if len(s.shape) != 1:
                        s = s[:,ref_channel]
                        
                        
                    _,y = scipy.io.wavfile.read(ypath)
                    y = y[:,ref_channel]
                    
                    _,x = scipy.io.wavfile.read(xpath)
                    
                    sl = min(len(y),len(s),len(x));
                    
                    s = s[:sl]
                    y = y[:sl]
                    x = x[:sl]
                    
                    s = np.float32(s)
                    y = np.float32(y)
                    x = np.float32(x)
                    
                    pesqy = NB_PESQ(s,y)
                    stoiy = STOI(s,y)
                    si_sdry = SI_SDR(s,y)
                    # si_sdry = SI_SDR(s,y)
                    sdry = SDR(s,y)
                    
                    
                    PESQ_mat[i,0]    += pesqy
                    STOI_mat[i,0]    += stoiy
                    SDR_mat[i,0]    += sdry
                    SI_SDR_mat[i,0]    += si_sdry
                    # si_sdr_list[i,0] += si_sdry;
                    
                    
                    stoix = STOI(s,x)
                    pesqx = NB_PESQ(s,x)
                    si_sdrx = SI_SDR(s,x)
                    sdrx = SDR(s,x)
                    
                    PESQ_mat[i,1]    += pesqx
                    STOI_mat[i,1]    += stoix
                    SDR_mat[i,1]    += sdrx
                    SI_SDR_mat[i,1]    += si_sdrx
                            
                PESQ_mat[i,:]    /= nfiles
                STOI_mat[i,:]    /= nfiles
                SDR_mat[i,:]     /= nfiles
                SI_SDR_mat[i,:]  /= nfiles
                
                
            res1path = predpath + array  + '/' 
            if not os.path.isdir(res1path):
                os.makedirs(res1path)
            
            
            
            # PESQ_mat = np.vstack((PESQ_mat,np.mean(PESQ_mat,axis=0)))
            
            
            
            PESQ_np = np.zeros((2,4))
            STOI_np = np.zeros((2,4))
            SDR_np = np.zeros((2,4))
            SI_SDR_np = np.zeros((2,4))
        
            for q in range(2):
                
                
                PESQ_np[q,0:3] = PESQ_mat[:,q]
                PESQ_np[q,3] = np.mean(PESQ_np[q,0:3])
                
                STOI_np[q,0:3] = STOI_mat[:,q]
                STOI_np[q,3] = np.mean(STOI_np[q,0:3])
                
                SDR_np[q,0:3] = SDR_mat[:,q]
                SDR_np[q,3] = np.mean(SDR_np[q,0:3])
                
                SI_SDR_np[q,0:3] = SI_SDR_mat[:,q]
                SI_SDR_np[q,3] = np.mean(SI_SDR_np[q,0:3])
                
            io.savemat(res1path+ 'metrics.mat', {'PESQ': PESQ_np, \
                                                 'STOI': STOI_np, \
                                                 'SDR': SDR_np, \
                                                 'SI_SDR': SI_SDR_np})
            

            PESQ_total[array_id] = PESQ_np[0,-1]
            PESQ_total[array_id+len(array_shapes)] = PESQ_np[1,-1]
            
            STOI_total[array_id] = STOI_np[0,-1]
            STOI_total[array_id+len(array_shapes)] = STOI_np[1,-1]
            
            SDR_total[array_id] = SDR_np[0,-1]
            SDR_total[array_id+len(array_shapes)] = SDR_np[1,-1]

            SI_SDR_total[array_id] = SI_SDR_np[0,-1]
            SI_SDR_total[array_id+len(array_shapes)] = SI_SDR_np[1,-1]
                          
        p_cols = array_shapes*2
        p_rows = ['PESQ','STOI', 'SDR', 'SI_SDR']
         
        metrics = np.vstack((PESQ_total,STOI_total,SDR_total,SI_SDR_total))
        
        df = pd.DataFrame(metrics, index=p_rows, columns=p_cols)
        df.to_csv("{}/metrics.csv".format(respath),sep=',')  
            
                

        
            
            
            
            

                            
                    