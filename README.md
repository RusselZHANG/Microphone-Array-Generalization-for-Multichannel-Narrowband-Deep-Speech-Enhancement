# Microphone-Array-Generalization-for-Multichannel-Narrowband-Deep-Speech-Enhancement

This repository for the official PyTorch implementation of [Microphone Array Generalization for Multichannel Narrowband Deep Speech Enhancement](https://arxiv.org/abs/2107.12601), accepted by InterSpeech 2021.

## Introduction
Our work addresses the problem of microphone array generalization for deep-learning-based end-to-end multichannel speech enhancement. We aim to train a unique potentially performing well on unseen microphone arrays. The goal is to make the network learn the universal information for speech enhancement that is available for any array geometry, rather than learn the one-array-dedicated characteristics.  To resolve this problem, a single network is trained using data recorded by various **VIRTUAL** microphone arrays of different geometries using RIR Generator[1] and simulated diffused noise[2]. We design three variants of our recently proposed original NarrowBand Deep Filtering(NBDF) [3] network to cope with the agnostic number of microphones.  
  

![figure 1](https://github.com/RusselZHANG/Microphone-Array-Generalization-for-Multichannel-Narrowband-Deep-Speech-Enhancement/blob/main/doc/fig.png)

## Key Features
* Simulated_RIR_Generator
* Network
  * original NBDF (CP-NBDF)
  * CC-NBDF
  * PW-NBDF   
* Train
* Inference
* Evaluation

## Get started
(1) Clone:
``` 
$ git clone https://github.com/atomicoo/Tacotron2-PyTorch.git
```
(2) Requirements:
``` 
$ pip install -r requirements.txt
```
[RIR Generator](https://github.com/ehabets/RIR-Generator) [1], [coherent multichannel noise generator](https://github.com/ehabets/ANF-Generator)[2] and [wind noise simulator](https://github.com/ehabets/Wind-Generator) [4] is required.


# Reference
[1] E. A. Habets, “Room impulse response generator,” Technische Universiteit Eindhoven, Tech. Rep, vol. 2, no. 2.4, p. 1, 2006.  
<br>
[2] E. A. Habets, I. Cohen, and S. Gannot, “Generating nonstationary multisensor signals under a spatial coherence constraint,” The Journal of the Acoustical Society of America, vol. 124, no. 5, pp. 2911–2917, 2008.  
<br>
[3] X. Li and R. Horaud, “Narrow-band deep filtering for multichannel speech enhancement,” arXiv preprint arXiv:1911.10791, 2019.  
<br>
[4]  D. Mirabilii and E. A. Habets, “Simulating multi-channel wind noise based on the corcos model,” in 2018 16th International Workshop on Acoustic Signal Enhancement (IWAENC).IEEE,2018, pp. 560–564.

