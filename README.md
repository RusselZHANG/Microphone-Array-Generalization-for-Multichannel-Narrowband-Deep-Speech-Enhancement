# Microphone-Array-Generalization-for-Multichannel-Narrowband-Deep-Speech-Enhancement

This repository for the official PyTorch implementation of [Microphone Array Generalization for Multichannel Narrowband Deep Speech Enhancement](https://arxiv.org/abs/2107.12601), accepted by InterSpeech 2021.

## Introduction
Our work addresses the problem of microphone array generalization for deep-learning-based end-to-end multichannel speech enhancement. We aim to train a unique potentially performing well on unseen microphone arrays. The goal is to make the network learn the universal information for speech enhancement that is available for any array geometry, rather than learn the one-array-dedicated characteristics.  To resolve this problem, a single network is trained using data recorded by various **VIRTUAL** microphone arrays of different geometries using RIR Generator[1] and simulated diffused noise[2]. We design three variants of our recently proposed original NarrowBand Deep Filtering(NBDF) network to cope with the agnostic number of microphones.
![figure 1](https://github.com/RusselZHANG/Microphone-Array-Generalization-for-Multichannel-Narrowband-Deep-Speech-Enhancement/blob/main/doc/fig.png)

