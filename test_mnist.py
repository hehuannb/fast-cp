#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 21:02:48 2020

@author: huan
"""

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from gcp_torch import spAGCP_torch

# def enron():
#     return load_tensor('enron.ten')

if __name__ == '__main__':
    batch_size_test=1000
    transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = datasets.MNIST(
        root="./data", train=True, transform=transform, download=True)
    testloader = torch.utils.data.DataLoader(
      train_dataset,
      batch_size=batch_size_test,num_workers=1)
    examples = enumerate(testloader)
    batch_idx, (example_data, example_targets) = next(examples)
    a = torch.squeeze(example_data)
    a = np.array(a)
    # a[a!=0]=1
    subs = np.vstack((a != 0).nonzero()).T
    vals = a[a!=0]
    vals = vals[:,np.newaxis]
    rank = 20
    X = torch.sparse_coo_tensor(subs.T, np.squeeze( vals),dtype=torch.double)
    model1 = spAGCP_torch(rank,len(subs))
    info1 = model1.fit(X, subs, torch.tensor(np.squeeze(vals)), alpha=3e-5, batch=1000, inner=1000, epoch=20, loss='gauss', sample='stratified', adam=True, acceleration=False, seed=0)

    model3 = spAGCP_torch(rank, len(subs))
    info3 = model3.fit(X, subs, torch.tensor(np.squeeze(vals)), alpha=3e-5, batch=1000, inner=1000, epoch=20, loss='gauss', sample='stratified', adam=True, acceleration=True, seed=0)

    
    # alst = [fit[i]['time'] for i in range(len(fit))]
    time1 = [info1[i]['time'] for i in range(len(info1))]
    # time2 = [info2[i]['time'] for i in range(len(info2))]
    time3 = [info3[i]['time'] for i in range(len(info3))]
    # time4 = [info4[i]['time'] for i in range(len(info4))]
    
    loss1 = [info1[i]['loss'] for i in range(len(info1))]
    # loss2 = [info2[i]['losst'] for i in range(len(info2))]
    loss3 = [info3[i]['loss'] for i in range(len(info3))]
    # loss4 = [info4[i]['losst'] for i in range(len(info4))]
    # # lossals = [fit[i]['fit'] for i in range(len(fit))]
    # # plt.semilogy(alst, lossals, label='ALS')
    plt.semilogy(time1, loss1,label='strat')
    # plt.semilogy(time2, loss2, label='uniform')
    plt.semilogy(time3, loss3,label='Acc-strat')
    # plt.semilogy(time4, loss4, label='Acc-uniform')
    # # plt.semilogy(time1, losst1,label='GCP t')
    # # plt.semilogy(time2, losst2, label='AGCP t')
    plt.legend()
    # plt.plot(dgcp.model.U[1])

    
    # recon1= reconstruct(model1.model.U)
    # recon2 = reconstruct(model3.model.U)
    
    # # aa=reconstruct(als.model.U)
    
    
    # fig = plt.figure()
    
    
    # for i in range(6):
    #   plt.subplot(2,3,i+1)
    #   plt.tight_layout()
    #   plt.imshow(a[i], interpolation='none')
    #   plt.title("Ground Truth: {}".format(example_targets[i]))
    #   plt.xticks([])
    #   plt.yticks([])
    # fig = plt.figure()
    # for i in range(6):
    #   plt.subplot(2,3,i+1)
    #   plt.tight_layout()
    #   plt.imshow(recon1[i], interpolation='none')
    #   plt.title("Reconstruct: {}".format(example_targets[i]))
    #   plt.xticks([])
    #   plt.yticks([])  
      

  
  