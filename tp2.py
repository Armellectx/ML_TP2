# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 10:39:37 2022

@author: armel
"""

# overfitting

import sys
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt

dn = 50.
h=10
nepochs=1000
lr = 0.0001


#Data: convert to tensor
with open("meteo/2019.csv","r") as f: ls=f.readlines()
trainx = torch.Tensor([float(l.split(',')[1])/dn for l in ls[:-7]]).view(1,-1,1)
trainy = torch.Tensor([float(l.split(',')[1])/dn for l in ls[7:]]).view(1,-1,1)
#print(trainx.shape)
#print(trainx[0,0,0])
with open("meteo/2020.csv","r") as f: ls=f.readlines()
testx = torch.Tensor([float(l.split(',')[1])/dn for l in ls[:-7]]).view(1,-1,1)
testy = torch.Tensor([float(l.split(',')[1])/dn for l in ls[7:]]).view(1,-1,1)

# trainx = 1, seqlen, 1
# trainy = 1, seqlen, 1


#Tensor to Dataset
    
trainds = torch.utils.data.TensorDataset(trainx, trainy)
trainloader = torch.utils.data.DataLoader(trainds, batch_size=1, shuffle=False)

testds = torch.utils.data.TensorDataset(testx, testy)
testloader = torch.utils.data.DataLoader(testds, batch_size=1, shuffle=False)

#on calcule la mean scare error à chaque fois 
crit = nn.MSELoss()




#Model: simple RNN with a linear regression layer:

class Mod(nn.Module):
    def __init__(self,nhid):
        super(Mod, self).__init__()
        self.rnn = nn.RNN(1,nhid)
        self.mlp = nn.Linear(nhid,1)

    def forward(self,x):
        # x = B, T, d
        xx = x.transpose(0,1)
        y,_=self.rnn(xx)
        T,B,H = y.shape
        
        y = self.mlp(y.view(T*B,H))
        y = y.view(T,B,-1)
        y = y.transpose(0,1)
        return y
    
    
    
#compute MSE on test:
def test(mod):
    testloss, nbatch = 0., 0
    for data2 in testloader:
        inputs2, goldy2 = data2
        haty2 = mod(inputs2)
        loss2 = crit(haty2,goldy2)
        testloss += loss2.item()
        nbatch += 1
    testloss /= float(nbatch)
    return testloss

#Training loop:
def train(mod):
    optim = torch.optim.Adam(mod.parameters(), lr)
    plot_values = {"epoch": [], "loss": [], "test_loss":[]}
    for epoch in range(nepochs):
        mod.train(False)
        testloss = test(mod)
        mod.train(True)
        totloss, nbatch = 0., 0
        for data in trainloader:
            inputs, goldy = data
            optim.zero_grad()
            haty = mod(inputs)
            loss = crit(haty,goldy)
            totloss += loss.item()
            nbatch += 1
            loss.backward()
            optim.step()
            


            

        totloss /= float(nbatch)
        plot_values["epoch"].append(epoch)
        plot_values["loss"].append(totloss)
        plot_values["test_loss"].append(testloss)
            
        print(epoch,"err",totloss,testloss)
        
    # Plot loss evolution
    _, ax = plt.subplots()
    ax.plot(plot_values["epoch"], plot_values["loss"], 'b', label='training loss')
    ax.plot(plot_values["epoch"], plot_values["test_loss"], 'g', label='validation loss')
    ax.set(xlabel='epoch', ylabel='MSE loss', title='Training supervision for lr = %s'%lr)
    ax.axis(ymin=0)
    ax.grid()
    ax.legend()
    
    print("fin",totloss,testloss,file=sys.stderr)


#The Main:
mod=Mod(h)
print("nparms",sum(p.numel() for p in mod.parameters() if p.requires_grad),file=sys.stderr)
train(mod)
plt.show()


#test du modèle 