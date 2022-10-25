# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 17:24:37 2022

@author: armel
"""

import torch

def favre():
    a=torch.ones(3,2)
    b=torch.sin(1.+torch.sqrt(3.*torch.diag(torch.Tensor([1,1]))+5.*torch.norm(a)))
    print(a)
    print(b)

def conv():
    s=[ torch.nn.Conv1d(1,1,3,1),
            torch.nn.ReLU(),
        torch.nn.Conv1d(1,2,2,2),
            torch.nn.ReLU(),
        torch.nn.Conv1d(2,3,2,1)
        ]
    # receptive field = 6
    mod = torch.nn.Sequential(*s)
    x = torch.zeros(10,1)
    y = mod(x.transpose(0,1))
    print(x.shape)
    print(y.shape)

def convrnn():
    N=3
    class RCNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = torch.nn.Conv1d(1,1,3,3)
            self.rnn = torch.nn.LSTM(1,10)
            self.mlp = torch.nn.Linear(10,N)

        def forward(self,x):
            #Â x = (B,T,D=1) need (B,D,T)
            print("X1",x.shape)
            z=self.cnn(x.transpose(1,2))
            z=torch.relu(z)
            #Â z = (B,D,T) need (T,B,D)
            print("Z1",z.shape)
            _,z=self.rnn(z.transpose(0,2).transpose(1,2))
            z=z[0]
            #Â z = (1,B,H)
            print("Z2",z.shape)
            z = z.view(-1,10)
            y = self.mlp(z)
            print("Y",y.shape)
            return y

    mod = RCNN()
    x = torch.randn(1,3000,1)
    y = mod(x)

convrnn()
exit()
