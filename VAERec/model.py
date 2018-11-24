import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

import argparse
torch.manual_seed(1)


class MultiVAE(nn.Module):
    def __init__(self,input_dim,is_training):
        super(MultiVAE,self).__init__()
        self.fc1=nn.Linear(input_dim,1000)
        self.fc2=nn.Linear(1000,500)
        self.fc31=nn.Linear(500,200)
        self.fc32=nn.Linear(500,200)

        self.fc4=nn.Linear(200,500)
        self.fc5=nn.Linear(500,1000)
        self.fc6=nn.Linear(1000,input_dim)

        self.is_training=is_training
    def encoder(self,x):
        if(self.is_training):
            x=F.dropout(x,0.5)
        x=F.tanh(self.fc2(F.tanh(self.fc1(x))))
        return self.fc31(x),self.fc32(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decoder(self,z):
        z = F.tanh(self.fc5(F.tanh(self.fc4(z))))
        x= self.fc6(z)
        x = F.log_softmax(x, dim=1)
        return x

    def forward(self,x):

        mu,logvar=self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x=self.decoder(z)
        return x,mu,logvar


