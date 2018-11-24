import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from scipy import sparse
import torch.optim as optim
import random
from dataLoader import trainLoader
import argparse
from model import MultiVAE
parser = argparse.ArgumentParser(description='VAE for recommendation')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')

parser.add_argument('--input_dim', type=int, default=100,
                    help='input dimension')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda:0" if args.cuda else "cpu")

trainData=trainLoader("processed/train.csv")

def loss_function(recon_x, x, mu, logvar):
    x=torch.tensor(x,dtype=torch.float)
    recon_x = torch.tensor(recon_x, dtype=torch.float)

    loss1=torch.mean(-torch.sum(recon_x*x,1).cpu())


    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),1).cpu())

    return loss1 + 0.1*KLD


def trainIter(model,model_optimizer,batch):

    x, mu, logvar=model(torch.tensor(batch,dtype=torch.float32,device=device))
    loss=loss_function(x,batch,mu,logvar)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
    model_optimizer.step()

    return loss
def train(model):
    model_optimizer = optim.Adam(model.parameters(), lr=0.005)


    idxs = [i for i in range(trainData.n_users)]
    for epoch in range(1, args.epochs + 1):
        losses=0
        random.shuffle(idxs)
        for batch_id, (start, end) in enumerate(zip(range(0, trainData.n_users, args.batch_size),range(args.batch_size, trainData.n_users, args.batch_size))):
            batch=trainData.get_batch(idxs[start:end])
            batch=torch.tensor(batch)
            batch=torch.squeeze(batch)

            loss=trainIter(model,model_optimizer,batch)
            losses+=loss.item()

        print(losses/trainData.n_users)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': model_optimizer.state_dict(),
            'loss': loss,
        }, "checkpoint-word_embedding" + str(epoch))


model=MultiVAE(trainData.n_items,True).to(device)
train(model)