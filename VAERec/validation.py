from evaluation import *

import torch
import torch.nn as nn
import random

from model import MultiVAE
from dataLoader import valLoader
import sys
import os
import numpy as np

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")



def evaluate():


    dataloader = valLoader("processed/test_tr.csv","processed/test_te.csv")
    idxs = [i for i in range(dataloader.n_users)]
    for i in range(1,21):
        model=MultiVAE(dataloader.n_items)
        checkpoint = torch.load("checkpoint-word_embedding" + str(i))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.is_training=False
        model.to(device)
        with torch.no_grad():
            n100_list, r20_list, r50_list = [], [], []
            for batch_id, (start, end) in enumerate(zip(range(0,dataloader.n_users, 64),range(64, dataloader.n_users,64))):
                batch,target=dataloader.get_batch(idxs[start:end])
                batch=np.squeeze(batch)
                target=np.squeeze(target)
                x, mu, logvar = model(torch.tensor(batch, dtype=torch.float32, device=device))
                x=x.cpu().numpy()
                x[batch.nonzero()] = -np.inf

                #n100_list.append(NDCG_binary_at_k_batch(x,target, k=100))
                r20_list.append(Recall_at_k_batch(x, target, k=20))
                r50_list.append(Recall_at_k_batch(x, target, k=50))

        #n100_list = np.concatenate(n100_list)
        r20_list = np.concatenate(r20_list)
        r50_list = np.concatenate(r50_list)

        #print("Test NDCG@100=%.5f (%.5f)" % (np.mean(n100_list), np.std(n100_list) / np.sqrt(len(n100_list))))
        print("Test Recall@20=%.5f (%.5f)" % (np.mean(r20_list), np.std(r20_list) / np.sqrt(len(r20_list))))
        print("Test Recall@50=%.5f (%.5f)" % (np.mean(r50_list), np.std(r50_list) / np.sqrt(len(r50_list))))
evaluate()