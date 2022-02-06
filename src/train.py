
import sys
import os
import json


import anndata as ad
import numpy as np
import pandas as pd


import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from scipy import sparse
from model import AE_gex,AE_adt

with open('config/train-params.json') as fh:
    train_cfg = json.load(fh)

# get parameters
epoch_gex = train_cfg['epoch_count_gex']
epoch_adt = train_cfg['epoch_count_adt']
batch_size = train_cfg['batch_size']
batch_size_coupled = train_cfg['batch_size_coupled']

# loss function of pairwise distance between latent space and original space
def pairwise(code,curbatch):
    d_embedding = torch.pdist(code)
    cur_embedding = torch.pdist(curbatch)
    los = nn.MSELoss()
    return los(d_embedding-cur_embedding)


def get_train_coupled(df1, df2):
    print("Starting coupled model training")
    print("Disclaimer: currently training on low number of Epochs to save runtime")
    # initialize model
    model_coupled = AE_coupled().to(device)
    
    # create an optimizer object
    optimizer_gex = optim.Adam(model_coupled.parameters(), lr=1e-3)
      
    # declare which losses to use
    criterion_mse = nn.MSELoss()
    criterion_pairwise = pairwise
    
    x,y = [],[]
    
    num_points = df1.shape[0]

    for epoch in range(5):
        loss = 0
        permutation = torch.randperm(num_points)
        for i in range(0,num_points, batch_size_coupled):
            indices = permutation[i:i+batch_size_coupled]
            cur_batch_gex = df1[indices]
            cur_batch_adt = df2[indices]
            optimizer_gex.zero_grad()


            #get predictions
            code_output_gex,outputs_gex_adt, outputs_gex_gex = model_coupled(cur_batch_gex)
            code_output_adt,outputs_adt_adt, outputs_adt_gex = model_coupled(cur_batch_adt)


            #train_loss mse
            train_loss_mse_gex = criterion_mse(outputs_gex_gex, cur_batch_gex)
            train_loss_mse_adt = criterion_mse(outputs_adt_adt, cur_batch_adt)

            #train_loss pairwise
            train_loss_pairwise_gex = criterion_pairwise(code_output_gex,cur_batch_gex)
            train_loss_pairwise_adt = criterion_pairwise(code_output_adt,cur_batch_adt)

            #train loss cross modal
            loss_adt_to_gex = criterion_mse(outputs_adt_gex,cur_batch_gex)
            loss_gex_to_adt = criterion_mse(outputs_gex_adt,cur_batch_adt)

            #final train loss
            train_loss = (train_loss_mse_gex+train_loss_mse_adt+train_loss_pairwise_gex
                          +train_loss_pairwise_adt+loss_adt_to_gex+loss_gex_to_adt)


            train_loss.backward()

            optimizer_gex.step()
            loss += train_loss.item()

        loss = loss / num_points

        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, 5, loss))
        x.append(loss)
        y.append(epoch+1)
        
    return model_coupled