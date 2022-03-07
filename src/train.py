
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
from model import AE_coupled

with open('config/train-params.json') as fh:
    train_cfg = json.load(fh)

# loss function of pairwise distance between latent space and original space

def pairwise(code,curbatch, epochs):
    d_embedding = torch.pdist(code)
    d_org = torch.pdist(curbatch)
    los = nn.MSELoss()
    return los(d_embedding,d_org)


def get_train_coupled(gex, adt):
    print("STARTING AUTOENCODER MODEL TRAINING")
    
    # initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_coupled = AE_coupled().to(device)
    
    # create an optimizer object
    optimizer_gex = optim.Adam(model_coupled.parameters(), lr=train_cfg['learning_rate'])
      
    # declare which losses to use
    criterion_mse = nn.MSELoss()
    criterion_pairwise = pairwise
    
    gex = gex.to(device)
    adt = adt.to(device)
    
    normalize_gex = normalize_data(gex)
    
    num_points = gex.shape[0]

    for epoch in range(epochs):
        loss = 0
        permutation = torch.randperm(df1.shape[0])
        
        for i in range(0, num_points, train_cfg['batch_size']):
            indices = permutation[i:i+train_cfg['batch_size']]
            cur_batch_gex = normalize_gex[indices]
            cur_batch_adt = adt[indices]
            org_cur_batch = gex[indices]
            optimizer_gex.zero_grad()

            #get predictions
            code_output_gex,outputs_gex_adt,outputs_gex_gex = model_coupled(cur_batch_gex, False)
            code_output_adt,outputs_adt_adt, outputs_adt_gex = model_coupled(cur_batch_adt, True)

            #train_loss mse
            train_loss_mse_gex = criterion_mse(outputs_gex_gex, org_cur_batch) * train_cfg['gex_mse_weight']
            train_loss_mse_adt = criterion_mse(outputs_adt_adt, cur_batch_adt)

            #train_loss pairwise
            train_loss_pairwise_gex = criterion_pairwise(code_output_gex,cur_batch_gex)
            train_loss_pairwise_adt = criterion_pairwise(code_output_adt,cur_batch_adt)

            #train loss cross modal
            loss_adt_to_gex = criterion_mse(outputs_adt_gex,org_cur_batch)
            loss_gex_to_adt = criterion_mse(outputs_gex_adt,cur_batch_adt)
            #+loss_adt_to_gex+loss_gex_to_adt

            #train loss cross modal latent space
            loss_mse_latent = criterion_mse(code_output_adt,code_output_gex)

            #final train loss
            train_loss = (train_loss_mse_adt+train_loss_mse_gex+
                          loss_mse_latent+train_loss_pairwise_gex+train_loss_pairwise_adt+loss_adt_to_gex+loss_gex_to_adt)

            train_loss.backward()

            optimizer_gex.step()
            loss += train_loss.item()

        loss = loss / num_points

    return model_coupled