
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

# loss function of pairwise distance between latent space and original space
def pairwise_loss(code,curbatch):
    d_embedding = torch.pdist(code)
    d_org = torch.pdist(curbatch)
    los = nn.MSELoss()
    denom = torch.add(d_embedding,d_org)
    ratio = torch.divide((torch.absolute(d_embedding-d_org)),denom)
    return torch.sum(torch.absolute(ratio))


def get_train_gex(df):
    # intialize model basics
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mod = AE_gex(input_shape=df.shape[1]).to(device)
    optimizer_gex = optim.Adam(mod.parameters(), lr=1e-4)
    criterion_mse = nn.MSELoss()
    criterion_pairwise = pairwise_loss

    x,y = [],[]
    num_points =df.shape[0]

    # begin training
    for epoch in range(epoch_gex):
        loss = 0
        permutation = torch.randperm(df.shape[0])
        for i in range(0,num_points,batch_size):
            indices = permutation[i:i+batch_size]
            cur_batch = df[indices]
         
            optimizer_gex.zero_grad()
            
           
            outputs = mod(cur_batch)[1]
            code_output = mod(cur_batch)[0]
            
            # compute loss
            train_loss = criterion_mse(outputs, cur_batch)
        
            train_loss.backward()
            
            
            optimizer_gex.step()
            
        
            loss += train_loss.item()
        
        # compute total loss of epoch
        loss = loss / num_points
        
        
        print("gex epoch : {}/{}, loss = {:.6f}".format(epoch + 1, 5, loss))
        x.append(loss)
        y.append(epoch+1)


    return mod

def get_train_adt(df):
    # model basics
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mod = AE_adt(input_shape=df.shape[1]).to(device)
    optimizer_adt = optim.Adam(mod.parameters(), lr=1e-4)
    criterion_mse = nn.MSELoss()
    criterion_pairwise = pairwise_loss

    x,y = [],[]
    for epoch in range(epoch_adt):
        loss = 0
        num_points = df.shape[0]
        permutation = torch.randperm(num_points)
        for i in range(0,num_points,batch_size):
            indices = permutation[i:i+batch_size]
            cur_batch = df[indices]
            optimizer_adt.zero_grad()
            
            outputs = mod(cur_batch)[1]
            code_output = mod(cur_batch)[0]
            
            train_loss_mse = criterion_mse(outputs, cur_batch)
            train_loss = criterion_pairwise(code_output,cur_batch)+train_loss_mse
            
            train_loss.backward()
            
            # step size
            optimizer_adt.step()
            
            # add loss
            loss += train_loss.item()
        
        # total loss for epoch
        loss = loss / num_points
        print("adt epoch : {}/{}, loss = {:.6f}".format(epoch + 1, 5, loss))

        x.append(loss)
        y.append(epoch+1)
    return mod