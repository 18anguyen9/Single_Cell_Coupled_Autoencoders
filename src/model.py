from scipy import sparse
import sys
import os
#import json


#import anndata as ad
import numpy as np
#import pandas as pd
#import logging
import random


import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics.pairwise import euclidean_distances
from scipy import sparse

    
# create coupled autoencoder model    
class AE_coupled(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # create layers
        # adt
        self.encoder_input_layer_adt = nn.Linear(in_features=134, out_features=100)
        self.encoder_layer64_adt = nn.Linear(in_features=100, out_features=64)
        self.encoder_layer32_adt= nn.Linear(in_features=64, out_features=32)
        
        self.decoder_layer32d_adt = nn.Linear(in_features=2, out_features=32)
        self.decoder_layer64d_adt = nn.Linear(in_features=32, out_features=64)
        self.decoder_layer100d_adt = nn.Linear(in_features=64, out_features=100)
        self.decoder_output_layer_adt = nn.Linear(in_features=100, out_features=134)
        
        #CODE LAYER
        self.encoder_layer_code = nn.Linear(in_features=32, out_features=2)
        
        #GEX
        self.encoder_input_layer_gex= nn.Linear(in_features=13953, out_features=10000)
        self.encoder_layer6000_gex= nn.Linear(in_features=10000, out_features=6000)
        self.encoder_layer3000_gex= nn.Linear(in_features=6000, out_features=3000)
        self.encoder_layer1000_gex= nn.Linear(in_features=3000, out_features=1000)
        self.encoder_layer256_gex= nn.Linear(in_features=1000, out_features=256)
        self.encoder_layer128_gex = nn.Linear(in_features=256, out_features=128)
        self.encoder_layer64_gex = nn.Linear(in_features=128, out_features=64)
        self.encoder_layer32_gex = nn.Linear(in_features=64, out_features=32)
        
        
        self.decoder_layer32d_gex = nn.Linear(in_features=2, out_features=32)
        self.decoder_layer64d_gex = nn.Linear(in_features=32, out_features=64)
        self.decoder_layer128d_gex = nn.Linear(in_features=64, out_features=128)
        self.decoder_layer256d_gex = nn.Linear(in_features=128, out_features=256)
        self.decoder_layer1000d_gex = nn.Linear(in_features=256, out_features=1000)
        self.decoder_layer3000d_gex = nn.Linear(in_features=1000, out_features=3000)
        self.decoder_layer6000d_gex = nn.Linear(in_features=3000, out_features=6000)
        self.decoder_layer10000d_gex = nn.Linear(in_features=6000, out_features=10000)
        self.decoder_output_layer_gex = nn.Linear(in_features=10000, out_features=13953)
    
    def adt_to_code(self,features):
        activation = self.encoder_input_layer_adt(features)
        activation = torch.relu(activation)
        activation = self.encoder_layer64_adt(activation)
        activation = torch.relu(activation)
        activation = self.encoder_layer32_adt(activation)
        activation = torch.relu(activation)
        code = self.encoder_layer_code(activation)
        return code
    
    def gex_to_code(self,features):
        activation = self.encoder_input_layer_gex(features)
        activation = self.encoder_layer6000_gex(activation)
        activation = torch.relu(activation)
        activation = self.encoder_layer3000_gex(activation)
        activation = torch.relu(activation)
        activation = self.encoder_layer1000_gex(activation)
        activation = torch.relu(activation)
        activation = self.encoder_layer256_gex(activation)
        activation = torch.relu(activation)
        activation = self.encoder_layer128_gex(activation)
        activation = torch.relu(activation)
        activation = self.encoder_layer64_gex(activation)
        activation = torch.relu(activation)
        activation = self.encoder_layer32_gex(activation)
        activation = torch.relu(activation)
        code = self.encoder_layer_code(activation)
        return code
    
    def code_to_adt(self,code):
        activation = torch.relu(code)
        activation = self.decoder_layer32d_adt(activation)
        activation = torch.relu(activation)
        activation = self.decoder_layer64d_adt(activation)
        activation = torch.relu(activation)
        activation = self.decoder_layer100d_adt(activation)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer_adt(activation)
        return activation
    
    def code_to_gex(self,code):
        activation = torch.relu(code)
        activation = self.decoder_layer32d_gex(activation)
        activation = torch.relu(activation)
        activation = self.decoder_layer64d_gex(activation)
        activation = torch.relu(activation)
        activation = self.decoder_layer128d_gex(activation)
        activation = torch.relu(activation)
        activation = self.decoder_layer256d_gex(activation)
        activation = torch.relu(activation)
        activation = self.decoder_layer1000d_gex(activation)
        activation = torch.relu(activation)
        activation = self.decoder_layer3000d_gex(activation)
        activation = torch.relu(activation)
        activation = self.decoder_layer6000d_gex(activation)
        activation = torch.relu(activation)
        activation = self.decoder_layer10000d_gex(activation)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer_gex(activation)
        return activation
    
    def forward(self, features):
        # encode
        num_dim = features.shape[1]
        if num_dim == 134:
            code= self.adt_to_code(features)
        elif num_dim==13953:
            code= self.gex_to_code(features)
        else:
            return "invalid input"
        
        #decode
        output_layer_adt = self.code_to_adt(code)
        output_layer_gex = self.code_to_gex(code)
        return code, output_layer_adt, output_layer_gex
   
    
#compute reconstruction loss
def predict_mod(mod,test_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = test_data.to(device)
    prediction = mod(data)[1]
    return nn.MSELoss()(data, prediction)

# compute reconstruction loss of a translation made from one modality
# to another, specifically for the coupled autoencoder

# modality has two possiblities: 'adt', 'gex', denoting the type
# passed into 'test_data'
def predict_crossmodal(mod, test_data, eval_data, modality):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data = test_data.to(device)
    translation = eval_data.to(device)
    
    if modality == 'gex':
        prediction = mod(data)[1]
        return nn.MSELoss()(translation, prediction)
        
    if modality == 'adt':
        prediction = mod(data)[-1]
        return nn.MSELoss()(translation, prediction)
    else:
        print('Please choose the modality of the test data.')
    
