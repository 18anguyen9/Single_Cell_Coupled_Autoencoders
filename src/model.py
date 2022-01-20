from scipy import sparse
import sys
import os
#import json


#import anndata as ad
import numpy as np
#import pandas as pd
#import logging
import random

'''
import plotly.io as plt_io
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='ticks')'''

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics.pairwise import euclidean_distances
from scipy import sparse

# create ADT autoencoder model
class AE_adt(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # create layers
        self.encoder_hidden_layer = nn.Linear(in_features=kwargs["input_shape"], out_features=100)
        self.encoder_output_layer100 = nn.Linear(in_features=100, out_features=64)
        self.normalize = nn.LayerNorm(64)
        self.encoder_output_layer64 = nn.Linear(in_features=64, out_features=32)
        
        self.encoder_output_layer32 = nn.Linear(in_features=32, out_features=2)
        
        self.decoder_hidden_layer32d = nn.Linear(in_features=2, out_features=32)
        self.decoder_hidden_layer64d = nn.Linear(in_features=32, out_features=64)
        self.decoder_hidden_layer100d = nn.Linear(in_features=64, out_features=100)
        self.decoder_output_layer = nn.Linear(in_features=100, out_features=kwargs["input_shape"])

    def forward(self, features):
        # create network
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        activation = self.encoder_output_layer100(activation)
        activation = torch.relu(activation)
        
        activation = self.normalize (activation)
        
        activation = self.encoder_output_layer64(activation)
        activation = torch.relu(activation)
        code = self.encoder_output_layer32(activation)
        
        activation = self.decoder_hidden_layer32d(code)
        activation = torch.relu(activation)
        activation = self.decoder_hidden_layer64d(activation)
        activation = torch.relu(activation)
        activation = self.decoder_hidden_layer100d(activation)
        activation = torch.relu(activation)
        
        activation = self.decoder_output_layer(activation)
        return [code,activation]
# create GEX autoencoder model
class AE_gex(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # create layers
        self.encoder_hidden_layer256 = nn.Linear(in_features=kwargs["input_shape"], out_features=1000)
        self.normalize = nn.LayerNorm(1000)
        self.encoder_hidden_layer128 = nn.Linear(in_features=1000, out_features=128)
        self.encoder_hidden_layer64 = nn.Linear(in_features=128, out_features=64)
        self.encoder_output_layer = nn.Linear(in_features=64, out_features=2)
        
        self.decoder_hidden_layer = nn.Linear(in_features=2, out_features=64)
        self.decoder_hidden_layer64d = nn.Linear(in_features=64, out_features=128)
        self.decoder_hidden_layer128d = nn.Linear(in_features=128, out_features=1000)
        self.decoder_output_layer = nn.Linear(in_features=1000, out_features=kwargs["input_shape"])

    def forward(self, features):
        # create network
        activation = self.encoder_hidden_layer256(features)
        activation = torch.relu(activation)
        activation = self.normalize(activation)
        activation = self.encoder_hidden_layer128(activation)
        activation = torch.relu(activation)
        activation = self.encoder_hidden_layer64(activation)
        code = self.encoder_output_layer(activation)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_hidden_layer64d(activation)
        activation = torch.relu(activation)
        activation = self.decoder_hidden_layer128d(activation)
        activation = torch.relu(activation)    
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return [code,activation]
    
#compute reconstruction loss
def predict_mod(mod,test_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prediction = mod(test_data.to(device))[1]
    return nn.MSELoss()(test_data.to(device),prediction)

