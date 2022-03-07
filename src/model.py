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


with open('config/model-params.json') as f:
    model_cfg = json.load(f)
    
# create coupled autoencoder model 

class AE_coupled(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        
        # create layers / design architecture
        
        # ADT / encoding
        self.encoder_input_layer_adt = nn.Linear(in_features=model_cfg['adt_dims'], \
                                                 out_features=model_cfg['adt_dims_layer1'])
        
        self.encoder_layer1_adt = nn.Linear(in_features=model_cfg['adt_dims_layer1'], 
                                            out_features=model_cfg['adt_dims_layer2'])
        
        self.encoder_layer2_adt= nn.Linear(in_features=model_cfg['adt_dims_layer2'], \
                                           out_features=model_cfg['latent_dims_input'])
        
        # ADT / decoding

        self.decoder_layer3_adt = nn.Linear(in_features=model_cfg['latent_dims'], \
                                            out_features=model_cfg['latent_dims_input'])
        
        self.decoder_layer2_adt = nn.Linear(in_features=model_cfg['latent_dims_input'], \
                                            out_features=model_cfg['adt_dims_layer2'])
        
        self.decoder_layer1_adt = nn.Linear(in_features=model_cfg['adt_dims_layer2'], \
                                            out_features=model_cfg['adt_dims_layer1'])
        
        self.decoder_output_layer_adt = nn.Linear(in_features=model_cfg['adt_dims_layer1'], \
                                                  out_features=model_cfg['adt_dims'])

        # CODE LAYER / LATENT SPACE
        self.encoder_layer_code = nn.Linear(in_features=model_cfg['latent_dims_input'], \
                                            out_features=model_cfg['latent_dims'])

        #GEX / encoding
        self.encoder_input_layer_gex= nn.Linear(in_features=model_cfg['gex_dims'], \
                                                out_features=model_cfg['gex_dims_layer1'])
        
        self.encoder_layer1_gex= nn.Linear(in_features=model_cfg['gex_dims_layer1'],\
                                           out_features=model_cfg['gex_dims_layer2'])
        
        self.encoder_layer2_gex= nn.Linear(in_features=model_cfg['gex_dims_layer2'], \
                                           out_features=model_cfg['gex_dims_layer3'])
        
        self.encoder_layer3_gex= nn.Linear(in_features=model_cfg['gex_dims_layer3'], \
                                           out_features=model_cfg['gex_dims_layer4'])
        
        self.encoder_layer4_gex= nn.Linear(in_features=model_cfg['gex_dims_layer4'], \
                                           out_features=model_cfg['gex_dims_layer5'])
        
        self.encoder_layer5_gex = nn.Linear(in_features=model_cfg['gex_dims_layer5'], \
                                            out_features=model_cfg['gex_dims_layer6'])
        
        self.encoder_layer6_gex = nn.Linear(in_features=model_cfg['gex_dims_layer6'], \
                                            out_features=model_cfg['gex_dims_layer7'])
        
        self.encoder_layer7_gex = nn.Linear(in_features=model_cfg['gex_dims_layer7'], \
                                            out_features=model_cfg['latent_dims_input'])
        
        #GEX / decoding

        self.decoder_layer8_gex = nn.Linear(in_features=model_cfg['latent_dims'], \
                                            out_features=model_cfg['latent_dims_input'],)
        
        self.decoder_layer7_gex = nn.Linear(in_features=model_cfg['latent_dims_input'],\
                                            out_features=model_cfg['gex_dims_layer7'])
        
        self.decoder_layer6_gex = nn.Linear(in_features=model_cfg['gex_dims_layer7'], \
                                            out_features=model_cfg['gex_dims_layer6'])
        
        self.decoder_layer5_gex = nn.Linear(in_features=model_cfg['gex_dims_layer6'], \
                                            out_features=model_cfg['gex_dims_layer5'])
        
        self.decoder_layer4_gex = nn.Linear(in_features=model_cfg['gex_dims_layer5'], \
                                            out_features=model_cfg['gex_dims_layer4'])
        
        self.decoder_layer3_gex = nn.Linear(in_features=model_cfg['gex_dims_layer4'], \
                                            out_features=model_cfg['gex_dims_layer3'])
        
        self.decoder_layer2_gex = nn.Linear(in_features=model_cfg['gex_dims_layer3'], \
                                            out_features=model_cfg['gex_dims_layer2'])
        
        self.decoder_layer1_gex = nn.Linear(in_features=model_cfg['gex_dims_layer2'], \
                                            out_features=model_cfg['gex_dims_layer1'])
        
        self.decoder_output_layer_gex = nn.Linear(in_features=model_cfg['gex_dims_layer1'], \
                                                  out_features=model_cfg['gex_dims'])

    # connect the layers together for encoding ADT
    def adt_to_code(self,features):
        activation = self.encoder_input_layer_adt(features)
        activation = torch.relu(activation)
        activation = self.encoder_layer1_adt(activation)
        activation = torch.relu(activation)
        activation = self.encoder_layer2_adt(activation)
        activation = torch.relu(activation)
        code = self.encoder_layer_code(activation)
        return code
    
    # connect the layers for encoding GEX
    def gex_to_code(self,features):
        activation = self.encoder_input_layer_gex(features)
        activation = self.encoder_layer1_gex(activation)
        activation = torch.relu(activation)
        activation = self.encoder_layer2_gex(activation)
        activation = torch.relu(activation)
        activation = self.encoder_layer3_gex(activation)
        activation = torch.relu(activation)
        activation = self.encoder_layer4_gex(activation)
        activation = torch.relu(activation)
        activation = self.encoder_layer5_gex(activation)
        activation = torch.relu(activation)
        activation = self.encoder_layer6_gex(activation)
        activation = torch.relu(activation)
        activation = self.encoder_layer7_gex(activation)
        activation = torch.relu(activation)
        code = self.encoder_layer_code(activation)
        return code
    
    # connect the layers for decoding ADT
    def code_to_adt(self,code):
        activation = torch.relu(code)
        activation = self.decoder_layer3_adt(activation)
        activation = torch.relu(activation)
        activation = self.decoder_layer2_adt(activation)
        activation = torch.relu(activation)
        activation = self.decoder_layer1_adt(activation)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer_adt(activation)
        return activation
    
    # connect the layers for decoding GEX
    def code_to_gex(self,code):
        activation = torch.relu(code)
        activation = self.decoder_layer8_gex(activation)
        activation = torch.relu(activation)
        activation = self.decoder_layer7_gex(activation)
        activation = torch.relu(activation)
        activation = self.decoder_layer6_gex(activation)
        activation = torch.relu(activation)
        activation = self.decoder_layer5_gex(activation)
        activation = torch.relu(activation)
        activation = self.decoder_layer4_gex(activation)
        activation = torch.relu(activation)
        activation = self.decoder_layer3_gex(activation)
        activation = torch.relu(activation)
        activation = self.decoder_layer2_gex(activation)
        activation = torch.relu(activation)
        activation = self.decoder_layer1_gex(activation)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer_gex(activation)
        return activation
    
    # normalize the GEX data
    def norm_batch(self, batch_gex):
        gex_torch_data_norm = torch.nn.functional.normalize(batch_gex,p=2, dim=1)
        return gex_torch_data_norm
    
    # take the learned factor for scaling GEX to unnormalize GEX to its
    # original scale
    
    def un_norm_gex(self,gex_output):
        main_chunk = gex_output[:, :model_cfg['gex_dims']]
        scale_col = gex_output[:, model_cfg['gex_dims']:model_cfg['gex_dims']+1]
        return main_chunk*scale_col
    
    
    # direct modality to the correct "path" of layers in the autoencoder
    def forward(self, features,to_adt):
        # encode
        num_dim = features.shape[1]
        
        if num_dim == model_cfg['adt_dims']:
            code = self.adt_to_code(features)
        elif num_dim==model_cfg['gex_dims']:

            code = self.gex_to_code((features))
        else:
            return "invalid input"

        output_layer_adt = self.code_to_adt(code)
        output_layer_gex = self.code_to_gex(code)

        output_layer_gex = self.un_norm_gex(output_layer_gex)

        return code,output_layer_adt,output_layer_gex
   
    
#compute reconstruction loss

def predict_mod(mod,test_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = test_data.to(device)
    prediction = mod(data)[1]
    return nn.MSELoss()(data, prediction)


# compute losses between modality and target modality (cross-modal prediction)

def predict_crossmodal(mod, test_data, eval_data, target_modality):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_metric = nn.MSELoss()
    
    data = test_data.to(device)
    translation = eval_data.to(device)
    
    if target_modality == 'adt':
        prediction = mod(data)[1]
        return np.sqrt(loss_metric(translation, prediction))
        
    if target_modality == 'gex':
        prediction = mod(data)[-1]
        return np.sqrt(loss_metric(translation, prediction))
    else:
        print('Please choose the modality of the test data.')
    
