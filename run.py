#!/usr/bin/env python

import sys
import os
import json

sys.path.insert(0, 'src')

from etl import read_data, get_data_test_adt, get_data_test_gex
from model import predict_mod, predict_crossmodal
from features import convert_sparse_matrix_to_sparse_tensor, normalize_data
from train import get_train_coupled
import torch
from torch import cuda

def main(targets):
    
    # reset our model by initializing it to None, if it has previously been made 
    coupled_model = None
    
    if 'test' in targets:
        
        if coupled_model != None:
            coupled_model == None
        
        with open('config/data-params.json') as fh:
            data_cfg = json.load(fh)
        with open('config/train-params.json') as ft:
            train_cfg = json.load(ft)
        
        # load training data
        data_gex = read_data(data_cfg['indir'], file='train_data_gex.npz')
        data_adt = read_data(data_cfg['indir'], file='train_data_adt.npz')
        
        #train model
        coupled_model = get_train_coupled(data_gex, data_adt, train_cfg['epoch_count'])
        
        # get the test data
        test_data_adt= get_data_test_adt()
        test_data_gex = get_data_test_gex()

        
        #predict cross modal losses 
        coupled_loss_gex_adt = predict_crossmodal(coupled_model, test_data_gex, test_data_adt, 'adt').item()
        coupled_loss_adt_gex = predict_crossmodal(coupled_model, test_data_adt, test_data_gex, 'gex').item()
        
        print("Loss of GEX to ADT: "+str(coupled_loss_gex_adt))
        print("Loss of ADT to GEX: "+str(coupled_loss_adt_gex))

        return [coupled_loss_gex_adt, coupled_loss_adt_gex]
    
    
    # training this autoencoder model on the cell data is complicated due to the size of the data, so
    # test-full is if you actually want to run the training on a sufficient amount of data. This will
    # take a very, very long time.
    
    if 'test-full' in targets:
        
        with open('config/data-params.json') as fh:
            data_cfg = json.load(fh)
        with open('config/train-params.json') as ft:
            train_cfg = json.load(ft)
        
        # load training data
        train_gex, train_adt, cell_types, \
        test_gex, test_adt, cell_types_test = process_full_data(data_cfg['indir'], data_cfg['training_size'])
        
        # train the model
        coupled_model = get_train_coupled(data_gex, data_adt, train_cfg['epoch_count'])
        
        # evaluate cross-modal prediction losses
        coupled_loss_gex_adt = predict_crossmodal(coupled_model, normalize(test_data_gex), test_data_adt, 'adt').item()
        coupled_loss_adt_gex = predict_crossmodal(coupled_model, test_data_adt, test_data_gex, 'gex').item()
        
        print("Loss of GEX to ADT: "+str(coupled_loss_gex_adt))
        print("Loss of ADT to GEX: "+str(coupled_loss_adt_gex))
    
        return [coupled_loss_gex_adt, coupled_loss_adt_gex]
    
    # this can be run if you get a cuda memory error.
    
    if 'clear-cache' in targets:
        
        torch.cuda.empty_cache()
        
if __name__ == '__main__':
    # run via:
    # python main.py data features model
    targets = sys.argv[1:]
    main(targets)