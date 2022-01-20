#!/usr/bin/env python

import sys
import os
import json

sys.path.insert(0, 'src')

from etl import read_data,get_data_test_adt,get_data_test_gex
from model import predict_mod
from features import convert_sparse_matrix_to_sparse_tensor
from train import get_train_gex, get_train_adt
import torch



def main(targets):
    # intialize our models to None 
    adt_model = None
    gex_model = None
    coupled_model = None
    
    if 'test' in targets:
        with open('config/data-params.json') as fh:
            data_cfg = json.load(fh)
        
        # load training data
        data_gex = read_data(**data_cfg, file='train_data_gex.npz')
        data_adt = read_data(**data_cfg, file='train_data_adt.npz')
    
        # train models if model not trained already
        if gex_model == None:
            gex_model = get_train_gex(data_gex)
            
        if adt_model== None:
            adt_model = get_train_adt(data_adt)
        
        if coupled_model == None:
            coupled_model = get_train_coupled(data_gex, data_adt)
        
        # get the test data
        test_data_adt= get_data_test_adt()
        test_data_gex = get_data_test_gex()

        # predict adt and compute loss
        loss_test_adt = predict_mod(adt_model, test_data_adt).item()
        print("loss of adt test set: " +str(loss_test_adt))

        # predict gex and compute loss
        loss_test_gex = predict_mod(gex_model, test_data_gex).item()
        print("loss of gex test set: " +str(loss_test_gex))
        
        # in the coupled autoencoder, predict the cross modal losses
        # (ADT to GEX, GEX to ADT) and the losses within modalities
        # (ADT to ADT, GEX to GEX)
        
        coupled_loss_gex_adt = predict_crossmodal(coupled_model, test_data_gex, test_data_adt, 'gex').item()
        coupled_loss_adt_gex = predict_crossmodal(coupled_model, test_data_adt, test_data_gex, 'adt').item()
        
        coupled_loss_adt_adt = predict_mod(coupled_model, test_data_adt).item()
        coupled_loss_gex_gex = predict_mod(coupled_model, test_data_gex).item()
              
        return [loss_test_adt, loss_test_gex, loss_test_gex_adt, coupled_loss_gex_adt, coupled_loss_adt_gex \
                coupled_loss_adt_adt, coupled_loss_gex_gex]

if __name__ == '__main__':
    # run via:
    # python main.py data features model
    targets = sys.argv[1:]
    main(targets)