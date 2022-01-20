from scipy import sparse
import sys
import os

import numpy as np
import pandas as pd


import torch
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics.pairwise import euclidean_distances
from scipy import sparse

# helper function to convert matrix to torch tensor

def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    torch_data = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()
    return torch_data

# vector normalizes each row in a data set (takes in a matrix of tensors)
def normalize_data(data):
    return torch.from_numpy(np.array([(tf.linalg.normalize(i)[0].numpy()) for i in gex_torch_data])).to(device)
    
    
    
    
    
    
    
    
    
    
    
    