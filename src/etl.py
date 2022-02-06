import json
import os
import sys

sys.path.insert(0, 'src')
import torch
import scipy
from scipy import sparse
from features import convert_sparse_matrix_to_sparse_tensor,normalize_data

# read data 
def read_data(outdir, file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    fp = os.path.join(outdir, file)
    torch_gex_data = convert_sparse_matrix_to_sparse_tensor(scipy.sparse.load_npz(fp)).to(device)
    #if torch_gex_data.shape[1]==13953:
        #return normalize_data(torch_gex_data).to(device)
    #else:
    return torch_gex_data

# get test data for adt
def get_data_test_adt():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    adt_subset = sparse.load_npz(os.path.abspath(os.getcwd())+"/test/testdata/test_data_adt.npz")
    torch_data = convert_sparse_matrix_to_sparse_tensor(adt_subset)
    return torch_data

# get test data for gex
def get_data_test_gex():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    gex_subset = sparse.load_npz(os.path.abspath(os.getcwd())+"/test/testdata/test_data_gex.npz")
    torch_data = convert_sparse_matrix_to_sparse_tensor(gex_subset).to(device)
    return (torch_data)