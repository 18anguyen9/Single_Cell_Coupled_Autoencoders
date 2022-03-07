import json
import os
import sys

sys.path.insert(0, 'src')
import scipy
from scipy import sparse
from features import convert_sparse_matrix_to_sparse_tensor,normalize_data

with open('data-params.json') as f:
    data_cfg = json.load(f)

# read data 
def read_data(loc, file):
    fp = os.path.join(loc, file)
    torch_gex_data = convert_sparse_matrix_to_sparse_tensor(scipy.sparse.load_npz(fp))
    return torch_gex_data

# get test data for adt
def get_data_test_adt():
    adt_subset = sparse.load_npz(os.path.abspath(os.getcwd())+"/test/testdata/test_data_adt.npz")
    torch_data = convert_sparse_matrix_to_sparse_tensor(adt_subset)
    return torch_data

# get test data for gex
def get_data_test_gex():
    gex_subset = sparse.load_npz(os.path.abspath(os.getcwd())+"/test/testdata/test_data_gex.npz")
    torch_data = convert_sparse_matrix_to_sparse_tensor(gex_subset)
    return torch_data

def process_full_data(loc, training_size):
    
    adata_gex = ad.read_h5ad(loc + "cite_gex_processed_training.h5ad")
    adata_adt = ad.read_h5ad(loc + "cite_adt_processed_training.h5ad")
    train_cells = adata_gex.obs_names[adata_gex.obs["batch"] != "s1d2"]
    test_cells  = adata_gex.obs_names[adata_gex.obs["batch"] == "s1d2"]
    input_train_mod1 = adata_gex[train_cells]
    input_train_mod2 = adata_adt[train_cells]
    input_test_mod1 =  adata_gex[test_cells]

    true_test_mod2 =  adata_adt[test_cells]
    number_of_rows = input_train_mod1.X.shape[0]
    random_indices = np.random.choice(number_of_rows, size=training_size, replace=False)
    gex_subset = input_train_mod1.X[random_indices, :]
    adt_subset = input_train_mod2.X[random_indices, :]

    gex_data = convert_sparse_matrix_to_sparse_tensor(gex_subset)
    adt_data = convert_sparse_matrix_to_sparse_tensor(adt_subset)
    cell_types = input_train_mod1.obs.iloc[random_indices, :]['cell_type'].values
    
    gex_test = convert_sparse_matrix_to_sparse_tensor(input_test_mod1.X)
    adt_test = convert_sparse_matrix_to_sparse_tensor(true_test_mod2.X)
    cell_types_test = input_test_mod1.obs['cell_type'].values
    
    return gex_train, adt_train, cell_types, gex_test, adt_test, cell_types_test

    