import pandas as pd 
import numpy as np
import sys
from collections import defaultdict
from datetime import datetime
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix, coo_matrix, vstack
from tqdm import tqdm


def generate_csr_matrix(meta_df, colname, ncols, alpha = 1., item_col='product_id'):
    ''' Generate Metadata-to-item mapping in the form of a CSR matrix. '''
    data = np.ones(meta_df.shape[0]) * alpha
    rows, cols = meta_df[colname].values, meta_df[item_col].values
    nrows = meta_df[colname].max() + 1
    return csr_matrix((data, (rows, cols)), shape = (int(nrows), int(ncols)))

def normalize_idf(X):
    ''' Normalize matrix X according to column-wise IDF. '''
    # Log-normalised Smoothed Inverse Document Frequency
    row_counts = X.sum(axis = 1)
    row_counts -= (row_counts.min() - 2.0) # Start from 0 for more expressive log-scale
    idf = (1.0 / np.log(row_counts)).A1.ravel()
    return csr_matrix(np.diag(idf)) @ X

def compute_sparsity(A):
    ''' Compute the sparsity level (% of non-zeros) of matrix A. '''
    return 1.0 - np.count_nonzero(A) / (A.shape[0] * A.shape[1])

def sparsify(B, rho = .95):
    ''' Get B to the required sparsity level by dropping out the rho % lower absolute values. '''
    min_val = np.quantile(np.abs(B), rho)
    B[np.abs(B) < min_val] = .0
    return B
