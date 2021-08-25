import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

def convert_matrix(gt, gt_map, alt_map):
    alt_keys = list(alt_map.keys())
    gt_indexing = [gt_map[i] for i in alt_keys]
    a = gt[:,gt_indexing][gt_indexing,:]
    return a

def knn_augment(b_mat, sim_mat, sim_item_map, train_item_map, test_items, data,
                n_neighbors, metric,
                gt_mat = None, gt_item_map = None, verbose=False,
                method = 'weighted'):
    '''
    Identifies KNN for test items and iteratively augments matrix B based on
    popularity of KNN (based on number of views in training set)
    
    inputs:
        b_mat: 2D array of matrix B, trained on training set
        sim_mat: 2D array of features matrix
        sim_item_map: pd.Series with index ItemId and corresponding index for sim_mat
        train_item_map: pd.Series with index ItemId and corresponding index for b_mat
        test_items: list, items to augment B matrix
        data: training data, for calculation of popularity
        
        n_neighbors: int, number of neighbors in KNN
        metric: euclidean / manhatten / chebyshev / minkowski (p)
        
        gt_mat: 2D array of ground truth matrix B, for calculation of MSE
        gt_item_map: dict of ItemId and corresponding index for gt_mat
        
        method: weighting method, average / weighted
        
    outputs:
        augmented_b: 2D array of matrix B, augmented with test_items
        augmented_item_map: dict of ItemId and corresponding index for augmented_b
        
    '''
    
    dists = []
    indexes = []
    pops = []
    
    # sim_item_map = dict(zip(sim_item_map.index.tolist(), sim_item_map.values.tolist()))
    # train_item_map = dict(zip(train_item_map.index.tolist(), train_item_map.values.tolist()))
    
    init_items = list(train_item_map.keys())
    init_items = [i for i in init_items if i in sim_item_map.index.tolist()]
    train_item_map2 = dict(map(reversed, train_item_map.items()))
    n_init = b_mat.shape[0]

    similarity_init = sim_mat[:, sim_item_map.loc[init_items]][sim_item_map.loc[init_items],:].copy()
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(similarity_init)
    
    for test_item in test_items:
        test_point = sim_mat[sim_item_map[test_item], sim_item_map[init_items]].reshape(1,-1)
        dist, nn_idx = neigh.kneighbors(test_point)
        dist, nn_idx = dist.tolist()[0], nn_idx.tolist()[0]
        nn_items = [train_item_map2[i] for i in nn_idx]
        
        products = data.groupby('ItemId').size()
        pop = sum(products[nn_items])
        
        dists.append(dist)
        indexes.append(nn_idx)
        pops.append(pop)
    
    nn_df = pd.DataFrame({'ItemId': test_items, 'Distances': dists, 'Indexes': indexes,
                          'Views': pops})
    nn_df = nn_df.sort_values(by='Views', ascending=False, ignore_index=True)
    
    augmented_items = train_item_map.index.tolist()
    augmented_b = b_mat.copy()
    
    if method == 'average':
        for i in range(len(test_items)):
            augmented_items += [nn_df.ItemId[i]]
            # add augmented row
            augmented_row = np.mean(augmented_b[nn_df.Indexes[i],:], axis=0)
            augmented_b = np.vstack((augmented_b, augmented_row))
            # add augmented col
            augmented_col = np.mean(augmented_b[:-1,nn_df.Indexes[i]], axis=1)
            diag_entry = np.mean([augmented_b[j,j] for j in nn_df.Indexes[i]])
            augmented_col = np.hstack((augmented_col, [diag_entry]))
            augmented_b = np.hstack((augmented_b, augmented_col.reshape(-1,1)))
    
    elif method == 'weighted':
        for i in range(len(test_items)):
            augmented_items += [nn_df.ItemId[i]]
            augmented_row = np.zeros(n_init+i)
            augmented_col = np.zeros(n_init+i)
            diag_entry = 0
            total_weights = 0
            
            for j in range(n_neighbors):    
                weight = 1/nn_df.Distances[i][j]
                nn_idx = nn_df.Indexes[i][j]
                total_weights += weight
                diag_entry += weight * augmented_b[nn_idx, nn_idx]
                augmented_row += weight * augmented_b[nn_idx,:]
                augmented_col += weight * augmented_b[:, nn_idx]
            # add augmented row
            augmented_row /= total_weights
            augmented_b = np.vstack((augmented_b, augmented_row))
            # add augmented col
            augmented_col = np.hstack((augmented_col, [diag_entry]))
            augmented_col /= total_weights
            augmented_b = np.hstack((augmented_b, augmented_col.reshape(-1,1)))
    
    augmented_item_map = pd.Series(data=np.arange(len(augmented_items)), index=augmented_items)
    # augmented_item_map2 = dict(map(reversed, augmented_item_map.items()))
    
    if gt_mat is None or gt_item_map is None:
        return augmented_b, augmented_item_map
    else:
        # calculate MSE
        # convert augmented_b to have the same index as train_b (ground truth)
        gt_mat = convert_matrix(gt_mat, gt_item_map, augmented_item_map)
        mse = np.mean((gt_mat - augmented_b)**2, axis=None)
        if verbose:
            print('MSE:', mse)
        return augmented_b, augmented_item_map, mse