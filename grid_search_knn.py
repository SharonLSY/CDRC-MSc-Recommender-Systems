from pathlib import Path
import sys
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import seaborn as sns

from algorithms.slist import SLIST
from evaluation.loader import load_data_session
from utils.knn_utils import *

'''
FILE PARAMETERS
'''
folder = 'test_date_trial/'
PATH_PROCESSED = './data/prepared/' + folder
FILE = 'test'

'''
SLIST OPTIMAL HYPERPARAMETERS
'''
alpha = 0.2 #[0.2, 0.4, 0.6, 0.8] 
direction = 'all' # sr / part / all
reg = 10
train_weight = 1 #0.5 #[0.125, 0.25, 0.5, 1, 2, 4, 8]
predict_weight = 1 #4 #[0.125, 0.25, 0.5, 1, 2, 4, 8]
session_weight = 1 #256 #[1, 2, 4, 8, 16, 32, 64, 128, 256]

'''
Item Features Matrix
'''
prod2vec_file = './data/prod2vec.csv'

# setting up ground truth B
train, test = load_data_session(PATH_PROCESSED, FILE, train_eval=False)
model = SLIST(alpha=alpha, direction=direction, reg=reg, train_weight=train_weight, 
                    predict_weight=predict_weight, session_weight=session_weight)
model.fit(train)
alt_item_map = model.itemidmap
alt_item_map2 = dict(map(reversed, alt_item_map.items()))
train_b = model.enc_w

# setting up prod2vec matrix & mapping
prod2vec_matrix = pd.read_csv(prod2vec_file, dtype={'ItemId': int})
prod2vec_matrix = prod2vec_matrix.loc[prod2vec_matrix['ItemId'].isin(model.itemidmap.index),:].reset_index(drop=True)
prod2vec_map = pd.Series(data=np.arange(prod2vec_matrix.shape[0]), index=prod2vec_matrix['ItemId'])
prod2vec_matrix.drop(['ItemId'], axis=1, inplace=True)
prod2vec_matrix = np.array(prod2vec_matrix.values)

nn_range = [100,200,500,800,1000]

for met in ['euclidean' , 'manhatten' , 'chebyshev']:
    mean_mse = []
    
    for nn in nn_range:
        mse_list = []
        print('n_neighbors =', nn)
        for i in tqdm(range(100)):
            n_items = model.n_items
            test_index = list(set([random.randint(0,n_items-1) for i in range(50)]))
            test_items = [alt_item_map2[i] for i in test_index]
            init_index = [i for i in list(alt_item_map2.keys()) if i not in test_index]
            init_items = [alt_item_map2[i] for i in init_index]
        
            init_b = train_b[:,init_index][init_index,:]
            
            n_init = len(init_items)
            init_item_map = pd.Series(index=init_items, data=range(n_init))
            init_item_map2 = dict(map(reversed, init_item_map.items()))
            
            _,_, mse = knn_augment(b_mat=init_b, sim_mat=prod2vec_matrix, 
                                   sim_item_map=prod2vec_map, train_item_map=init_item_map, 
                                   test_items = test_items, data=train,
                                   n_neighbors=nn, metric=met,
                                   gt_mat = train_b, gt_item_map=alt_item_map
                                   )
            mse_list.append(mse)
        mean_mse.append(np.mean(mse_list))
    sns.lineplot(x=nn_range, y=mean_mse, label = met)
plt.xlabel('Number of Neighbors')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.savefig('./plots/knn_grid_search.png', dpi=300, transparent=False, bbox_inches='tight')