import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import random
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import matplotlib.pyplot as plt

from algorithms.slist import SLIST
from trial_knn.knn_utils import *


path = './data/trial/'

'''
MODEL HYPERPARAMETER TUNING
'''
alpha = 0.2 #[0.2, 0.4, 0.6, 0.8] -- 1 to generate similarity matrix
direction = 'all' # sr / part / all
reg = 10
train_weight = 1 #0.5 #[0.125, 0.25, 0.5, 1, 2, 4, 8]
predict_weight = 1 #4 #[0.125, 0.25, 0.5, 1, 2, 4, 8]
session_weight = 1 #256 #[1, 2, 4, 8, 16, 32, 64, 128, 256]

# generating ground truth similarity matrix
data = pd.read_csv(path+'train.csv')

# remove sessions with single view
all_sessions = data.groupby('user_session').size()
all_sessions = all_sessions[all_sessions >= 2]

data = data.loc[data.user_session.isin(all_sessions.index)]

# remove items that appear less than 5 times
all_items = data.groupby('product_id').size()
all_items = all_items[all_items >= 5]

data = data.loc[data.product_id.isin(all_items.index)]

# remove sessions with single view
all_sessions = data.groupby('user_session').size()
all_sessions = all_sessions[all_sessions >= 2]

data = data.loc[data.user_session.isin(all_sessions.index)]
data.columns = ['Date','Time','ItemId','SessionId']

## generate SLIST
model = SLIST(alpha=0.2, direction=direction, reg=reg, train_weight=train_weight, 
                    predict_weight=predict_weight, session_weight=session_weight)
model.fit(data)
alt_item_map = model.itemidmap
alt_item_map2 = dict(map(reversed, alt_item_map.items()))
train_b = model.enc_w

## weighted knn
# first, randomly choose 5 items to remove from train_b
n_items = model.n_items
test_index = [1,2,3,4,5] #[random.randint(0,n_items-1) for i in range(5)]
test_items = [alt_item_map2[i] for i in test_index]
init_index = [i for i in list(alt_item_map2.keys()) if i not in test_index]
init_items = [alt_item_map2[i] for i in init_index]

init_b = train_b[:,init_index][init_index,:]

n_init = len(init_items)
init_item_map = dict(zip(init_items, range(n_init)))
init_item_map2 = dict(map(reversed, init_item_map.items()))

# secondly, get knn index and distance from similarity

nn_range = [200,500,1000,2000] # 1000 manhatten is the best

for met in ['euclidean' , 'manhatten' , 'chebyshev']:
    mean_mse = []
    
    for nn in nn_range:
        mse_list = []
        print('n_neighbors =', nn)
        for i in tqdm(range(10)):
            n_items = model.n_items
            test_index = list(set([random.randint(0,n_items-1) for i in range(20)]))
            test_items = [alt_item_map2[i] for i in test_index]
            init_index = [i for i in list(alt_item_map2.keys()) if i not in test_index]
            init_items = [alt_item_map2[i] for i in init_index]
        
            init_b = train_b[:,init_index][init_index,:]
            
            n_init = len(init_items)
            init_item_map = dict(zip(init_items, range(n_init)))
            init_item_map2 = dict(map(reversed, init_item_map.items()))
            
            _,_, mse = knn_augment(b_mat=init_b, sim_mat=similarity, 
                                   sim_item_map=gt_item_map, train_item_map=init_item_map, 
                                   test_items = test_items, data=data,
                                   n_neighbors=nn, metric=met,
                                   gt_mat = train_b, gt_item_map=alt_item_map
                                   )
            mse_list.append(mse)
        mean_mse.append(np.mean(mse_list))
    plt.plot(nn_range, mean_mse, label = met)

plt.legend()
plt.show()







