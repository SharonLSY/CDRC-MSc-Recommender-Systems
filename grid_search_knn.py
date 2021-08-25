from pathlib import Path
import sys
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from random import sample
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import load_npz

from algorithms.slist import SLIST
from evaluation.loader import load_data_session
from utils.knn_utils import *
from run_config import create_algorithms_dict

conf = 'conf/evaluation/evaluate_beauty_models.yml'

'''
FILE PARAMETERS
'''
folder = 'beauty_new_items/'
PATH_PROCESSED = './data/prepared/' + folder
FILE = 'browsing_data'

'''
SLIST OPTIMAL HYPERPARAMETERS
'''
#alpha = 0.1 #[0.2, 0.4, 0.6, 0.8] 
#direction = 'all' # sr / part / all
#reg = 10
#train_weight = 0.125 #0.5 #[0.125, 0.25, 0.5, 1, 2, 4, 8]
#predict_weight = 0.7673070073686803 #4 #[0.125, 0.25, 0.5, 1, 2, 4, 8]
#session_weight = 300 #256 #[1, 2, 4, 8, 16, 32, 64, 128, 256]

'''
Item Features Matrix
'''
prod2vec_file = PATH_PROCESSED+'X_meta.npz'
prod2vec_mapping = PATH_PROCESSED+'meta_item.csv'

# read optimal hyperparameters from conf file

file = Path(conf)
if file.is_file():

    print('Loading file')
    stream = open(str(file))
    c = yaml.load(stream)
    stream.close()
    
slist_params = list(filter(lambda x: x['key'] == 'slist', c['algorithms']))[0]['params']
alpha = slist_params['alpha']
direction = slist_params['direction']
train_weight = slist_params['alpha']
train_weight = slist_params['train_weight']
predict_weight = slist_params['predict_weight']
session_weight = slist_params['session_weight']


# setting up ground truth B
train, test = load_data_session(PATH_PROCESSED, FILE, train_eval=False)
model = SLIST(alpha=alpha, direction=direction, reg=10, train_weight=train_weight, 
                    predict_weight=predict_weight, session_weight=session_weight)
model.fit(train)
alt_item_map = model.itemidmap
alt_item_map2 = dict(map(reversed, alt_item_map.items()))
train_b = model.enc_w

# setting up prod2vec matrix & mapping
prod2vec_matrix = load_npz(prod2vec_file)
prod2vec_map = pd.read_csv(prod2vec_mapping)
prod2vec_map = pd.Series(index = prod2vec_map.ItemId, data = prod2vec_map.ItemIdx.tolist())
prod2vec_matrix = prod2vec_matrix[ prod2vec_map[model.itemidmap.index] ,]
prod2vec_map = pd.Series(data=np.arange(prod2vec_matrix.shape[0]), index=prod2vec_map[model.itemidmap.index].index.tolist())

nn_range = [10,20,50,100]
metrics = ['manhattan' , 'cosine', 'euclidean']

mean_mse = {}
for met in metrics:
    mean_mse[met] = {}
    for nn in nn_range:
        mean_mse[met][nn] = []

for i in tqdm(range(50)):
    n_items = model.n_items
    test_index = sample(range(0, n_items), 100) # generate 50 random numbers
    #test_index = list(set([random.randint(0,n_items-1) for i in range(50)]))
    test_items = [alt_item_map2[i] for i in test_index]
    init_index = [i for i in list(alt_item_map2.keys()) if i not in test_index]
    init_items = [alt_item_map2[i] for i in init_index]

    init_b = train_b[:,init_index][init_index,:]
    
    n_init = len(init_items)
    init_item_map = pd.Series(index=init_items, data=range(n_init))
    init_item_map2 = dict(map(reversed, init_item_map.items()))
    
    for met in metrics:
        for nn in nn_range:
            _,_, mse = knn_augment(b_mat=init_b, sim_mat=prod2vec_matrix, 
                                   sim_item_map=prod2vec_map, train_item_map=init_item_map, 
                                   test_items = test_items, data=train,
                                   n_neighbors=nn, metric=met,
                                   gt_mat = train_b, gt_item_map=alt_item_map
                                   )
#            print(met, nn, mse)
            mean_mse[met][nn].append(mse)

keep_mse = mean_mse.copy()

mean_mse = keep_mse
for met in metrics:
    mse_mean = []
    for nn in nn_range:
        mse_mean.append(np.mean(mean_mse[met][nn]))
        print(met, mse_mean)
    sns.lineplot(x=nn_range, y=mse_mean, label = met)
plt.xlabel('Number of Neighbors')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.savefig('./plots/knn_grid_search.png', dpi=300, transparent=False, bbox_inches='tight')
plt.clf()