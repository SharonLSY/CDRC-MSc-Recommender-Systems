import pandas as pd
import numpy as np
import time
from tqdm import tqdm

from algorithms.slist import SLIST

path = './data/trial/'

'''
MODEL HYPERPARAMETER TUNING
'''
alpha = 1 #[0.2, 0.4, 0.6, 0.8] -- 1 to generate similarity matrix
direction = 'all' # sr / part / all
reg = 10
train_weight = 1 #0.5 #[0.125, 0.25, 0.5, 1, 2, 4, 8]
predict_weight = 1 #4 #[0.125, 0.25, 0.5, 1, 2, 4, 8]
session_weight = 1 #256 #[1, 2, 4, 8, 16, 32, 64, 128, 256]

# generating ground truth similarity matrix
data = pd.read_csv(path+'data_full.csv')

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

# generate similarity matrix B
data.columns = ['Date','Time','ItemId','SessionId']

model = SLIST(alpha=1, direction=direction, reg=reg, train_weight=train_weight, 
                    predict_weight=predict_weight, session_weight=session_weight)
model.fit(data)
gt_item_map = model.itemidmap
gt_item_map2 = dict(map(reversed, gt_item_map.items()))
similarity = model.enc_w

model = SLIST(alpha=0.2, direction=direction, reg=reg, train_weight=train_weight, 
                    predict_weight=predict_weight, session_weight=session_weight)
model.fit(data)
gt_b = model.enc_w