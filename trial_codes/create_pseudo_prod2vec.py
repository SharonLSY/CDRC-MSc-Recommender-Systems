import numpy as np
import time
from tqdm import tqdm
from datetime import date, datetime

from algorithms.slist import SLIST
from evaluation.loader import load_data_session
from evaluation.metrics.accuracy import MRR, HitRate
from evaluation.metrics.popularity import Popularity
from evaluation.metrics.coverage import Coverage
import evaluation.evaluation as evaluation
import preprocessing.preprocessing as pp

'''
FILE PARAMETERS
'''
DATA_PATH = './data/'
folder = '28062021/'
PATH_PROCESSED = './data/prepared/' + folder
DATA_FILE = 'test'

'''
MODEL HYPERPARAMETER TUNING
'''
alpha = 1 #[0.2, 0.4, 0.6, 0.8] 
direction = 'all' # sr / part / all
reg = 10
train_weight = 1 #0.5 #[0.125, 0.25, 0.5, 1, 2, 4, 8]
predict_weight = 1 #4 #[0.125, 0.25, 0.5, 1, 2, 4, 8]
session_weight = 1 #256 #[1, 2, 4, 8, 16, 32, 64, 128, 256]
new_items_file=PATH_PROCESSED+'new_items'

MIN_SESSION_LENGTH = 2
MIN_ITEM_SUPPORT = 5

data, _ = load_data( DATA_PATH+DATA_FILE )
data = data.loc[data.Time >= datetime.strptime('2020/04/11', "%Y/%m/%d").timestamp()]
data = data.loc[data.Time <= datetime.strptime('2020/04/18', "%Y/%m/%d").timestamp()]
data = filter_data( data, MIN_ITEM_SUPPORT, MIN_SESSION_LENGTH )

# creating SLIS array
model = SLIST(alpha=alpha, direction=direction, reg=reg, train_weight=train_weight, 
              predict_weight=predict_weight, session_weight=session_weight)
model.fit(data)

prod2vec_mat = model.enc_w
prod2vec_map = model.itemidmap

mat_columns = ['col_'+ str(i) for i in range(model.enc_w.shape[1])]
prod2vec_df = pd.DataFrame(prod2vec_mat, columns = mat_columns)
prod2vec_df['ItemId'] = prod2vec_map.index.tolist()

prod2vec_df.to_csv('./data/prod2vec.csv', index=False)


