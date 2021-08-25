import numpy as np
import time
from tqdm import tqdm

from algorithms.slist_ext import SLIST_ext
from evaluation.loader import load_data_session
from evaluation.metrics.accuracy import MRR, HitRate
from evaluation.metrics.popularity import Popularity
from evaluation.metrics.coverage import Coverage
import evaluation.evaluation as evaluation

'''
FILE PARAMETERS
'''
folder = 'beauty_new_items/'
PATH_PROCESSED = './data/prepared/' + folder
FILE = 'browsing_data'


'''
MODEL HYPERPARAMETER TUNING
'''
alpha = 0.1 #[0.2, 0.4, 0.6, 0.8] 
direction = 'all' # sr / part / all
reg = 10
train_weight = 0.125 #0.5 #[0.125, 0.25, 0.5, 1, 2, 4, 8]
predict_weight = 0.7673070073686803 #4 #[0.125, 0.25, 0.5, 1, 2, 4, 8]
session_weight = 300 #256 #[1, 2, 4, 8, 16, 32, 64, 128, 256]
prod2vec_file = './data/prepared/beauty_new_items/X_meta.npz'
prod2vec_map = './data/prepared/beauty_new_items/meta_item.csv'
n_neighbors = 50
knn_metric = 'cosine' # manhatten / euclidean / chebyshev


# hyperparameter tuning
train, test = load_data_session(PATH_PROCESSED, FILE, train_eval=False)

model = SLIST_ext(alpha=alpha, direction=direction, reg=reg, train_weight=train_weight, 
              predict_weight=predict_weight, session_weight=session_weight,
              prod2vec_file=prod2vec_file, prod2vec_map=prod2vec_map,
              n_neighbors=n_neighbors, knn_metric=knn_metric)
model.fit(train, test)

# getting test sessions with new items
test_sessions = test.loc[test.ItemId.isin(model.new_items)].SessionId.unique()
test2 = test.loc[test.SessionId.isin(test_sessions)]

mrr = MRR(length=10)
hr = HitRate(length=10)

result = evaluation.evaluate_sessions(model, [mrr, hr], test2, train)

#########################################

from algorithms.slist import SLIST
folder = 'beauty_models/'
PATH_PROCESSED = './data/prepared/' + folder
FILE = 'browsing_data'

train, test = load_data_session(PATH_PROCESSED, FILE, train_eval=False)
model2 = SLIST(alpha=alpha, direction=direction, reg=reg, train_weight=train_weight, 
              predict_weight=predict_weight, session_weight=session_weight
              )
model2.fit(train, test)

# comparing against the same sessions
test = test.loc[test.SessionId.isin(test_sessions)]

mrr = MRR(length=10)
hr = HitRate(length=10)

result2 = evaluation.evaluate_sessions(model2, [mrr, hr], test2, train)


