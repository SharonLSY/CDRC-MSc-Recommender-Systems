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
folder = 'test_date_trial/'
PATH_PROCESSED = './data/prepared/' + folder
FILE = 'test'


'''
MODEL HYPERPARAMETER TUNING
'''
alpha = 0.2 #[0.2, 0.4, 0.6, 0.8] 
direction = 'all' # sr / part / all
reg = 10
train_weight = 1 #0.5 #[0.125, 0.25, 0.5, 1, 2, 4, 8]
predict_weight = 1 #4 #[0.125, 0.25, 0.5, 1, 2, 4, 8]
session_weight = 1 #256 #[1, 2, 4, 8, 16, 32, 64, 128, 256]
prod2vec_file = './data/prod2vec.csv'
n_neighbors = 500
knn_metric = 'manhatten' # manhatten / euclidean / chebyshev


# hyperparameter tuning
train, test = load_data_session(PATH_PROCESSED, FILE, train_eval=False)
model = SLIST_ext(alpha=alpha, direction=direction, reg=reg, train_weight=train_weight, 
              predict_weight=predict_weight, session_weight=session_weight,
              prod2vec_file=prod2vec_file, 
              n_neighbors=n_neighbors, knn_metric=knn_metric)
model.fit(train, test)

mrr = MRR(length=100)
hr = HitRate()

result = evaluation.evaluate_sessions(model, [mrr, hr], test, train)

