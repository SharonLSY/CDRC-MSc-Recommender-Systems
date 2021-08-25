import numpy as np
import time
from tqdm import tqdm

from algorithms.slist import SLIST
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


# hyperparameter tuning
train, val = load_data_session(PATH_PROCESSED, FILE, train_eval=True)
model = SLIST(alpha=alpha, direction=direction, reg=reg, train_weight=train_weight, 
              predict_weight=predict_weight, session_weight=session_weight)
model.fit(train, val)

mrr = MRR(length=100)
hr = HitRate()
pop = Popularity()
pop.init(train)
cov = Coverage()
cov.init(train)

result = evaluation.evaluate_sessions(model, [mrr, hr, pop, cov], val, train)

