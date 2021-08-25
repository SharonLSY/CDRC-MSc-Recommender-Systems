import numpy as np
import time
from tqdm import tqdm

from algorithms.ae.ae import *
from evaluation.loader import load_data_session
from evaluation.metrics.accuracy import MRR, HitRate
from evaluation.metrics.popularity import Popularity
from evaluation.metrics.coverage import Coverage
import evaluation.evaluation as evaluation

'''
FILE PARAMETERS
'''
folder = '28062021/'
PATH_PROCESSED = './data/prepared/' + folder
FILE = 'test'

'''
MODEL HYPERPARAMETER TUNING
'''


# hyperparameter tuning
train, val = load_data_session(PATH_PROCESSED, FILE, slice_num=0, train_eval=True)

conf_gamma = []
conf_lr = []
mrr_score = []

model = AutoEncoder()
model.fit(train, val)

mrr = MRR(length=10)

result = evaluation.evaluate_sessions(model, [mrr], val, train)