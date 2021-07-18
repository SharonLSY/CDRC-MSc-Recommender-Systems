import numpy as np
import time
from tqdm import tqdm

from algorithms.recvae.model import *
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
# hidden_dim
# latent_dim
# n_epochs
# n_enc_epochs
# n_dec_epochs
# batch_size, beta, gamma, lr,
# not_alternating

seed = 1337
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# hyperparameter tuning
train, val = load_data_session(PATH_PROCESSED, FILE, slice_num=0, train_eval=True)

conf_gamma = []
conf_lr = []
mrr_score = []

for g in [5e-4,1e-3,5e-3,1e-2,5e-2,1e-1]:
    for l in [5e-4,1e-3,5e-3,1e-2,5e-2,1e-1]:

        model = RecVAE(batch_size=50, gamma=g, lr=l)
        model.fit(train, val)
        
        mrr = MRR(length=10)
        
        result = evaluation.evaluate_sessions(model, [mrr], val, train)
        
        conf_gamma.append(g)
        conf_lr.append(l)
        mrr_score.append(result[0][1])

results_df = pd.DataFrame({'gamma':conf_gamma, 'lr': conf_lr, 'mrr':mrr_score})
