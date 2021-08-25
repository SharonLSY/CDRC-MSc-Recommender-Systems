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
folder = 'trial/'
PATH_PROCESSED = './data/prepared/' + folder
FILE = 'BDC.BROWSING_PRODUCT_VIEW'

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
train, val = load_data_session(PATH_PROCESSED, FILE, train_eval=True)

#conf_gamma = []
#conf_lr = []
#mrr_score = []

model = RecVAE(batch_size=500, n_epochs=1)
model.fit(train, val)
        
mrr = MRR(length=10)

result = evaluation.evaluate_sessions(model, [mrr], val, train)

#conf_gamma.append(g)
#conf_lr.append(l)
#mrr_score.append(result[0][1])

#results_df = pd.DataFrame({'gamma':conf_gamma, 'lr': conf_lr, 'mrr':mrr_score})
