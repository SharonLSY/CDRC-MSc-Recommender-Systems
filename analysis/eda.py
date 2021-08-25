import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing.preprocessing import *
from evaluation.loader import load_data_session

PATH_PROCESSED = './data/prepared/beauty_models/'
FILE = 'browsing_data'

df, _ = load_data_session(PATH_PROCESSED, FILE, train_eval=False)

# remove single pdp sessions
session_length = df.groupby('SessionId').size()
print(f'Single PDP sessions: {(session_length==1).sum()/session_length.shape[0]}')
session_length = session_length[session_length >1]
df = df.loc[df.SessionId.isin(session_length.index)]

# session length
session_length = session_length[session_length <= np.percentile(session_length, 99)]
sns.violinplot(session_length)
plt.xlabel('Session Length')
plt.savefig('./plots/session_length.png', dpi=300, transparent=False, bbox_inches='tight')
plt.clf()

# distribution of product views by total views
prod_views = df.groupby('ItemId').size()
prod_views = prod_views[prod_views <= np.percentile(prod_views, 99)]
sns.boxplot(prod_views)
plt.xlabel('Product Views')
#plt.savefig('./plots/knn_grid_search.png', dpi=300, transparent=False, bbox_inches='tight')
plt.clf()

# plotting popularity of products through the days


