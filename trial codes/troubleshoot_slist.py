import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from scipy import sparse
from scipy.sparse.linalg import inv
from scipy.sparse import csr_matrix, csc_matrix, vstack
from algorithms.slist import SLIST

# HYPERPARAMETERS
alpha = 0 #[0.2, 0.4, 0.6, 0.8] 
direction = 'sr' # sr / part / all
reg = 10
train_weight = 0.5 #[0.125, 0.25, 0.5, 1, 2, 4, 8]
predict_weight = 4 #[0.125, 0.25, 0.5, 1, 2, 4, 8]
session_weight = 256 #[1, 2, 4, 8, 16, 32, 64, 128, 256]

train2 = train.loc[train.SessionId.isin([312893, 903890])]
model = SLIST(alpha=alpha, direction=direction, reg=reg, train_weight=train_weight, 
                    predict_weight=predict_weight, session_weight=session_weight)
model.fit(train2)
n_items = train2.ItemId.nunique()

itemids = train2['ItemId'].unique()
itemidmap = pd.Series(data=np.arange(n_items), index=itemids)

 ## TESTING SLIS
input1, target1, row_weight1 = model.make_train_matrix(train2, weight_by='SLIS')
w2 = row_weight1
input_matrix = input1
target_matrix = target1

W2 = sparse.diags(w2, dtype=np.float32)
G = input_matrix.transpose().dot(W2).dot(input_matrix).toarray()
P = np.linalg.inv(G + np.identity(n_items, dtype=np.float32) * 10)

C = -P @ (input_matrix.transpose().dot(W2).dot(input_matrix-target_matrix).toarray())

mu = np.zeros(n_items)
mu += 10
mu_nonzero_idx = np.where(1 - np.diag(P)*10 + np.diag(C) >= 10)
mu[mu_nonzero_idx] = (np.diag(1 - 10 + C) / np.diag(P))[mu_nonzero_idx]

# B = I - Pλ + C
model_b = np.identity(n_items, dtype=np.float32) - P @ np.diag(mu) + C

 ## TESTING SLIST
input1, target1, row_weight1 = model.make_train_matrix(train2, weight_by='SLIS')
input2, target2, row_weight2 = model.make_train_matrix(train2, weight_by='SLIT')
input1.data = np.sqrt(alpha) * input1.data
target1.data = np.sqrt(alpha) * target1.data
input2.data = np.sqrt(1-alpha) * input2.data
target2.data = np.sqrt(1-alpha) * target2.data

input_matrix = vstack([input1, input2])
target_matrix = vstack([target1, target2])
w2 = row_weight1 + row_weight2  # 
w2 = np.square(w2)
W2 = sparse.diags(w2, dtype=np.float32)

# P = (X^T * X + λI)^−1 = (G + λI)^−1
# (A+B)^-1 = A^-1 - A^-1 * B * (A+B)^-1
# P =  G
G = input_matrix.transpose().dot(W2).dot(input_matrix).toarray()
print(f"G is made. \nSparsity:{(1 - np.count_nonzero(G)/(8**2))*100}%")
P = np.linalg.inv(G + np.identity(n_items, dtype=np.float32) * 10)
print("P is made")

model_b = P @ input_matrix.transpose().dot(W2).dot(target_matrix).toarray()

# PREDICT
## test_session only takes the first 3 events of a session in the training set
test_session = train.loc[train.SessionId == 312893].copy()
test_session = test_session.loc[test_session.ItemId.isin([1004249,1002524,4804056])]
test_session.sort_values(['SessionId', 'ItemId'], inplace=True)
items_to_predict = train2['ItemId'].unique()
metrics = [mrr, hr]

session_items = [1004249,1002524,4804056]
session = 312893
session_times = [1585758896.0, 1585758803.0, 1585759915.0]

session_items_new_id = itemidmap[session_items].values
predict_for_item_ids_new_id = itemidmap[itemids].values

W_test = np.ones_like(session_items, dtype=np.float32)
for i in range(len(W_test)):
    W_test[i] = np.exp(-abs(i+1-len(session_items))/predict_weight)

W_test = W_test if predict_weight > 0 else np.ones_like(W_test)
W_test = W_test.reshape(-1,1)
preds = model_b[session_items_new_id] * W_test
preds = np.sum(preds, axis=0)
preds = preds[predict_for_item_ids_new_id]

series = pd.Series(data=preds, index=itemids)

series = series / series.max()

##################################################################################

# MY SLIS
G = input_matrix.transpose().dot(np.square(W2)).dot(input_matrix).toarray()
P = np.linalg.inv(G + np.identity(n_items, dtype=np.float32) * 10)

gamma = np.zeros(n_items) + 10
gamma_idx = np.where(1 - np.diag(P)*10 + np.diag(C) >= 10)
gamma[gamma_idx] = ((1-10) / np.diag(P))[gamma_idx]
my_b = np.identity(n_items, dtype=np.float32) - P @ np.diag(gamma)

# MY SLIST
P = np.linalg.inv((input1.T @ np.diag(row_weight1) @ input1) + \
                  (input2.T @ np.diag(row_weight2) @ input2) + \
                  np.identity(n_items, dtype=np.float32) * 10)

my_b = np.identity(n_items, dtype=np.float32) - reg * P - \
       P @ input2.T @  np.diag(row_weight2) @ (input2 - target2)




