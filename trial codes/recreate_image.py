# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 10:21:48 2021

@author: sharo
"""

alpha = 0.5 #[0.2, 0.4, 0.6, 0.8] 
direction = 'sr' # sr / part / all
reg = 10
train_weight = 0 #[0.125, 0.25, 0.5, 1, 2, 4, 8]
predict_weight = 4 #[0.125, 0.25, 0.5, 1, 2, 4, 8]
session_weight = 0 #[1, 2, 4, 8, 16, 32, 64, 128, 256]

# creating train2 dataset to follow the image in PPT
time = np.arange(1,9,1)*10+1585758803.0
user_id = [1]*3 + [2]*3 + [3]*2
item_id = [1,2,3,2,3,4,3,4]
session_id = user_id
train2 = pd.DataFrame({'Time': time, 'UserId': user_id, 
                       'ItemId': item_id, 'SessionId': session_id})

itemids = train2['ItemId'].unique()
n_items = len(itemids)
itemidmap = pd.Series(data=np.arange(n_items), index=itemids)

# for d in ['sr','part','all']:
#     model = SLIST(alpha=alpha, direction=d, reg=reg, train_weight=train_weight, 
#                     predict_weight=predict_weight, session_weight=session_weight)
#     model.fit(train2)
#     model_b = model.enc_w
#     print('model b:\n', model_b)

model = SLIST(alpha=alpha, direction='part', reg=reg, train_weight=train_weight, 
              predict_weight=predict_weight, session_weight=session_weight,
              normalize=None)
model.fit(train2)
model_b = model.enc_w
print('model b:\n', model_b)

test_session = train2.loc[(train2.SessionId == 1)].iloc[:3,:].copy()
test_session.loc[3] = [max(test_session.Time)+10, 1, 1, 1] 
test_session.sort_values(['SessionId', 'Time'], inplace=True)
items_to_predict = train2['ItemId'].unique()

session_items = test_session.ItemId.unique().tolist()
session = 1
session_times = test_session.Time.tolist()
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