import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import datetime

path = './data/'

# data = pd.read_parquet(path + 'train.parquet', engine='auto',
#                        columns=['event_time', 'event_type', 'product_id', 'user_id',
#                                 'brand', 'price', 'user_session','cat_0', 'cat_1',
#                                 'cat_2', 'cat_3'])

data = pd.read_csv(path + '2020-Jan.csv')

# only take computer views
data = data.loc[data.event_type == 'view'].copy()
data['cat0'] = data['category_code'].apply(lambda x: str(x).split('.')[0])
data = data.loc[data.cat0 == 'computers'].copy()

data['event_time'] = pd.to_datetime(data['event_time'], 
                                    format='%Y-%m-%d %H:%M:%S UTC', 
                                    errors='ignore')

data['event_date'] = data['event_time'].apply(lambda x: x.date())
data['event_time'] = data['event_time'].apply(lambda x: x.timestamp())

products = data.groupby('product_id').event_date.min()
products = products.reset_index()
products_number = products.groupby('event_date').size()

data = data.loc[data.event_date >= datetime.date(2020,1,20),
                ['event_date','event_time','product_id','user_session']]

session_dates = data.groupby('user_session').event_date.min()

sessions = data.groupby('user_session').size()
sessions = sessions[sessions >= 2]
data = data.loc[data.user_session.isin(sessions.index.tolist())]

test_date = datetime.date(2020,1,26)
val_date = datetime.date(2020,1,25)
train_dates = [datetime.date(2020,1,26-d) for d in range(2,5)]

train_full = data.loc[data['event_date'].isin(train_dates + [val_date])].copy()
train_data = data.loc[data['event_date'].isin(train_dates)].copy()

train_sessions = train_data.user_session.unique()
val_data = train_full.loc[~train_full.user_session.isin(train_sessions)].copy()

val_sessions = val_data.user_session.unique()

test_data = data.loc[(~data.user_session.isin(val_sessions)) & 
                     (data['event_date'] == test_date)].copy()

# saving datasets
path_saved = './data/trial/'

train_full.to_csv(path_saved+'train_full.csv', index=False)
train_data.to_csv(path_saved+'train.csv', index=False)
val_data.to_csv(path_saved+'val.csv', index=False)
test_data.to_csv(path_saved+'test.csv', index=False)

data.loc[(data.event_date >= min(train_dates)) & (data.event_date <= test_date)].to_csv(path_saved+'data_full.csv', index=False)

