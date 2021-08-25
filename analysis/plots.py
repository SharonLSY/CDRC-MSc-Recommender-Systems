import pandas as pd
import numpy as np
from datetime import datetime

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

# plotting distribution of views across indiv products
data = df.groupby('ItemId').size().reset_index()
data.columns = ['PRODUCT_ID', 'num']
data = data.sort_values(by='num', axis=0, ascending=False).reset_index(drop=True)
data['perc'] = 100*data['num']/sum(data['num'])
data['cum_perc'] = np.cumsum(data['perc'])

x_value = [i*100/data.shape[0] for i in range(1,data.shape[0]+1)]
plt.plot(x_value, data.cum_perc)
plt.xlabel('% of Products')
plt.ylabel('Cumulative % of Views')
plt.savefig('./plots/product_views_dist.png', dpi=300, transparent=False, bbox_inches='tight')
plt.clf()

# plotting concentration of products across sessions
data = df.groupby('ItemId').size().reset_index()
data.columns = ['ItemId', 'num']
data = data.sort_values(by='num', axis=0, ascending=False).reset_index(drop=True)
data['perc'] = 100*data['num']/sum(data['num'])
data['cum_perc'] = np.cumsum(data['perc'])

x_value = [i*100/data.shape[0] for i in range(1,data.shape[0]+1)]
plt.plot(x_value, data.cum_perc)
plt.xlabel('% of Products')
plt.ylabel('Cumulative % of Views')
plt.savefig('./plots/product_views_dist.png', dpi=300, transparent=False, bbox_inches='tight')
plt.clf()


plt.plot(range(1,data.shape[0]+1), data.num)
plt.clf()

# Looking at % of users/sessions with % of items, starting from most popular items

product_views = df.groupby('ItemId').size().reset_index()
product_views.columns = ['ItemId','num_views']
product_views = product_views.sort_values(by='num_views', axis=0, ascending=False, ignore_index=True)

users = []
for i in range(100,0,-1):
    products = product_views.loc[product_views.num_views >= np.percentile(product_views.num_views, i),'ItemId'].tolist()
    num_users = df.loc[df.ItemId.isin(products),'UserId'].nunique()
    users.append(num_users)

users = [100*i/df.UserId.nunique() for i in users]

sessions = []
for i in range(100,0,-1):
    products = product_views.loc[product_views.num_views >= np.percentile(product_views.num_views, i),'ItemId'].tolist()
    num_sessions = df.loc[df.ItemId.isin(products),'SessionId'].nunique()
    sessions.append(num_sessions)

sessions = [100*i/df.SessionId.nunique() for i in sessions]

#plt.plot(range(1,101), users)
plt.plot(range(1,101), sessions)
plt.xlabel('% of Products')
plt.ylabel('Cumulative % of Sessions')
plt.savefig('./plots/sessions_by_products.png', dpi=300, transparent=False, bbox_inches='tight')
plt.clf()

# looking at item-item transitions
data = df.copy()
data['NextItemId'] = data['ItemId'].shift(-1)
data['NextSessionId'] = data['SessionId'].shift(-1)
data = data.loc[data.SessionId == data.NextSessionId]
data = data.loc[data.ItemId != data.NextItemId]
del data['NextSessionId']

## filter to just top 0.1% producrs
#product_views = data.groupby('ItemId').size().reset_index()
#product_views.columns = ['ItemId','num_views']
#products = product_views.loc[product_views.num_views >= np.percentile(product_views.num_views, 99.9),'ItemId'].tolist()
#data = data.loc[(data.ItemId.isin(products)) & (data.NextItemId.isin(products))]

data = data.loc[data.ItemId != data.NextItemId]
transition_table = data.groupby(['ItemId','NextItemId']).size().reset_index()
transition_table.columns = ['ItemId','NextItemId','value']
transition_table = transition_table.sort_values(by='value', axis=0, ascending=False, ignore_index=True)

# looking at top 0.1% of transitions
#transition_table = transition_table.loc[transition_table.value >= np.percentile(transition_table.value, 99.9)]

transition_table['perc'] = 100*transition_table['value']/sum(transition_table['value'])
transition_table['cum_perc'] = np.cumsum(transition_table['perc'])
x_value = [i*100/transition_table.shape[0] for i in range(1,transition_table.shape[0]+1)]
plt.plot(x_value, transition_table.cum_perc)

to_table = transition_table.loc[transition_table.ItemId < transition_table.NextItemId].copy()
rev_table = transition_table.loc[transition_table.NextItemId < transition_table.ItemId].copy()
to_table.columns = ['Item1','Item2','to_count']
rev_table.columns = ['Item2','Item1','rev_count']

transition_summary = pd.merge(to_table, rev_table, on=['Item1','Item2'], how='outer').fillna(0)

transition_matrix = pd.pivot_table(transition_table, values='value', index='ItemId',
                                   columns=['NextItemId'], aggfunc=np.sum, fill_value=0)

# looking at repeat items

data = df.copy()

rep_data = data.groupby(['UserId','SessionId','ItemId']).size().reset_index()
rep_data.columns = ['UserId','SessionId','ItemId','rep_count']

print(f'% of sessions with repeated items within session: {100*rep_data.loc[rep_data.rep_count >=2].SessionId.nunique()/data.SessionId.nunique()}')
print(f'% of users with repeated items within session: {100*rep_data.loc[rep_data.rep_count >=2].UserId.nunique()/data.UserId.nunique()}')

# repeated item consumption across sessions

rep_data2 = data.groupby(['UserId','ItemId']).SessionId.nunique().reset_index()
rep_data2.columns = ['UserId','ItemId','rep_count']
rep_cookies = rep_data2.UserId.unique()

sub = data.loc[data.UserId.isin(rep_cookies)].copy()

print(f'% of users with repeated items across session: {100*rep_data2.loc[rep_data2.rep_count >=2].UserId.nunique()/data.UserId.nunique()}')

data = df.groupby('UserId').SessionId.nunique()
print(f'% of users with more than 1 session: {100*data.loc[data>1].shape[0]/data.shape[0]}')

# looking at category of 1st and last itemn in each session

data = df.copy()

data = data.sort_values(by=['UserId','SessionId','Time'], ascending=True, ignore_index=True)
data['next_user'] = data['UserId'].shift(-1)
data['next_session'] = data['SessionId'].shift(-1)#
data['next_item'] = data['ItemId'].shift(-1)
data = data.loc[(data.UserId == data.next_user) & (data.SessionId != data.next_session)].reset_index(drop=True)
del data['next_user'], data['next_session']
data = data.loc[:,['Time', 'UserId', 'SessionId', 'ItemId', 'next_item']]
data.columns = ['Time', 'UserId', 'SessionId', 'item', 'next_item']

meta = pd.read_csv('data/item_dims1.csv')
meta2 = pd.read_csv('data/item_dims2.csv')
meta = pd.concat([meta, meta2], axis=0).drop_duplicates().reset_index(drop=True)
del meta2

meta['category'] = meta['merchandise_category_description']
#meta.loc[meta.item_hierarchy5_description.str.lower() == meta.brand_description.str.lower(),'category'] = meta.loc[meta.item_hierarchy5_description.str.lower() == meta.brand_description.str.lower(),'item_hierarchy4_description']
meta = meta[['sap_code_var_p','category']]

data = pd.merge(data, meta, left_on='item', right_on='sap_code_var_p')
del data['sap_code_var_p']
data = pd.merge(data, meta, left_on='next_item', right_on='sap_code_var_p')
del data['sap_code_var_p']

match_data = data.loc[data.category_x == data.category_y].copy()

matching_cat = data.loc[data.category_x == data.category_y].shape[0]
print(f'% of consecutive sessions with matching category: {100*matching_cat/data.shape[0]}' )

# plotting popularity of products through the days
num_products=5
df['Date'] = df['Time'].apply(lambda x: datetime.fromtimestamp(x).strftime("%Y-%m-%d"))
item_views = df.groupby('ItemId').size().reset_index()
item_views.columns = ['ItemId','num_views']
item_views = item_views.sort_values(by=['num_views'], ascending=False, ignore_index=True)
top_items = item_views[:num_products].ItemId.tolist()

daily_views = df.groupby('Date').size()
del daily_views['2021-06-20']

item_daily_views = df.groupby(by=['Date','ItemId']).size()
del item_daily_views['2021-06-20']

item_daily_views = item_daily_views.reset_index()
item_daily_views.columns = ['Date','ItemId','num_views']
daily_views = daily_views.reset_index()
daily_views.columns = ['Date','total_views']

item_daily_views = pd.merge(item_daily_views, daily_views, how='left')
item_daily_views = item_daily_views.loc[item_daily_views.ItemId.isin(top_items)]
item_daily_views['views_perc'] = 100*item_daily_views['num_views']/item_daily_views['total_views']

for i in top_items:
    sub = item_daily_views.loc[item_daily_views.ItemId == i].copy()
    plt.plot(sub.Date, sub.views_perc, label=i)
plt.show()
plt.clf()

# plotting subcategory views
num_subcats=10
item_daily_views = df.groupby(by=['Date','ItemId']).size()
del item_daily_views['2021-06-20']
item_daily_views = item_daily_views.reset_index()
item_daily_views.columns = ['Date','ItemId','num_views']

meta = pd.read_csv('data/item_dims1.csv')
meta2 = pd.read_csv('data/item_dims2.csv')
meta = pd.concat([meta, meta2], axis=0).drop_duplicates().reset_index(drop=True)
del meta2

meta = meta.loc[:,['sap_code_var_p','merchandise_category_description']]
meta.columns = ['ItemId','subcat']

item_daily_views = pd.merge(item_daily_views, meta, how='left')

subcat_views = item_daily_views.groupby('subcat').num_views.sum().reset_index()
subcat_views = subcat_views.sort_values(by=['num_views'], ascending=False, ignore_index=True)
top_subcats = subcat_views[:num_subcats].subcat.tolist()

daily_views = df.groupby('Date').size()
del daily_views['2021-06-20']
daily_views = daily_views.reset_index()
daily_views.columns = ['Date','total_views']

subcat_daily_views = item_daily_views.groupby(['Date','subcat']).num_views.sum().reset_index()
subcat_daily_views = pd.merge(subcat_daily_views, daily_views, how='left')
subcat_daily_views['views_perc'] = 100*subcat_daily_views['num_views']/subcat_daily_views['total_views']

sub = subcat_daily_views.loc[subcat_daily_views.subcat.isin(top_subcats)].copy()
sns.lineplot(x='Date', y='views_perc', data=sub, hue='subcat')
plt.show()
plt.clf()

i=1
start=0
while start < 0.8:
    start = subcat_views[:i].num_views.sum()/subcat_views.num_views.sum()
    if start >= 0.8:
        print(i)
        print(start)
    i+=1