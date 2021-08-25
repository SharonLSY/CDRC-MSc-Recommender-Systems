import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, date

data_filepath = './data/prepared/beauty_models/browsing_data'

#meta = pd.read_csv('data/item_dims1.csv')
#meta2 = pd.read_csv('data/item_dims2.csv')
#meta = pd.concat([meta, meta2], axis=0).drop_duplicates().reset_index(drop=True)
#del meta2
#
#brands = meta.brand_description.str.lower().unique().tolist()
#brands.append(['ysl','chanel beaute','rimmel','huda','cyo','ioma','dior','fenty','nyx','sleek'])

# plotting number of products released over time

meta = pd.read_csv('data/item_dims1.csv')
meta2 = pd.read_csv('data/item_dims2.csv')
meta = pd.concat([meta, meta2], axis=0).drop_duplicates().reset_index(drop=True)
del meta2

meta['valid_from_date'] = pd.to_datetime(meta['valid_from_date'], format='%d/%m/%Y')
meta['valid_from_date'] = meta['valid_from_date'].apply(lambda x: x.date())
meta = meta[meta['valid_from_date'] >= date(2020,7,1)]
meta = meta[meta['valid_from_date'] < date(2021,7,1)]

meta['item_introduction_date'] = pd.to_datetime(meta['item_introduction_date'], format='%d/%m/%Y')
meta['item_introduction_date'] = meta['item_introduction_date'].apply(lambda x: x.date())
meta['item_deletion_date'] = meta['item_deletion_date'].str.replace('9999','2099')
meta['item_deletion_date'] = pd.to_datetime(meta['item_deletion_date'], format='%d/%m/%Y')
meta['item_deletion_date'] = meta['item_deletion_date'].apply(lambda x: x.date())
meta = meta.loc[meta['item_introduction_date'] <= meta['valid_from_date']]
meta = meta[meta['item_deletion_date'] > meta['valid_from_date']]

release = meta.groupby('valid_from_date').size()

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator((1,4,7,10)))
plt.plot(release)
plt.gcf().autofmt_xdate()
plt.xlabel('Date')
plt.ylabel('Number of Products')
plt.savefig('./plots/products_release.png', dpi=300, transparent=False, bbox_inches='tight')
plt.clf()

###################################

# plotting distribution of products across main categories

meta = pd.read_csv('data/item_dims1.csv')
meta2 = pd.read_csv('data/item_dims2.csv')
meta = pd.concat([meta, meta2], axis=0).drop_duplicates().reset_index(drop=True)
del meta2

cat = meta.loc[meta.item_hierarchy2_description == 'Beauty'].groupby(['item_hierarchy3_description']).size()
cat = cat.reset_index()
cat.columns = ['Category','Number of Products']#
cat = cat.sort_values(by='Number of Products', axis=0, ascending=False)
#cat.loc[cat.Category == 'Consumables and general merchandise', 'Category'] = 'Consumables and\ngeneral merchandise'

#sum_value = cat.loc[cat.Category.isin(['Pharmacy services', 'Dispensing','Other']),'Number of Products'].sum()
#cat2 = pd.DataFrame([['Pharmacy services,\nDispensing and Other',sum_value]], columns=cat.columns)

#cat = cat.loc[~cat.Category.isin(['Pharmacy services', 'Dispensing','Other'])]
#cat = pd.concat([cat, cat2], axis=0)

ax = sns.barplot(x='Category', y='Number of Products', data=cat)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.savefig('./plots/products_cat_distribution.png', dpi=300, transparent=False, bbox_inches='tight')
plt.clf()

######################################

# plot distribution of views across main categories

data = pd.read_csv(data_filepath+'_train_full.txt', sep='\t')
data = data.groupby('ItemId').size().reset_index()
data.columns = ['PRODUCT_ID', 'num']

meta = pd.read_csv('data/item_dims1.csv')
meta2 = pd.read_csv('data/item_dims2.csv')
meta = pd.concat([meta, meta2], axis=0).drop_duplicates().reset_index(drop=True)
del meta2

meta = meta[['sap_code_var_p','item_hierarchy3_description']]
#meta.loc[meta.item_hierarchy2_description.isin(['Pharmacy services', 'Dispensing','Other']), 'item_hierarchy2_description'] = 'Pharmacy services,\nDispensing and Other'
#meta.loc[meta.item_hierarchy2_description == 'Consumables and general merchandise', 'item_hierarchy2_description'] = 'Consumables and\ngeneral merchandise'

views = pd.merge(data, meta, left_on='PRODUCT_ID', right_on='sap_code_var_p', how='left')
views = views.groupby('item_hierarchy3_description').num.sum().reset_index()
views.columns = ['Category', 'Number of Views']
views = views.sort_values(by='Number of Views', axis=0, ascending=False)

ax = sns.barplot(x='Category', y='Number of Views', data=views)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.savefig('./plots/products_cat_views.png', dpi=300, transparent=False, bbox_inches='tight')
plt.clf()

##########################################

# plotting distribution of views across indiv products

data = pd.read_csv(data_filepath+'_train_full.txt', sep='\t')
data = data.groupby('ItemId').size().reset_index()
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

#plt.plot(range(1,data.shape[0]+1), data.num)

###########################################

# plotting distributions of views across subcats

data = pd.read_csv(data_filepath+'_train_full.txt', sep='\t')
data = data.groupby('ItemId').size().reset_index()
data.columns = ['PRODUCT_ID', 'num']

meta = pd.read_csv('data/item_dims1.csv')
meta2 = pd.read_csv('data/item_dims2.csv')
meta = pd.concat([meta, meta2], axis=0).drop_duplicates().reset_index(drop=True)
del meta2

meta = meta[['sap_code_var_p','merchandise_category_description']]
views = pd.merge(data, meta, left_on='PRODUCT_ID', right_on='sap_code_var_p', how='left')
views = views.groupby('merchandise_category_description').num.sum().reset_index()
views.columns = ['Category', 'Number of Views']
views = views.sort_values(by='Number of Views', axis=0, ascending=False, ignore_index=True)

show = 10
views['Rank'] = range(1, views.shape[0]+1)
sum_views = views.loc[show:,'Number of Views'].sum()
views.loc[show,'Category'] = 'Others'
views.loc[show,'Number of Views'] = sum_views
views = views.loc[:show,:]

ax = sns.barplot(x='Category', y='Number of Views', data=views)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.savefig('./plots/products_subcat_views.png', dpi=300, transparent=False, bbox_inches='tight')
plt.clf()


###########################################

# Looking at session transitions (category)

data = pd.read_csv(data_filepath+'_train_full.txt', sep='\t')
data['NextItemId'] = data['ItemId'].shift(-1)
data['NextSessionId'] = data['SessionId'].shift(-1)
data = data.loc[data.SessionId == data.NextSessionId]
data = data.loc[data.ItemId != data.NextItemId]
del data['NextSessionId']

meta = pd.read_csv('data/item_dims1.csv')
meta2 = pd.read_csv('data/item_dims2.csv')
meta = pd.concat([meta, meta2], axis=0).drop_duplicates().reset_index(drop=True)
del meta2

meta['category'] = meta['merchandise_category_description']
#meta.loc[meta.item_hierarchy5_description.str.lower() == meta.brand_description.str.lower(),'category'] = meta.loc[meta.item_hierarchy5_description.str.lower() == meta.brand_description.str.lower(),'item_hierarchy4_description']
meta = meta[['sap_code_var_p','category']]

data = pd.merge(data, meta, left_on='ItemId', right_on='sap_code_var_p', how='left')
del data['sap_code_var_p']
data.columns = ['Time', 'UserId', 'ItemId', 'SessionId', 'NextItemId', 'Cat']
data = pd.merge(data, meta, left_on='NextItemId', right_on='sap_code_var_p', how='left')
del data['sap_code_var_p']
data.columns = ['Time', 'UserId', 'ItemId', 'SessionId', 'NextItemId', 'Cat', 'NextCat']
data['value'] = 1

data = data.loc[data.Cat != data.NextCat]
transition_table = data.groupby(['Cat','NextCat']).size().reset_index()

to_table = transition_table.loc[transition_table.Cat < transition_table.NextCat].copy()
rev_table = transition_table.loc[transition_table.NextCat < transition_table.Cat].copy()
to_table.columns = ['Cat1','Cat2','to_count']
rev_table.columns = ['Cat2','Cat1','rev_count']

transition_table = pd.merge(to_table, rev_table, on=['Cat1','Cat2'], how='outer').fillna(0)

#transition_matrix = pd.pivot_table(data, values='value', index='Cat',
#                    columns=['NextCat'], aggfunc=np.sum, fill_value=0)

###########################################

# Looking at session transitions (item)

data = pd.read_csv(data_filepath+'_train_full.txt', sep='\t')
data['NextItemId'] = data['ItemId'].shift(-1)
data['NextSessionId'] = data['SessionId'].shift(-1)
data = data.loc[data.SessionId == data.NextSessionId]
data = data.loc[data.ItemId != data.NextItemId]
del data['NextSessionId']

# filter to just top 0.1% producrs
#product_views = data.groupby('ItemId').size().reset_index()
#product_views.columns = ['ItemId','num_views']
#products = product_views.loc[product_views.num_views >= np.percentile(product_views.num_views, 99.9),'ItemId'].tolist()
#data = data.loc[(data.ItemId.isin(products)) & (data.NextItemId.isin(products))]

data = data.loc[data.ItemId != data.NextItemId]
transition_table = data.groupby(['ItemId','NextItemId']).size().reset_index()
transition_table.columns = ['ItemId','NextItemId','value']
transition_table = transition_table.sort_values(by='value', axis=0, ascending=False, ignore_index=True)

# looking at top 0.1% of transitions
transition_table = transition_table.loc[transition_table.value >= np.percentile(transition_table.value, 99.9)]


to_table = transition_table.loc[transition_table.ItemId < transition_table.NextItemId].copy()
rev_table = transition_table.loc[transition_table.NextItemId < transition_table.ItemId].copy()
to_table.columns = ['Item1','Item2','to_count']
rev_table.columns = ['Item2','Item1','rev_count']

transition_summary = pd.merge(to_table, rev_table, on=['Item1','Item2'], how='outer').fillna(0)

transition_matrix = pd.pivot_table(transition_table, values='value', index='ItemId',
                                   columns=['NextItemId'], aggfunc=np.sum, fill_value=0)

########################################################

# Looking at % of users/sessions with % of items, starting from most popular items

data = pd.read_csv(data_filepath+'_train_full.txt', sep='\t', dtype={'ItemId':'category'})

product_views = data.groupby('ItemId').size().reset_index()
product_views.columns = ['ItemId','num_views']
product_views = product_views.sort_values(by='num_views', axis=0, ascending=False, ignore_index=True)

users = []
for i in range(100,0,-1):
    products = product_views.loc[product_views.num_views >= np.percentile(product_views.num_views, i),'ItemId'].tolist()
    num_users = data.loc[data.ItemId.isin(products),'UserId'].nunique()
    users.append(num_users)

users = [100*i/data.UserId.nunique() for i in users]

sessions = []
for i in range(100,0,-1):
    products = product_views.loc[product_views.num_views >= np.percentile(product_views.num_views, i),'ItemId'].tolist()
    num_sessions = data.loc[data.ItemId.isin(products),'SessionId'].nunique()
    sessions.append(num_sessions)

sessions = [100*i/data.SessionId.nunique() for i in sessions]

#plt.plot(range(1,101), users)
plt.plot(range(1,101), sessions)
plt.xlabel('Top Percentile of Items by Number of Views')
plt.ylabel('% of Sessions')
plt.savefig('./plots/sessions_by_products.png', dpi=300, transparent=False, bbox_inches='tight')
plt.clf()

##########################################################

# looking at category of 1st and last itemn in each session

data = pd.read_csv(data_filepath+'_train_full.txt', sep='\t')

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

data = pd.read_csv(data_filepath+'_train_full.txt', sep='\t')
data = data.groupby('UserId').SessionId.nunique()

print(f'% of users with more than 1 session: {100*data.loc[data>1].shape[0]/data.shape[0]}')

#######################################################

# repeated item consumption within sessions

data = pd.read_csv(data_filepath+'_train_full.txt', sep='\t')

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

########################################################

data = pd.read_csv(data_filepath+'_train_full.txt', sep='\t')

meta = pd.read_csv('data/item_dims1.csv')
meta2 = pd.read_csv('data/item_dims2.csv')
meta = pd.concat([meta, meta2], axis=0).drop_duplicates().reset_index(drop=True)
del meta2

meta['category'] = meta['item_hierarchy5_description']
meta.loc[meta.item_hierarchy5_description.str.lower() == meta.brand_description.str.lower(),'category'] = meta.loc[meta.item_hierarchy5_description.str.lower() == meta.brand_description.str.lower(),'item_hierarchy4_description']
meta = meta[['sap_code_var_p','category','brand_description']]

data = pd.merge(data, meta, left_on='ItemId', right_on='sap_code_var_p', how='left')








