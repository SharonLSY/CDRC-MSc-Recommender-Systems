import argparse
import numpy as np
import os
import pandas as pd
import pickle
import string
import re
from datetime import datetime
from scipy.sparse import save_npz, vstack
from sklearn.preprocessing import LabelEncoder

from preprocessing.preprocessing import *
from utils.utils import *

proc_folder = './data/prepared/beauty_new_items/'

with open(proc_folder+'browsing_data_full_item_list.txt', 'rb') as pickle_file:
    item_list = pickle.load(pickle_file)


# Adapted from CEASE: https://github.com/olivierjeunen/ease-side-info-recsys-2020 
minsup = 3
maxsup = len(item_list) // 4

cols = ['sap_code_var_p','item_description','merchandise_category_description','item_hierarchy2_description',
        'item_hierarchy3_description', 'item_hierarchy4_description','item_hierarchy5_description',#
        'item_hierarchy6_description','ladder','sub_ladder','item_deletion_date']
meta = pd.read_csv('data/item_dims1.csv', usecols=cols, dtype={'sap_code_var_p':'category'})
meta2 = pd.read_csv('data/item_dims2.csv', usecols=cols, dtype={'sap_code_var_p':'category'})
meta = pd.concat([meta, meta2], axis=0).drop_duplicates(subset=['sap_code_var_p'], ignore_index=True)
meta = meta.loc[meta.sap_code_var_p.isin(item_list)]

meta['item_description'] = meta['item_description'].str.lower()
meta.dropna(axis=0, how='all', inplace=True)
meta.reset_index(drop=True, inplace=True)

meta.fillna('', inplace=True)

cat_cols = ['item_hierarchy2_description','item_hierarchy3_description', 
            'item_hierarchy4_description','item_hierarchy5_description',
            'item_hierarchy6_description']
meta['product_category'] = meta[cat_cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

meta['ladder'] = meta[['ladder','sub_ladder']].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

meta = meta.rename(columns={'sap_code_var_p': 'product_id', 'item_description': 'product_name'})
meta = meta[['product_id','product_name','product_category','merchandise_category_description','ladder']]

# product_name
pattern = re.compile('[^A-Za-z0-9 ]+')
meta['product_name'] = meta['product_name'].apply(lambda s: pattern.sub('', s).lower())
title = pd.DataFrame(meta.product_name.str.split(' ').tolist(), index = meta.product_id).stack().reset_index([0, 'product_id'])
title.columns = ['product_id','product_name']
title.drop_duplicates(inplace=True)
title.dropna(inplace = True)
title['product_name'] = title['product_name'].str.strip()
title = title.loc[~title.product_name.isin([''])]
title = title[~title['product_name'].str.contains('\d')]
word2count = title.groupby('product_name')['product_id'].apply(lambda x: len(set(x))).reset_index().rename(columns = {'product_id': 'count'})
word2count = word2count[word2count['count'] >= minsup]
word2count = word2count[word2count['count'] <= maxsup]
title = title.merge(word2count[['product_name']], on = 'product_name', how = 'right')
print(title['product_name'].value_counts())

# product_category
meta['product_category'] = meta['product_category'].apply(lambda s: pattern.sub('', s).lower())
cat = pd.DataFrame(meta.product_category.str.split(' ').tolist(), index = meta.product_id).stack().reset_index([0, 'product_id'])
cat.columns = ['product_id','product_category']
cat.drop_duplicates(inplace=True)
cat.dropna(inplace = True)
cat['product_category'] = cat['product_category'].str.strip()
cat = cat.loc[~cat.product_category.isin([''])]
cat = cat[~cat['product_category'].str.contains('\d')]
word2count = cat.groupby('product_category')['product_id'].apply(lambda x: len(set(x))).reset_index().rename(columns = {'product_id': 'count'})
word2count = word2count[word2count['count'] >= minsup]
word2count = word2count[word2count['count'] <= maxsup]
cat = cat.merge(word2count[['product_category']], on = 'product_category', how = 'right')
print(cat['product_category'].value_counts())

# merchandise_category_description
meta['merchandise_category_description'] = meta['merchandise_category_description'].apply(lambda s: pattern.sub('', s).lower())
merch = pd.DataFrame(meta.merchandise_category_description.str.split(' ').tolist(), index = meta.product_id).stack().reset_index([0, 'product_id'])
merch.columns = ['product_id','merchandise_category_description']
merch.drop_duplicates(inplace=True)
merch.dropna(inplace = True)
merch['merchandise_category_description'] = merch['merchandise_category_description'].str.strip()
merch = merch.loc[~merch.merchandise_category_description.isin([''])]
merch = merch[~merch['merchandise_category_description'].str.contains('\d')]
word2count = merch.groupby('merchandise_category_description')['product_id'].apply(lambda x: len(set(x))).reset_index().rename(columns = {'product_id': 'count'})
word2count = word2count[word2count['count'] >= minsup]
word2count = word2count[word2count['count'] <= maxsup]
merch = merch.merge(word2count[['merchandise_category_description']], on = 'merchandise_category_description', how = 'right')
print(merch['merchandise_category_description'].value_counts())

# ladder
meta['ladder'] = meta['ladder'].apply(lambda s: pattern.sub('', s).lower())
ladder = pd.DataFrame(meta.ladder.str.split(' ').tolist(), index = meta.product_id).stack().reset_index([0, 'product_id'])
ladder.columns = ['product_id','ladder']
ladder.drop_duplicates(inplace=True)
ladder.dropna(inplace = True)
ladder['ladder'] = ladder['ladder'].str.strip()
ladder = ladder.loc[~ladder.ladder.isin([''])]
ladder = ladder[~ladder['ladder'].str.contains('\d')]
word2count = ladder.groupby('ladder')['product_id'].apply(lambda x: len(set(x))).reset_index().rename(columns = {'product_id': 'count'})
word2count = word2count[word2count['count'] >= minsup]
word2count = word2count[word2count['count'] <= maxsup]
ladder = ladder.merge(word2count[['ladder']], on = 'ladder', how = 'right')
print(ladder['ladder'].value_counts())


# Ensure proper integer identifiers
item_enc = LabelEncoder()
cat_enc = LabelEncoder()
desc_enc = LabelEncoder()
title_enc = LabelEncoder()
brand_enc = LabelEncoder()
meta['product_idx'] = item_enc.fit_transform(meta['product_id'])
cat['product_id'] = item_enc.transform(cat['product_id'])
cat['product_category'] = cat_enc.fit_transform(cat['product_category'].astype(str))
merch['product_id'] = item_enc.transform(merch['product_id'])
merch['merchandise_category_description'] = desc_enc.fit_transform(merch['merchandise_category_description'])
title['product_id'] = item_enc.transform(title['product_id'])
title['product_name'] = title_enc.fit_transform(title['product_name'])
ladder['product_id'] = item_enc.transform(ladder['product_id'])
ladder['ladder'] = brand_enc.fit_transform(ladder['ladder'])



# Generate Metadata-to-item mapping
X_cat = generate_csr_matrix(cat, 'product_category', meta['product_idx'].max() + 1)
X_desc = generate_csr_matrix(merch, 'merchandise_category_description', meta['product_idx'].max() + 1)
X_title = generate_csr_matrix(title, 'product_name', meta['product_idx'].max() + 1)
X_brand = generate_csr_matrix(ladder, 'ladder', meta['product_idx'].max() + 1)
X_meta = vstack((X_cat,X_desc,X_title,X_brand)).T


itemidmap = pd.DataFrame({'ItemId': meta['product_id'], 'ItemIdx': meta['product_idx']})
itemidmap.to_csv(proc_folder+'meta_item.csv', index=False)

# Write out metadata-item matrix
print(datetime.now(), 'Writing out metadata-item matrix...')
print('Size of metadata-item matrix:', X_meta.shape)
save_npz(proc_folder+'X_meta.npz', X_meta)

