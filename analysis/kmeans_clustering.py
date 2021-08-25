import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from tqdm import tqdm
import re
import string

from evaluation.loader import load_data_session
from utils.utils import *

PATH_PROCESSED = './data/prepared/beauty_models/'
FILE = 'browsing_data'

df, _ = load_data_session(PATH_PROCESSED, FILE, train_eval=False)

# KMEANS
session_items = df.loc[:,['SessionId','ItemId']].drop_duplicates()

enc = LabelEncoder()
session_items['ItemIdx'] = enc.fit_transform(session_items.ItemId)
session_items['SessionIdx'] = enc.fit_transform(session_items.SessionId)
binary_matrix = generate_csr_matrix(session_items, 'SessionIdx', session_items['ItemIdx'].max() +1, item_col='ItemIdx')

sparsity = binary_matrix.nnz/np.prod(binary_matrix.shape)
print(f'Sparsity: {round(100*(1-sparsity),2)}%')

svd = TruncatedSVD().fit(binary_matrix)
plt.plot(np.cumsum(svd.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')

Sum_of_squared_distances = []
K = [100,200,300,400,500]
for k in tqdm(K):
    km = KMeans(n_clusters=k)
    km = km.fit(binary_matrix)
    Sum_of_squared_distances.append(km.inertia_)
    
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()

# TFIDF + KMEANS
session_items = df[['SessionId','ItemId']].drop_duplicates().sort_values(by=['SessionId','ItemId'], ignore_index=True).groupby('SessionId').ItemId.apply(lambda x: ' '.join(x)).reset_index()

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(session_items.ItemId)

svd = TruncatedSVD().fit(tfidf)
plt.plot(np.cumsum(svd.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')

Sum_of_squared_distances = []
for k in tqdm(K):
    km = KMeans(n_clusters=k)
    km = km.fit(tfidf)
    Sum_of_squared_distances.append(km.inertia_)
    
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()

# TFIDF + normalize + KMEANS

tfidf_normalized = tfidf / tfidf.sum(axis=1).reshape(-1,1)

Sum_of_squared_distances2 = []
for k in tqdm(K):
    km = KMeans(n_clusters=k)
    km = km.fit(tfidf_normalized)
    Sum_of_squared_distances2.append(km.inertia_)
    
plt.plot(K, Sum_of_squared_distances2, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()
