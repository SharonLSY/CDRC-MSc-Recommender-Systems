import argparse
import numpy as np
import os
import pandas as pd
import pickle
import string
import re
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.phrases import Phrases, Phraser
from gensim.utils import simple_preprocess#
from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
from spacy.lang.en.stop_words import STOP_WORDS
from datetime import datetime
from scipy.sparse import save_npz, vstack
from sklearn.preprocessing import LabelEncoder

from preprocessing.preprocessing import *
from utils.utils import *

# Adapted from: https://github.com/kcalizadeh/phil_nlp/blob/master/Notebooks/3_w2v.ipynb

df = pd.read_csv('data/BDC.BROWSING_PRODUCT_VIEW.csv', usecols=['PRODUCT_ID','PRODUCT_NAME'])
df.columns = [str(col).lower() for col in df.columns]
df = df.drop_duplicates()

df['name_length'] = df['product_name'].str.split(' ').apply(len)

# word cleaning
def clean_sentence(sentence):
    sentence = sentence.lower().strip()
    sentence = re.sub(r'[^a-z0-9\s]', '', sentence)
    return re.sub(r'\s{2,}', ' ', sentence)

def tokenize(sentence):
    return [token for token in sentence.split() if token not in STOP_WORDS]

def build_phrases(sentences, min_count=5, threshold=10):
    phrases = Phrases(sentences,
                      min_count=min_count,
                      threshold=threshold,
                      progress_per=1000)
    return Phraser(phrases)

def sentence_to_bi_grams(phrases_model, sentence):
    return ' '.join(phrases_model[sentence])

df['product_name'] = df['product_name'].apply(lambda x: clean_sentence(x))
sent = [tokenize(row) for row in df['product_name']]

df['gensim_tokenized'] = df['product_name'].map(lambda x: simple_preprocess(x.lower(), deacc=True, max_len=100))

#phrases_model = build_phrases(sent, min_count=10)
#sent = [sentence_to_bi_grams(phrases_model,s) for s in sent]
#sent = [row.split(' ') for row in df['product_name']]

#model = Word2Vec(sent, min_count=2, size=100, workers=3, window=5, sg = 0)
#
#def find_most_similar(s, n=5):
#    s = s.split(' ')
##    s = ' '.join(phrases_model[s])
#    return model.wv.most_similar(s)[:n]
#
#find_most_similar('benefit hoola matte bronzer')

from scipy import spatial

documents = df['gensim_tokenized']

# format the series to be used
stopwords = ['ml','g']

sentences = [sentence for sentence in documents]
cleaned = []
for sentence in sentences:
  cleaned_sentence = [word.lower() for word in sentence]
  cleaned_sentence = [word for word in sentence if word not in stopwords]
  cleaned.append(cleaned_sentence)

# get bigrams
bigram = Phrases(cleaned, min_count=7, threshold=13, delimiter=b' ')
bigram_phraser = Phraser(bigram)

bigramed_tokens = []
for sent in cleaned:
    tokens = bigram_phraser[sent]
    bigramed_tokens.append(tokens)

# run again to get trigrams
trigram = Phrases(bigramed_tokens, min_count=10, threshold=17, delimiter=b' ')
trigram_phraser = Phraser(trigram)

trigramed_tokens = []
for sent in bigramed_tokens:
    tokens = trigram_phraser[sent]
    trigramed_tokens.append(tokens)
    
glove_file = datapath('glove/glove.twitter.27B.100d.txt')
tmp_file = get_tmpfile("test_word2vec.txt")

_ = glove2word2vec(glove_file, tmp_file)

glove_vectors = KeyedVectors.load_word2vec_format(tmp_file)

#print(bigramed_tokens[2567])
#print(trigramed_tokens[2567])

# build a toy model to update with
base_model = Word2Vec(size=200, min_count=5, window=2, sg = 1)
base_model.build_vocab(trigramed_tokens)
total_examples = base_model.corpus_count

# add GloVe's vocabulary & weights
base_model.build_vocab([list(glove_vectors.vocab.keys())], update=True)

# train on our data
base_model.train(trigramed_tokens, total_examples=total_examples, epochs=base_model.epochs)
base_model_wv = base_model.wv
del base_model

############################################

def convert_phrase(p):
    p = simple_preprocess(p.lower(), deacc=True, max_len=100)
    p = [word for word in p if word not in stopwords]
    p = bigram_phraser[p]
    p = trigram_phraser[p]
    return p

s0 = 'loreal paris revitalift pro retinol hydrating smoothing serum 30ml'
s1 = 'loreal men expert hydra energetic shower gel 300ml'
s2 = 'kvd beauty lockit liquid foundation 30ml'
s3 = 'fenty beauty pro filtr soft matte longwear foundation'

def get_vector(s):
    return np.mean(np.array([base_model_wv[i] for i in convert_phrase(s) if i in base_model_wv.vocab]), axis=0)


print('s0 vs s1 ->',1 - spatial.distance.cosine(get_vector(s0), get_vector(s1)))
print('s0 vs s2 ->', 1 - spatial.distance.cosine(get_vector(s0), get_vector(s2)))
print('s0 vs s3 ->', 1 - spatial.distance.cosine(get_vector(s0), get_vector(s3)))
print('s2 vs s3 ->', 1 - spatial.distance.cosine(get_vector(s2), get_vector(s3)))








