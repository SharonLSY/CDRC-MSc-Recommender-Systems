import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import networkx as nx
import argparse

from utils.sr_gnn_utils import  convert_to_seq, build_graph, Data, split_validation
from evaluation.loader import load_data_session
from evaluation.metrics.accuracy import MRR, HitRate
import evaluation.evaluation as evaluation
from algorithms.sr_gnn.model import *


'''
FILE PARAMETERS
'''
folder = '28062021/'
PATH_PROCESSED = './data/prepared/' + folder
FILE = 'test'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
opt = parser.parse_args()
print(opt)

train, val = load_data_session(PATH_PROCESSED, FILE, slice_num=0, train_eval=True)
#train = train.loc[train.SessionId.isin([312893, 903890])]
#val = train.loc[(train.SessionId == 312893) & (train.ItemId.isin([1004249,1002524,4804056]))].copy()

itemids = train['ItemId'].unique()
item_dict = pd.Series(data=np.arange(len(itemids)), index=itemids).reset_index()
item_dict.columns = ['ItemId', 'ItemIdx']

train_seq = convert_to_seq(train, item_dict)
val_seq = convert_to_seq(val, item_dict)

train_data = Data(train_seq, shuffle=True)
val_data = Data(val_seq, shuffle=False)

n_node = len(itemids)
model = trans_to_cuda(SessionGraph(opt, n_node))

start = time.time()
best_result = [0, 0]
best_epoch = [0, 0]
bad_counter = 0
for epoch in range(opt.epoch):
    print('-------------------------------------------------------')
    print('epoch: ', epoch)
    hit, mrr = train_test(model, train_data, val_data)
    flag = 0
    if hit >= best_result[0]:
        best_result[0] = hit
        best_epoch[0] = epoch
        flag = 1
    if mrr >= best_result[1]:
        best_result[1] = mrr
        best_epoch[1] = epoch
        flag = 1
    print('Best Result:')
    print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d'% (best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
    bad_counter += 1 - flag
    if bad_counter >= opt.patience:
        break
print('-------------------------------------------------------')
end = time.time()
print("Run time: %f s" % (end - start))