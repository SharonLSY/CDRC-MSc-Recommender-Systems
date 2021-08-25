import importlib
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from evaluation.loader import load_data_session
from builtins import Exception
from run_config import run_file, create_algorithms_dict, create_metric_list, load_evaluation, eval_algorithm, write_results_csv

PATH_PROCESSED = './data/prepared/beauty_models/'
FILE = 'browsing_data'
conf = 'conf/evaluate_scalability.yml'



for perc in range(10,101,10):
    print(perc)
    train, test = load_data_session(PATH_PROCESSED, FILE, train_eval=False)
    
    file = Path(conf)
    if file.is_file():
    
        print('Loading file')
        stream = open(str(file))
        c = yaml.load(stream)
        stream.close()
        
    algorithms = create_algorithms_dict(c['algorithms'])
    metrics = create_metric_list(c['metrics'])
    evaluation = load_evaluation(c['evaluation'])
    
    # take the last perc of train
    train_sessions = train.SessionId.unique().tolist()
    total_sessions = len(train_sessions)
    train_sessions = train_sessions[int((1-perc/100)*total_sessions):]
    train = train[np.in1d(train.SessionId, train_sessions)]
    
    if conf == 'conf/evaluate_scalability.yml':
        # need to limit test set to only items in training set
        item_list = train.ItemId.unique().tolist()
        test = test[test.ItemId.isin(item_list)]
        tslength = test.groupby('SessionId').size()
        test = test[np.in1d(test.SessionId, tslength[tslength>=2].index)]
    else:
        new_items = [i for i in test.ItemId.unique() if i not in train.ItemId.unique()]
    
    c['key'] = 'scalability_' + str(perc)
    
    for m in metrics:
        m.init(train)
      
    results = {}
    
    for k, a in algorithms.items():
        eval_algorithm(train, test, k, a, evaluation, metrics, results, c, slice=None, iteration=None, out=False)
    
    
    results_keys = list(results.keys())
    for k in results_keys:
        results[k].append( ('perc:', perc) )
        results[k].append( ('num_training_items:', train.ItemId.nunique()) )
        if conf == 'conf/evaluate_scalability2.yml':
            results[k].append( ('num_new_items:', len(new_items)) )
        results[k].append( ('num_training_sessions:', train.SessionId.nunique()) )
        
    
    write_results_csv(results, c, iteration=None)
    
