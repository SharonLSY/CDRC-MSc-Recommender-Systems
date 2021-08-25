import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
import pickle
import os
from random import sample, seed

#data config (all methods)
DATA_PATH = './data/'
DATA_PATH_PROCESSED = './data/prepared/'
DATA_FILE = 'test'
NEW_ITEMS=True
SESSION_LENGTH = 30 * 60 # 30 min -- if next event is > 30 min away, then separate session
RANDOM_SAMPLE = None
LIMIT_BEAUTY=True

#filtering config (all methods)
MIN_SESSION_LENGTH = 2
MIN_ITEM_SUPPORT = 5

#min date config
MIN_DATE = '2014-04-01'
TEST_DATE = '2021-07-01'

#days test default config 
DAYS_TEST = 1

#slicing default config
NUM_SLICES = 5 #10
DAYS_OFFSET = 0
DAYS_TRAIN = 30
DAYS_TEST = 3
DAYS_SHIFT = DAYS_TRAIN + DAYS_TEST

seed(123)

# preprocessing from test date
def preprocess_from_test_date(path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED, 
                              min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH, 
                              test_date=TEST_DATE, days_train = DAYS_TRAIN, days_test=DAYS_TEST,
                              new_items=NEW_ITEMS, random_sample=None, limit_beauty=False
                              ):
    
    
    seed(123)
    os.makedirs(path_proc, exist_ok=True) # make directory if path doesn't exist
    
    data, buys = load_data( path+file, limit_beauty )
    # data = filter_data( data, min_item_support, min_session_length )
    split_from_test_date( data, path_proc+file, test_date, 
                         min_item_support, min_session_length,
                         days_train, days_test, new_items, random_sample)

#preprocessing from original gru4rec
def preprocess_org( path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH ):
    
    data, buys = load_data( path+file )
    data = filter_data( data, min_item_support, min_session_length )
    split_data_org( data, path_proc+file )

#preprocessing from original gru4rec but from a certain point in time
def preprocess_org_min_date( path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH, min_date=MIN_DATE ):
    
    data, buys = load_data( path+file )
    data = filter_data( data, min_item_support, min_session_length )
    data = filter_min_date( data, min_date )
    split_data_org( data, path_proc+file )

#preprocessing adapted from original gru4rec
def preprocess_days_test( path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH, days_test=DAYS_TEST ):
    
    data, buys = load_data( path+file )
    data = filter_data( data, min_item_support, min_session_length )
    split_data( data, path_proc+file, days_test )

#preprocessing from original gru4rec but from a certain point in time
def preprocess_days_test_min_date( path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH, days_test=DAYS_TEST, min_date=MIN_DATE ):
    
    data, buys = load_data( path+file )
    data = filter_data( data, min_item_support, min_session_length )
    data = filter_min_date( data, min_date )
    split_data( data, path_proc+file, days_test )

#preprocessing to create data slices with a window
def preprocess_slices( path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH,
                       num_slices = NUM_SLICES, days_offset = DAYS_OFFSET, days_shift = DAYS_SHIFT, days_train = DAYS_TRAIN, days_test=DAYS_TEST ):
    
    data, buys = load_data( path+file )
    data = filter_data( data, min_item_support, min_session_length )
    slice_data( data, path_proc+file, num_slices, days_offset, days_shift, days_train, days_test )
    
#just load and show info
def preprocess_info( path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH ):
    
    data, buys = load_data( path+file )
    data = filter_data( data, min_item_support, min_session_length )
    
def preprocess_save( path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH ):
    
    data, buys = load_data( path+file )
    data = filter_data( data, min_item_support, min_session_length )
    data.to_csv(path_proc + file + '_preprocessed.txt', sep='\t', index=False)
    
#preprocessing to create a file with buy actions
def preprocess_buys( path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED ): 
    data, buys = load_data( path+file )
    store_buys(buys, path_proc+file)
    
def load_data( file, limit_beauty=False ) : 
    
    #load csv
    data = pd.read_csv( file+'.csv', 
                       dtype={'product_id':'category'}
                       )
    
    if limit_beauty:
        meta = pd.read_csv('data/item_dims1.csv', usecols = ['sap_code_var_p','item_hierarchy2_description'])
        meta2 = pd.read_csv('data/item_dims2.csv', usecols = ['sap_code_var_p','item_hierarchy2_description'])
        meta = pd.concat([meta, meta2], axis=0).drop_duplicates().reset_index(drop=True)
        del meta2
        beauty_products = meta.loc[meta.item_hierarchy2_description == 'Beauty','sap_code_var_p'].unique()
        data = data.loc[data.product_id.isin(beauty_products)]
    
#    data.columns = [str(col).lower() for col in data.columns]
    data['Time'] = data['browse_date'] + ' ' + data['browse_time'] + ' UTC'

    #specify header names
    data.rename(columns={'cookie_id':'UserId','product_id':'ItemId'}, inplace=True)
    print(data.columns)
    data = data.loc[:,['Time','UserId','ItemId']]
    
    
    data['Time'] = pd.to_datetime(data['Time'], format='%Y-%m-%d %H:%M:%S UTC', errors='ignore')
    # data = data.loc[(data.Time >= '2020-02-20')]
    data['Time'] = data['Time'].apply(lambda x: x.timestamp())
    
    data.sort_values( ['UserId','Time'], ascending=True, inplace=True )
    
    #sessionize    
    data['TimeTmp'] = pd.to_datetime(data.Time, unit='s')
    
    data.sort_values( ['UserId','TimeTmp'], ascending=True, inplace=True )
#     users = data.groupby('UserId')

    
    # Assign different session if next event is > SESSION_LENGTH *OR* UserId is different
    data['TimeShift'] = data['TimeTmp'].shift(1)
    data['TimeDiff'] = (data['TimeTmp'] - data['TimeShift']).dt.total_seconds().abs()
    data['UserShift'] = data['UserId'].shift(1)    
    data['SessionIdTmp'] = ( (data['TimeDiff'] > SESSION_LENGTH) | (data['UserId'] != data['UserShift']) ).astype( int )
    data['SessionId'] = data['SessionIdTmp'].cumsum( skipna=False )
    
    # Remove duplicate consecutive product in the same session
    data['ItemShift'] = data['ItemId'].shift(1)
    data['SessionShift'] = data['SessionId'].shift(1)
    data = data.loc[~((data['ItemShift'] == data['ItemId']) & (data['SessionShift'] == data['SessionId']))]
    
    del data['SessionIdTmp'], data['TimeShift'], data['TimeDiff'], data['UserShift'], data['ItemShift'], data['SessionShift']
    
    
    data.sort_values( ['SessionId','Time'], ascending=True, inplace=True )
    
    # cart = data[data.Type == 'purchase']
    # data = data[data.Type == 'cart']
    # del data['Type']
    cart=None
    
    
    #output
    data_start = datetime.fromtimestamp( data.Time.min(), timezone.utc )
    data_end = datetime.fromtimestamp( data.Time.max(), timezone.utc )
    
    del data['TimeTmp']
    
    print('Loaded data set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format( len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.date().isoformat(), data_end.date().isoformat() ) )
    
    return data, cart;


def filter_data( data, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH ) : 
    
    # remove single length sessions before filtering by item and minimum session length
    session_lengths = data.groupby('SessionId').size()
    data = data[np.in1d(data.SessionId, session_lengths[ session_lengths>1 ].index)]
    
    # filter item support
    item_supports = data.groupby('ItemId').size()
    print(f'Removed {item_supports[ item_supports < min_item_support ].shape[0]} items')
    item_supports = item_supports[ item_supports>= min_item_support ].index.unique()
    #data = data[np.in1d(data.ItemId, item_supports[ item_supports>= min_item_support ].index)]
    data = data[data.ItemId.isin(item_supports)]
    
    # filter session length
    session_lengths = data.groupby('SessionId').size()
    print(f'Removed {session_lengths[ session_lengths < min_session_length ].shape[0]} sessions')
    data = data[np.in1d(data.SessionId, session_lengths[ session_lengths>= min_session_length ].index)]
    
    #output
    data_start = datetime.fromtimestamp( data.Time.min(), timezone.utc )
    data_end = datetime.fromtimestamp( data.Time.max(), timezone.utc )
    
    print('Filtered data set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format( len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.date().isoformat(), data_end.date().isoformat() ) )
    
    return data;

def filter_min_date( data, min_date='2014-04-01' ) :
    
    min_datetime = datetime.strptime(min_date + ' 00:00:00', '%Y-%m-%d %H:%M:%S')
    
    #filter
    session_max_times = data.groupby('SessionId').Time.max()
    session_keep = session_max_times[ session_max_times > min_datetime.timestamp() ].index
    
    data = data[ np.in1d(data.SessionId, session_keep) ]
    
    #output
    data_start = datetime.fromtimestamp( data.Time.min(), timezone.utc )
    data_end = datetime.fromtimestamp( data.Time.max(), timezone.utc )
    
    print('Filtered data set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format( len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.date().isoformat(), data_end.date().isoformat() ) )
    
    return data;


def split_data_org( data, output_file ) :
    
    tmax = data.Time.max()
    session_max_times = data.groupby('SessionId').Time.max()
    session_train = session_max_times[session_max_times < tmax-86400].index
    session_test = session_max_times[session_max_times >= tmax-86400].index
    train = data[np.in1d(data.SessionId, session_train)]
    test = data[np.in1d(data.SessionId, session_test)]
    test = test[np.in1d(test.ItemId, train.ItemId)]
    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength>=2].index)]
    print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.SessionId.nunique(), train.ItemId.nunique()))
    train.to_csv(output_file + '_train_full.txt', sep='\t', index=False)
    print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test), test.SessionId.nunique(), test.ItemId.nunique()))
    test.to_csv(output_file + '_test.txt', sep='\t', index=False)
    
    tmax = train.Time.max()
    session_max_times = train.groupby('SessionId').Time.max()
    session_train = session_max_times[session_max_times < tmax-86400].index
    session_valid = session_max_times[session_max_times >= tmax-86400].index
    train_tr = train[np.in1d(train.SessionId, session_train)]
    valid = train[np.in1d(train.SessionId, session_valid)]
    valid = valid[np.in1d(valid.ItemId, train_tr.ItemId)]
    tslength = valid.groupby('SessionId').size()
    valid = valid[np.in1d(valid.SessionId, tslength[tslength>=2].index)]
    print('Train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train_tr), train_tr.SessionId.nunique(), train_tr.ItemId.nunique()))
    train_tr.to_csv( output_file + '_train_tr.txt', sep='\t', index=False)
    print('Validation set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(valid), valid.SessionId.nunique(), valid.ItemId.nunique()))
    valid.to_csv( output_file + '_train_valid.txt', sep='\t', index=False)
    
    
    
def split_data( data, output_file, days_test ) :
    
    data_end = datetime.fromtimestamp( data.Time.max(), timezone.utc )
    test_from = data_end - timedelta( days_test )
    
    session_max_times = data.groupby('SessionId').Time.max()
    session_train = session_max_times[ session_max_times < test_from.timestamp() ].index
    session_test = session_max_times[ session_max_times >= test_from.timestamp() ].index
    train = data[np.in1d(data.SessionId, session_train)]
    test = data[np.in1d(data.SessionId, session_test)]
    test = test[np.in1d(test.ItemId, train.ItemId)]
    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength>=2].index)]
    print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.SessionId.nunique(), train.ItemId.nunique()))
    train.to_csv(output_file + '_train_full.txt', sep='\t', index=False)
    print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test), test.SessionId.nunique(), test.ItemId.nunique()))
    test.to_csv(output_file + '_test.txt', sep='\t', index=False)


def split_from_test_date(data, output_file, test_date, 
                         min_item_support, min_session_length,
                         days_train, days_test, new_items=False,
                         random_sample=None) :
    
    test_from = datetime.strptime(test_date + ' 00:00:00', '%Y-%m-%d %H:%M:%S')
    test_until = test_from + timedelta(days_test)
    train_from  = test_from - timedelta(days_train)
    valid_from = test_from - timedelta(days_test)
    
    session_max_times = data.groupby('SessionId').Time.max()
    session_all = session_max_times[ (session_max_times >= train_from.timestamp()) & (session_max_times < test_until.timestamp()) ].index
    data = data[np.in1d(data.SessionId, session_all)]
    
    data = filter_data( data, min_item_support, min_session_length )
    
    session_train = session_max_times[ (session_max_times >= train_from.timestamp()) & (session_max_times < test_from.timestamp()) ].index
    session_test = session_max_times[ (session_max_times >= test_from.timestamp()) & (session_max_times < test_until.timestamp()) ].index
    train = data[np.in1d(data.SessionId, session_train)]
    test = data[np.in1d(data.SessionId, session_test)]
    
    if random_sample != None:
        print(f'Sampling {random_sample} of all sessions')
        session_train = sample(train.SessionId.unique().tolist(), int(random_sample*train.SessionId.nunique()))
        session_test = sample(test.SessionId.unique().tolist(), int(random_sample*test.SessionId.nunique()))
        train = train[np.in1d(train.SessionId, session_train)]
        test = test[np.in1d(test.SessionId, session_test)]
        
    res = get_stats( pd.concat([train, test], ignore_index=True) )
    res.to_csv(output_file+'_stats.csv', index=False)
    
    if new_items:
        item_list = data.ItemId.unique().tolist()
        # write list into txt file
        with open(output_file + '_full_item_list.txt', 'wb') as file:
            pickle.dump(item_list, file)
        print(f'Kept {len(item_list) - train.ItemId.nunique()} new items')
    
    else:
        item_list = train.ItemId.unique().tolist()
        test = test[test.ItemId.isin(item_list)]
    
    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength>=2].index)]
    print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.SessionId.nunique(), train.ItemId.nunique()))
    train.to_csv(output_file + '_train_full.txt', sep='\t', index=False)
    print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test), test.SessionId.nunique(), test.ItemId.nunique()))
    test.to_csv(output_file + '_test.txt', sep='\t', index=False)
    
    session_train = session_max_times[ (session_max_times >= train_from.timestamp()) & (session_max_times < valid_from.timestamp()) ].index
    session_valid = session_max_times[ (session_max_times >= valid_from.timestamp()) & (session_max_times < test_from.timestamp()) ].index
    train_tr = train[np.in1d(train.SessionId, session_train)]
    valid = train[np.in1d(train.SessionId, session_valid)]
    if new_items is False:
        valid = valid[valid.ItemId.isin(train_tr.ItemId.unique())]
    tslength = valid.groupby('SessionId').size()
    valid = valid[np.in1d(valid.SessionId, tslength[tslength >= 2].index)]
    print('Train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train_tr), train_tr.SessionId.nunique(),
                                                                        train_tr.ItemId.nunique()))
    train_tr.to_csv(output_file + '_train_tr.txt', sep='\t', index=False)
    print('Validation set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(valid), valid.SessionId.nunique(),
                                                                             valid.ItemId.nunique()))
    valid.to_csv(output_file + '_train_valid.txt', sep='\t', index=False)
    
    print('{} / {} / {}'.format(train_from.date(), valid_from.date(), test_from.date()))

    
    
def slice_data( data, output_file, num_slices, days_offset, days_shift, days_train, days_test ): 
    
    for slice_id in range( 0, num_slices ) :
        split_data_slice( data, output_file, slice_id, days_offset+(slice_id*days_shift), days_train, days_test )

def split_data_slice( data, output_file, slice_id, days_offset, days_train, days_test, new_items=None ) :
    
    data_start = datetime.fromtimestamp( data.Time.min(), timezone.utc )
    data_end = datetime.fromtimestamp( data.Time.max(), timezone.utc )
    
    print('Full data set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}'.
          format( slice_id, len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.isoformat(), data_end.isoformat() ) )
    
    
    start = datetime.fromtimestamp( data.Time.min(), timezone.utc ) + timedelta( days_offset ) 
    middle =  start + timedelta( days_train )
    end =  middle + timedelta( days_test )
    
    #prefilter the timespan
    session_max_times = data.groupby('SessionId').Time.max()
    greater_start = session_max_times[session_max_times >= start.timestamp()].index
    lower_end = session_max_times[session_max_times <= end.timestamp()].index
    data_filtered = data[np.in1d(data.SessionId, greater_start.intersection( lower_end ))]
    
    print('Slice data set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {} / {}'.
          format( slice_id, len(data_filtered), data_filtered.SessionId.nunique(), data_filtered.ItemId.nunique(), start.date().isoformat(), middle.date().isoformat(), end.date().isoformat() ) )
    
    #split to train and test
    session_max_times = data_filtered.groupby('SessionId').Time.max()
    sessions_train = session_max_times[session_max_times < middle.timestamp()].index
    sessions_test = session_max_times[session_max_times >= middle.timestamp()].index
    
    train = data[np.in1d(data.SessionId, sessions_train)]
    
    print('Train set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}'.
          format( slice_id, len(train), train.SessionId.nunique(), train.ItemId.nunique(), start.date().isoformat(), middle.date().isoformat() ) )
    
    train.to_csv(output_file + '_train_full.'+str(slice_id)+'.txt', sep='\t', index=False)
    
    if new_items is None:
        keep_items = train.ItemId
    else:
        keep_items = list(set(train.ItemId + new_items))
    
    test = data[np.in1d(data.SessionId, sessions_test)]
    test = test[np.in1d(test.ItemId, train.ItemId)]
    
    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength>=2].index)]
    
    print('Test set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {} \n\n'.
          format( slice_id, len(test), test.SessionId.nunique(), test.ItemId.nunique(), middle.date().isoformat(), end.date().isoformat() ) )
    
    test.to_csv(output_file + '_test.'+str(slice_id)+'.txt', sep='\t', index=False)

    data_end = datetime.fromtimestamp(train.Time.max(), timezone.utc)
    valid_from = data_end - timedelta(days=days_test)
    session_max_times = train.groupby('SessionId').Time.max()
    session_train = session_max_times[session_max_times < valid_from.timestamp()].index
    session_valid = session_max_times[session_max_times >= valid_from.timestamp()].index
    train_tr = train[np.in1d(train.SessionId, session_train)]
    valid = train[np.in1d(train.SessionId, session_valid)]
    valid = valid[np.in1d(valid.ItemId, train_tr.ItemId)]
    tslength = valid.groupby('SessionId').size()
    valid = valid[np.in1d(valid.SessionId, tslength[tslength >= 2].index)]
    print('Train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train_tr), train_tr.SessionId.nunique(),
                                                                        train_tr.ItemId.nunique()))
    train_tr.to_csv(output_file + '_train_tr.'+ str(slice_id) + '.txt', sep='\t', index=False)
    print('Validation set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(valid), valid.SessionId.nunique(),
                                                                             valid.ItemId.nunique()))
    valid.to_csv(output_file + '_train_valid.'+ str(slice_id) + '.txt', sep='\t', index=False)


def store_buys( buys, target ):
    buys.to_csv( target + '_buys.txt', sep='\t', index=False )

def get_stats( dataframe ):
    print( 'get_stats ' )
    
    res = {}
    
    res['STATS'] = ['STATS']
#    res['name'] = [name]
    res['actions'] = [len(dataframe)]
    res['items'] = [ dataframe.ItemId.nunique() ]
    res['sessions'] = [ dataframe.SessionId.nunique() ]
    res['time_start'] = [ dataframe.Time.min() ]
    res['time_end'] = [ dataframe.Time.max() ]
    
    res['unique_per_session'] = dataframe.groupby('SessionId')['ItemId'].nunique().mean()
    
    res = pd.DataFrame(res)

    res['actions_per_session'] = res['actions'] / res['sessions']
    res['actions_per_items'] = res['actions'] / res['items']
    #res['sessions_per_action'] = res['sessions'] / res['actions']
    res['sessions_per_items'] = res['sessions'] / res['items']
    #res['items_per_actions'] = res['items'] / res['actions']
    res['items_per_session'] = res['items'] / res['sessions']
    res['span'] = res['time_end'] - res['time_start']
    res['days'] = res['span'] / 60 / 60 / 24
    
    return res

