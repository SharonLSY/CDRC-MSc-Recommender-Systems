import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
import pickle

#data config (all methods)
DATA_PATH = './data/'
DATA_PATH_PROCESSED = './data/prepared/test_date_trial/'
DATA_FILE = 'test'
NEW_ITEMS=True
SESSION_LENGTH = 30 * 60 # 30 min -- if next event is > 30 min away, then separate session

#filtering config (all methods)
MIN_SESSION_LENGTH = 2
MIN_ITEM_SUPPORT = 5

#min date config
MIN_DATE = '2014-04-01'
TEST_DATE = '2020-04-15'

#days test default config 
DAYS_TEST = 1

#slicing default config
NUM_SLICES = 5 #10
DAYS_OFFSET = 0
DAYS_TRAIN = 3
DAYS_TEST = 1
DAYS_SHIFT = DAYS_TRAIN + DAYS_TEST

# preprocessing from test date
def preprocess_from_test_date(path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED, 
                              min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH, 
                              test_date=TEST_DATE, days_train = DAYS_TRAIN, days_test=DAYS_TEST,
                              new_items=NEW_ITEMS):
    data, buys = load_data( path+file )
    # data = filter_data( data, min_item_support, min_session_length )
    split_from_test_date( data, path_proc+file, test_date, 
                         min_item_support, min_session_length,
                         days_train, days_test, new_items)

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
    
def load_data( file ) : 
    
    #load csv
    data = pd.read_parquet( file+'.parquet', engine='auto', 
                           columns=['event_time', 'user_id', 'event_type', 'product_id'],
                           )
    
    #specify header names
    data.columns = ['Time','UserId','Type','ItemId']
    
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
    
    cart = data[data.Type == 'purchase']
    #data = data[data.Type == 'cart']
    del data['Type']
    
    
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
    data = data[np.in1d(data.ItemId, item_supports[ item_supports>= min_item_support ].index)]
    
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
                         days_train, days_test, new_items=False ) :
    
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
    valid = valid[np.in1d(valid.ItemId, train_tr.ItemId)]
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




