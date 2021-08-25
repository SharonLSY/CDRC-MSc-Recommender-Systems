from evaluation.loader import load_data_session
from preprocessing.preprocessing import *

PATH = './data/'
PATH_PROCESSED = './data/prepared/beauty_models/'
PATH_PROCESSED2 = './data/prepared/beauty_new_items/'
PATH_PROCESSED3 = './data/prepared2/beauty_models/'
PATH_PROCESSED4 = './data/prepared2/beauty_new_items/'
FILE = 'browsing_data'
conf = 'conf/evaluate_scalability2.yml'

test_date = '2021-07-01'
days_train=10
days_test=1
min_item_support=5
min_session_length=2
new_items=True
limit_beauty=True



train, test = load_data_session(PATH_PROCESSED, FILE, train_eval=True)
train2, test2 = load_data_session(PATH_PROCESSED2, FILE, train_eval=False)
train3, test3 = load_data_session(PATH_PROCESSED3, FILE, train_eval=False)
train4, test4 = load_data_session(PATH_PROCESSED4, FILE, train_eval=False)

train_sessions = train.SessionId.unique().tolist()
train2_sessions = train2.SessionId.unique().tolist()
train3_sessions = train3.SessionId.unique().tolist()
train4_sessions = train4.SessionId.unique().tolist()

# all sessions in original dataset
test_sessions = test.SessionId.unique().tolist()
all_sessions = train_sessions + test_sessions

data, buys = load_data( PATH+FILE, limit_beauty )



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

train = train[np.in1d(train.SessionId, all_sessions)]
test = test[np.in1d(test.SessionId, all_sessions)]

output_file=PATH_PROCESSED2+FILE

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
