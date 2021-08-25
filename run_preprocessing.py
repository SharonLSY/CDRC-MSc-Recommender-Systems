import preprocessing.preprocessing as pp
import time


'''
preprocessing method ["info","org","days_test","slice"]
    info: just load and show info
    org: from gru4rec (last day => test set)
    org_min_date: from gru4rec (last day => test set) but from a minimal date onwards
    days_test: adapted from gru4rec (last N days => test set)
    slice: new (create multiple train-test-combinations with a window approach  
    buys: load buys and safe file to prepared
    test_date: splits data based on test_date and works backwards to obtain training set based on defined number of training days
'''
METHOD = "test_date"

'''
data config (all methods)
'''
PATH = './data/'
PATH_PROCESSED = './data/prepared2/beauty_models/'
FILE = 'browsing_data'
NEW_ITEMS = False # bool, if True then it keeps items in test set
RANDOM_SAMPLE = 0.05
LIMIT_BEAUTY=True
'''
org_min_date config
'''
#MIN_DATE = '2014-07-01'
TEST_DATE = '2021-07-01'

'''
filtering config (all methods)
'''
MIN_SESSION_LENGTH = 2
MIN_ITEM_SUPPORT = 5

'''
slicing default config
'''
#NUM_SLICES = 5
#DAYS_OFFSET = 0 #number of days the training start date is shifted after creating one slice
#each slice consists of...
DAYS_TRAIN = 10 #30
DAYS_TEST = 1 #3
#DAYS_SHIFT = DAYS_TRAIN + DAYS_TEST

if __name__ == '__main__':
    '''
    Run the preprocessing configured above.
    '''
    
    print( "START preprocessing ", METHOD )
    sc, st = time.perf_counter(), time.time()
    
    if METHOD == "info":
        pp.preprocess_info( PATH, FILE, MIN_ITEM_SUPPORT, MIN_SESSION_LENGTH )
    
    elif METHOD == "org":
        pp.preprocess_org( PATH, FILE, PATH_PROCESSED, MIN_ITEM_SUPPORT, MIN_SESSION_LENGTH )
     
    elif METHOD == "org_min_date":
        pp.preprocess_org_min_date( PATH, FILE, PATH_PROCESSED, MIN_ITEM_SUPPORT, MIN_SESSION_LENGTH, MIN_DATE )
        
    elif METHOD == "day_test":
        pp.preprocess_days_test( PATH, FILE, PATH_PROCESSED, MIN_ITEM_SUPPORT, MIN_SESSION_LENGTH, DAYS_TEST )
    
    elif METHOD == "slice":
        pp.preprocess_slices( PATH, FILE, PATH_PROCESSED, MIN_ITEM_SUPPORT, MIN_SESSION_LENGTH, NUM_SLICES, DAYS_OFFSET, DAYS_SHIFT, DAYS_TRAIN, DAYS_TEST )
        
    elif METHOD == "buys":
        pp.preprocess_buys( PATH, FILE, PATH_PROCESSED )
        
    elif METHOD == "test_date":
        pp.preprocess_from_test_date( PATH, FILE, PATH_PROCESSED, MIN_ITEM_SUPPORT, MIN_SESSION_LENGTH, TEST_DATE, DAYS_TRAIN, DAYS_TEST, NEW_ITEMS, RANDOM_SAMPLE, LIMIT_BEAUTY )
    
    else: 
        print( "Invalid method ", METHOD )
        
    print( "END preprocessing ", (time.perf_counter() - sc), "c ", (time.time() - st), "s" )