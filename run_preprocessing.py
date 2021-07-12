import preprocessing.preprocessing as pp
import time

'''
preprocessing method ["info","slice","buys"]
    info: just load and show info
    slice: create multiple train-test-combinations with a window approach  
    buys: load buys and safe file to prepared
'''
METHOD = "slice"

'''
data config (all methods)
'''
PATH = './recsys_data/'
PATH_PROCESSED = './recsys_data/prepared/28062021/'
FILE = 'test'

'''
org_min_date config
'''
MIN_DATE = '2014-07-01'

'''
filtering config (all methods)
'''
MIN_SESSION_LENGTH = 3 #2
MIN_ITEM_SUPPORT = 10 #5

'''
days test default config
'''
DAYS_TEST = 1

'''
slicing default config
'''
NUM_SLICES = 1 #5 #offset in days from the first date in the data set
DAYS_OFFSET = 0 #number of days the training start date is shifted after creating one slice
#each slice consists of...
DAYS_TRAIN = 4 #30
DAYS_TEST = 1
DAYS_SHIFT = DAYS_TRAIN + DAYS_TEST #31

if __name__ == '__main__':
    '''
    Run the preprocessing configured above.
    '''
    
    print( "START preprocessing ", METHOD )
    sc, st = time.perf_counter(), time.time()
    
    if METHOD == "info":
        pp.preprocess_info( PATH, FILE, MIN_ITEM_SUPPORT, MIN_SESSION_LENGTH )
    
    elif METHOD == "slice":
        pp.preprocess_slices( PATH, FILE, PATH_PROCESSED, MIN_ITEM_SUPPORT, MIN_SESSION_LENGTH, NUM_SLICES, DAYS_OFFSET, DAYS_SHIFT, DAYS_TRAIN, DAYS_TEST )
        
    elif METHOD == "buys":
        pp.preprocess_buys( PATH, FILE, PATH_PROCESSED )
    
    else: 
        print( "Invalid method ", METHOD )
        
    print( "END preproccessing ", (time.perf_counter() - sc), "c ", (time.time() - st), "s" )