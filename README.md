# CDRC-MSc-Recommender-Systems
 Recommender Systems for CDRC MSc


This is based on the session-based recommendation framework session-rec (https://github.com/rn5l/session-rec).


Algorithms included:
* SLIST (https://arxiv.org/abs/2103.16104)
* STAMP (https://dl.acm.org/doi/10.1145/3219819.3219950)
* STAN (https://dl.acm.org/doi/10.1145/3331184.3331322)
* RecVAE (https://arxiv.org/abs/1912.11160)
* GRU4Rec+ (https://arxiv.org/abs/1706.03847)

## How to Use

* Data Preprocessing: Performs data preprocessing and splitting, and generates the item features matrix.
    
    Data Preprocessing and splitting: `python run_preprocessing.py conf/preprocess/base_dataset.yml`
    
    Generate item features matrix: `python preprocessing/generate_item_matrix_cease.py`
    
* Hyperparameter Optimization: Performs Bayesian optimization and outputs the results for each iteration in a CSV file.
    
    Base models: `python run_config.py conf/opt/opt_beauty_slist.yml`
    
    KNN extension in SLIST Ext: `python grid_search_knn.py`
    
* Model Evaluation: Performs model evaluation and outputs the results in a CSV file.
    
    Example: `python run_config.py conf/evaluation/evaluate_beauty_models.yml`
