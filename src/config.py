#Global Variables and settings
import pandas as pd
from numpy import arange
from time import time
from datetime import datetime


from src import config_utils

ENABLE_TIMING = False #If true, print messages. #TODO: add logging module to do that
ENABLE_LOGGING = True
SAVE_FIGS = False

#Slice data while developing
DEV_START_DATE = "2024-05-01"
DEV_END_DATE = "2025-07-01"
PRED_COLUMN = "use_transfused"

#For random sampling reproducibility
SEED = 67

#Settings for multiprocessing
TOTAL_CORES = 4
TOTAL_RAM_GB = 10
RAM_PER_WORKER = TOTAL_RAM_GB / TOTAL_CORES 




#----------------------------------------------------------------------------------------------------
# MARK: Grid search options
#----------------------------------------------------------------------------------------------------

#Ranges of options for grid search

exog_types = {
    "uses" : ["use_discarded", "use_expired"],
    #"wards" : ['ward_AN', 'ward_CH', 'ward_I1', 'ward_I3', 'ward_Other', 'ward_UC'],
    "days" : ["workday_enc", "holiday_enc", "day_of_week", "day_of_year", "year"],
    "weather" : ["tlmin", "tlmax"]
}

exog_combinations = config_utils.get_exog_list_combinations(exog_types)
exog_combinations_list = [list[1] for list in exog_combinations]


gs_config_arima = {
    "prediction_column" : [PRED_COLUMN],
        
    "train_percent" : arange(0.6, 0.8, 0.1),
    "test_len" : [14],
    "start_date" : [pd.to_datetime(day) for day in ["2024-01-01", "2024-02-01", "2024-03-01"]],
    # "start_date" : [pd.to_datetime(day) for day in ["2024-01-01", "2015-01-01", "2022-01-01"]],

    "p" : list(range(0,7)),
    "d" : list(range(0,3)),
    "q" : list(range(0,7))
}



gs_config_sarimax = {
    "prediction_column" : [PRED_COLUMN],
    
    "exog_cols" : exog_combinations_list,
    
    "train_percent" : arange(0.6, 0.8, 0.1),
    "test_len" : [14],
    "start_date" : [pd.to_datetime(day) for day in ["2024-01-01", "2024-02-01", "2024-03-01"]],
    # "start_date" : [pd.to_datetime(day) for day in ["2024-01-01", "2015-01-01", "2022-01-01"]],

    "p" : [0, 1], #list(range(0,3)),
    "d" : [0, 1], #list(range(0,2)),
    "q" : [1], #list(range(0,7)),
    "P" : [0, 1], #list(range(0,2)),
    "D" : [0, 1], #list(range(0,2)),
    "Q" : [0, 1], #list(range(0,7)),
    "m" : [7] #list(range(0,7))
}



lstm_n_samples = 500
gs_config_lstm = {
    "prediction_column" : [PRED_COLUMN],

    "validation_type" : ["rolling"], #, "expanding"],
    
    "exog_cols" : exog_combinations_list,
    
    "train_percent" : [0.7, 0.8], #list(np.arange(0.6, 0.8, 0.1)), #wouldnt it make more sense to use int of days before to train? like train_days = 365*7 or 730 or something?
    "test_len" : [7, 14],
    "start_date" : [pd.to_datetime(day) for day in ["2024-01-01", "2024-02-01", "2024-03-01"]],
    # "start_date" : [pd.to_datetime(day) for day in ["2024-01-01", "2015-01-01", "2022-01-01"]],
    # "start_date" : [pd.to_datetime(day) for day in ["2008-01-01", "2012-01-01", "2016-01-01", "2020-01-01", "2024-01-01"]],
    
    "memory_cell" : [32, 64, 128],
    "epochs" : [20, 100],
    "batch_size" : [32], #64
    "pi_iterations" : [100, 200],
    "optimizer" : ["adam"],
    "loss" : ["mean_squared_error", "mean_absolute_error", "mean_squared_logarithmic_error"], #see description here:https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/

    "lower_limit" : [2.5],
    "upper_limit" : [97.5]
}


gs_config_prophet = {
    "prediction_column" : [PRED_COLUMN],
    
    "exog_cols" : exog_combinations_list,
    
    "train_percent" : arange(0.6, 0.8, 0.1),
    "test_len" : [14],
    "start_date" : [pd.to_datetime(day) for day in ["2024-01-01", "2024-02-01", "2024-03-01"]],
    # "start_date" : [pd.to_datetime(day) for day in ["2024-01-01", "2015-01-01", "2022-01-01"]],

    "lower_limit" : [2.5],
    "upper_limit" : [97.5]
}


