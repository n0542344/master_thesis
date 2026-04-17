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

#Set config for multiprocessing (doesnt work i think, but needs to be set)
TOTAL_CORES = 4
TOTAL_RAM_GB = 10
RAM_PER_WORKER = TOTAL_RAM_GB / TOTAL_CORES 

#For random sampling reproducibility
SEED = 67



#Same for all grids:
PRED_COLUMN = "use_transfused"
START_DATE = pd.to_datetime("2022-01-01")
TRAIN_PERCENT = 0.7 #list(np.arange(0.6, 0.8, 0.1)), 
TEST_LEN = 14
VALIDATION_TYPE = "rolling"
#----------------------------------------------------------------------------------------------------
# MARK: Grid search options
#----------------------------------------------------------------------------------------------------

#Get all combinations of exogenous types

#Wards are not implemented, because that would lead to data leakage, where sarimax would get a nearly perfect forecast!
#(Because S/arima, prophet need the exogenous variables for the prediction period!)
exog_types = {
    "uses" : ["use_discarded", "use_expired"],
    #"wards" : ['ward_AN', 'ward_CH', 'ward_I1', 'ward_I3', 'ward_Other', 'ward_UC'],
    "days" : ["workday_enc", "holiday_enc", "day_of_week", "day_of_year", "year"],
    "weather" : ["tlmin", "tlmax"]
}

exog_combinations = config_utils.get_exog_list_combinations(exog_types)
exog_combinations_list = [list[1] for list in exog_combinations] + [None] #add empty list, to run without exog
#exog_combinations_list = [[]] #delete!



#Ranges of options for grid search

#currently 147 combinations (just multiply n for each param: 1*1*1*1*7*3*7=147)
arima_n_samples = 1 #-1
gs_config_arima = {
    "prediction_column" : [PRED_COLUMN],
    "train_percent" : [TRAIN_PERCENT], #arange(0.6, 0.8, 0.1),
    "test_len" : [TEST_LEN],
    "start_date" : [START_DATE],


    "p" : list(range(0,7)),
    "d" : list(range(0,3)),
    "q" : list(range(0,7))
}


#currently 16384 combinations, if d/D is only set to one would be 4096
sarimax_n_samples = 1 #500 #set to <0 to get full grid (no sampling) or delete in main.py
gs_config_sarimax = {
    "prediction_column" : [PRED_COLUMN],
    "train_percent" : [TRAIN_PERCENT], #arange(0.6, 0.8, 0.1),
    "test_len" : [TEST_LEN],
    "start_date" : [START_DATE],
    # "start_date" : [pd.to_datetime(day) for day in ["2024-01-01", "2015-01-01", "2022-01-01"]],
    "exog_cols" : exog_combinations_list,

    "p" : list(range(0,5)), #[0, 1], 
    "d" : [0, 1], #list(range(0,2)),
    "q" : list(range(0,5)), #[1], 
    "P" : list(range(0,5)), #[0, 1],
    "D" : [0, 1], 
    "Q" : list(range(0,5)), #[0, 1],
    "m" : [7, 14] #list(range(0,7))
}



lstm_n_samples = 1 #-1 #set to <0 to get full grid (no sampling) or delete in main.py
#currently 256 combinations
gs_config_lstm = {
    "prediction_column" : [PRED_COLUMN],
    "train_percent" : [TRAIN_PERCENT], #arange(0.6, 0.8, 0.1),
    "test_len" : [TEST_LEN],
    "start_date" : [START_DATE],
    # "start_date" : [pd.to_datetime(day) for day in ["2024-01-01", "2015-01-01", "2022-01-01"]],
    "validation_type" : [VALIDATION_TYPE], #, "expanding"],
    
    "exog_cols" : exog_combinations_list,
    
    "inner_window" : [365], #365=default; in days (time steps)
    
    "memory_cell" : [32, 64, 128, 256],
    "epochs" : [20, 100],
    "batch_size" : [32, 64],
    "pi_iterations" : [100], # n iterations to generate uncertainty
    "optimizer" : ["adam"],
    "loss" : ["mean_squared_error", "mean_absolute_error"], #, "mean_squared_logarithmic_error"], #see description here:https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/

    "lower_limit" : [2.5],
    "upper_limit" : [97.5]
}

#currently 288 combinations
prophet_n_samples = 10
gs_config_prophet = {
    "prediction_column" : [PRED_COLUMN],
    "train_percent" : [TRAIN_PERCENT], #arange(0.6, 0.8, 0.1),
    "test_len" : [TEST_LEN],
    "start_date" : [START_DATE],
    # "start_date" : [pd.to_datetime(day) for day in ["2024-01-01", "2015-01-01", "2022-01-01"]],
    "validation_type" : [VALIDATION_TYPE], #, "expanding"],
        

    "exog_cols" : exog_combinations_list,

    "changepoint_prior_scale" : [0.001, 0.01, 0.05], #flexibility of trend changes
    "seasonality_mode" : ["additive", "multiplicative"], #default: additive
    "seasonality_prior_scale" : [0.1, 1, 10], #default: 10; modulates impact of seasonality effect, low=less impact
    "holidays_prior_scale" : [0.1, 1, 10], #default: 10; modulates impact of prior holidays

    "lower_limit" : [2.5],
    "upper_limit" : [97.5]
}


#Not run as grid search, so values not in list!
config_comparison = {
    "col" : PRED_COLUMN,
    "forecast_window" : 14, #for plotting
    
    "single_value" : 535, #for single_value
    "start_date" : START_DATE, #for mean
    "end_date" : pd.to_datetime("2025-07-03") #for mean

    # #individual comp. models (dataframes)
    # "single_value_df" : ,
    # "naive_df" : ,
    # "mean_df" : ,
    # "seasonal_naive_df" : None
}


