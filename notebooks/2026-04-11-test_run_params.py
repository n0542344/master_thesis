#test prophet, why results is so off!
#%% MARK: libs etc
#This needs to be at the top
import os
import argparse #to run different python files for lstm
#Set LSTM (tensorflow) threads
# os.environ["TF_DISABLE_MKL"] = "1"
# os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
# os.environ["TF_NUM_INTEROP_THREADS"] = "1"
# os.environ["TF_DATA_EXPERIMENTAL_DISABLE_THREADING"] = "1"

#TODO: Still necessary after changing to different python files for concurrent running?
os.environ["OMP_NUM_THREADS"] = "4"      # change to however many you want
os.environ["TF_NUM_INTRAOP_THREADS"] = "4"
os.environ["TF_NUM_INTEROP_THREADS"] = "4"

#ignore/suppress keras logs:
os.environ["KERAS_VERBOSE"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pandas as pd
import numpy as np
from numpy import nan
from time import time
from datetime import datetime
from matplotlib import pyplot as plt


from src import config
from src import data_model
from src import load
from src import model
from src import transform
from src import viz
from src import utils


from sklearn.model_selection import ParameterGrid


#%%
df_processed = pd.read_csv("./data/03_transformed/output_transformed.csv", index_col="date", parse_dates=True)
df = data_model.Data(data=df_processed)

#%%

full_grid_prophet = list(ParameterGrid(config.gs_config_prophet))
test_config = full_grid_prophet[0]


#%%
m_prophet = model.ModelProphet(df)
m_prophet.set_validation_rolling_window(**test_config)
m_prophet.set_model_parameters(**test_config)
m_prophet.set_exogenous_cols(**test_config)
m_prophet.set_prediction_column(**test_config)
m_prophet.model_run()

