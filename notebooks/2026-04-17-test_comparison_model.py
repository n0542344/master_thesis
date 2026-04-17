#test prophet, why results is so off!
#%% MARK: libs etc
#This needs to be at the top
import pandas as pd
from numpy import nan
import datetime

from src import config
from src import data_model
from src import load
from src import model
from src import transform
from src import viz
from src import utils


import importlib


#%%
df_processed = pd.read_csv("./data/03_transformed/output_transformed.csv", index_col="date", parse_dates=True)
df = data_model.Data(data=df_processed)



#%%
importlib.reload(model)

comp = model.ModelComparison(data=df)
comp.set_parameters(**config.config_comparison)
comp.model_run()
# %%


covid = pd.read_csv("./data/00_external_data/grippemeldedienst-interpolated.csv")
covid["date"] = covid["date"].to_datetime()
covid.index = covid["date"]
covid.drop(columns="date", inplace=True)
covid.index = pd.to_datetime(covid.index)

covid["daily_smooth"] = covid["influenza_daily"].rolling(7).mean()
covid["2017-08-01":].plot()