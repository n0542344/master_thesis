#Add new data from Abwassermonitoring to external data (only inlfuenza?)
# Merge it with influenza data from before (grippemeldediensst.csv)
# then interpolate it
#%% MARK: libs etc
#This needs to be at the top
import sys
sys.path.insert(1, './../')

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
df_abwasser = pd.read_csv(
    "../data/00_external_data/abwassermonitoring.csv", 
    index_col="Datum", 
    sep=";")
df_abwasser.index = pd.to_datetime(df_abwasser.index, format="%d.%m.%Y")
df_abwasser = df_abwasser.drop(df_abwasser.columns[-1], axis=1)
df_abwasser = df_abwasser.query("Target == 'SARS-CoV-2'")
df_abwasser = df_abwasser.rename({df_abwasser.columns[-1]:"covid_weekly", "Datum":"date"}, axis=1)
df_abwasser["covid_weekly"] = df_abwasser["covid_weekly"].str.replace(",", ".")
df_abwasser["covid_weekly"] = (pd.to_numeric(df_abwasser["covid_weekly"])
                                   .div(1e11))#scale down

#make wide, only use SARS-CoV-2, as others only start in 2022 
#and have some missing values
df_abwasser = df_abwasser.pivot_table(
        index=df_abwasser.index, 
        columns="Target", 
        values="covid_weekly"
)
df_abwasser = df_abwasser.rename({"SARS-CoV-2":"covid_weekly"}, axis=1)
    

#forward fill to daily:
df_abwasser = df_abwasser.resample("D").ffill() #no limit to fill gap in 2022
df_abwasser["covid_daily"] = df_abwasser["covid_weekly"].div(7).round(3)

#Smothing/interpolating --> ich mache besser nur forward fill!
# df_abwasser_daily["COV_smoothed"] = (df_abwasser
#     .resample("D")
#     .interpolate(method="linear")
# )
#%%
# Grippemeldedienst:
df_gmd = pd.read_csv("../data/00_external_data/grippemeldedienst-interpolated.csv", index_col="date", parse_dates=True)
df_gmd.plot()
df_abwasser.plot()


#%%


df_processed = pd.read_csv("../data/03_transformed/output_transformed.csv", index_col="date", parse_dates=True)
df = data_model.Data(data=df_processed)



#%%

df["influenza_weekly"] = df["influenza_weekly"].ffill()

