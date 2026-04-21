#check the counts for use_transfused, use_discarded, 
# use_expired, use_unkown, count against the PAT_BG_RH cols

#%% 
import pandas as pd
from numpy import nan
import datetime

from src import config
from src import config_cleaning
from src import data_model
from src import load
from src import model
from src import transform
from src import viz
from src import utils


import importlib


#%%
df_raw = load.load_data(path=f".{config_cleaning.RAW_DATA_PATH}")
df_clean = pd.read_csv("../data/02_cleaned/output_cleaned.csv", index_col="date", parse_dates=True)
df_processed = pd.read_csv("../data/03_transformed/output_transformed.csv", index_col="date", parse_dates=True)
df = data_model.Data(data=df_processed)


# %%

df[["count", "use_transfused", "use_discarded", "use_expired", "use_transfused"]].head()
df_processed.loc[:,df_processed.columns.str.startswith("PAT_BG_")].head().sum(axis=1)
df_clean.head()

# %%

# Eigentlich sollte keine Patient bloodgroup/RH sein wenn es discarded/expired ist.
# In raw data, bei ToD ist das aber praktisch immer der Fall! 
# (wobei hier PAT_BG nicht mit immer mit EC_BG übereinstimmt --> sollte so sein, d.h. 
# hat separate quelle, ist nicht einfach duplizierte werte).
# In ToD_N (wo abkürzungen, VRN, AUS, etc. verwendet werden), ist oft, nicht immer,
# PAT_BG_RH 'missing value' --> sollte so sein
# ==> wie weit ist das relevant? Weil use_transfused/discarded/expired sollten davon
# nicht betroffen sein, und v.a. nicht use_transfused, brauche nur das
# brauche die ganzen PAT_BG_RH_... nicht, oder?

#ToD
#Entferte werte ToD: "'Ausgegeben'"
df_raw[df_raw["ToD"].isin(['Abgelaufen', 'Verkauft'])][["ToD", "PAT_BG", "PAT_RH", "ToD_N", "PAT_BG_RH"]]


#ToD_N
#entfernte Werte ToD_N: 'VRN', "AUS"
df_raw[df_raw["ToD_N"].isin(['ABG', 'END', 'RET', 'RES'])][["ToD", "PAT_BG", "PAT_RH", "ToD_N", "PAT_BG_RH"]]