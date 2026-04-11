#Main pipeline for doing data exploration on raw/cleaed/transformed data
# (Not on results!)
#%%
import pandas as pd
import numpy as np
from numpy import nan
from time import time
from matplotlib import pyplot as plt
import pathlib
import importlib

import seaborn as sns

from src import clean
from src import config
from src import data_model
from src import load
from src import model
from src import transform
from src import viz
from src import utils
from src import config_cleaning

import pandas as pd
#%%
#--------------------------------------------------------------------------------
# MARK: LOAD DATA
# (raw, cleaned, transformed)
#----------------------------------------------------------------------------------
importlib.reload(clean)
importlib.reload(config)
importlib.reload(transform)

df_raw = load.load_data(path=config_cleaning.RAW_DATA_PATH)
df_clean = clean.clean_data(df_raw, new_file_path=config_cleaning.CLEANED_DATA_PATH)
df_processed = transform.transform_data(df_clean, new_file_path=config_cleaning.TRANSFORMED_DATA_PATH)
df = data_model.Data(data=df_processed)

#%%
#--------------------------------------------------------------------------------
# MARK: RAW DATA VIZ
#----------------------------------------------------------------------------------

# Read Data

# #%%
# load.show_info(df=df_raw)
# hidden_cols=["date", "EC_ID_I_hash", "EC_ID_O_hash", "T_ISO", "T_DE_T", "T_US", "T_DE_S", "T_US_T", "T_DE", "T_ISO_T", "T_XL"]
# for col in df_raw.columns:
#     if col not in hidden_cols:
#         tmp = df_raw[col].astype(str).unique()
#         tmp = np.array(sorted(tmp))
#         print(f"{col}:\n{tmp}\n")
# print(df_raw.columns)
# #TODO: move back to clean_data
# # df_raw = clean.clean_dates(df_raw) #TODO: remove here, enable again in clean_data()



#%%
#--------------------------------------------------------------------------------
# MARK: CLEANED DATA VIZ
#----------------------------------------------------------------------------------
#Runs only if no file exists at. If not existing, saves df to new file
#unify dates, columns etc. rename stuff
IMAGES_PATH_EXPLORATION = IMAGE_PATH + "/00-Data-Exploration/"

#TODO: Check what unique vals are present in df
clean.check_unique_values(df_clean.drop(["EC_ID_I_hash", "EC_ID_O_hash", "PAT_WARD"], axis=1))


#
# Plot frequency counts for unique values in every column
#TODO: move into viz.py


for col_name, col in df_clean.items():
    if col_name in ["EC_ID_O_hash", "EC_ID_I_hash"]:
        continue
    print(col.value_counts())
    col.value_counts()[:40].plot(kind="bar", title=col_name,)
    plt.savefig(fname=IMAGES_PATH_EXPLORATION+f"01-barcharts-value_counts-{col_name}")
    plt.show()


importlib.reload(viz)
##
# Plot each patient wards transfusion counts (for wards with >500 transfusions)
viz.plot_patient_wards(df_clean, n=500, save_figs=False, location=IMAGES_PATH_EXPLORATION, foldername="")






#%%
#--------------------------------------------------------------------------------
# MARK: TRANSFORMED DATA VIZ
# /PROCESSING
#----------------------------------------------------------------------------------
# make STATIONARY! (if all models need that, otherwise make it a member function)
# splitting in test/training etc. here or as extra step/model step?



# Proces....
#add external data (holidays weather (temp, precipitation), covid/influenca cases)
#NOTE: covid/grippe muss evnetuell imputiert werden da nur wöchentlich
#NOTE: kann gut zeigen, dass wien gleichen verlauf hat wie bundesländer, daher kann ich Ö-weite Daten
# nehmen, falls es keine wien-spezifischen Daten gibt.


#%%
# Plot seasonalities daily & weekly of processed df

importlib.reload(viz)

BG_RH_cols = ['EC_BG_RH_0_NB']#,
    #    'EC_BG_RH_0_Rh_negative', 'EC_BG_RH_0_Rh_positive', 'EC_BG_RH_A_NB',
    #    'EC_BG_RH_A_Rh_negative', 'EC_BG_RH_A_Rh_positive', 'EC_BG_RH_AB_NB',
    #    'EC_BG_RH_AB_Rh_negative', 'EC_BG_RH_AB_Rh_positive', 'EC_BG_RH_B_NB',
    #    'EC_BG_RH_B_Rh_negative', 'EC_BG_RH_B_Rh_positive', 'PAT_BG_RH_0_NB',
    #    'PAT_BG_RH_0_Rh_negative', 'PAT_BG_RH_0_Rh_positive', 'PAT_BG_RH_A_NB',
    #    'PAT_BG_RH_A_Rh_negative', 'PAT_BG_RH_A_Rh_positive', 'PAT_BG_RH_AB_NB',
    #    'PAT_BG_RH_AB_Rh_negative', 'PAT_BG_RH_AB_Rh_positive',
    #    'PAT_BG_RH_B_NB', 'PAT_BG_RH_B_Rh_negative', 'PAT_BG_RH_B_Rh_positive',
    #    'PAT_BG_RH_NB_NB', 'PAT_BG_RH_NB_Rh_negative',
    #    'PAT_BG_RH_NB_Rh_positive', 'PAT_BG_RH_Not_applicable']
for bg_rh in BG_RH_cols:
    # viz.seasonal_plot(df_processed, plot_type="weekly", col_name=bg_rh)
    viz.seasonal_plot(df_processed, plot_type="daily", col_name=bg_rh)



ward_cols = ['ward_AN', 'ward_CH', 'ward_I1', 'ward_I3', 'ward_Other', 'ward_UC']
for ward in ward_cols:
    viz.seasonal_plot(df_processed, plot_type="weekly", col_name=ward)
    viz.seasonal_plot(df_processed, plot_type="daily", col_name=ward)


#%%
#TODO: save data to csv
# # Plot daily/weekly cases influenza
# fig, ax = plt.subplots(1)
# ax.plot(df_processed["new_cases_daily"])
# ax.plot(df_processed["new_cases_weekly"], color="red")
# plt.show



#%%--------------------------------------------------------------------------------
# MARK: DATA VIZ 
# (EXPLORATION)
#----------------------------------------------------------------------------------
IMAGES_PATH_EXPLORATION = IMAGE_PATH + "/00-Data-Exploration/"
START_DATE_EXPLORATION = "2012-01-01" #"2020-01-01"
END_DATE_EXPLORATION = "2016-12-31" #"2020-01-01"
PRE_COVID_START = "2018-01-01"
PRE_COVID_END = "2020-01-01"

#%%
#TODO: save vizualisations to csv
importlib.reload(data_model)

df = data_model.Data(data=df_processed)

#%%


df["2023-01-01":"2023-12-31"].plot_seasonal(plot_type='daily', col_name='use_transfused')#, fig_location=IMAGES_PATH_EXPLORATION)
df[START_DATE_EXPLORATION:].plot_seasonal(plot_type='weekly', col_name='use_transfused')


##%%
#Boxplots for year, month, weekly and day of week
df[START_DATE_EXPLORATION:END_DATE_EXPLORATION].plot_boxplots(col_name='use_transfused')
df[START_DATE_EXPLORATION:END_DATE_EXPLORATION].plot_boxplots(col_name='use_discarded', max_y = 65)
df[START_DATE_EXPLORATION:END_DATE_EXPLORATION].plot_boxplots(col_name='use_expired')
df[START_DATE_EXPLORATION:END_DATE_EXPLORATION].plot_seasonal_subseries(col_name='use_transfused') #NOTE: i think it works, but not enough dummy data.
#TODO: check if seasonal subseries plot works with multi-year data


##%%
#Decompose
df[PRE_COVID_START:PRE_COVID_END].decompose_one(col_name='use_transfused')


#df.decompose_all("use_transfused")

#%%
#Glaub das verwende ich nicht
# mulitple decomposition (daily + weekly)
df[PRE_COVID_START:PRE_COVID_END].multiple_decompose(col_name="use_transfused", periods=[7, 365])
df[pd.to_datetime("2024-01-01"):pd.to_datetime("2024-12-31")].plot_daily_heatmap(col_name='use_transfused')


#%% Visualize counts for all plots (as of now only for those starting with 901AN) on top of each other, so that
# its visible, where naming of one ward starts/ends 


wards = df_clean["PAT_WARD"].unique()
fig, ax = plt.subplots(44,1, figsize=(6, 24))
fig.set_dpi(300)
fig.set_linewidth(5)
sel_wards = []
i = 0
for ward in wards:
    if str(ward).startswith("901AN"):
        sel_wards.append(ward)
        ax[i].plot(df[f"PAT_WARD_{ward}"], label=ward, linewidth=0.15)
        # ax[i].set_title(str(ward))
        ax[i].legend(loc="upper right", prop={"size":6}, frameon=False, framealpha=0.5)
        ax[i].set_yticklabels([])
        i = i+1
        
print(len(sel_wards))
print(sel_wards)
plt.savefig(fname="/".join([IMAGES_PATH_EXPLORATION + "show_inconsistency_wards"]))
plt.show()

#%%
# Get unique Letter-combinations (i guess top-wards) from all wards:

import re
unique_wards = df_clean["PAT_WARD"].unique()

pattern = re.compile(r'\d+([A-Za-z]+)\d+')  # number + letters + number

ward_results = set()  # only unique strings

for s in unique_wards:
    match = pattern.search(str(s))
    if match:
        ward_results.add(match.group(1))  # =letters

print(unique_wards)
print(len(unique_wards))
print(ward_results)
print(len(ward_results))

#%%--------------------------------------------------------------------------------
# MARK: STATIONARITY 
# Check for Stat./Make stationarys
#----------------------------------------------------------------------------------
#TODOLIST:
# 1. OG Data
#    Visual & statistical check: const mean, variance, no seasonal component 
#    --> then its stationary
# 1.1 Visual assessment
#    - Time series
#    - ACF, PACF
# 1.2 Statistical test
#    - Unit root test
# 
#  

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss

#%%
#Time series plots (acf, pacf etc)
df.plot_autocorrelation(col_name='use_transfused')
df.plot_partial_autocorrelation(col_name='use_transfused')

num_differencing = 2
i = 1
df_diff = df_processed.copy()

df_diff["use_transfused"].plot(lw=0.05)

plot_acf(df_diff["use_transfused"], title="No differentiation")
plot_pacf(df_diff["use_transfused"], title="No differentiation")

#ADF -- Augmented dickey-fuller test
adf_result_diff = adfuller(df_diff["use_transfused"][1:], autolag="AIC") #default
adf_result_diff2 = adfuller(df_diff["use_transfused"][1:], autolag="BIC")
adf_result = adfuller(df["use_transfused"], autolag="AIC") #default
adf_result2 = adfuller(df["use_transfused"], autolag="BIC")
adfuller(df["use_transfused"])
plt.plot(df["use_transfused"])
plt.plot(df_diff["use_transfused"])

#i think i can reject the H0 in both cases (df and df_diff), for df test staticstics is -11.49, far below 
#critical values of 1%, 5%, 10%, same  for df_diff with -21.07. therefore i reject H0 (data is 
# non stationary, has a unit root). H1 is true, data is stationary has no unit root. so no differencing 
# needed actually?

#Different params for autolag:
# no real differences in result, so doestn matter. (in gerneral AIC better for large datasets)


#KPSS
kpss_result_diff = kpss(df_diff["use_transfused"][1:])
kpss_result = kpss(df["use_transfused"])
kpss_result = kpss(df["use_transfused"])

#kpss is one-sided test, if test statistics is greater than critical value, then H0 is rejected. 
# H0 for KPSS is that ts is stationary, H1 ts is NOT stationary.
# for kopss_result_diff, p-val = 0.1, test stat. = 0.0104, 0.0104 is not greater than 0.347 (10%), so H0 cant
# be rejected, ts seems to be stationary.
# for kpss_result, with p value = 0.01, test statistic = 1.258 is greater than 0.739 (1%), so H0 is rejected,
# ts is not stationary. 

# So for differentiated data (df_diff), both ADF + KPSS suggest stationarity.
# For df (original data), ADF suggests stationarity, KPSS suggests non-stationarity.
# with this guide: https://www.statsmodels.org/dev/examples/notebooks/generated/stationarity_detrending_adf_kpss.html
# i should differntiate the original series (df), then check this for stationarity. i already did that, and
# both tests suggest that data is stationary. so differencing was right call.

print(adf_result[0])
print(f"No differencing: \nadf: {adf_result[0]}\np-value: {adf_result[1]}\ncritical vals: {adf_result[4]}")
while i <= num_differencing:
    df_diff["use_transfused"] = df_diff["use_transfused"].diff()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    df_diff["use_transfused"].plot(ax=ax1, lw=0.05)
    plot_acf(df_diff["use_transfused"].dropna(), ax=ax2)
    plot_pacf(df_diff["use_transfused"].dropna(), ax=ax3)
    fig.suptitle(f"differentiated {i}x")
    fig.show()

    adf_result = adfuller(df_diff["use_transfused"].dropna())
    print(f"No differencing: \nadf: {adf_result[0]}\np-value: {adf_result[1]}\ncritical vals: {adf_result[4]}")


    i += 1

#seasonal differencing = subtracting value of previous season 
# (i'll try week, so period = 7 (7 rows before in dataset))
num_differencing = 3
period = 7#365
i = 1
df_diff = df_processed.copy()

df_diff["use_transfused"].plot(lw=0.05)
plot_acf(df_diff["use_transfused"], title="No differentiation")
plot_pacf(df_diff["use_transfused"], title="No differentiation")

while i <= num_differencing:
    df_diff["use_transfused"] = df_diff["use_transfused"].diff(periods=period)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    df_diff["use_transfused"].plot(ax=ax1, lw=0.05)
    plot_acf(df_diff["use_transfused"].dropna(), ax=ax2)
    plot_pacf(df_diff["use_transfused"].dropna(), ax=ax3)
    fig.suptitle(f"differentiated {i}x")
    fig.show()
    i += 1



#%% 
# MARK: COMPARISON
#----------------------------------------------------------------------------------
import importlib
importlib.reload(model)
comp = model.ModelComparison(df)

comp.set_column()
comp.set_dates_mean()
comp.set_forecast_window()
comp.set_single_value()

comp.print_parameters()

comp.model_run()

comp.predictions

# comp.get_error_values()


for col in comp.predictions.columns:
    if col == "use_transfused":
        plt.plot(comp.predictions.loc["2024-01-01":"2024-07-31", col], linewidth=1.5, label=col)
    else:
        plt.plot(comp.predictions.loc["2024-01-01":"2024-07-31", col], linewidth=0.5, label=col)
    plt.legend()

comp.stats["id"] = 0000
comp.stats["model_name"] = "comparison"
comp.create_result_dir()
comp.save_results()