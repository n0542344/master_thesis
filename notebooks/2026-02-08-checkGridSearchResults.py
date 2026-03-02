#%%
import pandas as pd
import numpy as np
from datetime import datetime
import os
import json
from matplotlib import pyplot as plt

RESULT_PATH = "results"

PREFIX = "0210_"
RUN_DATE = "20260210"
ERROR_VAL = "RMSE"

MODEL= "Prophet" #Prophet Sarimax Arima LSTM
LOWER = "Lower" # Lower_CI (S/Arima) Lower (Prophet) Lower_PI (LSTM)
UPPER = "Upper" # Upper_CI (S/Arima) Upper (Prophet) Upper_PI (LSTM)

# MODEL= "Arima" 
# # MODEL= "Sarimax"
# LOWER = "Lower_CI" 
# UPPER = "Upper_CI"

# MODEL= "LSTM"
# LOWER = "Lower_PI"
# UPPER = "Upper_PI"


#%%


# arima_grid_results = pd.read_csv(f"{RESULT_PATH}/Arima/grid_search_results.csv", sep=",", index_col="id")
# sarimax_grid_results = pd.read_csv(f"{RESULT_PATH}/Sarimax/grid_search_result.csv", sep=",")
# lstm_grid_results = pd.read_csv(f"{RESULT_PATH}/LSTM/grid_search_result.csv", sep=",")
# prophet_grid_results = pd.read_csv(f"{RESULT_PATH}/Prophet/grid_search_result.csv", sep=",")

# best_fit = arima_grid_results["RMSE"].idxmin()

def main(model, error_val):
    pass


def get_lowest_model_id(prefix, model, error_val="RMSE"):

    gs_res_df = pd.read_csv(f"../{RESULT_PATH}/{prefix+model}/grid_search_results.csv", sep=",", index_col="id")
    best_fit_id = gs_res_df[error_val].idxmin()
    best_fit_values = gs_res_df.loc[[best_fit_id]]
    return int(best_fit_id), best_fit_values

best_id, best_errors = get_lowest_model_id(PREFIX, MODEL)
# %%

def get_id_results(prefix, model, id, date, day="Day_1"):
    #Date of model run as YYYYMMDD
    #day is the forecast day in format "Day_x"

    directory = f"../{RESULT_PATH}/{prefix+model}/{id}_{date}"

    best_df = pd.read_csv(f"{directory}/{day}.csv", sep=";", index_col=0, parse_dates=True) #Day_x
    best_fc_errors = pd.read_csv(f"{directory}/forecast_errors.csv", sep=";", index_col=0) 

    with open(f"{directory}/params.json") as json_file:
        params = json.load(json_file)

    with open(f"{directory}/stats.json") as json_file:
        stats = json.load(json_file)

    return best_df, best_fc_errors, params, stats


df, fc_errors, params, stats = get_id_results(PREFIX, MODEL, best_id, RUN_DATE)
# %%


def plot_forecast(df, lower: str, upper: str, forecast="Mean", actual="Actual"):
    #lower/upper should be strings of colnames 

    df[actual].plot()
    df[forecast].plot()
    df[lower].plot()
    df[upper].plot()
    df["Difference"].plot()
    plt.legend(loc="upper right")


plot_forecast(df, lower=LOWER, upper=UPPER)
# %%

fc_errors.drop("MSE", axis=1).plot()
plt.legend(loc="upper right")


# %%
df["Diff_upper"] = df[UPPER] - df["Actual"]
df[["Diff_upper", "Difference"]].plot()

# %%
