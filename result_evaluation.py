# Main pipeline for doing the evaluation of the results + viz of results.
#%%
import pandas as pd
from matplotlib import pyplot as plt
import os
import glob
import re
#%%
IMG_PATH = "./plots"

#Initial load of results
RESULTS_PATH = "./results"
PATH_ARIMA = f"{RESULTS_PATH}/Arima"
PATH_SARIMAX = f"{RESULTS_PATH}/Sarimax"
PATH_LSTM = f"{RESULTS_PATH}/LSTM"
PATH_PROPHET = f"{RESULTS_PATH}/Prophet"
SEP = ","
INDEX_COL = "id"

#Load all grid_search_result.csv files into dict
results_overview = {
    "arima" : pd.read_csv(filepath_or_buffer=f"{PATH_ARIMA}/grid_search_results.csv", sep=SEP, index_col=INDEX_COL),
    "sarimax" : pd.read_csv(filepath_or_buffer=f"{PATH_SARIMAX}/grid_search_results.csv", sep=SEP, index_col=INDEX_COL),
    "lstm" : pd.read_csv(filepath_or_buffer=f"{PATH_LSTM}/grid_search_results.csv", sep=SEP, index_col=INDEX_COL),
    "prophet" : pd.read_csv(filepath_or_buffer=f"{PATH_PROPHET}/grid_search_results.csv", sep=SEP, index_col=INDEX_COL)
}

#%%
# All functions
def parse_gs_results_csv():
    """???"""
    pass

def get_best_n_results(results: pd.DataFrame, error_val: str, n: int=1):
    """Get best n results of a model sorted by an error_val through grid_search_results.csv
    Drops empty rows first, so the no na-rows are on top after sorting.

    Args:
        results (pd.DataFrame): pandas df with a model (e.g. Arima) 
        already selected, containing the models grid_search_results.csv
        error_val (str): string to select column with error value
        n (int, optional): number of results to return. Defaults to 1.
    """
    results = results.dropna()
    res = results.sort_values(by=error_val).head(n)
    res[f"rank_{error_val}"] = res[error_val].rank().astype(int)
    return res

def load_model_resuls_by_id(model_name: str, result_id: int, results_path=RESULTS_PATH):
    """Load the directory into a single dictionary, with all day-ahead fc results as well as the stats, params etc.

    Args:
        model (_type_): String of the model, same as result directory name (so Arima, LSTM, etc)
        id (_type_): Id of the model run which is to be returned
    """
    path_pattern = os.path.join(f"{RESULTS_PATH}/{model_name}/{result_id}_*/Day_*.csv")
    files = sorted(glob.glob(path_pattern))

    # Fct to sort files numerically
    def get_day_number(filepath):
            filename = os.path.basename(filepath)
            match = re.search(r'Day_(\d+)', filename)
            return int(match.group(1)) if match else 0

    files.sort(key=get_day_number)
    
    data_dict = {}
    for file in files:
        # Create key: Day_1, Day_2, etc.
        key = os.path.splitext(os.path.basename(file))[0] 
        data_dict[key] = pd.read_csv(file, index_col=0, parse_dates=True, sep=";")
        
    return data_dict


def count_na_rows(model: pd.DataFrame):
    """Count the number of valid rows and the number of rows with na values,
    Mostly relevant for LSTM, to see how many of the runs were successful.

    Args:
        model (pd.DataFrame): _description_
    """
    n_total = len(model.index)
    n_valid = model.dropna().shape[0]
    n_na = n_total - n_valid

    return {"total": n_total, "valid_rows" : n_valid, "na_rows" : n_na}

def get_top_n_results(model: pd.DataFrame, error_val: str="RMSE", n: int=1):
    """Gets top n results of a model, the direct results -- not the overview.
    Returns a dictionary, where the first key is the rank of the model.

    Args:
        model (pd.DataFrame): _description_
    """
    top_n_ids = (
        model[error_val]
        .dropna()
        .sort_values()
        .index
        )

    detailed_res = {

    }

def add_params_to_overview():
    #Add columns with the parameters to the results
    pass


def plot_error_val_increase(res_dict, error_val: list=["RMSE"], n: int=100, img_name: str="05_RES_error_increase_by_rank"):
    #plot the increase of the error value by rank for all models.
    # errors are sorted individually by each line!
    # sharedy!


    fig, ax = plt.subplots(
        2, 1, 
        figsize=(16,8), 
        sharey=True, 
        sharex=True)
    
    fig.suptitle(f"Increase of {error_val}")

    for i, a in enumerate(ax.flatten()):
        for key in list(res_dict.keys()):
            data = (res_dict[key][error_val[i]]
                .sort_values()
                .reset_index(drop=True)
            )

            #Safety, if less rows than 'n'
            n_i = n
            if n > len(data):
                n_i = len(data)
            
            #plot
            (data
                .head(n=n_i)
                .plot(kind="line", label=key, ax=a)
            )

        key = list(res_dict.keys())[i]
        #a.set_title(error_val[i])
        a.set_ylabel(error_val[i])

    #unified legend
    handles, labels = a.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right', ncol=4)
    fig.supxlabel("Rank")
    fig.tight_layout()

    #with 4 subplots, one for each model and n lines for each error value.

    # fig, ax = plt.subplots(2, 2, figsize=(16,8), sharey=True, sharex=True)
    # fig.suptitle(f"Increase of {error_val}")
    # for i, a in enumerate(ax.flatten()):
    #     key = list(res_dict.keys())[i]

    #     for err in error_val:
    #         data = (res_dict[key][err]
    #             .sort_values()
    #             .reset_index(drop=True)
    #         )

    #         #Safety, if less rows than 'n'
    #         n_i = n
    #         if n > len(data):
    #             n_i = len(data)
            
    #         #plot
    #         (data.
    #         head(n=n_i)
    #         .plot(kind="line", label=err, ax=a)
    #         )
    #     a.set_title(key)
    # #unified legend
    # handles, labels = a.get_legend_handles_labels()
    # fig.legend(handles, labels, loc='lower center', ncol=2)
    
    fig.savefig(fname=f"{IMG_PATH}/{img_name}")

#%%
N = 5
# Get best for Arima -- overview (=grid_search_results_csv)
arima_overview_top_n = get_best_n_results(results_overview["arima"], "RMSE", n=N)
arima_top_id = arima_overview_top_n.index[0]
arima_top = arima_overview_top_n.loc[arima_top_id]
arima_best = load_model_resuls_by_id("Arima", result_id=arima_top_id)


# Get best for sarimax -- overview (=grid_search_results_csv)
sarimax_overview_top_n = get_best_n_results(results_overview["sarimax"], "RMSE", n=N)
sarimax_top_id = sarimax_overview_top_n.index[0]
sarimax_top = sarimax_overview_top_n.loc[sarimax_top_id]
sarimax_best = load_model_resuls_by_id("Sarimax", result_id=sarimax_top_id)


# Get best for lstm -- overview (=grid_search_results_csv)
lstm_overview_top_n = get_best_n_results(results_overview["lstm"], "RMSE", n=N)
lstm_top_id = lstm_overview_top_n.index[0]
lstm_top = lstm_overview_top_n.loc[lstm_top_id]
lstm_best = load_model_resuls_by_id("LSTM", result_id=lstm_top_id)


# Get best for prophet -- overview (=grid_search_results_csv)
prophet_overview_top_n= get_best_n_results(results_overview["prophet"], "RMSE", n=N)
prophet_top_id = prophet_overview_top_n.index[0]
prophet_top = prophet_overview_top_n.loc[prophet_top_id]
prophet_best = load_model_resuls_by_id("Prophet", result_id=prophet_top_id)

#%%
# Get best of all models
best_results_overview = pd.DataFrame(data=
    {
    "arima" : arima_top,
    "sarimax" : sarimax_top,
    "lstm" : lstm_top,
    "prophet" : prophet_top
    }
).transpose()


#

#%%
# Show decrease of RMSE/MAE for all models/sarimax only, when ordering by rank.
plot_error_val_increase(results_overview, error_val=["RMSE", "MAE"], n=1000)
# using grid_search_results.csv
# the ordering you get with this: head -n1 grid_search_results.csv && tail -n +2 grid_search_results.csv | sort -rt',' -k7,7n | head -n 30
# (for rmse)
# to showcase that many parameters lead to good results.
# Use in chapter Methods, section Gridsearch



# %%
# Plot time series with Actual, FC, upper/lower CI for all models:
def plot_fc_time_series(model: pd.DataFrame):
    pass



def forecast_blood_groups(model, params):
    #Use the best param set to forecast individual blood groups
    # ! Need to get use_transfused daily aggregate for individual groups!