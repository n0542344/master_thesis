# Main pipeline for doing the evaluation of the results + viz of results.

import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.legend_handler import HandlerBase
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import os
import pathlib
import glob
import re
import json
import numpy as np

from src import clean
from src import config_cleaning
from src import transform
from src import result_evaluation_config as rconf

from datetime import datetime


import importlib





#----------------------------------------------------------------------------------
# MARK: File reading + preprocessing
#----------------------------------------------------------------------------------

# Entry dates
def read_daily_entries(data_path="./data/01_raw", filename="Blood-data_complete_including_2025.tsv", top_N_wards=9):
    """Reads and cleans the file, that provides Entry and Use date for each EC,
     also combines wards to top_N_wards, rest to 'Other' """

    raw_df = pd.read_csv(f"{data_path}/{filename}", sep="\t")

    #dates
    raw_df[["T_DE_S", "T_entry_DE_S"]] = raw_df[["T_DE_S", "T_entry_DE_S"]].apply(pd.to_datetime, format="%m-%d-%y")
    
    #use
    raw_df["ToD_N"] = raw_df["ToD_N"].replace(config_cleaning.transfusion_status_map)
    
    #EC_TYPE
    allowed_ec_types = ["EKF", "EKFX"]
    raw_df.loc[~raw_df["EC_TYPE"].isin(allowed_ec_types), "EC_TYPE"] = "Other"

    #PAT_BG_RH
    #split str after index 2 into BG/RH
    raw_df = clean.split_BG_RH(raw_df, origin="EC_BG_RH", temp_cols=["EC_BG_temp", "EC_RH_temp"], target_cols=["EC_BG", "EC_RH"])
    raw_df = clean.split_BG_RH(raw_df, origin="PAT_BG_RH", temp_cols=["PAT_BG_temp", "PAT_RH_temp"], target_cols=["PAT_BG", "PAT_RH"])
    #replace strings as necessary
    raw_df["PAT_RH_temp"] = raw_df["PAT_RH_temp"].replace({"-":"Rh negative", "\+":"Rh positive"}, regex=True)
    raw_df["EC_BG_temp"] = raw_df["EC_BG_temp"].replace({"2":""}, regex=True) #some 'A2' values
    raw_df["EC_RH_temp"] = raw_df["EC_RH_temp"].replace({"-":"Rh negative", "\+":"Rh positive"}, regex=True)
    #merge back together
    raw_df["PAT_BG_RH"] = raw_df["PAT_BG_temp"] + " " + raw_df["PAT_RH_temp"]
    raw_df["EC_BG_RH"] = raw_df["EC_BG_temp"] + " " + raw_df["EC_RH_temp"]
    raw_df = raw_df.drop(["PAT_RH_temp", "PAT_BG_temp", "EC_RH_temp", "EC_BG_temp"], axis=1)


    #PAT_WARDS->'ward': merge Short-name of upper-level (Kostenstelle); get top 5, rest to Other
    raw_df = transform.combine_wards(raw_df.rename(columns={"T_entry_DE_S":"date"}), top_N=top_N_wards) #combine_wards needs 'date' column

    raw_df = raw_df.rename(
        columns={
            "T_DE_S":"date_use", 
            #"T_entry_DE_S":"date_entry", 
            "ToD_N":"use",
            "date":"date_entry"
            }
    )

    #Add storage_period column
    raw_df["storage_period"] = (raw_df["date_use"] - raw_df["date_entry"]).dt.days
    raw_df.index = raw_df["date_use"]
    return raw_df


def aggregate_daily_entries(df, by="date_entry"):
    """Daily aggregate by column 'date_entry', defaults to date_entry
    makes df wide
    NOTE: The countrs for use, ward, (PAT_BG_RH) are for the use date.
    So you can't see, e.g. how old EC were, when they were used in ward_GY (like average
    age of EC was XX days -> not possible with aggregated data, use raw/cleaned data)
    Can still be used, to see were a df ended up, like e.g. in ward_GY, 
    """
    cols_to_sum = list(df.columns)
    cols_to_sum.remove('date_entry')
    cols_to_sum.remove('date_use')
    
    df_wide = transform.aggregate_categorical_cols(df.rename(columns={by:"date"}), cols_to_sum)
    df_wide.index = df_wide.index.rename(by)
    #df_wide = df_wide.set_index(by)
    return df_wide

# Results
def parse_all_stats_params(result_dir="./results", save_dict=False):
    """Get a dict, where each model contains a (param+stats)-dict with ID as key
    Iterate over each models result directory, then over each subdirectory.
    Gather params.json and stats.json, to create a dict with keys for each model, 
    where each model contains dicts with key of each id, containing the jsons.
    Use to link id from results to stats, to filter for example only models without exogenous vars etc.
    """
    res = {
        "arima" : {},
        "sarimax" : {},
        "lstm" : {},
        "prophet" : {}
    }

    results_dir = pathlib.Path(result_dir)
    #iterate over models:
    for model_dir in results_dir.iterdir():
        i = 0

        if model_dir.is_dir():
            #iterate over a model's results_dir dirs:
            for i, run_dir in enumerate(model_dir.iterdir()):
                if run_dir.is_dir():
                    stats = None
                    params = None

                    #open files:
                    stats_file = run_dir / "stats.json"
                    params_file = run_dir / "params.json"

                    #Load files
                    if stats_file.exists():
                        with open(stats_file) as f:
                            stats = json.load(f)
                    if params_file.exists():
                        with open(params_file) as f:
                            params = json.load(f)

                    #combine both, add to result
                    if stats and params:
                        model = stats.pop("model_name").lower()
                        id = stats.pop("id")
                        combined = {**stats, **params}
                        res[model][id] = combined

    if save_dict:
        try:
            with open(f"{results_dir}/all_params.json", "x") as f:
                json.dump(res, fp=f, sort_keys=True, indent=3)    
        except FileExistsError:
            print("File already exists")


    return res



def parse_all_forecasts(result_dir="./results", save_dict=False, forecast_days=14):
    """Get a dict, where each model contains a (param+stats)-dict with ID as key
    Iterate over each models result directory, then over each subdirectory.
    Gather params.json and stats.json, to create a dict with keys for each model, 
    where each model contains dicts with key of each id, containing the jsons.
    Use to link id from results to stats, to filter for example only models without exogenous vars etc.
    """
    res = []

    results_dir = pathlib.Path(result_dir)
    #iterate over models:
    for model_dir in results_dir.iterdir():
        #i = 0

        if model_dir.is_dir():
            #iterate over a model's results_dir dirs:
            for i, run_dir in enumerate(model_dir.iterdir()):

                if i >= 50:
                    continue


                #read stats+params+Day_x (all days)
                if run_dir.is_dir():

                    #open stats+params, to directly merge
                    stats = None
                    params = None

                    #open files:
                    stats_file = run_dir / "stats.json"
                    params_file = run_dir / "params.json"

                    #Load files
                    if stats_file.exists():
                        with open(stats_file) as f:
                            stats = json.load(f)
                    if params_file.exists():
                        with open(params_file) as f:
                            params = json.load(f)

                    #combine both, add to result
                    if stats and params:
                        model = stats.pop("model_name").lower()
                        id = stats.pop("id")
                        combined = {**stats, **params}
                    if model == 'sarimax':
                        print(f"Open: {model}; {id}")

                    #get Day_X forecast csv
                    day_x = None

                    for day in range(1, forecast_days+1):
                        #open files:
                        day_file = run_dir / f"Day_{day}.csv"
                        #Load files
                        if day_file.exists():
                            day_x = (pd.read_csv(day_file, sep=";", parse_dates=True, index_col=0)
                                     .rename_axis("date")
                                     .rename(columns={"Upper_CI":"Upper", "Lower_CI":"Lower"})
                                     .assign(day=day, id=id, model=model)
                            )
                            res.append(day_x)
                #maybe preliminarily concatenate?

    df = pd.concat(res)

    # if save_dict:
    #     try:
    #         with open(f"{results_dir}/all_results.json", "x") as f:
    #             json.dump(res, fp=f, sort_keys=True, indent=3)    
    #     except FileExistsError:
    #         print("File already exists")


    return df








def parse_gs_results_csv():
    #??? what was the idea? maybe to parse the top x or
    # something lke that?
    pass


#Loads whole forecast results (Day_XX) for specified id+model
#TODO: REMOVE?
def load_model_resuls_by_id_as_dict(model_name: str, result_id: int, get_all_days=True, results_path=rconf.RESULTS_PATH):
    """Loads whole forecast results (Day_XX) for specified id+model
    Load the directory into a single dictionary, with all day-ahead fc results as well as the stats, params etc.

    Args:
        model (_type_): String of the model, same as result directory name (so Arima, LSTM, etc)
        id (_type_): Id of the model run which is to be returned
    """
    model_name = model_name.capitalize() #Important!
    path_pattern = os.path.join(f"{results_path}/{model_name}/{result_id}_*/Day_*.csv")
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


def load_fc_error_by_id_as_df(result_id: int, model_name: str, results_path=rconf.RESULTS_PATH)->pd.DataFrame:
    """Loads the file forecast_errors.csv, containing all (14) day-ahead fc errors,
    returns a dataframe, adding ID and Model name

    Args:
        result_id (int): Id of result to return
        model_name (str): Model name of result

    Returns:
        pd.DataFrame: Dataframe of all fc errors per day, adding ID and model name
    """
    if model_name.lower() == "lstm":
        model_name_path = model_name.upper()
    else:
        model_name_path = model_name.capitalize() #Important!

    path = glob.glob(f"{results_path}/{model_name_path}/{result_id}_*/forecast_errors.csv")[0]

    fc_err_df = pd.read_csv(path, index_col=0, sep=";")
    fc_err_df.assign(id=result_id, model=model_name)

    return fc_err_df




def load_model_resuls_by_id_as_df(model_name: str, result_id: int, get_all_days=True, results_path=rconf.RESULTS_PATH):
    """Loads whole forecast results (Day_XX) for specified id+model
    Load the directory into a single dictionary, with all day-ahead fc results as well as the stats, params etc.

    Args:
        model (_type_): String of the model, same as result directory name (so Arima, LSTM, etc)
        id (_type_): Id of the model run which is to be returned
    """
    if model_name.lower() == "lstm":
        model_name = model_name.upper()
    else:
        model_name = model_name.capitalize() #Important!

    print(model_name, result_id)
    path_pattern = os.path.join(f"{results_path}/{model_name}/{result_id}_*/Day_*.csv")
    files = sorted(glob.glob(path_pattern))

    # Fct to sort files numerically
    def get_day_number(filepath):
            filename = os.path.basename(filepath)
            match = re.search(r'Day_(\d+)', filename)
            return int(match.group(1)) if match else 0

    files.sort(key=get_day_number)
    
    data_list = []
    for file in files:
        # Create key: Day_1, Day_2, etc.
        key = os.path.splitext(os.path.basename(file))[0] 
        day_num = get_day_number(file)

        df = pd.read_csv(file, index_col=0, parse_dates=True, sep=";")
        df = (df
        .assign(model=model_name.lower(), id=result_id, day=day_num)
        .rename(columns={"Upper_CI":"Upper", "Lower_CI":"Lower"}, errors="ignore")
        )

        data_list.append(df)

    data_df = pd.concat(data_list)
        
    return data_df




 
#----------------------------------------------------------------------------------
# MARK: Mergers
# Merge dicts/dfs
#----------------------------------------------------------------------------------

def merge_stats_params_to_gs_errors(df: pd.DataFrame, stats_params_dict, params=None)->pd.DataFrame:
    """Adds columns for all params+stats to individual forecast error results.

    merges the specified param(s) to the df of a model
    stats_params_dict is stats_params, from parse_all_stats_params
    model is the model name as string, as specified in stats_params_dict/stats_params


    Args:
        df (pd.DataFrame): _description_
        stats_params_dict (_type_): dict generated by parse_all_stats_params(), containing each models 
        params (_type_, optional): Name of a parameter. If specified, subsets to only these params. Defaults to None.

    Returns:
        pd.DataFrame: Dataframe with all 
    """
    
    models = ["arima", "sarimax", "lstm", "prophet"]
    for model in models:
        params_df = pd.DataFrame.from_dict(stats_params_dict[model], orient="index")
        params_df = params_df.rename(columns={"Upper_CI":"Upper", "Lower_CI":"Lower"}, errors="ignore")
        params_df.index.name = "id"
        
        if params:
            params_df = params_df[params]

        #in df, strings are complete lowercase
        df[model.lower()] = df[model.lower()].join(other=params_df, how="left")

    return df


def merge_stats_params_to_id(df: pd.DataFrame, stats_params_dict: dict, key_map: dict)->pd.DataFrame:
    model = df["model"][0]
    id = df["id"][0]
    res = stats_params_dict[model][id]

    stats_params_df = (pd.DataFrame(
        [stats_params_dict[model][id]],
        index=[id])
        .pipe(add_exog_key, key_map=key_map)
    )
    
    df = pd.merge(df, stats_params_df, how="left", left_on="id", right_index=True)
    return df

def merge_model_overviews(results_overview: dict)->pd.DataFrame:
    #Merges the individual key:value pairs of the model into a single df.
    # Input: dict with models as keys, and dfs (with gs error+stats+params) as values
    res = []
    for model, df in results_overview.items():
        df = df.assign(model=model)
        res.append(df)

    res_df = pd.concat(res)

    return res_df



 
#----------------------------------------------------------------------------------
# MARK: Getters
# Gets data from dfs/dicts (best, etc)
#----------------------------------------------------------------------------------

#n best results from grid_search_results
def get_best_n_results(results: pd.DataFrame, error_val: str, n: int=1):
    """n best results from grid_search_results
    Get best n results of a model sorted by an error_val through grid_search_results.csv
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


#How many grid sets were nonsensical
def count_na_rows(model: pd.DataFrame):
    """How many grid sets were nonsensical
    Count the number of valid rows and the number of rows with na values,
    Mostly relevant for LSTM, to see how many of the runs were successful.

    Args:
        model (pd.DataFrame): _description_
    """
    n_total = len(model.index)
    n_valid = model.dropna().shape[0]
    n_na = n_total - n_valid

    return {"total": n_total, "valid_rows" : n_valid, "na_rows" : n_na}


# INCOMPLETE!
def get_top_n_results(model: pd.DataFrame, error_val: str="RMSE", n: int=1):
    """INCOMPLETE! Gets top n results of a model, the direct results -- not the overview.
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


# MISSING! Add columns with the parameters to the results
def add_params_to_overview():
    """MISSING! Add columns with the parameters to the results"""
    # is already implemented with 
    pass


# Over/Underprediction day counts, Max Over/Underprediction values
def get_forecast_by_id():
    #input: model name, id as dict/df for best model
    pass


#Get best by exog_cols combination:
def get_best_by_exog_cols_combination(res_dict: pd.DataFrame, forecast_error=["RMSE"], top_N=1, chapter="05", save_dict=False, results_dir=rconf.IMG_PATH):
    """Get best by exog_cols combination
    Gets the best result for each model, grouped by exog_cols combination, 
    returns a single df with best with added model name
    and saves dict to /plots.
    NOTE: resulting df will have columns for all parameters that are passed in res_dict.
    Set forecast error to sort by, by supplying a list. order matters!
    Input: dict with models as keys, results overview df as value"""

    res = []

    for model in res_dict.keys():
        df = res_dict[model]
        if "exog_cols" in df.columns:

            #fill exog_cols na with ["None"], then remove rows withtout values (arbitrary cols)
            df["exog_cols"] = df["exog_cols"].apply(lambda x: x if isinstance(x, list) else ["None"])
            df = df.dropna(subset=["ME", "MAE"], axis=0) #subset is arbitrary

            #convert lists in exog_cols to sets
            df["exog_key"] = df["exog_cols"].apply(lambda x: frozenset(x) if x is not None else None)
            #group by exog_key, sort and get top_N
            subset = (df
                      .sort_values(forecast_error)
                      .groupby("exog_key")
                      .head(top_N) #top of each group(exog_key)
            )
            subset["model"] = model
            res.append(subset)

        else:
            subset = df.sort_values(forecast_error).head(top_N)
            subset["model"] = model
            subset["exog_cols"] = "No exogenous variables"
            res.append(subset)

    #concatenate to df:
    res_df = pd.concat(res)

    if save_dict:
        try:
            res_df.to_csv(f"{results_dir}/{chapter}-{today}-Best_Results_by_Exog_combo.csv", sep=";")
        except FileExistsError:
            print("File already exists")

    return res_df

def get_overprediction_underprediction_days(df, day_ahead: int=1)->pd.DataFrame:
    """For LATEX TABLE, get count of days from Day_X ahead forecast with underprediction/overprediction,
    also get values of maximum overprediction/underprediction

    Args:
        df (pd.DataFrame): Containing all forecast results for all (14) days

    Returns:
        pd.DataFrame: Dataframe with column value, and rows for days overpredicted, days underprediction, 
        maximum overprediction, maximum underprediction
    """
    if len(df["model"].unique()) == 1:
        model_name = df["model"][0] #NOTE: no check for validity!
    else:
        raise ValueError("Multiple names in 'model' column")
    

    #Day counts:
    overprediction = df.query("day == @day_ahead and Mean > Actual")
    underprediction = df.query("day == @day_ahead and Mean < Actual")

    over_count = int(len(overprediction))
    under_count = int(len(underprediction))

    #Max values
    max_overprediction = df.query("day == @day_ahead")["Difference"].max()
    min_overprediction = df.query("day == @day_ahead")["Difference"].min()

    #Make df:
    res_df = pd.DataFrame(data={
        "Overprediction (Days)" : [f"{over_count:.0f}"],
        "Underprediction (Days)" : [f"{under_count:.0f}"],
        "Maximum Overprediction" : [f"{max_overprediction:.2f}"],
        "Maximum Underprediction" : [f"{min_overprediction:.2f}"],
    }).T.rename(columns={0: model_name})



    return res_df

 
#----------------------------------------------------------------------------------
# MARK: LATEX TABLES
# some are made directly in analysis.py
#----------------------------------------------------------------------------------

def make_latex_table_over_underprediction_days(df: pd.DataFrame, chapter="05", 
                                               tbl_path: str=rconf.TBL_PATH,
                                               tbl_name: str="table_over_underprediction")->None:
    """Uses output table created from get_overprediction_underprediction_days()
    to create and save a latex table

    Args:
        df (pd.DataFrame): _description_

    Returns:
        _type_: _description_
    """
    latex_tbl = (
        df
        .rename_axis(None, axis=1)
        .rename(columns=lambda x: x.capitalize() if x != 'lstm' else x.upper())
        .style
        .to_latex(hrules=True)
    )

    # #inject header: model name spans all columns
    # latex_tbl = latex_tbl.replace(
    #     "\\toprule",  "\\multicolumn{2}{c}{\\textbf{" + model_name.capitalize() + "}} \\\\\\midrule"
    # )

    with open(f"{tbl_path}/{chapter}_{tbl_name}_all.txt", "w") as f:
        f.write(latex_tbl)

    

 
#----------------------------------------------------------------------------------
# MARK: Helper functions
# Generate dfs necessary for overview tables
#----------------------------------------------------------------------------------

def add_exog_key(df: pd.DataFrame, key_map: dict)->pd.DataFrame:
    """Adds column 'exog_key', which can be grouped (exog_cols cannot)"""
    from numpy import nan
    df["exog_cols"] = df["exog_cols"].fillna(value="")
    df["exog_key"] = df["exog_cols"].map(lambda x: key_map[tuple(sorted(x)) if x else ()])
    return df


 
#----------------------------------------------------------------------------------
# MARK: Plotting
#----------------------------------------------------------------------------------

def forecast_blood_groups(model, params):
    """Use the best param set to forecast individual blood groups"""
    # ! Need to get use_transfused daily aggregate for individual groups!
    pass



#MISSING!plot day_1 forecast for each model, for each exogenous combination?
def plot_results_by_exogenous():
    """MISSING! plot day_1 forecast for each model, for each exogenous combination"""    
    #Get top for each exogenous combo

    pass





#plot each model error values by rank to show decrease by ordering
def plot_error_val_increase(res_dict, error_val: list=["RMSE"], n: int=100, img_name: str="05_RES_error_increase_by_rank"):
    """plot each model error values by rank to show decrease by ordering

    Args:
        res_dict (_type_): dict of dataframes
        error_val (list, optional): which error values to plot. Defaults to ["RMSE"].
        n (int, optional): head() count of rank. Defaults to 100.
        img_name (str, optional): filename for saving. Defaults to "05_RES_error_increase_by_rank".
    """
    # plot the increase of the error value by rank for all models.
    # errors are sorted individually by each line!
    # sharedy!


    fig, axes = plt.subplots(
        2, 1, 
        figsize=(16,8), 
        sharey=True, 
        sharex=True)
    
    fig.suptitle(f"Ranked error values")

    for i, ax in enumerate(axes.flatten()):
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
                .plot(kind="line", label=key, ax=ax)
            )

        key = list(res_dict.keys())[i]
        ax.set_title(error_val[i])
        ax.set_ylabel(error_val[i])
    
    plt.subplots_adjust(hspace=0.6)

    #unified legend
    handles, labels = ax.get_legend_handles_labels()
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



def plot_one_day_ahead_Diff(df: pd.DataFrame, day_ahead: int=1, diff_color=True, start_date: str="2025-01-01", end_date: str=None, 
                       save_fig=True, img_path: str=rconf.IMG_PATH, img_name: str="Day_one_fc_with_Diff")->None:
    """plots one X-day-ahead forecast with difference filled between fc and actual
    can set time range and which day ahead (usually one-day-ahead)

    Args:
        df (pd.DataFrame): Containing all forecasts for all days ahead.
        day_ahead (int, optional): Which day ahead to plot. Defaults to 1.
        diff_color (bool, optional): If True, has two distinct colors for positive or negative. If False, is a single color
        start_date (str, optional): As string. Defaults to "2025-01-01".
        end_date (str, optional): Automatically detects last day for this day ahead (Note: different last date for each day-ahead). Defaults to None.
        save_fig (bool, optional): _description_. Defaults to True.
        img_path (str, optional): _description_. Defaults to rconf.IMG_PATH.
        img_name (str, optional): _description_. Defaults to "Day_one_fc_with_CI".
    """
    
    if not end_date:
        end_date = df[df["day"] == day_ahead].index.max() #different day_ahead have different last date

    model_name = df["model"][0]
    model_id = df["id"][0]
    #filter df:
    df = (df
          .sort_index()
          .query("day == @day_ahead")
          .loc[start_date:end_date]
          )

    #Plotting
    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)

    ax.plot(df["Actual"], label="Actual", lw=2, color=(0.1, 0.1, 0.1))
    ax.plot(df["Mean"], label="Forecast", lw=2, color="violet")

    if diff_color:
        # Postiv/Neg untersch. fill-between
        ax.fill_between(x=df.index, y1=df["Actual"], y2=df["Mean"], where=(df["Actual"]>df["Mean"]), interpolate=True, label="Underprediction", color="violet", alpha=0.3)
        ax.fill_between(x=df.index, y1=df["Actual"], y2=df["Mean"], where=(df["Actual"]<df["Mean"]), interpolate=True, label="Overprediction", color="lightblue", alpha=0.3)
    else:
        # Einfärbiger fill-between
        ax.fill_between(x=df.index, y1=df["Actual"], y2=df["Mean"], label="Difference", color="blue", alpha=0.25)

    ax.set_xlabel("Date")
    ax.set_ylabel("EC transfused")
    ax.legend(loc="lower right", ncol=4)

    fig.subplots_adjust(bottom=0.75)
    fig.suptitle(f"Day {day_ahead} forecast with difference between forecasted and actual transfusion counts for {model_name.capitalize()} (ID: {model_id})")

    if save_fig:
        save_plot(fig, img_name, model_name, img_path)



def plot_one_day_ahead_Diff_bars(df: pd.DataFrame, day_ahead: int=1, diff_color=True, start_date: str="2025-01-01", end_date: str=None, 
                       save_fig=True, img_path: str=rconf.IMG_PATH, img_name: str="Day_one_fc_with_Diff_bars")->None:
    """plots one X-day-ahead forecast with difference filled between fc and actual as BARS
    can set time range and which day ahead (usually one-day-ahead)

    Args:
        df (pd.DataFrame): Containing all forecasts for all days ahead.
        day_ahead (int, optional): Which day ahead to plot. Defaults to 1.
        diff_color (bool, optional): If True, has two distinct colors for positive or negative. If False, is a single color
        start_date (str, optional): As string. Defaults to "2025-01-01".
        end_date (str, optional): Automatically detects last day for this day ahead (Note: different last date for each day-ahead). Defaults to None.
        save_fig (bool, optional): _description_. Defaults to True.
        img_path (str, optional): _description_. Defaults to rconf.IMG_PATH.
        img_name (str, optional): _description_. Defaults to "Day_one_fc_with_CI".
    """
    
    if not end_date:
        end_date = df[df["day"] == day_ahead].index.max() #different day_ahead have different last date

    model_name = df["model"][0]
    model_id = df["id"][0]
    
    #filter df:
    df = (df
          .sort_index()
          .query("day == @day_ahead")
          .loc[start_date:end_date]
          )

    bottom = np.minimum(df["Actual"], df["Mean"])
    height = np.abs(df["Actual"] - df["Mean"])
    colors = np.where(df["Actual"] > df["Mean"], "violet", "lightblue")

    #Plotting
    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    
    # Plot floating bars
    ax.bar(
        df.index,
        height,
        bottom=bottom,
        color=colors,
        width=0.7,          # < 1 creates gaps between bars
        edgecolor="white",  # white edge fakes a small gap
        linewidth=0.5,
        )
    
    ax.plot(df["Actual"], label="Actual", lw=1, color=(0.1, 0.1, 0.1), alpha=0.5)

    ax.set_xlabel("Date")
    ax.set_ylabel("EC transfused")

    from matplotlib.patches import Patch
    legend_elements = [
        ax.get_lines()[0],
        #ax.get_lines()[1],
        Patch(facecolor="violet", alpha=0.7, label="Underprediction"),
        Patch(facecolor="lightblue", alpha=0.7, label="Overprediction"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", ncol=4)
    fig.subplots_adjust(bottom=0.75)
    fig.suptitle(f"Day {day_ahead} forecast with difference between forecasted and actual demand for {model_name.capitalize()} (ID: {model_id})")

    if save_fig:
        save_plot(fig, img_name, model_name, img_path)


def plot_one_day_ahead_CI(df, day_ahead: int=1, start_date: str="2025-01-01", end_date: str=None, 
                       save_fig=True, img_path: str=rconf.IMG_PATH, img_name: str="Day_one_fc_with_CI")->None:
    """plots one X-day-ahead forecast with upper and lower limits
    can set time range and which day ahead (usually one-day-ahead)

    Args:
        df (pd.DataFrame): Containing all forecasts for all days ahead.
        day_ahead (int, optional): Which day ahead to plot. Defaults to 1.
        start_date (str, optional): As string. Defaults to "2025-01-01".
        end_date (str, optional): Automatically detects last day for this day ahead (Note: different last date for each day-ahead). Defaults to None.
        save_fig (bool, optional): _description_. Defaults to True.
        img_path (str, optional): _description_. Defaults to rconf.IMG_PATH.
        img_name (str, optional): _description_. Defaults to "Day_one_fc_with_CI".
    """
    
    if not end_date:
        end_date = df[df["day"] == day_ahead].index.max() #different day_ahead have different last date

    model_name = df["model"][0]
    model_id = df["id"][0]
    #filter df:
    df = (df
          .sort_index()
          .query("day == @day_ahead")
          .loc[start_date:end_date]
          )

    #Plotting
    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)

    ax.plot(df["Actual"], label="Actual", lw=2, color=(0.1, 0.1, 0.1))
    ax.plot(df["Mean"], label="Forecast", lw=2, color="mediumvioletred")
    ax.plot(df["Upper"], color="violet", lw=0.15, alpha=0.5)
    ax.plot(df["Lower"], color="violet", lw=0.15, alpha=0.5)
    ax.fill_between(x=df.index, y1=df["Lower"], y2=df["Upper"], label="95% CI", color="violet", alpha=0.2)

    ax.set_xlabel("Date")
    ax.set_ylabel("EC transfused")
    ax.legend(loc="lower right", ncol=4)

    fig.subplots_adjust(bottom=0.75)
    fig.suptitle(f"Day {day_ahead} forecast with confidence intervals for {model_name.capitalize()} (id {model_id})")

    if save_fig:
        save_plot(fig, img_name, model_name, img_path)


def plot_all_fc_days(df, start_date: str="2025-01-01", end_date: str=None, 
                     save_fig=True, img_path: str=rconf.IMG_PATH, img_name: str="All_days")->None:
    #Plots time series, with all (14) fc days at once, overlapping each other
    #df needs to contain all forecasts of all days ahead
    if not end_date:
        end_date = df.index.max()

    if len(df["model"].unique()) & len(df["id"].unique()) == 1:
        model_name = df["model"][0]
        model_id = df["id"][0]
    else:
        raise ValueError("Either 'model' or 'id' are not unique!")

    n_days = len(df["day"].unique())

    #Plotting
    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)

    ax.plot(df["Actual"], label="Actual", lw=1, color=(0.1, 0.1, 0.1))


    cmap = plt.get_cmap("winter_r")
    norm = mcolors.Normalize(vmin=1, vmax=n_days)
    # for day in range(n_days+1, 0, -1): #count down to get Day_1 on top
    for day in range(1, n_days+1):
        ax.plot(df.query("day == @day")["Mean"], color=cmap(norm(day)), label=None, lw=1, zorder=n_days-day) #label=f"Day {day}",  cmap(day/n_days)

    ax.set_xlabel("Date")
    ax.set_ylabel("EC transfused")


    #Colorbar for legend
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    axins = inset_axes(
        ax,
        width="5%",  # width: 5% of parent_bbox width
        height="50%",  # height: 50%
        loc="lower left",
        bbox_to_anchor=(0.15, -0.1, 1, 0.025), #l/b/r/h
        bbox_transform=ax.transAxes,
        borderpad=0
    )

    cbar = fig.colorbar(sm, cax=axins,  
                        orientation="horizontal", location="bottom", 
                        aspect=13, shrink=0.1, 
                        pad=0.005, ticks=[1, n_days])
    cbar.outline.set_visible(False) # remove border
    cbar.set_ticklabels(["Day 1", "Day 14"])
    cbar.ax.set_position([0.55, 0.0, 0.3, 0.015]) #left/bottow/right/height

    ax.legend(
        loc="lower left",
        frameon=False,
        ncol=1,
        bbox_to_anchor=(0, -0.15)
    )#, ncol=(n_days)//2 + 1)

    # fig.subplots_adjust(bottom=0.15)
    fig.suptitle(f"{n_days}-Day forecast for {model_name.capitalize()} (ID: {model_id})")

    if save_fig:
        save_plot(fig, img_name, model_name, img_path)




def save_plot(fig, name: str, model: str, path: str, chapter="05")->None:
    print(f"Saved plot to {path}/{chapter}_{model.capitalize()}_{name}.png")
    fig.savefig(f"{path}/{chapter}_{model.capitalize()}_{name}.png")



# Plot time series with Actual, FC, upper/lower CI for all models:
def plot_fc_time_series(model: pd.DataFrame):
    """Plot time series with Actual, FC, upper/lower CI for all models:"""
    pass






#ordered bar chart of count for age at use
def plot_age_at_usage(df: pd.DataFrame, chapter="05", save_fig=False, save_path="./plots"):
    """Ordered bar chart of count for age at use
    Two plots, because transfused has too high scale difference to rest"""

    df = (df
     .query('storage_period < 50 and storage_period >= 0')
     .value_counts(subset=["use", "storage_period"])
     .sort_index()
     .unstack()
     .transpose()
     .drop("unknown", axis=1)
    
    )

    fig, axes = plt.subplots(2,1, sharex=True)
    fig.suptitle("Age of EC at usage")
    
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, col in enumerate(df.columns):
        color = colors[i % len(colors)]
        if col == "transfused":
            axes[0].scatter(x=df.index, y=df[col], s=7, label=col, color=color)
        else:
            axes[1].scatter(x=df.index, y=df[col], s=7, label=col, color=color)
            # ax.bar(df.index, height=df[col], label=col)
    
    axes[0].set_title("Transfused")
    axes[1].set_title("Discarded & Expired")

    #shared legend setup:
    handles1, labels1 = axes[0].get_legend_handles_labels()
    handles2, labels2 = axes[1].get_legend_handles_labels()

    fig.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="lower center",
        ncol=len(df.columns),
        markerscale=3
    )
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.2, hspace=0.6)

    for ax in axes:
        ax.set_xlabel("Days of storage")
        ax.set_ylabel("Count")
    

    if save_fig:
        today = datetime.today().strftime('%Y_%m_%d')
        fig.savefig(fname=f"{save_path}/{chapter}-{today}-Age_at_usage.png") #05 is results chapter in latex



def plot_exog_combination_results(df: pd.DataFrame, forecast_error: str="RMSE", highlight="None", save_fig=False, save_path="./plots", chapter="05"):
    #not sure how to implement: 
    # wanted to make one subplot for each model, showing difference in result
    # with all 7 exog combinations. maybe use top 10 by exog-combo, 
    # and do a ranked line plot? Or just top 1 per exog-combo and bar chart?

    fig, axes = plt.subplots(2, 2)
    fig.suptitle("Influence of exogenous variables")


    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, model in enumerate(df["model"].unique()):
        color = colors[i % len(colors)]
        hl_color = colors[-1] 
        
        #Best value for each exogenous
        if model != "arima":
            df_ex = (
                df
                .query(f"model == '{model}'")
                .sort_values(["RMSE"])
                .head(7)
                .filter(items=[f"{forecast_error}", "model", "exog_cols"])
            )
        else:
            df_ex = (
                df
                .query(f"model == '{model}'")
                .sort_values(["RMSE"])
                .head(7)
                .filter(items=[f"{forecast_error}", "model"])
            )


    
    for i, col in enumerate(df.columns):
        ax = axes.flatten()[i]

        ax.set_title("Transfused")
        ax.set_title("Discarded & Expired")

        #shared legend setup:
        handles1, labels1 = axes[0].get_legend_handles_labels()
        handles2, labels2 = axes[1].get_legend_handles_labels()

        fig.legend(
            handles1 + handles2,
            labels1 + labels2,
            loc="lower center",
            ncol=len(df.columns),
            markerscale=3
        )
        fig.tight_layout()
        plt.subplots_adjust(bottom=0.2, hspace=0.6)

        for ax in axes:
            ax.set_xlabel("Days of storage")
            ax.set_ylabel("Count")
        

        if save_fig:
            today = datetime.today().strftime('%Y_%m_%d')
            fig.savefig(fname=f"{save_path}/{chapter}-{today}-Age_at_usage.png") #05 is results chapter in latex
