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

from datetime import datetime, timedelta

import textwrap

import importlib
import copy
from dieboldmariano import dm_test
from itertools import combinations




#----------------------------------------------------------------------------------
# MARK: File reading + preprocessing
#----------------------------------------------------------------------------------

# Entry dates
def read_daily_entries(data_path=rconf.PATH_RAW_DATA, filename="Blood-data_complete_including_2025.tsv", top_N_wards=9):
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
def parse_all_stats_params(result_dir=rconf.RESULTS_PATH, save_dict=False):
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
                print(f"Save new file to {results_dir}/all_params.json")
                json.dump(res, fp=f, sort_keys=True, indent=3)    
        except FileExistsError:
            print(f"File already exists at {results_dir}/all_params.json")


    return res



def parse_all_forecasts(result_dir=rconf.RESULTS_PATH, save_dict=False, forecast_days=14):
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
    fc_err_df = fc_err_df.assign(id=result_id, model=model_name)

    return fc_err_df




def load_model_results_by_id_as_df(model_name: str, result_id: int, get_all_days=True, results_path=rconf.RESULTS_PATH):
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
        # .pipe(add_exog_key, key_map=key_map)
    )
    if model != "arima":
        stats_params_df = stats_params_df.pipe(add_exog_key, key_map=key_map)
    
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
        # .rename(columns=lambda model_name: rconf.mmap[model_name]['name'])
        .rename(columns=lambda c: rconf.mmap.get(c, {}).get("name", c.capitalize()))
        # .rename(columns=lambda x: x.capitalize() if x != 'lstm' else x.upper())
        .style
        .to_latex(hrules=True)
    )

    # #inject header: model name spans all columns
    # latex_tbl = latex_tbl.replace(
    #     "\\toprule",  "\\multicolumn{2}{c}{\\textbf{" + model_name.capitalize() + "}} \\\\\\midrule"
    # )

    with open(f"{tbl_path}/{chapter}_all_{tbl_name}.txt", "w") as f:
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


def plot_forecast_errors_per_day(df: pd.DataFrame, save_fig=True, img_path: str=rconf.IMG_PATH, img_name: str="forecast_error_per_day_ahead")->None:
    #Takes dataframe, containing a single models forecast errors for each day ahead

    if len(df["model"].unique()) & len(df["id"].unique()) == 1:
        model_name = df["model"][0]
        model_id = df["id"][0]
    else:
        raise ValueError("Either 'model' or 'id' are not unique!")

    #Get extra day column, to sort by:
    df = df.assign(day=lambda r: r.index.str.split("_").str[1].astype(int))
    df = (df
          .set_index("day")
          .sort_index()
          .drop(["model", "id", "MSE", "MaxError"], axis=1) #MSE is too high to compare and already in RMSE
          .assign(MAPE=lambda c: c["MAPE"]*100)
    )
    # n_days = len(df["day"].unique())

    #plotting
    fig, ax = plt.subplots(figsize=(14, 8), constrained_layout=True)
    for col in df.columns:
        ax.plot(df[col], label=col)

    ax.set_xlabel("Days ahead")
    ax.set_ylabel("Error value")
    ax.set_ylim(ymin=min(0, ax.get_ylim()[0])) #either zero or previous value

    leg = fig.legend(
        loc="upper center", 
        ncol=3, #df.shape[1],
        frameon=False,
        bbox_to_anchor=(0.5, -0.05) #l/b/r/h
    )

    for legobj in leg.legend_handles:
        legobj.set_linewidth(rconf.LEGENDLW)

    fig.subplots_adjust(bottom=0.75)

    figure_title = f"Forecast errors for all Day-ahead forecasts \nfor {rconf.mmap[model_name]['name']} (ID: {model_id})" #model_name.capitalize()
    fig.suptitle(textwrap.fill(figure_title, width=30))#rconf.LINEBREAK))


    if save_fig:
        save_plot(fig, img_name, model_name, img_path)



#plot each model error values by rank to show decrease by ordering
def plot_error_val_increase(res_dict, error_val: list=["RMSE"], n: int=100, 
                            img_name: str="05_RES_error_increase_by_rank"):
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
    
    figure_title = f"Ranked error values"
    fig.suptitle(textwrap.fill(figure_title, width=rconf.LINEBREAK))


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
                .plot(
                    kind="line", 
                    label=rconf.mmap[key]['name'], #key, 
                    ax=ax,
                    lw=2
                )
            )

        key = list(res_dict.keys())[i]
        ax.set_title(error_val[i])
        ax.set_ylabel(error_val[i])

    
    plt.subplots_adjust(hspace=0.6)

    #unified legend
    handles, labels = ax.get_legend_handles_labels()
    legend_handles = [copy.copy(h) for h in handles]
    for handle in legend_handles:
        handle.set_linewidth(8)

    fig.legend(
        legend_handles, 
        labels, 
        loc='center', 
        bbox_to_anchor=(0.5, -0.05),
        frameon=False,
        ncol=4)
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
    
    fig.savefig(fname=f"{rconf.IMG_PATH}/{img_name}")



def plot_one_day_ahead_Diff(df: pd.DataFrame, day_ahead: int=1, diff_color=True, start_date: str=rconf.START_DATE, end_date: str=None, 
                       save_fig=True, img_path: str=rconf.IMG_PATH, img_name: str="Day_one_fc_with_Diff")->None:
    """plots one X-day-ahead forecast with difference filled between fc and actual
    can set time range and which day ahead (usually one-day-ahead)

    Args:
        df (pd.DataFrame): Containing all forecasts for all days ahead.
        day_ahead (int, optional): Which day ahead to plot. Defaults to 1.
        diff_color (bool, optional): If True, has two distinct colors for positive or negative. If False, is a single color
        start_date (str, optional): As string. Defaults to rconf.START_DATE.
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
    fig, ax = plt.subplots(figsize=(20, 8), constrained_layout=True)

    ax.plot(df["Actual"], label="Actual", lw=2, color=(0.1, 0.1, 0.1))
    ax.plot(df["Mean"], label="Forecast", lw=2, color=rconf.mmap[model_name]["col"])#"violet")
    ax.set_ylim(ymin=0)

    if diff_color:
        # Postiv/Neg untersch. fill-between
        ax.fill_between(x=df.index, y1=df["Actual"], y2=df["Mean"], where=(df["Actual"]>df["Mean"]), interpolate=True, label="Underprediction", color="violet", alpha=0.3)
        ax.fill_between(x=df.index, y1=df["Actual"], y2=df["Mean"], where=(df["Actual"]<df["Mean"]), interpolate=True, label="Overprediction", color="lightblue", alpha=0.3)
    else:
        # Einfärbiger fill-between
        ax.fill_between(x=df.index, y1=df["Actual"], y2=df["Mean"], label="Difference", color=rconf.mmap[model_name]["col"], alpha=0.25) #color="blue"

    ax.set_xlabel("Date")
    ax.set_ylabel("EC transfused")
    ax.legend(loc="lower right", ncol=4)

    fig.subplots_adjust(bottom=0.75)

    figure_title = f"Day {day_ahead} forecast with difference between forecasted and actual transfusion counts for {rconf.mmap[model_name]['name']} (ID: {model_id})" #model_name.capitalize()
    fig.suptitle(textwrap.fill(figure_title, width=rconf.LINEBREAK))

    if save_fig:
        save_plot(fig, img_name, rconf.mmap[model_name]['name'], img_path)



def plot_one_day_ahead_Diff_bars(df: pd.DataFrame, day_ahead: int=1, diff_color=True, start_date: str=rconf.START_DATE, end_date: str=None, 
                       save_fig=True, img_path: str=rconf.IMG_PATH, img_name: str="Day_one_fc_with_Diff_bars")->None:
    """plots one X-day-ahead forecast with difference filled between fc and actual as BARS
    can set time range and which day ahead (usually one-day-ahead)

    Args:
        df (pd.DataFrame): Containing all forecasts for all days ahead.
        day_ahead (int, optional): Which day ahead to plot. Defaults to 1.
        diff_color (bool, optional): If True, has two distinct colors for positive or negative. If False, is a single color
        start_date (str, optional): As string. Defaults to rconf.START_DATE.
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
    fig, ax = plt.subplots(figsize=(20, 8), constrained_layout=True)
    
    # Plot floating bars
    ax.bar(
        df.index,
        height,
        bottom=bottom,
        color=colors,
        width=1,          # < 1 creates gaps between bars
        edgecolor="white",  # white edge fakes a small gap
        linewidth=0.5,
        )
    
    ax.plot(df["Actual"], label="Actual", lw=1, color=(0.1, 0.1, 0.1), alpha=0.5)

    ax.set_xlabel("Date")
    ax.set_ylabel("EC transfused")
    ax.set_ylim(ymin=0)

    from matplotlib.patches import Patch
    legend_elements = [
        ax.get_lines()[0],
        #ax.get_lines()[1],
        Patch(facecolor="violet", alpha=0.7, label="Underprediction"),
        Patch(facecolor="lightblue", alpha=0.7, label="Overprediction"),
    ]
    ax.legend(
        handles=legend_elements, 
        loc="lower right", 
        ncol=4,
        frameon=False)
    fig.subplots_adjust(bottom=0.75)

    figure_title = f"Day {day_ahead} forecast with difference between forecasted and actual demand for {rconf.mmap[model_name]['name']} (ID: {model_id})"#model_name.capitalize()
    fig.suptitle(textwrap.fill(figure_title, width=rconf.LINEBREAK))

    if save_fig:
        save_plot(fig, img_name, model_name, img_path)


def plot_one_day_ahead_CI(df, day_ahead: int=1, start_date: str=rconf.START_DATE, end_date: str=None, 
                       save_fig=True, img_path: str=rconf.IMG_PATH, img_name: str="Day_one_fc_with_CI")->None:
    """plots one X-day-ahead forecast with upper and lower limits
    can set time range and which day ahead (usually one-day-ahead)

    Args:
        df (pd.DataFrame): Containing all forecasts for all days ahead.
        day_ahead (int, optional): Which day ahead to plot. Defaults to 1.
        start_date (str, optional): As string. Defaults to rconf.START_DATE.
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
    fig, ax = plt.subplots(figsize=(20, 8), constrained_layout=True)

    ax.plot(df["Actual"], label="Actual", lw=2, color=(0.1, 0.1, 0.1))
    ax.plot(df["Mean"], label="Forecast", lw=2,  color=rconf.mmap[model_name]["col"])#color="mediumvioletred")
    ax.plot(df["Upper"], color=rconf.mmap[model_name]["col"], lw=0.15, alpha=0.5) #color="violet",
    ax.plot(df["Lower"], color=rconf.mmap[model_name]["col"], lw=0.15, alpha=0.5) #color="violet",
    ax.fill_between(x=df.index, y1=df["Lower"], y2=df["Upper"], label="95% CI", color=rconf.mmap[model_name]["col"], alpha=0.2) #color="violet",

    ax.set_xlabel("Date")
    ax.set_ylabel("EC transfused")
    ax.set_ylim(ymin=0)

    ax.legend(
        loc="lower right", 
        ncol=4,
        frameon=False)

    fig.subplots_adjust(bottom=0.75)

    figure_title = f"Day {day_ahead} forecast with confidence intervals for {rconf.mmap[model_name]['name']} (id {model_id})"
    fig.suptitle(textwrap.fill(figure_title, width=rconf.LINEBREAK))

    if save_fig:
        save_plot(fig, img_name, model_name, img_path)


def plot_all_fc_days(df, start_date: str=rconf.START_DATE, end_date: str=None, 
                     save_fig=True, img_path: str=rconf.IMG_PATH, img_name: str="All_days")->None:
    #Plots time series, with all (14) fc days at once (of one model), overlapping each other
    #df needs to contain all forecasts of all days ahead
    if not end_date:
        end_date = df.index.max()

    if len(df["model"].unique()) & len(df["id"].unique()) == 1:
        model_name = str(df["model"][0])
        model_id = df["id"][0]
    else:
        raise ValueError("Either 'model' or 'id' are not unique!")

    df = df.sort_index()
    df = df[start_date:end_date]
    n_days = len(df["day"].unique()) #amount of forecast days in data

    #Plotting
    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)

    ax.plot(df["Actual"], label="Actual", lw=1, color=(0.1, 0.1, 0.1))


    cmap = plt.get_cmap("winter_r")
    norm = mcolors.Normalize(vmin=1, vmax=n_days)


    #only use colormap, if more than one line to plot
    if n_days > 1:
        # for day in range(n_days+1, 0, -1): #count down to get Day_1 on top
        for day in range(1, n_days+1):
            ax.plot(df.query("day == @day")["Mean"], color=cmap(norm(day)), label=None, lw=1, zorder=n_days-day) #label=f"Day {day}",  cmap(day/n_days)

        #Colorbar for legend
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        axins = inset_axes(
            ax,
            width="100%",  # width: 5% of parent_bbox width
            height="100%",  # height: 50% of bbox_ height
            loc="upper left",
            bbox_to_anchor=(0.75, -0.2, 0.25, 0.035), #, 0, 0), #x/y/breite/höhe?
            bbox_transform=ax.transAxes,
            borderpad=0
        )

        cbar = fig.colorbar(sm, cax=axins,  
                            orientation="horizontal", 
                            location="bottom", 
                            ticks=[1, n_days])
        cbar.outline.set_visible(False) # remove border
        cbar.set_ticklabels(["Day 1", f"Day {n_days}"])
        # cbar.ax.set_position([0.55, 0.0, 0.3, 0.015]) #left/bottow/right/height

        n_col = 1
    else:
        #only really the case for COMPARISON MODEL!
        for day in range(1, n_days+1):
            ax.plot(df.query("day == @day")["Mean"], label=model_name, lw=1, zorder=n_days-day) #label=f"Day {day}",  cmap(day/n_days)
        n_col = 2

    ax.set_xlabel("Date")
    ax.set_ylabel("EC transfused")
    ax.set_ylim(ymin=0)

    leg = ax.legend(
        loc="upper left",
        frameon=False,
        ncol=n_col,
        bbox_to_anchor=(0, -0.1)
    )#, ncol=(n_days)//2 + 1)

    for legobj in leg.legend_handles:
        legobj.set_linewidth(6.0)

    # fig.subplots_adjust(bottom=0.15)
    if model_id != "":
        figure_title = f"{n_days}-Day forecast for {rconf.mmap[model_name]['name']} (ID: {model_id})"
        fig.suptitle(textwrap.fill(figure_title, width=rconf.LINEBREAK))

    else:
        print(model_name)
        print(type(model_name))
        figure_title = f"{n_days}-Day forecast for {rconf.mmap[model_name]['name']}" #model_name.replace('_', ' ').capitalize()}"
        fig.suptitle(textwrap.fill(figure_title, width=rconf.LINEBREAK))


    if save_fig:
        save_plot(fig, img_name, model_name, img_path)


def plot_all_model_forecasts(best_models_id_name: pd.DataFrame, day_ahead: int=1, start_date: str=rconf.START_DATE, end_date: str=None, 
                     save_fig=True, img_path: str=rconf.IMG_PATH, img_name: str="models_best_forecast")->None:
    """Plots time series, with all (4) models best (passed) results on the same time scale, overlapping each other.
    df needs to contain all forecasts of all days ahead for all models.
    Takes a df with model and id columns as input, then iterates over them and loads data for each model sequentially using 
    load_model_results_by_id_as_df
    df: pd.DataFrame containing columns id and model
    day_ahead: which day to plot, usually only first"""

    n_models = best_models_id_name.shape[0]
    #Plotting
    fig, ax = plt.subplots(figsize=(20, 8), constrained_layout=True)

    #cmap = plt.get_cmap("Set1")
    #norm = mcolors.Normalize(vmin=1, vmax=n_models)

    for i, model_name in enumerate(best_models_id_name["model"]):
        #individual data prep
        df = load_model_results_by_id_as_df(model_name=model_name, result_id=best_models_id_name.query("model == @model_name")["id"].values[0])
        print(df.head())
        if not end_date:
            end_date = df.index.max()

        df = df.query("day == @day_ahead").loc[start_date:end_date, ]

        #plot Actual value only once
        if i == 0:
            ax.plot(df["Actual"], label="Actual", lw=2, color=(0.1, 0.1, 0.1), zorder=999)

        #individual plotting
        ax.plot(df["Mean"], label=rconf.mmap[model_name]['name'], lw=1) #model.capitalize(),, color=cmap(i),  #label=f"Day {day}",  cmap(day/n_days)cmap(norm(i)),


    ax.set_xlabel("Date")
    ax.set_ylabel("EC transfused")
    ax.set_ylim(ymin=0)


    leg = ax.legend(
        loc="upper center",
        frameon=False,
        # ncol=1,
        ncol=n_models + 1,
        bbox_to_anchor=(0.5, -0.1)
        # bbox_to_anchor=(0.5, -0.15)
    )

    for legobj in leg.legend_handles:
        legobj.set_linewidth(6.0)

    # fig.subplots_adjust(bottom=0.15)
    figure_title = f"{day_ahead}-Day forecast for all models"
    fig.suptitle(textwrap.fill(figure_title, width=rconf.LINEBREAK))

    if save_fig:
        model_name = "All"
        save_plot(fig, img_name, model_name, img_path)



def save_plot(fig, name: str, model: str, path: str, chapter="05")->None:
    print(f"Saved plot to {path}/{chapter}_{model.capitalize()}_{name}.png")
    fig.savefig(f"{path}/{chapter}_{model.capitalize()}_{name}.png", bbox_inches="tight")



# Plot time series with Actual, FC, upper/lower CI for all models:
def plot_fc_time_series(model: pd.DataFrame):
    """Plot time series with Actual, FC, upper/lower CI for all models:"""
    pass






#ordered bar chart of count for age at use
def plot_age_at_usage(df: pd.DataFrame, chapter="05", start_date=rconf.PIPE_START, end_date=rconf.PIPE_END,
                      save_fig=False, img_path: str=rconf.IMG_PATH, img_name: str="age_at_usage")->None:
    """Ordered bar chart of count for age at use
    Two plots, because transfused has too high scale difference to rest"""

    df = (df
          .sort_index()
          .loc[start_date:end_date, ]
          .query('storage_period < 50 and storage_period >= 0')
          .value_counts(subset=["use", "storage_period"])
          .sort_index()
          .unstack()
          .transpose()
          .drop("unknown", axis=1)
    
    )

    #NEW VERSION: showing only Transfused

    fig, ax = plt.subplots(figsize=(12,8))

    figure_title = "Age of EC at transfusion"
    fig.suptitle(textwrap.fill(figure_title, width=rconf.LINEBREAK))

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    ax.bar(x=df.index, height=df["transfused"], width=1, label="Transfused", color=colors[0])
    #ax.scatter(x=df.index, y=df["transfused"], s=45, label="Transfused", color=colors[0])
    ax.set_xlim(right=42)
    # fig.legend(
    #     loc="center right",
    #     bbox_to_anchor=(1, 0.8),
    #     frameon=False,
    #     markerscale=3
    # )


    # OLD VERSION: 2 subplots, first showing transfused, second expired & discarded
    # fig, axes = plt.subplots(2,1, sharex=True)

    # figure_title = "Age of EC at usage"
    # fig.suptitle(textwrap.fill(figure_title, width=rconf.LINEBREAK))

    # colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    # for i, col in enumerate(df.columns):
    #     color = colors[i % len(colors)]
    #     if col == "transfused":
    #         axes[0].scatter(x=df.index, y=df[col], s=7, label=col, color=color)
    #     else:
    #         axes[1].scatter(x=df.index, y=df[col], s=7, label=col, color=color)
    #         # ax.bar(df.index, height=df[col], label=col)
    
    # axes[0].set_title("Transfused")
    # axes[1].set_title("Discarded & Expired")

    # #shared legend setup:
    # handles1, labels1 = axes[0].get_legend_handles_labels()
    # handles2, labels2 = axes[1].get_legend_handles_labels()
    # 
    # fig.legend(
    #     handles1 + handles2,
    #     labels1 + labels2,
    #     loc="lower center",
    #     ncol=len(df.columns),
    #     frameon=False,
    #     markerscale=3
    # )
    # fig.tight_layout()
    # plt.subplots_adjust(bottom=0.2, hspace=0.6)

    # for ax in axes:
    #     ax.set_xlabel("Days of storage")
    #     ax.set_ylabel("Count")

    ax.set_xlabel("Days of storage")
    ax.set_ylabel("Count")
    
    fig.tight_layout()

    if save_fig:
        fig.savefig(fname=f"{img_path}/{chapter}_{img_name}.png") #05 is results chapter in latex



def plot_exog_combination_results(df: pd.DataFrame, forecast_error: str="RMSE", highlight="None", save_fig=False, save_path=rconf.IMG_PATH, chapter="05"):
    #not sure how to implement: 
    # wanted to make one subplot for each model, showing difference in result
    # with all 7 exog combinations. maybe use top 10 by exog-combo, 
    # and do a ranked line plot? Or just top 1 per exog-combo and bar chart?

    fig, axes = plt.subplots(2, 2)

    figure_title = "Influence of exogenous variables"
    fig.suptitle(textwrap.fill(figure_title, width=rconf.LINEBREAK))


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
            frameon=False,
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



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Plot forecast vs actual and entry_count (daily arrived EC) vs actual
# one is line plot with the difference filled, the other is a zero-centered bar chart (better IMO).

def plot_actual_fc_mean_diff(df: pd.DataFrame, start_date=rconf.SUBSET_START, end_date=rconf.SUBSET_END,
                             save_fig=True, img_path: str=rconf.IMG_PATH, img_name: str="actual_vs_fc_vs_entry")->None:
    """Plot Actual value vs Forecast (Mean) vs Entry data in three different subplots."""

    df = df.sort_index().loc[start_date:end_date]

    if len(df["model"].unique()) & len(df["id"].unique()) == 1:
        model_name = str(df["model"][0])
        model_id = df["id"][0]
    else:
        raise ValueError("Either 'model' or 'id' are not unique!")



    #fill-between with actual vs forecast, actual vs entry, fc vs actual:
    fig, ax = plt.subplots(3, 1, figsize=(16, 12), sharex=True, sharey=True)
    colors = {"Actual": "C0", "Mean": "C1", "entry_count": "C2"}

    import itertools
    for i, pair in enumerate(list(itertools.combinations(df[["Actual", "Mean", "entry_count"]], 2))):

        ax[i].plot(df[pair[0]], lw=0.75, label=pair[0], color=colors[pair[0]])
        ax[i].plot(df[pair[1]], lw=0.75, label=pair[1], color=colors[pair[1]])
        #fill between
        ax[i].fill_between(
            x=df.index,
            y1=df[pair[0]],
            y2=df[pair[1]],
            alpha=0.5,
            interpolate=True
        )
        ax[i].set_title(f"{pair[0]} vs. {pair[1]}")
    #unified legend
    handles_by_label = {}
    for a in ax:
        for h, l in zip(*a.get_legend_handles_labels()):
            handles_by_label[l] = h  # dict naturally deduplicates

    fig.legend(
        handles_by_label.values(),
        handles_by_label.keys(),
        loc="lower center",
        ncol=len(handles_by_label),
        frameon=False,
        bbox_to_anchor=(0.5, -0.025),
    )
    fig.supxlabel("Date")

    # figure_title = model_name
    # fig.suptitle(textwrap.fill(figure_title, width=rconf.LINEBREAK))

    fig.tight_layout()

    if save_fig:
        save_plot(fig, img_name, model_name, img_path)





def plot_actual_fc_mean_diff_bars_centered(df: pd.DataFrame, start_date=rconf.FC_START_DAY_1, end_date=rconf.FC_END_DAY_1,
                                           save_fig=True, img_path: str=rconf.IMG_PATH, img_name: str="actual_vs_fc_vs_entry_bars")->None:
    #Difference plot between forecast (passed as df), but with bars instead of lines.
    # no 'day' column present anymore (removed in join_entry_count)
    df = df.sort_index().loc[start_date:end_date]
    # df = df.sort_index().loc[start_date:end_date]

    if len(df["model"].unique()) & len(df["id"].unique()) == 1:
        model_name = str(df["model"][0])
        model_id = df["id"][0]
    else:
        raise ValueError("Either 'model' or 'id' are not unique!")

    fig_names = {
        "Actual" : "Actual",
        "Mean" : f"{rconf.mmap[model_name]['name']}", #model_name.capitalize()}",
        "entry_count" : "EC Arrivals"
    }

    # import itertools
    # for i, pair in enumerate(list(itertools.combinations(sarimax_best_res_fc_entry[["Actual", "Mean", "entry_count"]], 2))):

    pairs = [("Actual", "Mean"), ("Actual", "entry_count")]

    #fill-between with actual vs forecast, actual vs entry, fc vs actual:
    fig, ax = plt.subplots(len(pairs), 1, figsize=(18, 12), sharex=True, sharey=True)
    colors = {"Actual": "C0", "Mean": "C1", "entry_count": "C2"}

    for i, pair in enumerate(pairs):

        pair_name = f"{fig_names[pair[0]]} - {fig_names[pair[1]]}"

        df[pair_name] = df[pair[1]] - df[pair[0]]
        positive = df[pair_name] >= 0

        ax[i].bar(x=df.index[positive], height=df[pair_name][positive], color="green", label="overprediction")
        ax[i].bar(x=df.index[~positive], height=df[pair_name][~positive], color="red", label="underprediction")
        ax[i].axhline(0, color="black", lw=1)

        ax[i].set_title(f"{fig_names[pair[0]]} vs. {fig_names[pair[1]]}")

    #unified legend
    handles_by_label = {}
    for a in ax:
        for h, l in zip(*a.get_legend_handles_labels()):
            handles_by_label[l] = h  # dict naturally deduplicates

    fig.legend(
        handles_by_label.values(),
        handles_by_label.keys(),
        loc="upper center",
        ncol=len(handles_by_label),
        frameon=False,
        bbox_to_anchor=(0.5, -0.025),
    )
    fig.supxlabel("Date")
    fig.supylabel("Difference [EC]")
    # figure_title = model_name
    # fig.suptitle(textwrap.fill(figure_title, width=rconf.LINEBREAK))
    fig.tight_layout()

    if save_fig:
        save_plot(fig, img_name, model_name, img_path)


def join_entry_count(df:pd.DataFrame, entry_df:pd.DataFrame, day: int=1, start_date=rconf.SUBSET_START, end_date=rconf.SUBSET_END)->None:
    """Removes all columns besides 'Acutal', 'Mean', 'model', 'id'"""
    df_joined = (df
                 .query("day == @day")
                 .filter(items=["Actual", "Mean", "model", "id"])
                 .join(entry_df[["count"]]
                 .rename(columns={"count":"entry_count"}),
                 how="left")
    )
    return df_joined


def plot_ts_actual_fc_entry(df:pd.DataFrame, day: int=1, start_date=rconf.FC_START_DAY_1, end_date=rconf.FC_END_DAY_1,
                     save_fig=True, img_path: str=rconf.IMG_PATH, img_name: str="actual_vs_fc_vs_entry_TIMESERIES")->None:
    #df needs to have entry_count joined!
    # no 'day' column present anymore (removed in join_entry_count)
    df = df.loc[start_date:end_date]

    if len(df["model"].unique()) & len(df["id"].unique()) == 1:
        model_name = str(df["model"][0])
        model_id = df["id"][0]
    else:
        raise ValueError("Either 'model' or 'id' are not unique!")

    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["Actual"], label="actual", lw=2)
    ax.plot(df["Mean"], label="forecast", lw=1)
    ax.plot(df["entry_count"], label="entry_count", lw=1)

    fig.legend(frameon=False,)

    if save_fig:
        save_plot(fig, img_name, model_name, img_path)


def plot_cumsum_actual_fc_entry(df:pd.DataFrame, day: int=1, start_date=rconf.FC_START_DAY_1, end_date=rconf.FC_END_DAY_1,
                     save_fig=True, img_path: str=rconf.IMG_PATH, img_name: str="actual_vs_fc_vs_entry_CUMSUM")->None:
    #plot cumsum for actual, forecast and entry
    # df needs to have entry_count joined!
    # no 'day' column present anymore (removed in join_entry_count)

    if len(df["model"].unique()) & len(df["id"].unique()) == 1:
        model_name = str(df["model"][0])
        model_id = df["id"][0]
    else:
        raise ValueError("Either 'model' or 'id' are not unique!")

    df["Diff_entry"] = df.apply(lambda x: x.loc["Actual"] - x.loc["entry_count"], axis=1)# difference actual - entry
    df[["cumsum_actual", "cumsum_fc", "cumsum_entry"]] = df[["Actual", "Mean", "entry_count"]].cumsum()


    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df["cumsum_actual"], label="Actual")
    ax.plot(df["cumsum_fc"], label=rconf.mmap[model_name]["name"])
    ax.plot(df["cumsum_entry"], label="Entry")
    
    leg = fig.legend(
        frameon=False,
        loc="center left",
        bbox_to_anchor=(0.15, 0.7)
    )
    for legobj in leg.legend_handles:
        legobj.set_linewidth(6)

    figtitle = f"Cumulative EC count"
    fig.suptitle(textwrap.fill(figtitle, width=rconf.LINEBREAK))

    fig.tight_layout()

    if save_fig:
        save_plot(fig, img_name, model_name, img_path)


def calculate_cumsum_diff(df:pd.DataFrame)->int:
    #takes as input a dataframe containing Actual Value, Entry Value (total count), Day-1 forecast.

    # caclulates the cumulative difference between Actual and Entry count, Actual and Day-1 forecast.


    if len(df["model"].unique()) & len(df["id"].unique()) == 1:
        model_name = df["model"][0]
        model_id = df["id"][0]
    else:
        raise ValueError("Either 'model' or 'id' are not unique!")



    diff_fc = df["Mean"].sum() - df["Actual"].sum()
    diff_entry = df["entry_count"].sum() - df["Actual"].sum()
    # diff_fc = (df["Mean"].sum() - df["Actual"]).sum()
    # diff_entry = (df["entry_count"].sum() - df["Actual"]).sum()

    #wrap in [] to fix ValueError for scalars
    res_df = pd.DataFrame({
        "Model": [model_name, "Entry count"],
        "id":[model_id, "-"],
        "Difference Forecast" : [diff_fc, diff_entry]
        # "Difference entry" : [diff_entry]
        # "Reduction (%)" : diff_fcdf["entry_count"].sum() / df["Mean"].sum() #reduction of totally bought ECs, not of the FC diff!
        # "Reduction (%)" : df["entry_count"].sum() / df["Mean"].sum() #reduction of totally bought ECs, not of the FC diff!
    })



    return res_df


# def plot_single_fourteen_days(df: pd.DataFrame, start_date: str="2025-06-01", historic_start: str="2025-05-01", days_ahead: int=14,
#                               save_fig=True, img_path: str=rconf.IMG_PATH, img_name: str="single_fc_FOURTEEN_DAYS")->None:
#     #plots fourteen-day forecast for a single time point
#     # df needs to contain all days ( best_res_fc in the loop)
#     # historic_start is the date, when the Actual plot startt (used for reference!)


#     if len(df["model"].unique()) & len(df["id"].unique()) == 1:
#         model_name = df["model"][0]
#         model_id = df["id"][0]
#     else:
#         raise ValueError("Either 'model' or 'id' are not unique!")

#     df.index.name = "date" #necessary for merge

#     #create merge df
#     df_merge = pd.DataFrame({
#         "date" : pd.date_range(start_date, periods=days_ahead, freq='D'),
#         "day" : list(range(1, days_ahead + 1))
#     }).set_index("date")

#     #merge
#     df_res = pd.merge(
#         left=df_merge, 
#         right=df, 
#         how="left", 
#         left_on=["date", "day"],
#         right_on=["date", "day"]
#     )

#     df_actual = df.query("day == 1").loc[historic_start:df_res.index.max()]

#     fig, ax = plt.subplots(figsize=(16, 12))

#     ax.plot(df_actual["Actual"], label="Actual", lw=2.5)
#     #ax.plot(df_res["Actual"], color=actual_color, linestyle="dashed")
#     ax.plot(df_res["Mean"], label=f"{rconf.mmap[model_name]['name']} (ID: {model_id})", lw=3.5) #model_name.capitalize()
#     fc_color = ax.get_lines()[-1].get_c() #get color of last line
#     ax.fill_between(x=df_res.index, y1=df_res["Lower"], y2=df_res["Upper"], alpha=0.15, color=fc_color, label="95% CI", lw=0)

#     leg = fig.legend(
#         frameon=False,
#         loc="upper center",
#         ncol=3,
#         bbox_to_anchor=(0.5, -0.05))

#     for legobj in leg.legend_handles:
#         legobj.set_linewidth(rconf.LEGENDLW)

#     figtitle = f"Single fourteen-day forecast for {rconf.mmap[model_name]['name']}" #{model_name.capitalize()}"
#     fig.suptitle(textwrap.fill(figtitle, width=rconf.LINEBREAK))

#     fig.tight_layout()
#     if save_fig:
#         save_plot(fig, img_name, model_name, img_path)


def plot_single_fourteen_days(all: list, start_date: str="2025-06-01", historic_start: str="2025-05-01", days_ahead: int=14,
                              save_fig=True, img_path: str=rconf.IMG_PATH, img_name: str="single_fc_FOURTEEN_DAYS")->None:
    #plots fourteen-day forecast for a single time point, for all models, on separate subplots (axis)
    # 'all' is list, containing dfs of Actual|Mean|Upper|Lower ( best_res_fc in the loop)
    # historic_start is the date, when the Actual plot startt (used for reference!)
    # loop over models, then create a merge_df, containing dates from start-end_date, and day column of corresponding day

    end_date = (datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
    df = pd.concat(all)

    # if len(df["model"].unique()) != 2 & len(df["id"].unique()) != 2: #TODO: change to 4 (both values!)
    #     raise ValueError("Either 'model' or 'id' are not unique!")


    df.index.name = "date" #necessary for merge

    models = df["model"].unique()
    
    fig, axes = plt.subplots(len(models), 1, figsize=(16, 12))

    for i, model_name in enumerate(models):

        ax = axes.flatten()[i]

        #df for plotting Actual
        df_actual = df.query("day == 1 and model == @model_name").loc[historic_start:end_date]#df_res.index.max()]
        #create merge df
        df_merge = pd.DataFrame({
            "date" : pd.date_range(start_date, periods=days_ahead, freq='D'),
            "day" : list(range(1, days_ahead + 1))
        }).set_index("date")

        data = df.sort_index().query("model == @model_name")
        data.index.name = "date" #necessary for merge
        model_id = data["id"].unique()[0]

        #merge
        df_res = pd.merge(
            left=df_merge, 
            right=data, 
            how="left", 
            left_on=["date", "day"],
            right_on=["date", "day"]
        )
       
        ax.plot(df_actual["Actual"], label="Actual", lw=2.5)

        ax.plot(df_res["Mean"], label=f"{rconf.mmap[model_name]['name']} (ID: {model_id})", color=rconf.mmap[model_name]["col"], lw=3.5) #model_name.capitalize()
        fc_color = ax.get_lines()[-1].get_c() #get color of last line
        ax.fill_between(x=df_res.index, y1=df_res["Lower"], y2=df_res["Upper"], alpha=0.15, color=fc_color, label="95% CI", lw=0)
        ax.set_title(rconf.mmap[model_name]["name"])

    #Remove duplicate legend entries:
    #Get all handles/labels
    handles, labels = [], []
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    # Remove duplicates while preserving order
    unique_labels = {}
    for h, l in zip(handles, labels):
        if l not in unique_labels:
            unique_labels[l] = h

    leg = fig.legend(
        unique_labels.values(),
        unique_labels.keys(),
        frameon=False,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.05))

    for legobj in leg.legend_handles:
        legobj.set_linewidth(rconf.LEGENDLW)

    fig.supylabel("ECs transfused")
    figtitle = f"Single fourteen-day forecast for all models"
    fig.suptitle(textwrap.fill(figtitle, width=rconf.LINEBREAK))

    fig.tight_layout()
    if save_fig:
        model_name = "all"
        save_plot(fig, img_name, model_name, img_path)



def plot_rank_by_exog_key(df: pd.DataFrame, threshold: int=30, metric: str="RMSE", rank_nums=120,
                          save_fig=True, img_path: str=rconf.IMG_PATH, img_name: str="fc_error_by_EXOG_KEY")->None:
    #Plot ranked by RMSE/metric, grouped by exog_key, one plot per model:
    # So for each model, the runs are grouped by exog_key and each group gets a line, with RMSE (y) and Rank (x)
    #threshold sets the cutoff point for RMSE/metric
    fig, axes = plt.subplots(2, 2, figsize=(18, 16), sharex=True, sharey=True)

    #fixed colors
    exog_keys = df["exog_key"].unique()
    colors = {key: color for key, color in zip(exog_keys, plt.cm.tab10.colors)}

    for i, model in enumerate(df["model"].unique()):

        model_name = rconf.mmap[model]["name"]

        ax = axes.flatten()[i]

        for exog_key, group in df.query("model==@model").groupby("exog_key"):
            sorted_group = (group
                            .sort_values(metric)
                            .reset_index(drop=True)
                            .query(f"{metric} <= @threshold")
            )
            ax.plot(sorted_group.index, sorted_group[metric], label=exog_key, color=colors[exog_key], lw=2)
        ax.set_xlim(0, rank_nums)
        ax.set_xlabel("Rank")
        ax.set_ylabel(metric)
        ax.set_title(model_name)

    figure_title = f"Error by 'exog_key', ranked by RMSE"
    fig.suptitle(textwrap.fill(figure_title, width=rconf.LINEBREAK))

    #unified legend
    handles, labels = axes.flatten()[-1].get_legend_handles_labels() #all axes are same
    # handles, labels = axes.get_legend_handles_labels()
    legend_handles = [copy.copy(h) for h in handles]
    for handle in legend_handles:
        handle.set_linewidth(8)

    fig.legend(
        legend_handles, 
        labels, 
        frameon=False,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.05),
        bbox_transform=fig.transFigure
        )

    fig.tight_layout()

    if save_fig:
        model_name = "all"
        save_plot(fig, img_name, model_name, img_path)


# def plot_rank_by_exog_key(df: pd.DataFrame, threshold: int=30, metric: str="RMSE", rank_nums=120,
#                           save_fig=True, img_path: str=rconf.IMG_PATH, img_name: str="fc_error_by_EXOG_KEY")->None:
#     #Plot ranked by RMSE/metric, grouped by exog_key, one plot per model:
#     # So for each model, the runs are grouped by exog_key and each group gets a line, with RMSE (y) and Rank (x)
#     #threshold sets the cutoff point for RMSE/metric
#     for model in df["model"].unique():

#         model_name = rconf.mmap[model]["name"]

#         fig, ax = plt.subplots(figsize=(12, 12))

#         for exog_key, group in df.query("model==@model").groupby("exog_key"):
#             sorted_group = (group
#                             .sort_values(metric)
#                             .reset_index(drop=True)
#                             .query(f"{metric} <= @threshold")
#             )
#             ax.plot(sorted_group.index, sorted_group[metric], label=exog_key, lw=2)

#         ax.set_xlim(0, rank_nums)
#         ax.set_xlabel("Rank")
#         ax.set_ylabel(metric)
#         figure_title = f"Error by 'exog_key' for {model_name}"
#         fig.suptitle(textwrap.fill(figure_title, width=rconf.LINEBREAK))

#         if model in ["lstm", "prophet"]:
#             loc = "center left"
#             bbox = (0.45, 0.5)
#         else:
#             loc = "upper left"
#             bbox = (0.1, 0.9)

#         leg = fig.legend(
#             frameon=False,
#             loc=loc,
#             ncol=1,
#             bbox_to_anchor=bbox
#             )

#         for legobj in leg.legend_handles:
#             print(legobj)
#             legobj.set_linewidth(10)

#         if save_fig:
#             save_plot(fig, img_name, model_name, img_path)




def make_latex_table_best_run_fc_errs(df: pd.DataFrame, save_fig=True, img_path: str=rconf.IMG_PATH, 
                                      img_name: str="tbl_best_run_fourteen_fc_ERRORS")->None:
    #uses best_res_fcerr (best run per model), makes latex table of it.
    if len(df["model"].unique()) == 1:
        model_name = df["model"][0] 
        model_id = str(df["id"][0]) 
    else:
        raise ValueError("Multiple names in 'model' column")

    best_model_fc_errs_latex = (
        df
        #.sort_index()
        .loc[:, ["RMSE", "MAE", "ME","MAPE", "MaxError"]]
        .rename_axis("Days ahead")
        .reset_index()
        .assign(MAPE=lambda x: x["MAPE"] * 100)
        .rename(columns=lambda c: c.replace("_", r"\_")) #escape underscore in col names
        .assign(**{"Days ahead": lambda c: c["Days ahead"].str.replace("_", " ")}) 
        .rename(columns={"Days ahead" : ""}) #remove colname
        .style
        .hide(axis="index")
        .format(
            #{"MAPE": "{:.2f}"},
            precision=2)
        .to_latex(
            hrules=True
        )
    )

    #inject multiline header with "model"
    best_model_fc_errs_latex = best_model_fc_errs_latex.replace(
        "\\toprule",  "\\multicolumn{6}{c}{\\textbf{" + rconf.mmap[model_name]["name"] + " (ID: " + model_id + ")" + "}} \\\\\\midrule"
        # "\\toprule",  "\\multicolumn{colnum}}{c}{\\textbf{" + model.capitalize() + "}} \\\\\\midrule"
    )

    with open(f"{rconf.TBL_PATH}/05_{model_name}_{img_name}.txt", "w") as f:
        f.write(best_model_fc_errs_latex)



# Test diebold-mariano test
# i use best_res_fc_entry, because its availalbe and already contains entry_count
# see:https://pypi.org/project/dieboldmariano/ 
# and: https://real-statistics.com/time-series-analysis/forecasting-accuracy/diebold-mariano-test/

#A tuple of two values. The first is the test statistic, the second is the p-value.
#(-13.270838036003001, 8.010252107985669e-33)
# test statistic: -13.27
# p-value: 8.01e-33
#Also see:
# The null hypothesis is that the two methods have the same forecast accuracy
# "the null hypothesis [is] that the forecast errors coming from the two forecasts bring about the same loss:"

# Test statistic (value):
# Suppose that the significance level of the test is α = 0.05.
# Because this is a two-tailed test 0.05 must be split such that
# 0.025 is in the upper tail and another 0.025 in the lower. The
# z-value that corresponds to -0.025 is -1.96, which is the lower
# critical z-value. The upper value corresponds to 1-0.025, or
# 0.975, which gives a z-value of 1.96.
# The null hypothesis of no difference will be rejected if the
# computed DM statistic falls outside the range of -1.96 to 1.96
# from https://www.lem.sssup.it/phd/documents/Lesson19.pdf

def test_diebold_mariano_all_models(df_list: list):
    #diebold_mariano for each model against each other
    #input: list containing best_res_fc_entry for all models:
    # input var name is all_best_res_fc_entry

    #create combinations (pairs) of indices
    n_models = list(range(0, len(df_list)))
    combos = list(combinations(n_models, r=2))
    results = []

    #Test each model against entry_count:
    for df in df_list:

        true = df["Actual"]

        #forecast to compare
        fc1 = df["Mean"]
        m1_name = rconf.mmap[df['model'][0]]["name"]
        m1_id = str(df['id'][0])
        
        #forecast 2 (entry_count) to compare
        fc2 = df["entry_count"]
        m2_name = "Entry"

        dm_res = dm_test(V=true, P1=fc1, P2=fc2, h=1, one_sided=False) #default loss: mse


        results.append(pd.DataFrame({
            "first model" : f"{m1_name} (ID: {m1_id})",
            "second model" : f"{m2_name}",
            "Test statistic" : [dm_res[0]],
            "p-value" : [dm_res[1]]
        }))
        
    #Test each models pair:
    for pair in combos:
        df1 = df_list[pair[0]]
        df2 = df_list[pair[1]]

        true = df1["Actual"]

        #forecast 1 to compare
        fc1 = df1["Mean"]
        m1_name = rconf.mmap[df1['model'][0]]["name"]
        m1_id = str(df1['id'][0])
        
        #forecast 2 to compare
        fc2 = df2["Mean"]
        m2_name = rconf.mmap[df2['model'][0]]["name"]
        m2_id = str(df2['id'][0])

        dm_res = dm_test(V=true, P1=fc1, P2=fc2, h=1, one_sided=False) #default loss: mse


        results.append(pd.DataFrame({
            "first model" : f"{m1_name} (ID: {m1_id})",
            "second model" : f"{m2_name} (ID: {m2_id})",
            "Test statistic" : [dm_res[0]],
            "p-value" : [dm_res[1]]
        }))

    return pd.concat(results)

def make_latex_tbl_diebold_mariano_test(df: pd.DataFrame, save_fig=True, img_path: str=rconf.IMG_PATH, 
                                      img_name: str="tbl_diebold_mariano_TEST")->None:
    #uses output from test_diebold_mariano_all_models(), which uses all_best_res_fc_entry
    diebold_mariano_latex = (
        df
        .style
        .hide(axis="index")
        .format({
            "Test statistic":"{:.2f}", 
            "p-value":"{:.2e}"})
        .to_latex(
            hrules=True
        )
    )
    if save_fig:
        with open(f"{rconf.TBL_PATH}/05_all_{img_name}.txt", "w") as f:
            f.write(diebold_mariano_latex)
