# Main pipeline for doing the evaluation of the results + viz of results.
#%%
import pandas as pd
from matplotlib import pyplot as plt
import os
import pathlib
import glob
import re
import json

from src import clean
from src import config_cleaning
from src import transform

from datetime import datetime


import importlib


#%%
IMG_PATH = "./plots"
TBL_PATH = "../thesis_text/tables"

#Initial load of results
RESULTS_PATH = "./results_FirstRun"
PATH_ARIMA = f"{RESULTS_PATH}/Arima"
PATH_SARIMAX = f"{RESULTS_PATH}/Sarimax"
PATH_LSTM = f"{RESULTS_PATH}/LSTM"
PATH_PROPHET = f"{RESULTS_PATH}/Prophet"
SEP = ","
INDEX_COL = "id"
today = datetime.today().strftime('%Y_%m_%d')


#%%
#----------------------------------------------------------------------------------
# MARK: Function definitions
#----------------------------------------------------------------------------------

def read_daily_orders(data_path="./data/01_raw", filename="Blood-data_complete_including_2025.tsv", top_N_wards=9):
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




#%%
def aggregate_daily(df, by="date_entry"):
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

#%%


def parse_all_stats_params(result_dir="./results", save_dict=False):
    """Get a dict, where each model contains a (param+stats)-dict with ID as key
    Iterate over each models result directory, then over each subdirectory.
    Gather params.json and stats.json, to create a dict with keys for each model, 
    where each model contains dicts with key of each id, containing the jsons.
    Use to link id from results to stats, to filter for example only models without exogenous vars etc.
    """
    res = {
        "Arima" : {},
        "Sarimax" : {},
        "LSTM" : {},
        "Prophet" : {}
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
                        model = stats.pop("model_name")
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

#%%
def parse_gs_results_csv():
    #??? what was the idea? maybe to parse the top x or
    # something lke that?
    pass


#%%
# def merge_params_to_df(df: pd.DataFrame, all_stats, model, params=None):
#     """
#     merges the specified param(s) to the df of a model
#     all_stats is stats_params, from parse_all_stats_params
#     model is the model name as string, as specified in all_stats/stats_params
#     """
#     params_df = pd.DataFrame.from_dict(all_stats[model], orient="index")
#     params_df.index.name = "id"
    
#     if params:
#         params_df = params_df[params]


#     df = df.join(other=params_df, how="left")

#     return df


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
    
    models = ["Arima", "Sarimax", "LSTM", "Prophet"]
    for model in models:
        params_df = pd.DataFrame.from_dict(stats_params_dict[model], orient="index")
        params_df = params_df.rename(columns={"Upper_CI":"Upper", "Lower_CI":"Lower"}, errors="ignore")
        params_df.index.name = "id"
        
        if params:
            params_df = params_df[params]

        #in df, strings are complete lowercase
        df[model.lower()] = df[model.lower()].join(other=params_df, how="left")

    return df




#%%


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

#Loads whole forecast results (Day_XX) for specified id+model
#TODO: REMOVE?
def load_model_resuls_by_id_as_dict(model_name: str, result_id: int, get_all_days=True, results_path=RESULTS_PATH):
    """Loads whole forecast results (Day_XX) for specified id+model
    Load the directory into a single dictionary, with all day-ahead fc results as well as the stats, params etc.

    Args:
        model (_type_): String of the model, same as result directory name (so Arima, LSTM, etc)
        id (_type_): Id of the model run which is to be returned
    """
    model_name = model_name.capitalize() #Important!
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


def load_model_resuls_by_id_as_df(model_name: str, result_id: int, get_all_days=True, results_path=RESULTS_PATH):
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

    path_pattern = os.path.join(f"{RESULTS_PATH}/{model_name}/{result_id}_*/Day_*.csv")
    files = sorted(glob.glob(path_pattern))

    # Fct to sort files numerically
    def get_day_number(filepath):
            filename = os.path.basename(filepath)
            match = re.search(r'Day_(\d+)', filename)
            print(int(match.group(1)))
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


def forecast_blood_groups(model, params):
    """Use the best param set to forecast individual blood groups"""
    # ! Need to get use_transfused daily aggregate for individual groups!
    pass


#Get best by exog_cols combination:
def get_best_by_exog_cols_combination(res_dict: pd.DataFrame, forecast_error=["RMSE"], top_N=1, chapter="05", save_dict=False, results_dir="./plots"):
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


#%%
# Over/Underprediction day counts, Max Over/Underprediction values
def get_forecast_by_id():
    #input: model name, id as dict/df for best model
    pass



#%% 
#----------------------------------------------------------------------------------
# MARK: Plotting
#----------------------------------------------------------------------------------

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


#%% 
#----------------------------------------------------------------------------------
# MARK: Table generators
# Generate dfs necessary for overview tables
#----------------------------------------------------------------------------------





#%% 
#----------------------------------------------------------------------------------
# MARK: Function execution
#----------------------------------------------------------------------------------


# GENERAL LOADING/TRANSFORMATION FOR ALL MODELS
#%%
#Load all grid_search_result.csv files into dict
all_gs_errors = {
    "arima" : pd.read_csv(filepath_or_buffer=f"{PATH_ARIMA}/grid_search_results.csv", sep=SEP, index_col=INDEX_COL),
    "sarimax" : pd.read_csv(filepath_or_buffer=f"{PATH_SARIMAX}/grid_search_results.csv", sep=SEP, index_col=INDEX_COL),
    "lstm" : pd.read_csv(filepath_or_buffer=f"{PATH_LSTM}/grid_search_results.csv", sep=SEP, index_col=INDEX_COL),
    "prophet" : pd.read_csv(filepath_or_buffer=f"{PATH_PROPHET}/grid_search_results.csv", sep=SEP, index_col=INDEX_COL)
}




#%%
stats_params = parse_all_stats_params(RESULTS_PATH, save_dict=True)
#Join param(s) to results:

#%%
results_overview_dict = merge_stats_params_to_gs_errors(
    df=all_gs_errors, 
    stats_params_dict=stats_params
)


def merge_model_overviews(results_overview: dict)->pd.DataFrame:
    #Merges the individual key:value pairs of the model into a single df.
    # Input: dict with models as keys, and dfs (with gs error+stats+params) as values
    res = []
    for model, df in results_overview.items():
        df = df.assign(model=model)
        res.append(df)

    res_df = pd.concat(res)

    return res_df




#TODO: fix the exog_key column, get better querying.
#Needs to be alphabetically sorted!
key_labels_map = {
    (): "none",
    ("tlmax", "tlmin"): "temp",
    ("use_discarded", "use_expired"): "use",
    # ('covid_daily_scaled', 'influenza_daily_scaled'): "respiratory",
    ("day_of_week", "day_of_year", "holiday_enc", "workday_enc", "year"): "date",
    # ("tlmax", "tlmin", 'covid_daily_scaled', 'influenza_daily_scaled'): "temp+respiratory",
    ("tlmax", "tlmin", "use_discarded", "use_expired"): "temp+usage",
    # ("day_of_week", "day_of_year", "holiday_enc", "tlmax", "tlmin", "workday_enc", "year_scaled"): "temp+date",
    ("day_of_week", "day_of_year", "holiday_enc", "tlmax", "tlmin", "workday_enc", "year"): "temp+date",
    # ("day_of_week", "day_of_year", "holiday_enc", 'covid_daily_scaled', 'influenza_daily_scaled', "workday_enc", "year_scaled"): "respiratory+date",
    ("day_of_week", "day_of_year", "holiday_enc", "use_discarded", "use_expired", "workday_enc", "year"): "usage+date",
    # ("day_of_week", "day_of_year", "holiday_enc", "tlmax", "tlmin", 'covid_daily_scaled', 'influenza_daily_scaled', "workday_enc", "year_scaled"): "all",
    ("day_of_week", "day_of_year", "holiday_enc", "tlmax", "tlmin", "use_discarded", "use_expired", "workday_enc", "year"): "all",
}

def add_exog_key(df: pd.DataFrame)->pd.DataFrame:
    """Adds column 'exog_key', which can be grouped (exog_cols cannot)"""
    from numpy import nan
    df["exog_cols"] = df["exog_cols"].fillna(value="")
    df["exog_key"] = df["exog_cols"].map(lambda x: key_labels_map[tuple(sorted(x)) if x else ()])
    return df


#combine into one massive df (of Day_1 fc errors)
results_overview_df = merge_model_overviews(results_overview_dict)

#create exog_key, to make exog_cols sortable
results_overview_df = add_exog_key(results_overview_df)


#%%
#Get best results for all models + exog_combos
best_model_ids = (results_overview_df
                  .dropna(subset=["MAE", "RMSE", "run_duration"], axis=0)
                  #.query("exog_key == 'temp+date'")
                  #.sort_values(["RMSE", "model"])
                  .groupby(["model", "exog_key"])
                  .tail(1)
)



#using STYLE
for model in best_model_ids["model"].unique():
    best_model_ids_latex = (
        best_model_ids
        .query(f"model == '{model}'")
        .sort_values("RMSE")
        .reset_index()
        .loc[:, ["id", "RMSE", "MAE", "ME","MAPE", "MaxError", "exog_key"]]
        .rename(columns=lambda c: c.replace("_", r"\_")) #escape underscore!
        .style
        .hide(axis="index")
        .format(
            {"MAPE": "{:.3f}"},
            precision=2)
        .to_latex(
            hrules=True
            # captions need to be put inplace inside latex, so this can generate only the \begin[tabular]
            # part and be used within \begin[table]\centering\input
        )
    )
    #inject multiline header with "model"
    best_model_ids_latex = best_model_ids_latex.replace(
        "\\toprule",  "\\multicolumn{7}{c}{\\textbf{" + model.capitalize() + "}} \\\\\\midrule"
    )

    with open(f"{TBL_PATH}/05_tbl_overview_grouped_exog_cols_{model}.txt", "w") as f:
        f.write(best_model_ids_latex)



#using standard pandas to_latex
for model in best_model_ids["model"].unique():
    (best_model_ids
     .query(f"model == '{model}'")
     .rename(columns=lambda c: c.replace("_", r"\_")) #escape underscore!
     .to_latex(
        buf=f"{TBL_PATH}/05_tbl_overview_grouped_exog_cols_{model}.txt", 
        columns=["ME", "MAE", "RMSE", "MAPE", "MaxError", "model", "exog\_key"], #escape underscore!
        float_format="{:0.2f}".format,
        label=f"""Best results for {model}, grouped by the combination of 
        exogenous variables. For the mapping of 'exog\_key', see table \\ref{{tbl:todo}} """
        )
    )




#%%
#----------------------------------------------------------------------------------
# MARK: INDIVIDUAL MODELS: 
# Function execution
#----------------------------------------------------------------------------------
N = 5


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# SARIMAX

#%%
# Get best for sarimax -- overview (=grid_search_results_csv)
sarimax_overview_top_n = get_best_n_results(results_overview_dict["sarimax"], "RMSE", n=N)
sarimax_top_id = sarimax_overview_top_n.index[0]
sarimax_top = sarimax_overview_top_n.loc[sarimax_top_id]
sarimax_best = load_model_resuls_by_id_as_dict("Sarimax", result_id=sarimax_top_id)

#%%
#%%
best_by_exog_col = get_best_by_exog_cols_combination(results_overview_dict)

#Get full forecast results from top models (over all exog combos):
best_runs = best_by_exog_col.sort_values("RMSE").groupby("model").head(1)
best_runs_forecasts = {}
for _, row in best_runs.iterrows():
    #best_runs has only 4 target rows
    print(row["model"], row.name)
    best_runs_forecasts[row["model"]] = load_model_resuls_by_id_as_dict(model_name=row["model"], result_id=row.name)

best_run_forecasts = []
for _, row in best_runs.iterrows():
    print(row["model"], row.name)
    best_run_forecasts.append(load_model_resuls_by_id_as_df(model_name=row["model"], result_id=row.name))
best_runs_df = pd.concat(best_run_forecasts)


best_runs_list = []
for model, day in best_runs_forecasts.items():
    for day, df in day.items():
        df = (df
              .rename(columns={"Upper_CI":"Upper", "Lower_CI":"Lower"}, errors="ignore")
              .assign(model=model, day=day)
        )
        best_runs_list.append(df)
best_runs_df = pd.concat(best_runs_list)

best_runs_df = pd.concat(
    [df.assign(model=key) for key, df in best_runs_forecasts.items()]
)






#%%
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# MARK: ARIMA
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 



# # Get best for Arima -- overview (=grid_search_results_csv)
# # get
# arima_overview_top_n = get_best_n_results(results_overview_dict["arima"], "RMSE", n=N)
# arima_top_id = arima_overview_top_n.index[0]
# arima_top = arima_overview_top_n.loc[arima_top_id]
# arima_best = load_model_resuls_by_id("Arima", result_id=arima_top_id)



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# MARK: LSTM
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


# # Get best for lstm -- overview (=grid_search_results_csv)
# lstm_overview_top_n = get_best_n_results(results_overview_dict["lstm"], "RMSE", n=N)
# lstm_top_id = lstm_overview_top_n.index[0]
# lstm_top = lstm_overview_top_n.loc[lstm_top_id]
# lstm_best = load_model_resuls_by_id("LSTM", result_id=lstm_top_id)



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# MARK: PROPHET
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


# # Get best for prophet -- overview (=grid_search_results_csv)
# prophet_overview_top_n= get_best_n_results(results_overview_dict["prophet"], "RMSE", n=N)
# prophet_top_id = prophet_overview_top_n.index[0]
# prophet_top = prophet_overview_top_n.loc[prophet_top_id]
# prophet_best = load_model_resuls_by_id("Prophet", result_id=prophet_top_id)


#%%
# Get best results, for each exogenous combination:


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


# %%




# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# MARK: EC AGE
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 



#%%
ec_entry_raw = read_daily_orders()
ec_entry_aggregated = aggregate_daily(ec_entry_raw)



#%%
plot_age_at_usage(ec_entry_raw, save_fig=True)

plot_error_val_increase(all_gs_results, error_val=["RMSE", "MAE"], n=100)


