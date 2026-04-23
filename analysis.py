#%%
import importlib
from src import result_evaluation as eval
from src import result_evaluation_config as rconf
from src import model
import pandas as pd









#%% 
#----------------------------------------------------------------------------------
# MARK: Function execution
#----------------------------------------------------------------------------------


# GENERAL LOADING/TRANSFORMATION FOR ALL MODELS
#%%
#Load all grid_search_result.csv files into dict
all_gs_errors = {
    "arima" : pd.read_csv(filepath_or_buffer=f"{rconf.PATH_ARIMA}/grid_search_results.csv", sep=rconf.SEP, index_col=rconf.INDEX_COL),
    "sarimax" : pd.read_csv(filepath_or_buffer=f"{rconf.PATH_SARIMAX}/grid_search_results.csv", sep=rconf.SEP, index_col=rconf.INDEX_COL),
    "lstm" : pd.read_csv(filepath_or_buffer=f"{rconf.PATH_LSTM}/grid_search_results.csv", sep=rconf.SEP, index_col=rconf.INDEX_COL),
    "prophet" : pd.read_csv(filepath_or_buffer=f"{rconf.PATH_PROPHET}/grid_search_results.csv", sep=rconf.SEP, index_col=rconf.INDEX_COL)
}
stats_params = eval.parse_all_stats_params(rconf.RESULTS_PATH, save_dict=True)

#Join param(s) to results:
results_overview_dict = eval.merge_stats_params_to_gs_errors(
    df=all_gs_errors, 
    stats_params_dict=stats_params
)



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%%
#TODO: fix the exog_key column, get better querying.
#Needs to be alphabetically sorted!
key_labels_map = {
    (): "none",
    ("tlmax", "tlmin"): "temp",
    ("use_discarded", "use_expired"): "use",
    # ('covid_daily_scaled', 'influenza_daily_scaled'): "respiratory",
    ("day_of_week", "day_of_year", "holiday_enc", "workday_enc", "year"): "date",
    # ("tlmax", "tlmin", 'covid_daily_scaled', 'influenza_daily_scaled'): "temp+respiratory",
    ("tlmax", "tlmin", "use_discarded", "use_expired"): "temp+use",
    # ("day_of_week", "day_of_year", "holiday_enc", "tlmax", "tlmin", "workday_enc", "year_scaled"): "temp+date",
    ("day_of_week", "day_of_year", "holiday_enc", "tlmax", "tlmin", "workday_enc", "year"): "temp+date",
    # ("day_of_week", "day_of_year", "holiday_enc", 'covid_daily_scaled', 'influenza_daily_scaled', "workday_enc", "year_scaled"): "respiratory+date",
    ("day_of_week", "day_of_year", "holiday_enc", "use_discarded", "use_expired", "workday_enc", "year"): "use+date",
    # ("day_of_week", "day_of_year", "holiday_enc", "tlmax", "tlmin", 'covid_daily_scaled', 'influenza_daily_scaled', "workday_enc", "year_scaled"): "all",
    ("day_of_week", "day_of_year", "holiday_enc", "tlmax", "tlmin", "use_discarded", "use_expired", "workday_enc", "year"): "all",
}

#Save key_labels_map as latex table
key_labels_df = pd.DataFrame(
    [(v, ", ".join(k) if k else "-") for k, v in key_labels_map.items()],
    columns=["key", "exogenous columns"])
(key_labels_df
 .assign(**{"exogenous columns": lambda d: d["exogenous columns"].str.replace("_", r"\_")}) #escape underscore!
 .style
 .hide(axis="index")
 .to_latex(
     buf=f"{rconf.TBL_PATH}/05_tbl_keys_label_map.txt",
     column_format="p{2cm}p{10cm}", #for line wrapping
     hrules=True)
)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%%
#combine into one massive df (of Day_1 fc errors)
all_errors_df = eval.merge_model_overviews(results_overview_dict)

#create exog_key, to make exog_cols sortable
all_errors_df = eval.add_exog_key(all_errors_df, key_labels_map)



#Get one massive df with all forecasts + stats + params for all models
# Dont do this, is unnessecary
#all_forecasts_df = eval.parse_all_forecasts(result_dir="./results_FirstRun")




# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%%
#Get best results for all models + exog_combos
best_model_ids = (all_errors_df
                  .dropna(subset=["MAE", "RMSE", "run_duration"], axis=0)
                  #.query("exog_key == 'temp+date'")
                  #.sort_values(["RMSE", "model"])
                  .groupby(["model", "exog_key"])
                  .tail(1)
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 




#%%
#----------------------------------------------------------------------------------
# MARK: INDIVIDUAL MODELS: 
# Function execution
#----------------------------------------------------------------------------------


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# ALL MODELS OVERVIEW

#%%
# LATEX TABLE: Convert best results per model+exog-combo to latex and save in thesis_code/tables using STYLE
for model in best_model_ids["model"].unique():
    best_model_ids_latex = (
        best_model_ids
        .query(f"model == '{model}'") #and ~exog_key.str.contains('use|temp')
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

    with open(f"{rconf.TBL_PATH}/05_{model}_tbl_overview_grouped_exog_cols.txt", "w") as f:
        f.write(best_model_ids_latex)



# LATEX TABLE: Counts for days with over/underprediction, Max overprediction, max underprediction
best_models_id_name = (best_model_ids
                #.query("~exog_key.str.contains('use|temp'"")
                .sort_values("RMSE")
                .reset_index()
                .groupby("model")
                .head(1)
                .loc[:, ["id", "model"]]
)

# # #Now get for each the fc days
# sarimax_best_fc = eval.load_model_results_by_id_as_df(model_name="sarimax", result_id=best_models_id_name.query("model == 'sarimax'")["id"].values[0])
# sarimax_over_underpred_counts = eval.get_overprediction_underprediction_days(sarimax_best_fc, day_ahead=1)
# eval.make_latex_table_over_underprediction_days(sarimax_over_underpred_counts)#, model_name=sarimax_over_underpred_counts.index.name)
# eval.plot_all_fc_days(sarimax_best_fc)
# importlib.reload(eval)
# eval.plot_all_model_forecasts(best_models_id_name)


#LATEX TABLE: Gets data &  creates latex table for day over/underprediction + maximum deviance (one table)
#PLOT: plots all (14) days ahead at one time series (one plot per model)
#PLOT: all 4 models fc results on one time series (one plot)
#%%
#Make one table for all 4 models: data prep
over_underpred_counts = []
for model in best_models_id_name["model"]:
    best_fc = eval.load_model_results_by_id_as_df(model_name=model, result_id=best_models_id_name.query("model == @model")["id"].values[0])
    over_underpred_counts.append(eval.get_overprediction_underprediction_days(best_fc, day_ahead=1))
    #Plot all (14) forecast days, one plot per model
    eval.plot_all_fc_days(best_fc)
over_underpred_counts_df = pd.concat(over_underpred_counts, axis=1)
#Actually create the table
eval.make_latex_table_over_underprediction_days(over_underpred_counts_df)

#Plot all 4 best models forecast (Day_1) on one time series
eval.plot_all_model_forecasts(best_models_id_name)

# eval.plot_all_fc_days()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# SARIMAX

# Get df with 14-Day fc errors as well as df with all forecasts
sarimax_best_model, sarimax_best_id = best_model_ids.query("model == 'sarimax'").sort_values("RMSE").head(1).pipe(lambda d: (d["model"].values[0], d.index.values[0])) #for just the best
#Filter for exog_key to (not) contain something
# sarimax_best_model, sarimax_best_id = best_model_ids.query("model == and ~exog_key.str.contains('use|temp')").sort_values("RMSE").head(1).pipe(lambda d: (d["model"].values[0], d.index.values[0])) #for just the best
sarimax_best_res_fc = eval.load_model_results_by_id_as_df(model_name="Sarimax", result_id=sarimax_best_id) 
sarimax_best_res_fc = eval.merge_stats_params_to_id(sarimax_best_res_fc, stats_params, key_labels_map)

sarimax_best_res_fcerr = eval.load_fc_error_by_id_as_df(result_id=sarimax_best_id, model_name=sarimax_best_model)

importlib.reload(eval)
eval.plot_forecast_errors_per_day(df=sarimax_best_res_fcerr)

#Plot 1-day forecast plus upper/lower:
eval.plot_one_day_ahead_CI(sarimax_best_res_fc)
eval.plot_one_day_ahead_Diff(sarimax_best_res_fc)
eval.plot_one_day_ahead_Diff_bars(sarimax_best_res_fc)




# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%%
# As Loop:
importlib.reload(eval)
importlib.reload(rconf)
for model in ["arima", "sarimax", "lstm", "prophet"]:     
    print("Creating tables and plots for", model)
    # Get df with 14-Day fc errors as well as df with all forecasts
    best_model, best_id = best_model_ids.query("model == @model").sort_values("RMSE").head(1).pipe(lambda d: (d["model"].values[0], d.index.values[0])) #for just the best
    #Filter for exog_key to (not) contain something
    # best_model, best_id = best_model_ids.query("model == and ~exog_key.str.contains('use|temp')").sort_values("RMSE").head(1).pipe(lambda d: (d["model"].values[0], d.index.values[0])) #for just the best
    print(best_model, best_id)
    best_res_fc = eval.load_model_results_by_id_as_df(model_name=model, result_id=best_id) 
    best_res_fc = eval.merge_stats_params_to_id(best_res_fc, stats_params, key_labels_map)

    #Plot 1-day forecast plus upper/lower:
    eval.plot_one_day_ahead_CI(best_res_fc)
    eval.plot_one_day_ahead_Diff(best_res_fc)
    eval.plot_one_day_ahead_Diff_bars(best_res_fc)

    best_res_fcerr = eval.load_fc_error_by_id_as_df(result_id=best_id, model_name=best_model)
    eval.plot_forecast_errors_per_day(df=best_res_fcerr)


#%%












#%%
# Get best for sarimax -- overview (=grid_search_results_csv)
sarimax_overview_top_n = eval.get_best_n_results(results_overview_dict["sarimax"], "RMSE", n=N)
sarimax_top_id = sarimax_overview_top_n.index[0]
sarimax_top = sarimax_overview_top_n.loc[sarimax_top_id]
sarimax_best = eval.load_model_resuls_by_id_as_dict("Sarimax", result_id=sarimax_top_id)

#%%
#%%
best_by_exog_col = eval.get_best_by_exog_cols_combination(results_overview_dict)

#Get full forecast results from top models (over all exog combos):
best_runs = best_by_exog_col.sort_values("RMSE").groupby("model").head(1)
best_runs_forecasts = {}
for _, row in best_runs.iterrows():
    #best_runs has only 4 target rows
    print(row["model"], row.name)
    best_runs_forecasts[row["model"]] = eval.load_model_resuls_by_id_as_dict(model_name=row["model"], result_id=row.name)

best_run_forecasts = []
for _, row in best_runs.iterrows():
    print(row["model"], row.name)
    best_run_forecasts.append(eval.load_model_results_by_id_as_df(model_name=row["model"], result_id=row.name))
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
# MARK: EC ENTRY
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 



#%%
ec_entry_raw = eval.read_daily_entries()
ec_entry_aggregated = eval.aggregate_daily_entries(ec_entry_raw)



#%%
eval.plot_age_at_usage(ec_entry_raw, save_fig=True)

eval.plot_error_val_increase(all_gs_errors, error_val=["RMSE", "MAE"], n=100)


