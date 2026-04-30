#%%
import importlib
from src import result_evaluation as eval
from src import result_evaluation_config as rconf
from src import model
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
importlib.reload(rconf)
importlib.reload(eval)





#%% 
#----------------------------------------------------------------------------------
# MARK: Get Values
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
    # ("use_discarded", "use_expired"): "use",
    ('covid_daily_scaled', 'influenza_daily_scaled'): "respiratory",
    ("day_of_week", "day_of_year", "holiday_enc", "workday_enc", "year_scaled"): "date",
    ('covid_daily_scaled', 'influenza_daily_scaled', "tlmax", "tlmin"): "temp+respiratory",
    # ("tlmax", "tlmin", "use_discarded", "use_expired"): "temp+use",
    ("day_of_week", "day_of_year", "holiday_enc", "tlmax", "tlmin", "workday_enc", "year_scaled"): "temp+date",
    # ("day_of_week", "day_of_year", "holiday_enc", "tlmax", "tlmin", "workday_enc", "year"): "temp+date",
    ('covid_daily_scaled', "day_of_week", "day_of_year", "holiday_enc", 'influenza_daily_scaled', "workday_enc", "year_scaled"): "respiratory+date",
    # ("day_of_week", "day_of_year", "holiday_enc", "use_discarded", "use_expired", "workday_enc", "year"): "use+date",
    ('covid_daily_scaled', "day_of_week", "day_of_year", "holiday_enc", 'influenza_daily_scaled', "tlmax", "tlmin", "workday_enc", "year_scaled"): "all",
    # ("day_of_week", "day_of_year", "holiday_enc", "tlmax", "tlmin", "use_discarded", "use_expired", "workday_enc", "year"): "all",
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
     column_format="p{4cm}p{10cm}", #for line wrapping
     hrules=True)
)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%%
#combine into one massive df (of Day_1 fc errors)
all_errors_df = eval.merge_model_overviews(results_overview_dict)

#create exog_key, to make exog_cols sortable
all_errors_df = eval.add_exog_key(all_errors_df, key_labels_map)

importlib.reload(eval)
#PLOT RMSE grouped by exog_key, for each model
eval.plot_rank_by_exog_key(df=all_errors_df)


#Get one massive df with all forecasts + stats + params for all models
# Dont do this, is unnessecary
#all_forecasts_df = eval.parse_all_forecasts(result_dir="./results_FirstRun")


#%%
# Load entry counts for actually arrived EC
ec_entry_raw = eval.read_daily_entries()
ec_entry_aggregated = eval.aggregate_daily_entries(ec_entry_raw)




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
# LATEX: table "id | params" for best model ids:
cols_to_drop = ["RMSE", "MAE", "ME", "MAPE", "MedAE", "MSE", "MaxError", "start_date", "end_date", "window_num", "split_date", "prediction_column", "exog_cols", "model"]
col_order = {
    "arima" : ["id", "exog_key", "p", "d", "q", "run_duration"],
    "sarimax" : ["id", "exog_key", "p", "d", "q", "P", "D", "Q", "m", "run_duration"],
    "lstm" : ["id", "exog_key", "memory_cells", "epochs", "batch_size", "run_duration"],
    "prophet" : ["id", "exog_key", "seasonality_mode", "seasonality_prior_scale", "holidays_prior_scale", "run_duration"],
}
for model in best_model_ids["model"].unique():
    best_model_ids_latex_params = (
        best_model_ids
        .query(f"model == '{model}'") #and ~exog_key.str.contains('use|temp')
        .sort_values("RMSE")
        .reset_index()
        # .drop(columns=cols_to_drop) #remove fc errors
        # .dropna(axis=1) #drop missing columns (different params per model!)
        .loc[:, col_order[model]]
        # .pipe(lambda df: df[["id", "exog_key"] + [c for c in df.columns if c not in ["id", "exog_key"]]]) #bring id/exog_keys to front
        #.assign(MAPE=lambda x: x["MAPE"] * 100)
        #.set_index("id")
        .assign(run_duration=lambda d: d["run_duration"] / 60)
        .rename(columns={"run_duration" : "run_duration (min)"})
        .rename(columns=lambda c: c.replace("_", r"\_")) #escape underscore!
        .style
        .hide(axis="index")
        .format(
            {"run\_duration": "{:.1f}"},
            precision=0)
        .to_latex(
            hrules=True
            # captions need to be put inplace inside latex, so this can generate only the \begin[tabular]
            # part and be used within \begin[table]\centering\input
        )
    )

    colnum = len(col_order[model])

    #inject multiline header with "model"
    best_model_ids_latex_params = best_model_ids_latex_params.replace(
        "\\toprule",  "\\multicolumn{" + str(colnum) + "}{c}{\\textbf{" + rconf.mmap[model]["name"] + "}} \\\\\\midrule"
        # "\\toprule",  "\\multicolumn{colnum}}{c}{\\textbf{" + model.capitalize() + "}} \\\\\\midrule"
    )

    with open(f"{rconf.TBL_PATH}/05_{model}_tbl_overview_grouped_exog_cols_PARAMS.txt", "w") as f:
        f.write(best_model_ids_latex_params)


#%%
#----------------------------------------------------------------------------------
# MARK: PLOTTING/TABLES: 
# Function execution
#----------------------------------------------------------------------------------

eval.plot_error_val_increase(all_gs_errors, error_val=["RMSE", "MAE"], n=100)


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
        .assign(MAPE=lambda x: x["MAPE"] * 100)
        .rename(columns=lambda c: c.replace("_", r"\_")) #escape underscore!
        .style
        .hide(axis="index")
        .format(
            #{"MAPE": "{:.2f}"},
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

#LATEX: same table (fc errors) but for all models best run each:
all_best_model_ids_latex = (
    best_model_ids
    .sort_values("RMSE")
    .reset_index()
    .groupby("model")
    .head(1)
    .loc[:, ["model", "id", "RMSE", "MAE", "ME","MAPE", "MaxError", "exog_key"]]
    .assign(MAPE=lambda x: x["MAPE"] * 100)
    .assign(model=lambda x: np.where(
        x["model"] != "prophet",
        x["model"].str.upper(),
        x["model"].str.capitalize()
    ))
    # .sort_values("model") #if i want alphabetical?
    .rename(columns=lambda c: c.replace("_", r"\_")) #escape underscore!
    .style
    .hide(axis="index")
    .format(
        #{"MAPE": "{:.f}"},
        precision=2)
    .to_latex(
        hrules=True
        # captions need to be put inplace inside latex, so this can generate only the \begin[tabular]
        # part and be used within \begin[table]\centering\input
    )
)
# #inject multiline header with "model"
# best_model_ids_latex = best_model_ids_latex.replace(
#     "\\toprule",  "\\multicolumn{7}{c}{\\textbf{" + model.capitalize() + "}} \\\\\\midrule"
# )

with open(f"{rconf.TBL_PATH}/05_all_tbl_overview_best_fc_errs.txt", "w") as f:
    f.write(all_best_model_ids_latex)


# LATEX TABLE: Counts for days with over/underprediction, Max overprediction, max underprediction
best_models_id_name = (best_model_ids
                #.query("~exog_key.str.contains('use|temp'"")
                .sort_values("RMSE")
                .reset_index()
                .groupby("model")
                .head(1)
                .loc[:, ["id", "model"]]
)



# LATEX: 

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
#1y ago#%%
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
# # SARIMAX
# # Comment out??


# # Get df with 14-Day fc errors as well as df with all forecasts
# sarimax_best_model, sarimax_best_id = best_model_ids.query("model == 'sarimax'").sort_values("RMSE").head(1).pipe(lambda d: (d["model"].values[0], d.index.values[0])) #for just the best
# #Filter for exog_key to (not) contain something
# # sarimax_best_model, sarimax_best_id = best_model_ids.query("model == and ~exog_key.str.contains('use|temp')").sort_values("RMSE").head(1).pipe(lambda d: (d["model"].values[0], d.index.values[0])) #for just the best
# sarimax_best_res_fc = eval.load_model_results_by_id_as_df(model_name="Sarimax", result_id=sarimax_best_id) 
# sarimax_best_res_fc = eval.merge_stats_params_to_id(sarimax_best_res_fc, stats_params, key_labels_map)

# sarimax_best_res_fcerr = eval.load_fc_error_by_id_as_df(result_id=sarimax_best_id, model_name=sarimax_best_model)

# importlib.reload(eval)
# eval.plot_forecast_errors_per_day(df=sarimax_best_res_fcerr, save_fig=False)

# #Plot 1-day forecast plus upper/lower:
# eval.plot_one_day_ahead_CI(sarimax_best_res_fc, save_fig=False)
# eval.plot_one_day_ahead_Diff(sarimax_best_res_fc, save_fig=False)
# eval.plot_one_day_ahead_Diff_bars(sarimax_best_res_fc, save_fig=False)




# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%%
# As Loop:
importlib.reload(rconf)
importlib.reload(eval)
diff_entry_fc = []
all_best_res_fc = []
all_best_res_fc_entry = []
for model in ["arima", "sarimax", "lstm", "prophet"]:     
    print("Creating tables and plots for", model)
    # Get df with 14-Day fc errors as well as df with all forecasts
    best_model, best_id = best_model_ids.query("model == @model").sort_values("RMSE").head(1).pipe(lambda d: (d["model"].values[0], d.index.values[0])) #for just the best
    #Filter for exog_key to (not) contain something
    # best_model, best_id = best_model_ids.query("model == and ~exog_key.str.contains('use|temp')").sort_values("RMSE").head(1).pipe(lambda d: (d["model"].values[0], d.index.values[0])) #for just the best
    print(best_model, best_id)
    best_res_fc = eval.load_model_results_by_id_as_df(model_name=model, result_id=best_id) 
    best_res_fc = eval.merge_stats_params_to_id(best_res_fc, stats_params, key_labels_map)
    all_best_res_fc.append(best_res_fc)
    #Plot 1-day forecast plus upper/lower:
    eval.plot_one_day_ahead_CI(best_res_fc)
    eval.plot_one_day_ahead_Diff(best_res_fc)
    eval.plot_one_day_ahead_Diff_bars(best_res_fc)


    #Plot Actual vs Forecast (Mean) and Actual vs entry_count
    #join entry_count (removes other columns besides Actual, Mean, model, id (and entry_count))
    best_res_fc_entry = eval.join_entry_count(df=best_res_fc, entry_df=ec_entry_aggregated, day=1)
    all_best_res_fc_entry.append(best_res_fc_entry) #for dm_test

    #test diebold-mariano, save as LATEX:
    dm_test_results = eval.test_diebold_mariano_all_models(all_best_res_fc_entry)
    eval.make_latex_tbl_diebold_mariano_test(dm_test_results)

    #with centered bars
    eval.plot_actual_fc_mean_diff_bars_centered(best_res_fc_entry)#, start_date=rconf.SUBSET_START, end_date=rconf.SUBSET_END)
    # as lineplot with filled difference
    eval.plot_actual_fc_mean_diff(best_res_fc_entry, start_date=rconf.SUBSET_START, end_date=rconf.SUBSET_END)
    #Plot time series of actual, forecast ('Mean'), entry_count
    eval.plot_ts_actual_fc_entry(best_res_fc_entry)
    #Plot cumulative sum of actual, forecast ('Mean'), entry_count
    eval.plot_cumsum_actual_fc_entry(best_res_fc_entry)

    #Get difference between Actual/entry_count, Actual/Mean (fc)
    diff_entry_fc.append(eval.calculate_cumsum_diff(best_res_fc_entry))

    #Plot forecast error per day
    best_res_fcerr = eval.load_fc_error_by_id_as_df(result_id=best_id, model_name=best_model)
    eval.plot_forecast_errors_per_day(df=best_res_fcerr)
    eval.make_latex_table_best_run_fc_errs(df=best_res_fcerr) #LATEX table

#Plot 14-day forecast streak, 4 subplots:
eval.plot_single_fourteen_days(all_best_res_fc, start_date="2025-06-01", historic_start="2025-05-01")

#%%

# LATEX: table of difference Actual vs. entry_count, Actual vs. FC, taken from last value of cumsum
diff_entry_fc_latex = (
    pd.concat(diff_entry_fc)
    .drop_duplicates(keep="last") #diff_entry_fc contains entry_count 1x per model
    .reset_index(drop=True)
    .rename(columns={
        "id": "ID",
        # "Difference entry" : r"\makecell{Cumulative \\Difference\\Entry}"
        # "Difference Forecast" : r"\makecell{Cumulative \\Difference\\Forecast}"
        "Difference Forecast" : "Cumulative Difference"
    })
    .rename(columns=lambda c: c.replace("_", r"\_")) #escape underscore!
    .rename(columns=lambda c: c.replace("entry", "Entry")) 
    # .rename(columns=lambda c: c.replace("Difference", "Cumulative Difference")) 
    .assign(Model=lambda c: c["Model"].replace({k:v["name"] for k,v in rconf.mmap.items()}))
    # .assign(Model=lambda c: c["Model"].map(lambda m: rconf.mmap[m]["name"]))
    .style
    .hide(axis="index")
    .format(
        #{"MAPE": "{:.f}"},
        precision=0)
    .to_latex(
        hrules=True
        # captions need to be put inplace inside latex, so this can generate only the \begin[tabular]
        # part and be used within \begin[table]\centering\input
    )
)
with open(f"{rconf.TBL_PATH}/05_all_tbl_cumsum_end_DIFF.txt", "w") as f:
    f.write(diff_entry_fc_latex)




#%%
# Comment out??


# # Get best for sarimax -- overview (=grid_search_results_csv)
# sarimax_overview_top_n = eval.get_best_n_results(results_overview_dict["sarimax"], "RMSE", n=N)
# sarimax_top_id = sarimax_overview_top_n.index[0]
# sarimax_top = sarimax_overview_top_n.loc[sarimax_top_id]
# sarimax_best = eval.load_model_resuls_by_id_as_dict("Sarimax", result_id=sarimax_top_id)



#%%
# Comment out??

# best_by_exog_col = eval.get_best_by_exog_cols_combination(results_overview_dict)

# #Get full forecast results from top models (over all exog combos):
# best_runs = best_by_exog_col.sort_values("RMSE").groupby("model").head(1)
# best_runs_forecasts = {}
# for _, row in best_runs.iterrows():
#     #best_runs has only 4 target rows
#     print(row["model"], row.name)
#     best_runs_forecasts[row["model"]] = eval.load_model_resuls_by_id_as_dict(model_name=row["model"], result_id=row.name)

# best_run_forecasts = []
# for _, row in best_runs.iterrows():
#     print(row["model"], row.name)
#     best_run_forecasts.append(eval.load_model_results_by_id_as_df(model_name=row["model"], result_id=row.name))
# best_runs_df = pd.concat(best_run_forecasts)


# best_runs_list = []
# for model, day in best_runs_forecasts.items():
#     for day, df in day.items():
#         df = (df
#               .rename(columns={"Upper_CI":"Upper", "Lower_CI":"Lower"}, errors="ignore")
#               .assign(model=model, day=day)
#         )
#         best_runs_list.append(df)
# best_runs_df = pd.concat(best_runs_list)

# best_runs_df = pd.concat(
#     [df.assign(model=key) for key, df in best_runs_forecasts.items()]
# )





# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# MARK: EC ENTRY
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 





#%%
# TS Scatter plot by age at use
importlib.reload(eval)
eval.plot_age_at_usage(ec_entry_raw, save_fig=True)


# plot_actual_fc_mean_diff_bars_centered()
#fig.legend()




#%%


# for model in all_errors_df["model"].unique():
ax.plot(all_errors_df.query("model == 'sarimax'")["RMSE"], color=all_errors_df["exog_key"])




# %%


# def test_diebold_mariano_against_entry(df: pd.DataFrame):
#     #diebold_mariano for each model against entry_count
#     # uses best_res_fc_entry as df

#     #negative test statistic says, that the first (Mean) time series fits the actual data better
#     # than the second (entry_count), with a p-value close to zero, its confidence is high.
#     # the Null hypothesis (both are roughly equal) can be rejected -> fc1 performs better
#     true = df["Actual"]
#     #forecast 1 to compare
#     fc1 = df["Mean"]
#     #forecast 2 to compare
#     fc2 = df["entry_count"]

#     dm_res = dm_test(V=true, P1=fc1, P2=fc2, h=1, one_sided=False) #default loss: mse

#     test_res = pd.DataFrame(data={
#         "model" : [df["model"][0]],
#         "id" : [df["id"][0]],
#         "Test statistic" : [dm_res[0]],
#         "p-value" : [dm_res[1]]
#     })

#     return test_res



