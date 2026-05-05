#Analysis/image generation for abstract which should be submitted to Conference.

#%%
import importlib
from src import result_evaluation as eval
from src import result_evaluation_config as rconf
from src import config_cleaning
from src import model

from src import load
from src import clean
from src import data_model
from src import transform

from statsmodels.tsa.seasonal import seasonal_decompose
import textwrap

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import pandas as pd

SAVE_PATH = "../Conference_Abstract"
START_DATE = "2020-07-03"
END_DATE = "2025-07-03"

STD_COL = "use_transfused"
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# DATA EXPLORATION DATA
df_raw = load.load_data(path=config_cleaning.RAW_DATA_PATH)
df_clean = clean.clean_data(df_raw, new_file_path=config_cleaning.CLEANED_DATA_PATH)
df_processed = transform.transform_data(df_clean, new_file_path=config_cleaning.TRANSFORMED_DATA_PATH)
df = data_model.Data(data=df_processed)



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

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

#combine into one massive df (of Day_1 fc errors)
all_errors_df = eval.merge_model_overviews(results_overview_dict)

#create exog_key, to make exog_cols sortable
all_errors_df = eval.add_exog_key(all_errors_df, key_labels_map)



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
                  .sort_values(["RMSE"])
                  .groupby(["model", "exog_key"])
                  .head(1)
)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Load entry counts for actually arrived EC
ec_entry_raw = eval.read_daily_entries()
ec_entry_aggregated = eval.aggregate_daily_entries(ec_entry_raw)





# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# FUNCTIONS
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# %%
def get_table_values(days_df: pd.DataFrame, error_df: pd.DataFrame):

    mean_fc = days_df["Mean"].mean()
    mean_upper = days_df["Upper"].mean()
    mean_lower = days_df["Lower"].mean()
    RMSE = error_df["RMSE"]

    model_name = days_df["model"][0]
    len_fc = days_df.query("day == 1").shape[0]


    #Actual is ABOVE forecast/upper CI
    days_above_fc = days_df.query("day == 1 and Actual > Mean").shape[0]
    prct_above_fc = days_above_fc/len_fc * 100

    days_above_upper = days_df.query("day == 1 and Actual > Upper").shape[0]
    prct_above_upper = days_above_upper/len_fc * 100

    #Actual is BELOW forecast/lower CI
    days_below_fc = days_df.query("day == 1 and Actual < Mean").shape[0]
    prct_below_fc = days_below_fc/len_fc * 100

    days_below_lower = days_df.query("day == 1 and Actual < Lower").shape[0]
    prct_below_lower = days_below_lower/len_fc * 100

    res_df = pd.DataFrame({
        "Model" : rconf.mmap[model_name]["name"],
        "Mean Forecast" : [round(mean_fc, 2)],
        "Mean Upper CI" : [round(mean_upper, 2)],
        "Mean Lower CI" : [round(mean_lower, 2)],
        "RMSE" : [round(RMSE, 2)],
        "Days above forecast" : [days_above_fc],
        "Days above forecast (%)" : [round(prct_above_fc, 2)],
        "Days above Upper CI" : [days_above_upper],
        "Days above Upper CI (%)" : [round(prct_above_upper, 2)],

        "Days below forecast" : [days_above_fc],
        "Days below forecast (%)" : [round(prct_above_fc, 2)],
        "Days below Upper CI" : [days_above_upper],
        "Days below Upper CI (%)" : [round(prct_above_upper, 2)]
    }).set_index("Model")

    return res_df

#%%
def add_simple_CI_to_SV(df: pd.DataFrame, critical=1.96, start_date=rconf.PIPE_START, end_date=rconf.PIPE_END)->pd.DataFrame:
    #make a simple 95% two-sided CI for single_value forecast, no rolling window, just taking the whole series.
    import numpy as np
    historical = df.loc[start_date:end_date, "Actual"]
    fixed_value = df["Mean"][0] #is already renamed to Mean

    sigma = np.std(historical)

    lower = fixed_value - critical * sigma
    upper = fixed_value + critical * sigma

    df = df.assign(Lower=lower, Upper=upper)


    return df
# %%
def plot_decomposition(df: pd.DataFrame, col: str=STD_COL, start_date: str=START_DATE, end_date: str=END_DATE, 
                       fig=None, axes=None, model: str='additive', period: int=7, name: str="decomposition", path: str=rconf.IMG_PATH, 
                       chapter="05_EXPLRT")->None:
    result = seasonal_decompose(df.loc[start_date:end_date, col], model=model, period=period)


    if axes is None:
        fig, axes = plt.subplots(4, 1, figsize=(21, 14), sharex=True)
    else:
        return_fig = False

    axes[0].plot(result.observed, lw=1.5)
    axes[1].plot(result.trend, lw=1.5)
    axes[2].plot(result.seasonal, lw=1.5)
    axes[3].scatter(result.resid.index, result.resid, s=10)#, alpha=0.5)

    for ax, title in zip(axes, ['Observed', 'Trend', 'Seasonal', 'Residual']):
        ax.set_ylabel(title)

    fig.axes[-1].set_xlabel('date')
    axes[0].set_title(f"Seasonal decomposition ({start_date} to {end_date})")
    fig.tight_layout()

    eval.save_plot(fig, name=name, path=path, chapter=chapter)

    if return_fig:
        return fig, axes
    
#%%

def plot_one_day_ahead_CI(df, day_ahead: int=1, start_date: str=rconf.START_DATE, end_date: str=None, 
                       ax=None, save_fig=True, img_path: str=rconf.IMG_PATH, img_name: str="Day_one_fc_with_CI")->None:
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
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 8), constrained_layout=True)
    else:
        fig = ax.get_figure()
        return_fig = False

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
        fontsize=18,

        frameon=False)

    # fig.subplots_adjust(bottom=0.75)

    figure_title = f"Day {day_ahead} forecast with confidence intervals for {rconf.mmap[model_name]['name']} (id {model_id})"
    ax.set_title(textwrap.fill(figure_title, width=rconf.LINEBREAK))

    if save_fig:
        eval.save_plot(fig, img_name, model_name, img_path)

    if return_fig:
        return fig, ax

#%%
def plot_one_day_ahead_Diff_bars(df: pd.DataFrame, day_ahead: int=1, diff_color=True, start_date: str=rconf.START_DATE, end_date: str=None, 
                       ax=None, save_fig=True, img_path: str=rconf.IMG_PATH, img_name: str="Day_one_fc_with_Diff_bars")->None:
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
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 8), constrained_layout=True)
    else:
        fig = ax.get_figure()
        return_fig = False
        
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
        fontsize=18,
        frameon=False)
    # fig.subplots_adjust(bottom=0.75)

    figure_title = f"Day {day_ahead} forecast with difference between forecasted and actual demand for {rconf.mmap[model_name]['name']} (ID: {model_id})"#model_name.capitalize()
    ax.set_title(textwrap.fill(figure_title, width=rconf.LINEBREAK))

    if save_fig:
        eval.save_plot(fig, img_name, model_name, img_path)
    
    if return_fig:
        return fig, ax





# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# OUTPUT
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# %%



#
# %%
best_res_fc = []
table_res = []
for model in ["arima", "sarimax", "lstm", "prophet"]:     
    # Get df with 14-Day fc errors as well as df with all forecasts
    best_model, best_id = (best_model_ids
                           .query("model == @model")
                           .sort_values("RMSE")
                           .head(1)
                           .pipe(lambda d: (d["model"].values[0], d.index.values[0])) #for just the best
    )
    best_error = all_gs_errors[model].loc[best_id] #one-day ahead
    best_res_fc = eval.load_model_results_by_id_as_df(model_name=model, result_id=best_id) 
    best_res_fc_entry = eval.join_entry_count(df=best_res_fc, entry_df=ec_entry_aggregated, day=1)
    
    #make bar plot
    #eval.plot_one_day_ahead_Diff_bars(best_res_fc, img_path=SAVE_PATH, save_fig=False)#, start_date=rconf.SUBSET_START, end_date=rconf.SUBSET_END)

    #make CI plot
    #eval.plot_one_day_ahead_CI(best_res_fc, img_path=SAVE_PATH, save_fig=False)

    #Get values for table:
    table_res.append(get_table_values(best_res_fc, best_error))

# Load comparison model from results
sv_errors = pd.read_csv("./results/Comparison/forecast_errors.csv", sep=";", index_col=0).loc["single_value"]
sv_fc = pd.read_csv("./results/Comparison/single_value.csv", sep=";", index_col=0, parse_dates=True)
sv_fc = sv_fc.join(other=df["use_transfused"], how="left").rename(columns={"use_transfused":"Actual", "single_value":"Mean"})
sv_fc = add_simple_CI_to_SV(sv_fc)
sv_fc = sv_fc.loc[rconf.FC_START_DAY_1:rconf.FC_END_DAY_1].assign(model="single_value", day=1)

table_res.append(get_table_values(sv_fc, sv_errors))


result_df = pd.concat(table_res)
result_df.to_csv(f"{SAVE_PATH}/results_table.csv")

#%%
# plot_decomposition(df, start_date="2025-01-01", end_date="2025-06-30", path=SAVE_PATH, name="decomposition_abstract_conference")

fig, axd = plt.subplot_mosaic(
    [["decomp0", "ci"],
     ["decomp1", "ci"],
     ["decomp2", "diff"],
     ["decomp3", "diff"]],
    figsize=(24, 14),
    gridspec_kw={"width_ratios": [1, 2]},
    constrained_layout=True
)

# gs = fig.add_gridspec(4, 2)
# ax1 = fig.add_subplot(gs[:, 0:1])
# ax2 = fig.add_subplot(gs[0:2, 1])
# ax3 = fig.add_subplot(gs[2:4, 1])



plot_one_day_ahead_CI(best_res_fc, ax=axd["diff"], save_fig=False)
plot_one_day_ahead_Diff_bars(best_res_fc, ax=axd["ci"], save_fig=False)
plot_decomposition(df, start_date=rconf.SUBSET_START, end_date=rconf.SUBSET_END, fig=fig, axes=[axd["decomp0"], axd["decomp1"], axd["decomp2"], axd["decomp3"]])
fig.tight_layout()
# %%
def plot_combined(df_fc: pd.DataFrame, df_raw: pd.DataFrame, day_ahead: int = 1,
                  start_date: str = "2025-01-01", end_date: str = "2025-06-30",
                  save_fig: bool = True, img_path: str = rconf.IMG_PATH,
                  img_name: str = "combined", chapter: str = "05_EXPLRT", label_pos: str = "bottom") -> tuple:

    from matplotlib import gridspec

    # fig, axd = plt.subplot_mosaic(
    #     [["decomp0", "ci"],
    #      ["decomp1", "ci"],
    #      ["decomp2", "diff"],
    #      ["decomp3", "diff"]],
    #     figsize=(24, 12),
    #     # constrained_layout=True,
    #     gridspec_kw={"width_ratios": [1, 2]}  # adjust as needed
    # )
    fig = plt.figure(figsize=(26, 14))
    gs_outer = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 2], wspace=0.08)

    # Left: 4 tightly spaced rows
    gs_left = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs_outer[0], hspace=0.20)
    decomp_axes = [fig.add_subplot(gs_left[i]) for i in range(4)]

    # Right: 2 rows with more spacing
    gs_right = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_outer[1], hspace=0.25)
    ax_ci   = fig.add_subplot(gs_right[0])
    ax_diff = fig.add_subplot(gs_right[1])

    axd = {"decomp0": decomp_axes[0], 
           "decomp1": decomp_axes[1],
           "decomp2": decomp_axes[2], 
           "decomp3": decomp_axes[3],
           "ci": ax_ci, 
           "diff": ax_diff}

    # --- Decomposition (left column) ---
    result = seasonal_decompose(df_raw.loc["2025-01-01":"2025-06-30", STD_COL], model='additive', period=7)
    decomp_axes = [axd["decomp0"], axd["decomp1"], axd["decomp2"], axd["decomp3"]]

    decomp_axes[0].plot(result.observed, lw=1.5)
    decomp_axes[1].plot(result.trend, lw=1.5)
    decomp_axes[2].plot(result.seasonal, lw=1.5)
    decomp_axes[3].scatter(result.resid.index, result.resid, s=10)

    for ax, title in zip(decomp_axes, ['Observed', 'Trend', 'Seasonal', 'Residual']):
        ax.set_ylabel(title)
        ax.set_xlim(pd.to_datetime(start_date), pd.to_datetime(end_date))
    decomp_axes[-1].set_xlabel('Date')
    decomp_axes[0].set_title("Seasonal decomposition", pad=20)


    # --- CI plot (top right) ---
    model_name = df_fc["model"][0]
    model_id = df_fc["id"][0]
    end_fc = df_fc[df_fc["day"] == day_ahead].index.max()
    df = df_fc.sort_index().query("day == @day_ahead").loc[rconf.START_DATE:end_fc]

    ax_ci = axd["ci"]
    ax_ci.plot(df["Actual"], label="Actual", lw=2, color=(0.1, 0.1, 0.1))
    ax_ci.plot(df["Mean"], label="Forecast", lw=2, color=rconf.mmap[model_name]["col"])
    ax_ci.plot(df["Upper"], color=rconf.mmap[model_name]["col"], lw=0.15, alpha=0.5)
    ax_ci.plot(df["Lower"], color=rconf.mmap[model_name]["col"], lw=0.15, alpha=0.5)
    ax_ci.fill_between(df.index, df["Lower"], df["Upper"], label="95% CI",
                       color=rconf.mmap[model_name]["col"], alpha=0.2)
    ax_ci.set_ylabel("EC transfused")
    ax_ci.set_ylim(ymin=0, ymax=160)
    ax_ci.legend(loc="lower right", ncol=4, frameon=False, columnspacing=1.2)
    ax_ci.set_title(textwrap.fill(
        f"Day {day_ahead} forecast with CI for {rconf.mmap[model_name]['name']} (ID: {model_id})",
        width=rconf.LINEBREAK), pad=20)


    # --- Diff bars (bottom right) ---
    bottom = np.minimum(df["Actual"], df["Mean"])
    height = np.abs(df["Actual"] - df["Mean"])
    colors = np.where(df["Actual"] > df["Mean"], "violet", "lightblue")

    ax_diff = axd["diff"]
    ax_diff.bar(df.index, height, bottom=bottom, color=colors,
                width=1, edgecolor="white", linewidth=0.5)
    ax_diff.plot(df["Actual"], label="Actual", lw=1, color=(0.1, 0.1, 0.1), alpha=0.5)
    ax_diff.set_xlabel("Date")
    ax_diff.set_ylabel("EC transfused")
    ax_diff.set_ylim(ymin=0, ymax=160)

    from matplotlib.patches import Patch
    # from matplotlib.lines import Line2D

    # legend_elements = [
    #     Line2D([0], [0], color=(0.1, 0.1, 0.1), lw=1, alpha=0.5, label="Actual"),
    #     Patch(facecolor="violet", alpha=0.7, label="Underprediction"),
    #     Patch(facecolor="lightblue", alpha=0.7, label="Overprediction"),
    # ]
    # ax_diff.legend(handles=legend_elements, loc="lower right", ncol=3, frameon=False, columnspacing=0.8)

    leg = ax_diff.legend(
        handles=[ax_diff.get_lines()[0],
                 Patch(facecolor="violet", alpha=0.7, label="Underprediction"),
                 Patch(facecolor="lightblue", alpha=0.7, label="Overprediction")],
        loc="lower right", ncol=3, frameon=False, columnspacing=1.2)
    # leg.set_clip_on(True)
    # ax_diff.set_title(textwrap.fill(
    #     f"Day {day_ahead} forecast errors for {rconf.mmap[model_name]['name']} (ID: {model_id})",
    #     width=60), pad=20)#rconf.LINEBREAK))
    ax_diff.text(0.5, 0.95,
                textwrap.fill(f"Day {day_ahead} forecast errors for {rconf.mmap[model_name]['name']} (ID: {model_id})", width=60),
                transform=ax_diff.transAxes,
                ha="center", va="bottom",
                fontsize=plt.rcParams["axes.titlesize"],
                fontweight=plt.rcParams["axes.titleweight"])

    # --- Panel labels (a), (b), (c) ---
    # label_configs = [
    #     (decomp_axes[-1], "(a)", -50),  # more offset, has xlabel "Date"
    #     (ax_ci,           "(b)", -15),  # less offset, no xlabel
    #     (ax_diff,         "(c)", -50),  # more offset, has xlabel "Date"
    # ]
    # if label_pos == "top_left":
    #     label_configs[0] = (decomp_axes[0], "(a)", -50)


    # for ax, label, y_offset in label_configs:
    #     if label_pos == "bottom":
    #         ax.annotate(label,
    #                     xy=(0.5, 0),
    #                     xycoords="axes fraction",
    #                     xytext=(0, y_offset),
    #                     textcoords="offset points",
    #                     ha="center", va="top",
    #                     fontsize=20)#, fontweight="bold")
    #     elif label_pos == "top_left":
    #         ax.annotate(label,
    #                     xy=(0, 1),
    #                     xycoords="axes fraction",
    #                     xytext=(15, -5),
    #                     textcoords="offset points",
    #                     ha="left", va="top",
    #                     fontsize=20)#, fontweight="bold")
    label_configs = [
        (decomp_axes[-1], "(a)", -50),
        (ax_ci,           "(b)", None),  # always inside, avoids layout interference
        (ax_diff,         "(c)", -50),
    ]
    if label_pos == "top_left":
        label_configs[0] = (decomp_axes[0], "(a)", -50)


    for ax, label, y_offset in label_configs:
        if label_pos == "bottom":
            ax.annotate(label,
                        xy=(0.5, 0), xycoords="axes fraction",
                        xytext=(0, y_offset), textcoords="offset points",
                        ha="center", va="top", fontsize=20)
        elif label_pos == "top_left":
            ax.annotate(label,
                        xy=(0, 1), xycoords="axes fraction",
                        xytext=(5, -5), textcoords="offset points",
                        ha="left", va="top", fontsize=20)

    fig.get_layout_engine().set(
        w_pad=0.35,    # horizontal padding between columns (in inches)
        h_pad=0.35,   # vertical padding between rows (in inches)
        # hspace=0.16,  # additional fractional vertical space between rows
        # wspace=0.08,  # additional fractional horizontal space between columns
    )

    if save_fig:
        eval.save_plot(fig, img_name, model_name, img_path)

    return fig, axd

fig, axd = plot_combined(best_res_fc, df, save_fig=False, img_path=SAVE_PATH, label_pos="top_left")
# fig.get_layout_engine().set(hspace=0.08, h_pad=0.5)
# %%
