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
import textwrap

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acovf

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

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
from src import result_evaluation_config as rconf
from src import result_evaluation as eval

import pandas as pd
plt.style.use('seaborn-v0_8-pastel')


#%%
#--------------------------------------------------------------------------------
# MARK: LOAD DATA
# (raw, cleaned, transformed)
#----------------------------------------------------------------------------------
importlib.reload(clean)
importlib.reload(config)
importlib.reload(transform)
START_DATE = "2020-07-03"
END_DATE = "2025-07-03"
STD_COL = "use_transfused"
df_raw = load.load_data(path=config_cleaning.RAW_DATA_PATH)
df_clean = clean.clean_data(df_raw, new_file_path=config_cleaning.CLEANED_DATA_PATH)
df_processed = transform.transform_data(df_clean, new_file_path=config_cleaning.TRANSFORMED_DATA_PATH)
df = data_model.Data(data=df_processed)
df = df[START_DATE:]


def save_plot(fig, name: str, path: str, chapter="05_EXPLRT")->None:
    print(f"Saved plot to {path}/{chapter}_{name}.png")
    fig.savefig(f"{path}/{chapter}_{name}.png")


#%%
#----------------------------------------------------------------------------------
# MARK: Comparison Model
#----------------------------------------------------------------------------------


m_comp = model.ModelComparison(df)
m_comp.set_parameters(col="use_transfused",
                      single_value=100,
                      forecast_window=14,
                      start_date=START_DATE,
                      end_date=END_DATE)
m_comp.dir_name = "Comparison"
m_comp.file_path = "./results/"+m_comp.dir_name
m_comp.model_run()
m_comp.save_results()

m_comp.predictions
m_comp.forecast_errors


# Tests:
# - Augmented dickey fuller test (stationarity) <- df
# - autocovariance <- df[use_transfused] -> NICHT MACHEN, too much!
# - 

# Plots
# - (P)ACF <- df[use_transfused]


#%%
#%%
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


# LATEX TABLE: iterate over COMPARISON rows
comp_model_latex = (
    m_comp.forecast_errors
    .sort_values("RMSE")
    .rename_axis("Type", axis=0)
    .loc[:, ["RMSE", "MAE", "ME", "MAPE", "MaxError"]]
    .assign(MAPE = lambda c: c["MAPE"] * 100 )
    .rename(columns=lambda c: c.replace("_", r"\_")) #escape underscore!
    .rename(index=lambda c: c.replace("_", r" ").capitalize()) #escape underscore!
    .style
    .format(
        {"MAPE": "{:.3f}"},
        precision=2)
    .to_latex(
        hrules=True
        # captions need to be put inplace inside latex, so this can generate only the \begin[tabular]
        # part and be used within \begin[table]\centering\input
    )
)
# #inject multiline header with "model"
# comp_model_latex = comp_model_latex.replace(
#     "\\toprule",  "\\multicolumn{6}{c}{\\textbf{" + model_name.capitalize() + "}} \\\\\\midrule"
# )
with open(f"{rconf.TBL_PATH}/05_COMPARISON_tbl_overview.txt", "w") as f:
    f.write(comp_model_latex)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


# LATEX TABLE: Counts for days with over/underprediction, Max overprediction, max underprediction
COMP_fc_SV = m_comp.predictions[["Actual", "single_value"]]
COMP_fc_Naive = m_comp.predictions[["Actual", "naive"]]
COMP_fc_Mean = m_comp.predictions[["Actual", "mean"]]
COMP_fc_SNaive = m_comp.predictions[["Actual", "seasonal_naive"]]

COMP_fc_all = [(COMP_fc_SV, "single_value"), (COMP_fc_Naive, "naive"), (COMP_fc_Mean, "mean"), (COMP_fc_SNaive, "seasonal_naive")]

COMP_over_underpred_counts = []
for i, (model_data, COMP_name) in enumerate(COMP_fc_all):
    model_data = (model_data
                  .loc[START_DATE:END_DATE]
                  .assign(day=1) #dummy, needed for get_overpred...()
                  .assign(model=COMP_name) #dummy, needed for get_overpred...()
                  .assign(id="") #dummy, needed for get_overpred...()
                  .assign(Difference=lambda x: model_data["Actual"] - model_data[COMP_name]) #is wrong way around, but same as actual models!
                  .rename(columns={COMP_name:"Mean"})
                  )
    COMP_over_underpred_counts.append(eval.get_overprediction_underprediction_days(model_data, day_ahead=1))
    
COMP_over_underpred_counts_df = (pd.concat(COMP_over_underpred_counts, axis=1)
                              .rename(columns=lambda c: c.replace("_", r" ").capitalize()) #escape underscore!
)
eval.make_latex_table_over_underprediction_days(COMP_over_underpred_counts_df, tbl_name="COMP_table_over_underprediction")


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


# PLOT: 4 subplots of time series for each comparison model
def plot_all_COMP_forecasts(df, start_date: str="2025-01-01", end_date: str=None, 
                     save_fig=True, img_path: str=rconf.IMG_PATH, img_name: str="COMP_all_ts_overview")->None:
    #Plots time series, with all 4 comparison models forecasts on individual subplots.
    # Takes df with Actual value and each comparison models forecast in separate column.

    if not end_date:
        end_date = df.index.max()

    df = df[start_date:end_date]

    #Plotting
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True, sharex=True, sharey=True)
    ax = ax.flatten()
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, col in enumerate(df.columns[df.columns != "Actual"]):
        actual_label = "Actual" if i == 0 else None
        ax[i].plot(df["Actual"], label=actual_label, lw=1.5, color=(0.1, 0.1, 0.1))
        ax[i].plot(df[col], label=col.replace("_", " ").capitalize(), color=colors[i], lw=1.5) #label=f"Day {day}",  cmap(day/n_days)
        ax[i].set_ylim(ymin=0)
        ax[i].set_title(col.replace("_", " ").capitalize())
        #handles, labels = ax[0].get_legend_handles_labels()


    legend = fig.legend(#handles, labels,
        loc="lower center",
        frameon=False,
        ncol=5,
        bbox_to_anchor=(0.5, -0.05)
    )
    #thicker legend lines
    for line in legend.get_lines():
        line.set_linewidth(3)

    fig.suptitle(f"Forecasts of all comparison models")
    fig.supxlabel("Date")
    fig.supylabel("EC transfused")
    if save_fig:
        save_plot(fig, img_name, img_path)

plot_all_COMP_forecasts(m_comp.predictions, start_date="2024-01-01", end_date="2024-03-31")


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 





#%%----------------------------------------------
# Cleaned Data

# Time series of selected wards, to show them dis/re-appearing
def plot_wards_timeseries(df: pd.DataFrame=df_clean)->None:
    #Only do if i have time
    pass


def plot_whole_series_weekends(df: pd.DataFrame, name: str="whole_series", path: str=rconf.IMG_PATH, chapter="05_EXPLRT")->None:
    """Plot 3 subplots above each other:
    One showing use_transfused+discarded+expired for 2005-25
    One for the data period used 2020-25
    One for like a year, but highlight the weekends+holidays


    Args:
        df (pd.DataFrame): processed df
    """
    fig, ax = plt.subplots(3, 1, figsize=(12,18))

    ax[0].plot(df_processed["use_transfused"], label="transfused", lw=0.1)
    ax[0].plot(df_processed["use_discarded"], color="orange", label="discarded", lw=0.1)
    ax[0].plot(df_processed["use_expired"], color="red", label="expired", lw=0.1)
    ax[0].set_title("Whole collected data")


    df["waste"] = df["use_discarded"] + df["use_expired"]
    ax[1].plot(df.loc[rconf.PIPE_START:rconf.PIPE_END, "use_transfused"], label="transfused", lw=0.5)
    ax[1].plot(df.loc[rconf.PIPE_START:rconf.PIPE_END, "use_discarded"], color="orange", label="discarded", lw=0.5)
    ax[1].plot(df.loc[rconf.PIPE_START:rconf.PIPE_END, "use_expired"], color="red", label="expired", lw=0.5)
    # ax[1].plot(df.loc[rconf.PIPE_START:rconf.PIPE_END, "waste"], label="expired+discarded", lw=0.3)
    ax[1].set_title("Used time period")

    #Slice to one year:
    df = df["2024-01-01":"2024-05-31"]
    ax[2].bar(x=df.index, height=df["use_transfused"], label="transfused", width=1)
    ax[2].bar(x=df.index, height=df["use_discarded"], label="discarded", color="orange", width=1)
    ax[2].bar(x=df.index, height=df["use_expired"], label="expired", color="red", width=1, bottom=df["use_discarded"])#stack onto discarded

    # ax[2].plot(df.loc[rconf.PIPE_START:rconf.PIPE_END, "use_transfused"], label="transfused", lw=0.5)
    # ax[2].plot(df.loc[rconf.PIPE_START:rconf.PIPE_END, "use_discarded"], label="discarded", lw=0.5)
    # ax[2].plot(df.loc[rconf.PIPE_START:rconf.PIPE_END, "use_expired"], label="expired", lw=0.5)
    # ax[2].plot(df.loc[rconf.PIPE_START:rconf.PIPE_END, "waste"], label="expired", lw=0.5)
    ax[2].set_title("Highlighting weekends and public holidays for 2024")
    
    #highlight non-working days:
    for day in df[~df["is_workday"]].index:
        ax[2].axvspan(day - pd.Timedelta(hours=12), day + pd.Timedelta(hours=12), alpha=0.1)


    fig.suptitle("Daily aggregated time series")
    fig.supxlabel('date', y=0.06)
    fig.supylabel('Use (Units EC)', x=0.05)
    fig.subplots_adjust(bottom=0.1, top=0.95, hspace=0.3)

    #unified legend:
    handles, labels = [], []
    for a in ax.flatten():
        h, l = a.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                labels.append(li)
                handles.append(hi)

    leg = fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, 0.04), frameon=False)
    #set legend lines thicker
    for line in leg.get_lines():
        line.set_linewidth(2.0)
    # fig.legend()

    save_plot(fig, name=name, path=path, chapter=chapter)


plot_whole_series_weekends(df, name="whole_series", path=rconf.IMG_PATH, chapter="05_EXPLRT")



#%%
# ACF, PACF:
def plot_ACF_PACF(df, col: str=STD_COL, start_date: str=START_DATE, end_date: str=END_DATE, acf: bool=True, name: str="period2020_2025", path: str=rconf.IMG_PATH, chapter="05_EXPLRT")->None:
    #Plot acf if 'acf' is true, otherwise plot pacf.
    
    fig, ax = plt.subplots()

    plot_fct = plot_acf if acf else plot_pacf
    plot_title = "ACF" if acf else "PACF"
    name = plot_title + name
    plot_fct(df.loc[rconf.PIPE_START:rconf.PIPE_END, col], ax=ax)
    
    ax.set_title(f"{plot_title} for {col}, between ")
    plt.show()
    save_plot(fig, name=name, path=path, chapter=chapter)


plot_ACF_PACF(df, acf=True)
plot_ACF_PACF(df, acf=False)

#%%
def plot_decomposition(df: pd.DataFrame, col: str=STD_COL, start_date: str=START_DATE, end_date: str=END_DATE, 
                       model: str='additive', period: int=7, name: str="decomposition", path: str=rconf.IMG_PATH, 
                       chapter="05_EXPLRT")->None:
    result = seasonal_decompose(df.loc[start_date:end_date, col], model=model, period=period)
    # res = {
    #     "Raw": result.observed,
    #     "Trend": result.trend,
    #     "Seasonal": result.seasonal,
    #     "Residuals": result.resid
    #     }

    # Plotting:
    # fig, ax = plt.subplots(4, 1, figsize=(12, 12))
    
    # result.observed.plot(ax=ax[0], linewidth=0.2)
    # result.trend.plot(ax=ax[1], linewidth=0.2)
    # result.seasonal.plot(ax=ax[2], linewidth=0.2)
    # result.resid.plot(ax=ax[3], linewidth=0.2)

    # ax[0].set_ylabel("Observed")
    # ax[1].set_ylabel("Trend")
    # ax[2].set_ylabel("Seasonal")
    # ax[3].set_ylabel("Residual")

    #fig = result.plot()



    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(result.observed, lw=0.5)
    axes[1].plot(result.trend, lw=0.5)
    axes[2].plot(result.seasonal, lw=0.5)
    axes[3].scatter(result.resid.index, result.resid, s=2)#, alpha=0.5)

    for ax, title in zip(axes, ['Observed', 'Trend', 'Seasonal', 'Residual']):
        ax.set_ylabel(title)

    fig.axes[-1].set_xlabel('date')
    fig.suptitle(f"Seasonal decomposition ({start_date} to {end_date})")
    fig.tight_layout()

    save_plot(fig, name=name, path=path, chapter=chapter)


plot_decomposition(df, start_date=rconf.SUBSET_START, end_date=rconf.SUBSET_END)




#%%
#









#%%
#%%
# Weekly seasonality: Line plot over one week
importlib.reload(rconf)
#Copied from viz.py:
def plot_seasonal(data, plot_type: str, col_name = "use_transfused",
                  save_fig=True, img_path: str=rconf.IMG_PATH, img_name: str="EXPLRTN_seasonal")->None: 
    #New version, copied from data_model.plot_seasonal:


    accepted_types = ["daily", "weekly"]
    if plot_type not in accepted_types:
        raise ValueError("'plot_type' must be 'daily' or 'weekly'")
    
    series = data[col_name].to_frame()


    #Get data depending on 'plot_type'
    # Days in a week
    if plot_type == 'daily':
        #Set plotting & naming values:      
        x = 'day_of_week'
        ref_frame = 'week' #comparison period; name of column
        ref_frame_str = 'week_str' #comparison period string;  name of column
        xlabel = 'Day of week'
        xticks_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        title = 'Daily'
        print(type(series))
        print(series.head())
        #Resample daily:
        df = series.resample("D").sum()
        print(df)
        #Add new columns:
        df[x] = df.index.dayofweek
        df[ref_frame] = df.index.isocalendar().week
        df[ref_frame_str] = df[ref_frame].astype(str) #need string for 'hue'


    #Weeks in year
    elif plot_type == 'weekly':
        #Set plotting & naming values:
        x = 'week_of_year'
        ref_frame = 'year' #comparison period; name of column
        ref_frame_str = 'year_str' #comparison period string;  name of column
        xlabel = 'Week number'
        xticks_labels = [str(week) for week in range(1,53)]
        title = 'Weekly'

        #Resample weekly:
        df = series.resample('W').sum()
        #Add new columns:
        df[x] = df.index.isocalendar().week
        df[ref_frame] = df.index.year
        df[ref_frame_str] = df[ref_frame].astype('str')
    # NOTE: could add daily in year, daily in month

    #Settings for plot:
    num_of_lines = df[ref_frame_str].nunique()
    color_palette = sns.color_palette("mako", n_colors=num_of_lines)
    color_palette_reversed = color_palette[::-1]

    #Plotting:
    fig, ax = plt.subplots()
    ax = sns.lineplot(x=x, y=col_name, data=df, hue=ref_frame_str, errorbar=('ci', False), palette=color_palette_reversed, linewidth=0.75)
    axtitle = f"{title} seasonality plot for {col_name} ({df.index.min().strftime(rconf.DATE_FRMT)} - {df.index.max().strftime(rconf.DATE_FRMT)})"
    ax.set_title(textwrap.fill(axtitle, width=40))
    ax.set_xlabel(xlabel)
    ax.set_ylabel('value')
    ax.set_xticks(ticks=range(len(xticks_labels)), labels=xticks_labels)
    ax.legend(title=plot_type, loc='upper right', bbox_to_anchor=(1, 1))

    if plot_type == 'daily':
        ax.legend([],[], frameon=False)


    #if more than 12 xticks, show only every third label
    if len(ax.get_xticklabels()) > 12:
        for i, label in enumerate(ax.xaxis.get_major_ticks()):
            if i % 3 != 0:
                label.set_visible(False)
                

            
    ax.grid(True)
    plt.tight_layout()
    plt.show()
    if save_fig:
        save_plot(fig, img_name, img_path)


plot_seasonal(df_processed[rconf.SUBSET_START:rconf.SUBSET_END], plot_type="daily", col_name="use_transfused", img_name="seasonal_weekly")
# viz.seasonal_plot(df_processed["2024-01-01":"2025-01-01"], plot_type="daily", col_name="use_transfused")







# Processed Data

# Augmented dickey fuller test (stationarity)

adf = adfuller(df["use_transfused"])


# (-6.3030896239505, -> adf test statistic
#  3.3745414799057215e-08, -> pvalue
#  23, -> used lags number
#  1803, -> number of observations (nobs)
#  {'1%': -3.4339820768018106, -> critical values
#   '5%': -2.8631443597478143,
#   '10%': -2.567624108684946},
#  15809.457156008557) -> maximized information criterion, if autolag is not None

# test statistic below critical values (1, 5, 10%) -> fail to reject H0 => has unit root (H0=has unit root)
# pvalue is above critical values -> cannot reject that ther is a unit root
# 

#https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.adfuller.html
# The null hypothesis of the Augmented Dickey-Fuller is that there is a unit root, 
# with the alternative that there is no unit root. If the pvalue is above a critical size, 
# then we cannot reject that there is a unit root.
# The p-values are obtained through regression surface approximation from MacKinnon 1994, 
# but using the updated 2010 tables. If the p-value is close to significant, 
# then the critical values should be used to judge whether to reject the null.




# #%% 
# # MARK: COMPARISON
# #----------------------------------------------------------------------------------
# import importlib
# importlib.reload(model)
# comp = model.ModelComparison(df)

# comp.set_column()
# comp.set_dates_mean()
# comp.set_forecast_window()
# comp.set_single_value()

# comp.print_parameters()

# comp.model_run()

# comp.predictions

# # comp.get_error_values()


# for col in comp.predictions.columns:
#     if col == "use_transfused":
#         plt.plot(comp.predictions.loc["2024-01-01":"2024-07-31", col], linewidth=1.5, label=col)
#     else:
#         plt.plot(comp.predictions.loc["2024-01-01":"2024-07-31", col], linewidth=0.5, label=col)
#     plt.legend()

# comp.stats["id"] = 0000
# comp.stats["model_name"] = "comparison"
# comp.create_result_dir()
# comp.save_results()
























# #%%
# #--------------------------------------------------------------------------------
# # MARK: RAW DATA VIZ
# #----------------------------------------------------------------------------------




# #%%
# #--------------------------------------------------------------------------------
# # MARK: CLEANED DATA VIZ
# #----------------------------------------------------------------------------------
# IMAGES_PATH_EXPLORATION = IMAGE_PATH + "/00-Data-Exploration/"


# #
# #%% Plot frequency counts for unique values in every column
# #TODO: move into viz.py


# for col_name, col in df_clean.items():
#     if col_name in ["EC_ID_O_hash", "EC_ID_I_hash"]:
#         continue
#     print(col.value_counts())
#     col.value_counts()[:40].plot(kind="bar", title=col_name,)
#     plt.savefig(fname=IMAGES_PATH_EXPLORATION+f"01-barcharts-value_counts-{col_name}")
#     plt.show()



# #%%
# # Plot each patient wards transfusion counts (for wards with >500 transfusions)
# importlib.reload(viz)
# ##
# viz.plot_patient_wards(df_clean, n=500, save_figs=False, location=IMAGES_PATH_EXPLORATION, foldername="")






# #%%
# #--------------------------------------------------------------------------------
# # MARK: TRANSFORMED DATA VIZ
# # /PROCESSING
# #----------------------------------------------------------------------------------

# #





# #%%
# # Plot seasonalities daily & weekly of processed df

# importlib.reload(viz)

# BG_RH_cols = ['EC_BG_RH_0_NB']#,
#     #    'EC_BG_RH_0_Rh_negative', 'EC_BG_RH_0_Rh_positive', 'EC_BG_RH_A_NB',
#     #    'EC_BG_RH_A_Rh_negative', 'EC_BG_RH_A_Rh_positive', 'EC_BG_RH_AB_NB',
#     #    'EC_BG_RH_AB_Rh_negative', 'EC_BG_RH_AB_Rh_positive', 'EC_BG_RH_B_NB',
#     #    'EC_BG_RH_B_Rh_negative', 'EC_BG_RH_B_Rh_positive', 'PAT_BG_RH_0_NB',
#     #    'PAT_BG_RH_0_Rh_negative', 'PAT_BG_RH_0_Rh_positive', 'PAT_BG_RH_A_NB',
#     #    'PAT_BG_RH_A_Rh_negative', 'PAT_BG_RH_A_Rh_positive', 'PAT_BG_RH_AB_NB',
#     #    'PAT_BG_RH_AB_Rh_negative', 'PAT_BG_RH_AB_Rh_positive',
#     #    'PAT_BG_RH_B_NB', 'PAT_BG_RH_B_Rh_negative', 'PAT_BG_RH_B_Rh_positive',
#     #    'PAT_BG_RH_NB_NB', 'PAT_BG_RH_NB_Rh_negative',
#     #    'PAT_BG_RH_NB_Rh_positive', 'PAT_BG_RH_Not_applicable']
# for bg_rh in BG_RH_cols:
#     # viz.seasonal_plot(df_processed, plot_type="weekly", col_name=bg_rh)
#     viz.seasonal_plot(df_processed, plot_type="daily", col_name=bg_rh)



# ward_cols = ['ward_AN', 'ward_CH', 'ward_I1', 'ward_I3', 'ward_Other', 'ward_UC']
# for ward in ward_cols:
#     viz.seasonal_plot(df_processed, plot_type="weekly", col_name=ward)
#     viz.seasonal_plot(df_processed, plot_type="daily", col_name=ward)



# #%%
# #Simple line plot time series

# def plot_timeseries(df: pd.DataFrame, columns: list, title: str):
#     fig, ax = plt.subplots(figsize=(12,6))

#     for col in columns:
#         ax.plot(df.index, df[col], label=col, linewidth=0.25)

#     ax.set_title(title)
#     ax.set_xlabel("Date")
#     ax.legend()
#     plt.tight_layout()
#     plt.show

# plot_timeseries(df, columns=["use_transfused", "use_expired", "use_discarded"], title="Qucik plot")



# #%%





# #%%--------------------------------------------------------------------------------
# # MARK: DATA VIZ 
# # (EXPLORATION)
# #----------------------------------------------------------------------------------
# IMAGES_PATH_EXPLORATION = IMAGE_PATH + "/00-Data-Exploration/"
# START_DATE_EXPLORATION = "2012-01-01" #"2020-01-01"
# END_DATE_EXPLORATION = "2016-12-31" #"2020-01-01"
# PRE_COVID_START = "2018-01-01"
# PRE_COVID_END = "2020-01-01"

# #%%
# #TODO: save vizualisations to csv
# importlib.reload(data_model)

# df = data_model.Data(data=df_processed)

# #%%


# df["2023-01-01":"2023-12-31"].plot_seasonal(plot_type='daily', col_name='use_transfused')#, fig_location=IMAGES_PATH_EXPLORATION)
# df[START_DATE_EXPLORATION:].plot_seasonal(plot_type='weekly', col_name='use_transfused')


# ##%%
# #Boxplots for year, month, weekly and day of week
# df[START_DATE_EXPLORATION:END_DATE_EXPLORATION].plot_boxplots(col_name='use_transfused')
# df[START_DATE_EXPLORATION:END_DATE_EXPLORATION].plot_boxplots(col_name='use_discarded', max_y = 65)
# df[START_DATE_EXPLORATION:END_DATE_EXPLORATION].plot_boxplots(col_name='use_expired')
# df[START_DATE_EXPLORATION:END_DATE_EXPLORATION].plot_seasonal_subseries(col_name='use_transfused') #NOTE: i think it works, but not enough dummy data.
# #TODO: check if seasonal subseries plot works with multi-year data


# ##%%
# #Decompose
# df[PRE_COVID_START:PRE_COVID_END].decompose_one(col_name='use_transfused')


# #df.decompose_all("use_transfused")

# #%%
# #Glaub das verwende ich nicht
# # mulitple decomposition (daily + weekly)
# df[PRE_COVID_START:PRE_COVID_END].multiple_decompose(col_name="use_transfused", periods=[7, 365])
# df[pd.to_datetime("2024-01-01"):pd.to_datetime("2024-12-31")].plot_daily_heatmap(col_name='use_transfused')


# #%% Visualize counts for all plots (as of now only for those starting with 901AN) on top of each other, so that
# # its visible, where naming of one ward starts/ends 


# wards = df_clean["PAT_WARD"].unique()
# fig, ax = plt.subplots(44,1, figsize=(6, 24))
# fig.set_dpi(300)
# fig.set_linewidth(5)
# sel_wards = []
# i = 0
# for ward in wards:
#     if str(ward).startswith("901AN"):
#         sel_wards.append(ward)
#         ax[i].plot(df[f"PAT_WARD_{ward}"], label=ward, linewidth=0.15)
#         # ax[i].set_title(str(ward))
#         ax[i].legend(loc="upper right", prop={"size":6}, frameon=False, framealpha=0.5)
#         ax[i].set_yticklabels([])
#         i = i+1
        
# print(len(sel_wards))
# print(sel_wards)
# plt.savefig(fname="/".join([IMAGES_PATH_EXPLORATION + "show_inconsistency_wards"]))
# plt.show()

# #%%
# # Get unique Letter-combinations (i guess top-wards) from all wards:

# import re
# unique_wards = df_clean["PAT_WARD"].unique()

# pattern = re.compile(r'\d+([A-Za-z]+)\d+')  # number + letters + number

# ward_results = set()  # only unique strings

# for s in unique_wards:
#     match = pattern.search(str(s))
#     if match:
#         ward_results.add(match.group(1))  # =letters

# print(unique_wards)
# print(len(unique_wards))
# print(ward_results)
# print(len(ward_results))

# #%%--------------------------------------------------------------------------------
# # MARK: STATIONARITY / Statistics
# # Check for Stat./Make stationarys
# #----------------------------------------------------------------------------------
# #TODOLIST:
# # 1. OG Data
# #    Visual & statistical check: const mean, variance, no seasonal component 
# #    --> then its stationary
# # 1.1 Visual assessment
# #    - Time series
# #    - ACF, PACF
# # 1.2 Statistical test
# #    - Unit root test
# # 

# #%% 
# # Auto_arima:
# from pmdarima import auto_arima


# autoarima_arima = auto_arima(y=df.loc["2020-01-01": ,config.PRED_COLUMN], seasonal=False, trace=True)
# autoarima_sarima = auto_arima(y=df.loc["2020-01-01": ,config.PRED_COLUMN], seasonal=True, m=7, trace=True)
# autoarima_sarimax = auto_arima(
#     y=df.loc["2020-01-01": ,config.PRED_COLUMN], 
#     x=df.loc["2020-01-01": , config.exog_combinations_list[-1]],
#     seasonal=True, 
#     m=7, 
#     trace=True)


# #%%


# from statsmodels.graphics.tsaplots import plot_acf
# from statsmodels.graphics.tsaplots import plot_pacf

# from statsmodels.tsa.stattools import adfuller
# from statsmodels.tsa.stattools import kpss

# #%%
# #Time series plots (acf, pacf etc)
# df.plot_autocorrelation(col_name='use_transfused')
# df.plot_partial_autocorrelation(col_name='use_transfused')

# num_differencing = 2
# i = 1
# df_diff = df_processed.copy()

# df_diff["use_transfused"].plot(lw=0.05)

# plot_acf(df_diff["use_transfused"], title="No differentiation")
# plot_pacf(df_diff["use_transfused"], title="No differentiation")

# #ADF -- Augmented dickey-fuller test
# adf_result_diff = adfuller(df_diff["use_transfused"][1:], autolag="AIC") #default
# adf_result_diff2 = adfuller(df_diff["use_transfused"][1:], autolag="BIC")
# adf_result = adfuller(df["use_transfused"], autolag="AIC") #default
# adf_result2 = adfuller(df["use_transfused"], autolag="BIC")
# adfuller(df["use_transfused"])
# plt.plot(df["use_transfused"])
# plt.plot(df_diff["use_transfused"])

# #i think i can reject the H0 in both cases (df and df_diff), for df test staticstics is -11.49, far below 
# #critical values of 1%, 5%, 10%, same  for df_diff with -21.07. therefore i reject H0 (data is 
# # non stationary, has a unit root). H1 is true, data is stationary has no unit root. so no differencing 
# # needed actually?

# #Different params for autolag:
# # no real differences in result, so doestn matter. (in gerneral AIC better for large datasets)


# #KPSS
# kpss_result_diff = kpss(df_diff["use_transfused"][1:])
# kpss_result = kpss(df["use_transfused"])
# kpss_result = kpss(df["use_transfused"])

# #kpss is one-sided test, if test statistics is greater than critical value, then H0 is rejected. 
# # H0 for KPSS is that ts is stationary, H1 ts is NOT stationary.
# # for kopss_result_diff, p-val = 0.1, test stat. = 0.0104, 0.0104 is not greater than 0.347 (10%), so H0 cant
# # be rejected, ts seems to be stationary.
# # for kpss_result, with p value = 0.01, test statistic = 1.258 is greater than 0.739 (1%), so H0 is rejected,
# # ts is not stationary. 

# # So for differentiated data (df_diff), both ADF + KPSS suggest stationarity.
# # For df (original data), ADF suggests stationarity, KPSS suggests non-stationarity.
# # with this guide: https://www.statsmodels.org/dev/examples/notebooks/generated/stationarity_detrending_adf_kpss.html
# # i should differntiate the original series (df), then check this for stationarity. i already did that, and
# # both tests suggest that data is stationary. so differencing was right call.

# print(adf_result[0])
# print(f"No differencing: \nadf: {adf_result[0]}\np-value: {adf_result[1]}\ncritical vals: {adf_result[4]}")
# while i <= num_differencing:
#     df_diff["use_transfused"] = df_diff["use_transfused"].diff()

#     fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
#     df_diff["use_transfused"].plot(ax=ax1, lw=0.05)
#     plot_acf(df_diff["use_transfused"].dropna(), ax=ax2)
#     plot_pacf(df_diff["use_transfused"].dropna(), ax=ax3)
#     fig.suptitle(f"differentiated {i}x")
#     fig.show()

#     adf_result = adfuller(df_diff["use_transfused"].dropna())
#     print(f"No differencing: \nadf: {adf_result[0]}\np-value: {adf_result[1]}\ncritical vals: {adf_result[4]}")


#     i += 1

# #seasonal differencing = subtracting value of previous season 
# # (i'll try week, so period = 7 (7 rows before in dataset))
# num_differencing = 3
# period = 7#365
# i = 1
# df_diff = df_processed.copy()

# df_diff["use_transfused"].plot(lw=0.05)
# plot_acf(df_diff["use_transfused"], title="No differentiation")
# plot_pacf(df_diff["use_transfused"], title="No differentiation")

# while i <= num_differencing:
#     df_diff["use_transfused"] = df_diff["use_transfused"].diff(periods=period)

#     fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
#     df_diff["use_transfused"].plot(ax=ax1, lw=0.05)
#     plot_acf(df_diff["use_transfused"].dropna(), ax=ax2)
#     plot_pacf(df_diff["use_transfused"].dropna(), ax=ax3)
#     fig.suptitle(f"differentiated {i}x")
#     fig.show()
#     i += 1






# %%
