#%% MARK: libs etc

import pandas as pd
import numpy as np
from numpy import nan
from time import time
from datetime import datetime
from matplotlib import pyplot as plt
import seaborn as sns
import os



from src import clean
from src import config
from src import data_model
from src import load
from src import model
from src import transform
from src import viz


from itertools import product
from sklearn.model_selection import ParameterGrid
import multiprocessing
import traceback
import gc



#For developing purposes:
import importlib
print(load.__file__)
print(clean.__file__)

IMAGE_PATH = "plots/2025_10_10-Plots_for_Meeting/"
RUN_ALL = False

importlib.reload(data_model)
df_raw = load.load_data(path="data/01_raw/blood-data_complete_2025-07-16.tsv")
df_clean = clean.clean_data(df_raw)
df_processed = transform.transform_data(df_clean)
df = data_model.Data(data=df_processed)

#%%--------------------------------------------------------------------------------
# MARK: INPUT EXISTING
#----------------------------------------------------------------------------------
PATH_RAW = "data/01_raw/blood-data_complete_2025-07-16.tsv"

#TODO: make backwards file loading: try processed file, if not exists try cleaned, if not exists try raw
#TODO: Alternative for now -- just load transformed file, because i know it exists.


#%%--------------------------------------------------------------------------------
# MARK: INPUT
#----------------------------------------------------------------------------------

# Read Data
df_raw = load.load_data(path="data/01_raw/blood-data_complete_2025-07-16.tsv")
# df_raw = load.load_data(path="data/01_raw/testdaten.tsv")

#%%
#TODO: remove
if RUN_ALL == True:
    load.show_info(df=df_raw)
    hidden_cols=["date", "EC_ID_I_hash", "EC_ID_O_hash", "T_ISO", "T_DE_T", "T_US", "T_DE_S", "T_US_T", "T_DE", "T_ISO_T", "T_XL"]
    for col in df_raw.columns:
        if col not in hidden_cols:
            tmp = df_raw[col].astype(str).unique()
            tmp = np.array(sorted(tmp))
            print(f"{col}:\n{tmp}\n")
    print(df_raw.columns)
    #TODO: move back to clean_data
    # df_raw = clean.clean_dates(df_raw) #TODO: remove here, enable again in clean_data()



#%%--------------------------------------------------------------------------------
# MARK: CLEANING 
#----------------------------------------------------------------------------------
#Runs only if no file exists at. If not existing, saves df to new file
#unify dates, columns etc. rename stuff
IMAGES_PATH_EXPLORATION = IMAGE_PATH + "/00-Data-Exploration/"
importlib.reload(clean)
importlib.reload(config)
df_clean = clean.clean_data(df_raw)
# df_clean.sort_index(inplace=True)
# #TODO: remove 5 lines:
# start_date = pd.to_datetime("2018-01-01")
# start_date = pd.to_datetime("2024-12-31")
# mask = (df_clean.index >= "2018-01-01") & (df_clean.index <= "2024-12-31")
#df_clean = df_clean.loc[mask]
#df_clean = df_clean['2018-01-01':'2024-12-31'] #only works on monotonic (==daily aggregated, no duplicate days) df

#%%
#TODO: remove
if RUN_ALL == True:
    #TODO: Check what unique vals are present in df
    clean.check_unique_values(df_clean.drop(["EC_ID_I_hash", "EC_ID_O_hash", "PAT_WARD"], axis=1))


#%%
# Plot frequency counts for unique values in every column
#TODO: move into viz.py

#TODO: remove
if RUN_ALL == True:
    for col_name, col in df_clean.items():
        if col_name in ["EC_ID_O_hash", "EC_ID_I_hash"]:
            continue
        print(col.value_counts())
        col.value_counts()[:40].plot(kind="bar", title=col_name,)
        plt.savefig(fname=IMAGES_PATH_EXPLORATION+f"01-barcharts-value_counts-{col_name}")
        plt.show()


    importlib.reload(viz)
    ##%%
    # Plot each patient wards transfusion counts (for wards with >500 transfusions)
    viz.plot_patient_wards(df_clean, n=500, save_figs=False, location=IMAGES_PATH_EXPLORATION, foldername="")







#%%--------------------------------------------------------------------------------
# MARK: TRANSFORMING
# /PROCESSING
#----------------------------------------------------------------------------------
# make STATIONARY! (if all models need that, otherwise make it a member function)
# splitting in test/training etc. here or as extra step/model step?



# Proces....
#add external data (holidays weather (temp, precipitation), covid/influenca cases)
#NOTE: covid/grippe muss evnetuell imputiert werden da nur wöchentlich
#NOTE: kann gut zeigen, dass wien gleichen verlauf hat wie bundesländer, daher kann ich Ö-weite Daten
# nehmen, falls es keine wien-spezifischen Daten gibt.
importlib.reload(transform)

# make daily aggregations for categorical variables
df_processed = transform.transform_data(df_clean)


#%% #Plot seasonalities daily & weekly of processed df
#TODO: remove
if RUN_ALL == True:
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
START_DATE_EXPLORATION = "2020-01-01"
PRE_COVID_START = "2018-01-01"
PRE_COVID_END = "2020-01-01"

#TODO: save vizualisations to csv
importlib.reload(data_model)

df = data_model.Data(data=df_processed)

#%%
#TODO: remove
if RUN_ALL == True:


    #df.print_head()
    df[START_DATE_EXPLORATION:].plot_seasonal(plot_type='daily', col_name='use_transfused', fig_location=IMAGES_PATH_EXPLORATION)
    df[START_DATE_EXPLORATION:].plot_seasonal(plot_type='weekly', col_name='use_transfused')


    ##%%
    #Boxplots
    df[START_DATE_EXPLORATION:].plot_boxplots(col_name='use_transfused')
    df[START_DATE_EXPLORATION:].plot_seasonal_subseries(col_name='use_transfused') #NOTE: i think it works, but not enough dummy data.
    #TODO: check if seasonal subseries plot works with multi-year data


    ##%%
    #Decompose
    df[PRE_COVID_START:PRE_COVID_END].decompose_one(col_name='use_transfused')


    #df.decompose_all("use_transfused")

    # mulitple decomposition (daily + weekly)
    df[PRE_COVID_START:PRE_COVID_END].multiple_decompose(col_name="use_transfused", periods=[7, 365])




    ##%%



    df[pd.to_datetime("2024-01-01"):pd.to_datetime("2024-12-31")].plot_daily_heatmap(col_name='use_transfused')


#%% Visualize counts for all plots (as of now only for those starting with 901AN) on top of each other, so that
# its visible, where naming of one ward starts/ends 
#TODO: remove
if RUN_ALL == True:
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
#TODO: remove
if RUN_ALL == True:
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
#TODO: remove
if RUN_ALL == True:
    from statsmodels.graphics.tsaplots import plot_acf
    from statsmodels.graphics.tsaplots import plot_pacf

    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.stattools import kpss

    ##%%
    #Time series plots (acf, pacf etc)
    df.plot_autocorrelation(col_name='count')
    df.plot_partial_autocorrelation(col_name='count')

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
    period = 365
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




#%%--------------------------------------------------------------------------------
# MARK: MODEL BUILDING
#----------------------------------------------------------------------------------

#TODO: add multiple runs, to test for different parameters
#all of the above could be grouped into sarima.run() (if certain stuff is set up before, like vals for params and split!)
#TODO: add parameter grid, like p = 0-7, q=0, d=1-2 (for example), and then try all combinations of these params.
#TODO: Add simple forecasts: naive, naive seasonal, fixed line (like its now) as forecasts for comparison.
#TODO: add CI to plots? (Only one day stepwise ahead).
#TODO: change color scheme + x axis texts for plot_stepwise_forecast_errors
#TODO: save data to csv
#TODO: load data from csv



#%% 
# MARK: COMPARISON
#----------------------------------------------------------------------------------
importlib.reload(model)
importlib.reload(config) #for DEV_START_DATE

comp = model.ModelComparison(df)

comp.set_column()
comp.set_dates_mean()
comp.set_forecast_window()
comp.set_single_value()

comp.print_parameters()

comp.model_run()

comp.predictions

comp.get_error_values()


for col in comp.predictions.columns:
    if col == "use_transfused":
        plt.plot(comp.predictions.loc["2024-01-01":"2024-07-31", col], linewidth=1.5, label=col)
    else:
        plt.plot(comp.predictions.loc["2024-01-01":"2024-07-31", col], linewidth=0.5, label=col)
    plt.legend()
#%% 
# MARK: ARIMA
#----------------------------------------------------------------------------------

importlib.reload(model)
importlib.reload(config) #for DEV_START_DATE


arima = model.ModelArima(df, id=9998)
# Test runs (it works as expected)
# arima.set_validation_expanding_window(train_percent=0.992, test_len=7, start_date="2022-01-01")
# arima.set_validation_single_split(train_percent=0.75)
arima.set_prediction_column(prediction_column=config.COLUMN)
arima.set_validation_rolling_window(train_percent=0.95, test_len=14, start_date="2024-05-01")#config.DEV_START_DATE) #TODO: change date/remove it

arima.set_model_parameters(7, 1, 1) #7,1,1, #TODO: add hyperparam grid
fc_errors_day_1 = arima.model_run()
#%%
#Try out stepwise error measurements (now only mae):
arima.plot_stepwise(plot_type="forecast") #forecast
arima.plot_stepwise(plot_type="forecast difference") #forecast
arima.plot_stepwise(df=arima.stepwise_forecast_difference, plot_type="difference", comparison=False) #forecast difference
arima.plot_stepwise_forecast_errors()
print(arima.stepwise_forecast_errors)


#%% MARK: GRID SEARCH (ARIMA)
#TODO: ARIMA with grid search:
#dummy var names for now (partly)

#TODO: remove
if RUN_ALL == True:

    arima_gs = model.ModelArima(df)
    arima_gs_result = {}
    grid_params = {
        col: (config.COLUMN),
        p: [0, 7],
        d: [0, 3],
        q: [0, 7],
        "rolling_type" : ["expanding", "rolling"]
    }
    #TODO: convert setted grid min/max/choice to list of possible values
    #idea: min/max always as set, rest as list.
    grid_params_list = convert_gs_to_list(grid_params)


    for count, grid in enumerate(grid_params_list): #suppose this iterates over our min/max or possible values
        params = grid #store parameters

        if grid["rolling_type"] == "expanding":
            arima.set_validation_expanding(df)
        elif grid["rolling_type"] == "rolling":
            arima.set_validation_rolling(df)

        #run model with params:
        arima.set_model_parameters(7, 1, 1) #7,1,1, #TODO: add hyperparam grid
        arima.model_run(col=config.COLUMN)

        #save values
        res_fc = arima.forecastt #store forecast result
        res_og = arima.actual_values #store which actual values where used for calc errors
        res_errors = arima_gs.stepwise_forecast_errors

        #store in dict
        arima_gs_result.append({count: {
                "params": params,
                "forecast":res_fc,
                "actual":res_og,
                "errors":res_errors
            }}
        )







#%%
# MARK: SARIMAX
#----------------------------------------------------------------------------------


importlib.reload(model)
importlib.reload(config) #for DEV_START_DATE


sarima = model.ModelSarimax(df)
# Test runs (it works as expected)
# arima.set_validation_expanding_window(train_percent=0.992, test_len=7, start_date="2022-01-01")
# arima.set_validation_single_split(train_percent=0.75)
sarima.set_validation_rolling_window(train_percent=0.8, test_len=14, start_date=config.DEV_START_DATE) #TODO: change date/remove it
sarima.set_prediction_column("use_transfused")
sarima.set_exogenous_cols(exog_cols=["tlmin", "workday_enc", "holiday_enc", "day_of_week", "day_of_year"])
sarima.set_model_parameters(p=7, d=1, q=1, P=0, D=0, Q=2, m=7) #7,1,1, #TODO: add hyperparam grid
#%%
sarima.model_run()#, exog=["PAT_BG_0", "PAT_BG_A", "PAT_BG_AB", "PAT_BG_B"])
#%%
#Try out stepwise error measurements (now only mae):
sarima.plot_stepwise(plot_type="forecast") #forecast
sarima.plot_stepwise(df=sarima.stepwise_forecast_difference, comparison=False, plot_type="forecast difference") #forecast difference
sarima.plot_stepwise_forecast_errors()
print(sarima.stepwise_forecast_errors)



 
#%%----------------------------------------------------------------------------------
# MARK: LSTM
#----------------------------------------------------------------------------------

importlib.reload(model)
importlib.reload(config) #for DEV_START_DATE


m_lstm = model.ModelLSTM(df)


#TODO: finish grid search
#TODO: implement automatic filling of grid search var (list of dicts with all possible combinations)
#TODO: grid_search_list/params could be a df, where index is a consecutive number and columns are the test params.
# That way, would be easy to assign number to saved csv file.

#TODO: this would set the options/borders/steps for grid searching
exog_cols = ["use_discarded", "use_expired", 'ward_AN', 'ward_CH', 'ward_I1', 'ward_I3', 'ward_Other', 'ward_UC', "workday_enc", "holiday_enc", "day_of_week", "day_of_year", "year", "tlmin", "tlmax"]



#Simple run (for testing, beofre implementing grid search)
m_lstm.set_validation_rolling_window(
    #TODO: store validation_sets as df: index + columns train start/train end/test start/test end
    #TODO: add option to choose days for train and test period.
    train_percent=0.95,#9,#985,#975,
    test_len=7, 
    start_date="2024-01-01"
)

#%%
m_lstm.set_model_parameters(
    inner_window = 128, #365*2, #365,#200,#365, #365*2 #365 to capture at least 1 year, #for training length

    memory_cells=64,#64
    epochs=20,#20
    batch_size=32, #32
    dropout=0.5,
    pi_iterations=10, #100 #how often to run, to calculate prediction intervals
    optimizer="adam",
    loss="mae",
    activation_fct="relu",
    lower_limit=2.5,
    upper_limit=97.5
)
m_lstm.set_exogenous_cols(exog_cols = exog_cols)
m_lstm.set_prediction_column(prediction_column="use_transfused")

m_lstm.print_params()

m_lstm.get_params_df()
#%%
#Run model
m_lstm.model_run()

#%%
#Get error values + plotting



#%% MARK: GridSearch:



#%%

importlib.reload(model)
importlib.reload(config)



#TODO: move to utils.py?
def run_worker(args):
    gc.collect()
    gc.collect()

    ModelClass, params = args
    job_id = params.get("id")

    try:
        model_instance = ModelClass(df, id=job_id)
        model_instance.set_prediction_column(**params)
        if ModelClass.__name__ != "ModelArima":
            model_instance.set_exogenous_cols(**params)
        model_instance.set_validation_rolling_window(**params)
        model_instance.set_model_parameters(**params) 
        
        run_results = model_instance.model_run()
        print(f"Job {job_id} finished in {model_instance.stats['run_duration']}")

        return (job_id, True, ModelClass.__name__, run_results)
    
    except Exception as e:
        print(f"Error in {ModelClass.__name__}: {job_id}: {e}")
        empty_df = pd.DataFrame(
            data={
                "ME": None, 
                "MAE" : None, 
                "MedAE" : None, 
                "MAPE" : None, 
                "MSE" : None, 
                "RMSE" : None, 
                "MaxError" : None,
                "id" : job_id},
            index=["Day_1"]
        )
        traceback.print_exc()
        return (job_id, False, ModelClass.__name__, empty_df) 



global_job_id = 1 #TODO: maybe move to top?
all_jobs = [] #TODO: maybe move to top?

#TODO: probably better in utils.py?
def add_model_grid_to_all_jobs(grid, model):
    #grid is a df, model a specific modle class, like model.ModelArima
    global all_jobs
    global global_job_id
    for g in grid:
        g["id"] = global_job_id
        all_jobs.append( (model, g) )
        global_job_id += 1


# from pmdarima import auto_arima
# autoarima_res = auto_arima(y=df[config.PRED_COLUMN], seasonal=True, m=7, trace=True)


importlib.reload(config)

grid_arima = list(ParameterGrid(config.gs_config_arima))
grid_sarimax = list(ParameterGrid(config.gs_config_sarimax))
grid_lstm = list(ParameterGrid(config.gs_config_lstm))
grid_prophet = list(ParameterGrid(config.gs_config_prophet))




add_model_grid_to_all_jobs(grid_arima, model.ModelArima)
add_model_grid_to_all_jobs(grid_sarimax, model.ModelSarimax)
add_model_grid_to_all_jobs(grid_lstm, model.ModelLSTM)
add_model_grid_to_all_jobs(grid_prophet, model.ModelProphet)


cores = max(1, multiprocessing.cpu_count() - 2)

#%%

with multiprocessing.Pool(cores) as pool:
    result_list_arima = pool.map(run_worker, [job for job in all_jobs[0:5]]) #For testing only a few

valid_results_arima = [res[3] for res in result_list_arima if res is not None]
final_result_df_arima = pd.concat(valid_results_arima).set_index("id")
final_result_df_arima.to_csv("./results/Arima/grid_search_results.csv")
print("Done with arima")

with multiprocessing.Pool(cores) as pool:
    result_list_sarimax = pool.map(run_worker, [job for job in all_jobs[1323:1323+5]]) #all_jobs[0:8])

valid_results_sarimax = [res[3] for res in result_list_sarimax if res is not None]
final_result_df_sarimax = pd.concat(valid_results_sarimax).set_index("id")
final_result_df_sarimax.to_csv("./results/Sarimax/grid_search_results.csv")
print("Done with sarimax")


with multiprocessing.Pool(cores) as pool:
    result_list_lstm = pool.map(run_worker, [job for job in all_jobs[5644:5644+5]]) #all_jobs[0:8])

valid_results_lstm = [res[3] for res in result_list_lstm if res is not None]
final_result_df_lstm = pd.concat(valid_results_lstm).set_index("id")
final_result_df_lstm.to_csv("./results/LSTM/grid_search_results.csv")
print("Done with lstm")


with multiprocessing.Pool(cores) as pool:
    result_list_prophet = pool.map(run_worker, [job for job in all_jobs[6480:6480+5]]) #all_jobs[0:8])

valid_results_prophet = [res[3] for res in result_list_prophet if res is not None]
final_result_df_prophet = pd.concat(valid_results_prophet).set_index("id")
final_result_df_prophet.to_csv("./results/Prophet/grid_search_results.csv")
print("Done with prophet")

print(__name__)

print("done")


#%%
m_prophet = model.ModelProphet(df)
m_prophet.set_validation_rolling_window(**test_worker[1])
m_prophet.set_model_parameters(**test_worker[1])
m_prophet.set_exogenous_cols(**test_worker[1])
m_prophet.set_prediction_column(**test_worker[1])
m_prophet.model_run()


#%%Try with gs_config:

test_worker = all_jobs[0] #arima
test_worker = all_jobs[1323] #Sarimax
test_worker = all_jobs[5644] #lstm
test_worker = all_jobs[12123] #prophet

run_worker(test_worker)

sarima = model.ModelSarimax(df)
# Test runs (it works as expected)
# arima.set_validation_expanding_window(train_percent=0.992, test_len=7, start_date="2022-01-01")
# arima.set_validation_single_split(train_percent=0.75)
sarima.set_validation_rolling_window(train_percent=0.8, test_len=14, start_date=config.DEV_START_DATE) #TODO: change date/remove it
sarima.set_prediction_column("use_transfused")
sarima.set_exogenous_cols(exog_cols=["tlmin", "workday_enc", "holiday_enc", "day_of_week", "day_of_year"])
sarima.set_model_parameters(p=7, d=1, q=1, P=0, D=0, Q=2, m=7) #7,1,1, #TODO: add hyperparam grid
#%%
sarima.model_run()#, exog=["PAT_BG_0", "PAT_BG_A", "PAT_BG_AB", "PAT_BG_B"])


#%%


# #TODO: grid search -- this is what a possible list of dicts could look like (missing exog_cols):
# grid_search_lstm = [{
#     "validation_type" : "rolling",
#     "train_prct" : 0.98,
#     "test_len" : 7,
#     "start_date" : pd.to_datetime("2020-01-01"),
    
#     "memory_cell" : 64,
#     "epochs" : 20,
#     "batch_size" : 32,
#     "batch_size" : 0.5,
#     "pi_iterations" : 100,
#     "optimizer" : "adam",
#     "loss" : "mae"
#     },
#     {
#     "validation_type" : "expanding",
#     "train_prct" : 0.97,    
#     "test_len" : 7,
#     "start_date" : pd.to_datetime("2015-01-01"),
    
#     "memory_cell" : 128,
#     "epochs" : 22,
#     "batch_size" : 36,
#     "batch_size" : 0.25,
#     "pi_iterations" : 120,
#     "optimizer" : "adam",
#     "loss" : "mae"
#     }
# ]


# #TODO: implement grid search for lstm (rought idea):
# for p in grid_search_lstm:
#     if p["validation_type"] == "rolling":
#         lstm_m.set_validation_rolling_window(train_percent=p["train_prct"], test_len=p["test_len"], start_date=p["start_date"])
#     elif p["validation_type"] == "expanding":
#         lstm_m.set_validation_expanding_window(train_percent=p["train_prct"], test_len=p["test_len"], start_date=p["start_date"])

#     lstm_m.set_model_parameters(**p) #should auto-unpack/-match params of p(grid search lstm)
#     lstm_m.model_run()
#     lstm_m.save_all() #TODO: saves model, prediction results (df), params, error values







#%% 
# MARK: PROPHET
#----------------------------------------------------------------------------------


importlib.reload(model)
importlib.reload(config) #for DEV_START_DATE


m_prophet = model.ModelProphet(df)

#TODO: this would set the options/borders/steps for grid searching
exog_cols = ["use_discarded", "use_expired", 'ward_AN', 'ward_CH', 'ward_I1', 'ward_I3', 'ward_Other', 'ward_UC', "workday_enc", "holiday_enc", "day_of_week", "day_of_year", "year", "tlmin", "tlmax"]



#Simple run (for testing, beofre implementing grid search)
m_prophet.set_validation_rolling_window(
    #TODO: store validation_sets as df: index + columns train start/train end/test start/test end
    #TODO: add option to choose days for train and test period.
    train_percent=0.7,#9,#985,#975,
    test_len=14, 
    start_date="2023-01-01"
)

m_prophet.set_model_parameters(
    lower_limit=2.5,
    upper_limit=97.5
)
m_prophet.set_exogenous_cols(exog_cols = exog_cols)
m_prophet.set_prediction_column(prediction_column="use_transfused")

m_prophet.print_params()

lstm_params = m_prophet.get_params_df()

#Run model
m_prophet.model_run()

#%%


#%% #Trying to run wihtout Object oriented

from prophet import Prophet

start_date = pd.to_datetime("2020-01-01")
split_date = pd.to_datetime("2023-12-31")
end_date = pd.to_datetime("2024-12-31")

pred_col = config.PRED_COLUMN
regressor_cols = ['use_discarded', 'use_expired',   
        'ward_CH', 'ward_I1', 'ward_I3', 'ward_Other', 'ward_UC', 'count',
        'is_workday', 'workday_enc', 'holiday', 'holiday_enc', 'day_of_week',
        'day_of_year', 'year']
    #    'EC_RH_Rh_positive', 'EC_TYPE_EKF', 'EC_TYPE_EKFX', 'EC_TYPE_Other',
    #    'PAT_BG_0', 'use_discarded', 'use_expired', 'use_transfused']
sel_cols = [pred_col] + regressor_cols


train_df = df.loc[start_date:split_date, sel_cols]
test_df = df.loc[split_date:end_date, sel_cols]


df.info()
train_df.info()
test_df.info()

prophet_m = model.ModelProphet(df)


#%%
#try to run model (Prophet)
prophet_train = (
    train_df[pred_col]
    .reset_index()
    .rename(columns={"date":"ds", config.PRED_COLUMN:"y"})
    )

prophet_train.info()
print(prophet_train.head())

#%%
m = Prophet(weekly_seasonality=True, interval_width=0.95)
m.add_country_holidays(country_name="Austria")

m.fit(prophet_train)
future_dates = m.make_future_dataframe(periods=365, freq="D")

fc = m.predict(future_dates)
#%%
m.plot(fc, uncertainty=True)

fc.head()
#.loc["2024-01-01":"2024-03-01"]

# m.add_regressor()

#%%
fc_subset = fc.set_index("ds")
fc_subset.index = pd.to_datetime(fc_subset.index)
fc_plotting = fc_subset["2020-01-01":"2020-05-31"]
fc_plotting = fc_plotting.join(
    df["use_transfused"]
)
# plt.plot(fc_plotting.index, fc_plotting["yhat_lower"])
# plt.plot(fc_plotting.index, fc_plotting["yhat_upper"])
plt.plot(fc_plotting.index, fc_plotting["yhat"])
plt.plot(fc_plotting.index, fc_plotting["use_transfused"])


#%%
import sklearn.metrics as metrics #error metrics (mae, mape etc)

prophet_fc = fc_plotting
# check accuracy
print("prophet")
print("RMSE:", metrics.root_mean_squared_error(y_pred=prophet_fc["yhat"], y_true=prophet_fc[config.PRED_COLUMN]))
print("MAPE:", metrics.mean_absolute_percentage_error(y_pred=prophet_fc["yhat"], y_true=prophet_fc[config.PRED_COLUMN]))
print("MAE: ", metrics.mean_absolute_error(y_pred=prophet_fc["yhat"], y_true=prophet_fc[config.PRED_COLUMN]))
print("MdAE:", metrics.median_absolute_error(y_pred=prophet_fc["yhat"], y_true=prophet_fc[config.PRED_COLUMN]))
print("MaxE:", metrics.max_error(y_pred=prophet_fc["yhat"], y_true=prophet_fc[config.PRED_COLUMN]))








#%%--------------------------------------------------------------------------------
# MARK: VIZ RESULTS
# DATA VIZ (FINISHED MODEL) 
#----------------------------------------------------------------------------------
# Plot prediction vs actual

# If OOP:
# sarima.plot_time()
# sarima.plot_polar()

# lstm.plot_time()
# etc. (could even loop: for obj in [sarima, lstm]: obj.plot_time() obj.plot_polar())

# if functional:

#plot_time(sarima)  # could also usy apply or similar to use list or loop: for mod in models: plot_time(mod)

#TODO: save to csv





#%%--------------------------------------------------------------------------------
# MARK: EVALUATION
#----------------------------------------------------------------------------------

# TODO: load data from csv

# evaluate ....
# print evaluations/tests like mae, mape, etc.


# TODO: save data to csv