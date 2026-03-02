#__ MARK: libs etc
#This needs to be at the top
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["STAN_NUM_THREADS"] = "1"


import pandas as pd
import numpy as np
from numpy import nan
from time import time
from datetime import datetime
from matplotlib import pyplot as plt

import seaborn as sns

from src import clean
from src import config
from src import data_model
from src import load
from src import model
from src import transform
from src import viz
from src import utils


from itertools import product
from sklearn.model_selection import ParameterGrid
import random
import multiprocessing

import gc
import traceback

import logging
from logging.handlers import RotatingFileHandler
from multiprocessing_logging import install_mp_handler
import sys
import resource

#For developing purposes:
# import importlib
# print(load.__file__)
# print(clean.__file__)

#turn off pandas warnings:
from warnings import filterwarnings
filterwarnings('ignore')

#turn off prophet log:
prophet_logger = logging.getLogger('cmdstanpy')
prophet_logger.addHandler(logging.NullHandler())
prophet_logger.propagate = False
prophet_logger.setLevel(logging.CRITICAL)

# IMAGE_PATH = "plots/2025_10_10-Plots_for_Meeting/"
RUN_ALL = False
RUN_DATE = datetime.today().strftime('%Y%m%d-%H_%M')

TOTAL_CORES = 2
TOTAL_RAM_GB = 10
RAM_PER_WORKER = TOTAL_RAM_GB / TOTAL_CORES 


#__
def run_worker(args):
    gc.collect()
    gc.collect()

    #limit_memory(3.3) 
    print("Worker started")

    ModelClass, params, df = args
    job_id = params.get("id")
    print(job_id, flush=True)
    # logger = logging.getLogger(__name__)
    # logger.info(f"Worker {ModelClass.__name__}/{job_id} started")


    try:
        model_instance = ModelClass(df, id=job_id)
        model_instance.set_prediction_column(**params)
        if ModelClass.__name__ != "ModelArima":
            model_instance.set_exogenous_cols(**params)
        model_instance.set_validation_rolling_window(**params)
        model_instance.set_model_parameters(**params) 
        # logger.info(f"{ModelClass.__name__}/{job_id} Start day {params['start_date'].date()} with {model_instance.stats['window_num']} windows")
        run_results = model_instance.model_run()
        # logger.info(f"{ModelClass.__name__}/{job_id} finished in {model_instance.stats['run_duration']}s")
        return (job_id, True, ModelClass.__name__, run_results)
    
    except Exception as e:
        print("Got an error: ", flush=True)
        print(f"Error in {ModelClass.__name__}: {job_id}: {e}", flush=True)
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
        print("worker returns")
        return (job_id, False, ModelClass.__name__, empty_df) 


def get_start_id(all_jobs):
    return sum(len(v) for v in all_jobs.values())

def add_model_grid_to_all_jobs(grid, model, df, start_id=0):
    #grid is a df, model a specific modle class, like model.ModelArima
    global_job_id = start_id
    jobs = []

    for param_set in grid:
        param_set["id"] = global_job_id
        # param_set["id"] = sum(len(v) for v in all_jobs.values()) #consecutive ID across all models
        jobs.append( (model, param_set, df) )
        global_job_id += 1

    return jobs

def initialize_worker(mem_limit_gb):
    """This runs once inside every one of the 24 worker processes."""
    # 1. Set RAM Limit (per worker)
    # RLIMIT_AS is 'Address Space' - the total memory the process can seize
    limit_bytes = int(mem_limit_gb * 1024 * 1024 * 1024)
    resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))

def limit_memory(maxsize_gb):
    # Convert GB to bytes
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    limit = maxsize_gb * 1024 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit, hard))


def main():
    multiprocessing.set_start_method("spawn", force=True)

    log_file = f"./logs/pipeline_{RUN_DATE}.log"
    formatter = logging.Formatter('%(asctime)s [%(processName)s] [%(levelname)s] %(message)s')

    #Console handler for docker logs
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    #File handler for persistent log files
    file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3)
    file_handler.setFormatter(formatter)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[console_handler, file_handler]
    )

    install_mp_handler()
    logger = logging.getLogger(__name__)
    logger.info(f"Starting pool: {TOTAL_CORES} workers, {RAM_PER_WORKER:.2f}GB each.")


    df_processed = pd.read_csv("./data/03_transformed/output_transformed.csv", index_col="date", parse_dates=True)
    df = data_model.Data(data=df_processed)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(processName)s] %(message)s',
        handlers=[logging.StreamHandler()]
    )

    # 3. Use a Pool or Processes normally
    logger = logging.getLogger(__name__)
    logger.info("---Starting pipeline---")

    # from pmdarima import auto_arima
    # autoarima_res = auto_arima(y=df[config.PRED_COLUMN], seasonal=True, m=7, trace=True)

    def sample_grid(full_grid, n_samples, seed=config.SEED):
        random.seed(seed)

        sampled_grid = random.sample(full_grid, min(n_samples, len(full_grid)))
        return sampled_grid

    #Create grids, sample
    full_grid_arima = list(ParameterGrid(config.gs_config_arima))
    full_grid_sarimax = list(ParameterGrid(config.gs_config_sarimax))
    full_grid_lstm = list(ParameterGrid(config.gs_config_lstm))
    sampled_grid_lstm = sample_grid(full_grid_lstm, n_samples=10) #only sample lstm, others are fast enough to fully search
    full_grid_prophet = list(ParameterGrid(config.gs_config_prophet))



    #TODO: sample after creating all_jobs to create ids refering to full grid (instead of ids continuous for sampled grid)
    all_jobs = {}

    all_jobs["arima"] = add_model_grid_to_all_jobs(full_grid_arima, model.ModelArima, df, get_start_id(all_jobs))
    all_jobs["sarimax"] = add_model_grid_to_all_jobs(full_grid_sarimax, model.ModelSarimax, df, get_start_id(all_jobs))
    all_jobs["lstm"] = add_model_grid_to_all_jobs(sampled_grid_lstm, model.ModelLSTM, df, get_start_id(all_jobs))
    all_jobs["prophet"] = add_model_grid_to_all_jobs(full_grid_prophet, model.ModelProphet, df, get_start_id(all_jobs))

    # cores = 24#max(1, multiprocessing.cpu_count() - 1)
    # logger.info(f"---Using {cores} cores---")
    logger.info(f"---Using {TOTAL_CORES} cores with max {TOTAL_RAM_GB} GB RAM---")


    # logger.info("Starting ARIMA")
    with multiprocessing.Pool(
        processes=TOTAL_CORES, 
        initializer=initialize_worker, 
        initargs=(RAM_PER_WORKER,)
    ) as pool:
        result_list_arima = pool.map(run_worker, all_jobs["arima"][0:5])

    valid_results_arima = [res[3] for res in result_list_arima if res is not None]
    final_result_df_arima = pd.concat(valid_results_arima).set_index("id")
    final_result_df_arima.to_csv("./results/Arima/grid_search_results.csv")
    logger.info("---Finished ARIMA---")


    #------Run models (with multiprocessing) --------

    logger.info("Starting SARIMAX")
    with multiprocessing.Pool(
        processes=TOTAL_CORES, 
        initializer=initialize_worker, 
        initargs=(RAM_PER_WORKER,)
    ) as pool:
        result_list_sarimax = pool.map(run_worker, all_jobs["sarimax"][0:5]) #all_jobs["sarimax"][0:8])

    valid_results_sarimax = [res[3] for res in result_list_sarimax if res is not None]
    final_result_df_sarimax = pd.concat(valid_results_sarimax).set_index("id")
    final_result_df_sarimax.to_csv("./results/Sarimax/grid_search_results.csv")
    logger.info("---Finished SARIMAX---")


    logger.info("Starting Prophet")
    with multiprocessing.Pool(
        processes=TOTAL_CORES, 
        initializer=initialize_worker, 
        initargs=(RAM_PER_WORKER,)
    ) as pool:
        try:
            result_list_prophet = pool.map(run_worker, all_jobs["prophet"][0:100]) #all_jobs["sarimax"][0:8])
        except Exception as e:
            print(f"Model run Prophet failed: {e}", flush=True)
            traceback.print_exc()
        result_list_prophet = pool.map(run_worker, all_jobs["prophet"][0:100]) #all_jobs["sarimax"][0:8])

    print("finished with prophet, getting results")
    valid_results_prophet = [res[3] for res in result_list_prophet if res is not None]
    final_result_df_prophet = pd.concat(valid_results_prophet).set_index("id")
    final_result_df_prophet.to_csv("./results/Prophet/grid_search_results.csv")

    logger.info("---Finished Prophet---")
    
    
    logger.info("Starting LSTM")
    with multiprocessing.Pool(
        processes=TOTAL_CORES, 
        initializer=initialize_worker, 
        initargs=(RAM_PER_WORKER,)
    ) as pool:
        # result_list_lstm = pool.map(run_worker, all_jobs["lstm"][0:5]) #all_jobs["sarimax"][0:8])
        try:
            result_list_lstm = pool.map(run_worker, all_jobs["lstm"][0:10]) #all_jobs["sarimax"][0:8])
        except Exception as e:
            print(e)

    valid_results_lstm = [res[3] for res in result_list_lstm if res is not None]
    final_result_df_lstm = pd.concat(valid_results_lstm).set_index("id")
    final_result_df_lstm.to_csv("./results/LSTM/grid_search_results.csv")
    logger.info("---Finished LSTM---")


    logger.info("Finished Running the pipeline")


#__
if __name__ == "__main__":
    main()



#__
# m_prophet = model.ModelProphet(df)
# m_prophet.set_validation_rolling_window(**test_worker[1])
# m_prophet.set_model_parameters(**test_worker[1])
# m_prophet.set_exogenous_cols(**test_worker[1])
# m_prophet.set_prediction_column(**test_worker[1])
# m_prophet.model_run()





#TODO: remove
if RUN_ALL == True:
#--------------------------------------------------------------------------------
# MARK: INPUT EXISTING
#----------------------------------------------------------------------------------
    PATH_RAW = "data/01_raw/blood-data_complete_2025-07-16.tsv"

    #TODO: make backwards file loading: try processed file, if not exists try cleaned, if not exists try raw
    #TODO: Alternative for now -- just load transformed file, because i know it exists.


    #--------------------------------------------------------------------------------
    # MARK: INPUT
    #----------------------------------------------------------------------------------

    # Read Data
    df_raw = load.load_data(path="data/01_raw/blood-data_complete_2025-07-16.tsv")
    # df_raw = load.load_data(path="data/01_raw/testdaten.tsv")

    #__
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



    #--------------------------------------------------------------------------------
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

    #

    #TODO: Check what unique vals are present in df
    clean.check_unique_values(df_clean.drop(["EC_ID_I_hash", "EC_ID_O_hash", "PAT_WARD"], axis=1))


    #
    # Plot frequency counts for unique values in every column
    #TODO: move into viz.py


    for col_name, col in df_clean.items():
        if col_name in ["EC_ID_O_hash", "EC_ID_I_hash"]:
            continue
        print(col.value_counts())
        col.value_counts()[:40].plot(kind="bar", title=col_name,)
        plt.savefig(fname=IMAGES_PATH_EXPLORATION+f"01-barcharts-value_counts-{col_name}")
        plt.show()


    importlib.reload(viz)
    ##
    # Plot each patient wards transfusion counts (for wards with >500 transfusions)
    viz.plot_patient_wards(df_clean, n=500, save_figs=False, location=IMAGES_PATH_EXPLORATION, foldername="")







    #--------------------------------------------------------------------------------
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


    #__ #Plot seasonalities daily & weekly of processed df

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



    #__--------------------------------------------------------------------------------
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

    #__


    #df.print_head()
    df[START_DATE_EXPLORATION:].plot_seasonal(plot_type='daily', col_name='use_transfused', fig_location=IMAGES_PATH_EXPLORATION)
    df[START_DATE_EXPLORATION:].plot_seasonal(plot_type='weekly', col_name='use_transfused')


    ##__
    #Boxplots
    df[START_DATE_EXPLORATION:].plot_boxplots(col_name='use_transfused')
    df[START_DATE_EXPLORATION:].plot_seasonal_subseries(col_name='use_transfused') #NOTE: i think it works, but not enough dummy data.
    #TODO: check if seasonal subseries plot works with multi-year data


    ##__
    #Decompose
    df[PRE_COVID_START:PRE_COVID_END].decompose_one(col_name='use_transfused')


    #df.decompose_all("use_transfused")

    # mulitple decomposition (daily + weekly)
    df[PRE_COVID_START:PRE_COVID_END].multiple_decompose(col_name="use_transfused", periods=[7, 365])




    ##__



    df[pd.to_datetime("2024-01-01"):pd.to_datetime("2024-12-31")].plot_daily_heatmap(col_name='use_transfused')


    #__ Visualize counts for all plots (as of now only for those starting with 901AN) on top of each other, so that
    # its visible, where naming of one ward starts/ends 


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

    #__
    # Get unique Letter-combinations (i guess top-wards) from all wards:

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

    #__--------------------------------------------------------------------------------
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

    from statsmodels.graphics.tsaplots import plot_acf
    from statsmodels.graphics.tsaplots import plot_pacf

    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.stattools import kpss

    ##__
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



    #__ 
    # MARK: COMPARISON
    #----------------------------------------------------------------------------------
    import importlib
    importlib.reload(model)
    comp = model.ModelComparison(df)

    comp.set_column()
    comp.set_dates_mean()
    comp.set_forecast_window()
    comp.set_single_value()

    comp.print_parameters()

    comp.model_run()

    comp.predictions

    # comp.get_error_values()


    for col in comp.predictions.columns:
        if col == "use_transfused":
            plt.plot(comp.predictions.loc["2024-01-01":"2024-07-31", col], linewidth=1.5, label=col)
        else:
            plt.plot(comp.predictions.loc["2024-01-01":"2024-07-31", col], linewidth=0.5, label=col)
        plt.legend()

    comp.stats["id"] = 0000
    comp.stats["model_name"] = "comparison"
    comp.create_result_dir()
    comp.save_results()