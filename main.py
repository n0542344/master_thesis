#__ MARK: libs etc
#This needs to be at the top
import os
import argparse #to run different python files for lstm
#Set LSTM (tensorflow) threads
# os.environ["TF_DISABLE_MKL"] = "1"
# os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
# os.environ["TF_NUM_INTEROP_THREADS"] = "1"
# os.environ["TF_DATA_EXPERIMENTAL_DISABLE_THREADING"] = "1"

#Limit cores per tensorflow python process
#TODO: Still necessary after changing to different python files for concurrent running?
os.environ["OMP_NUM_THREADS"] = "4"      # change to however many you want
os.environ["TF_NUM_INTRAOP_THREADS"] = "4"
os.environ["TF_NUM_INTEROP_THREADS"] = "4"

#ignore/suppress keras logs:
os.environ["KERAS_VERBOSE"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

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

# TOTAL_CORES = 4
# TOTAL_RAM_GB = 10
# RAM_PER_WORKER = TOTAL_RAM_GB / TOTAL_CORES 


#__
def run_worker(args):
    gc.collect()
    gc.collect()

    #limit_memory(3.3) 
    ModelClass, params, df = args
    job_id = params.get("id")
    print(f"{job_id} -- worker started", flush=True)
    # logger = logging.getLogger(__name__)
    # logger.info(f"Worker {ModelClass.__name__}/{job_id} started")


    try:
        model_instance = ModelClass(df, id=job_id)
        model_instance.set_prediction_column(**params)
        if ModelClass.__name__ != "ModelArima":
            model_instance.set_exogenous_cols(**params)
        model_instance.set_validation_rolling_window(**params)
        model_instance.set_model_parameters(**params) 
        #model.print_params(params=["exog_cols", "inner_window"]) #delete
        # logger.info(f"{ModelClass.__name__}/{job_id} Start day {params['start_date'].date()} with {model_instance.stats['window_num']} windows")
        run_results = model_instance.model_run()
        # logger.info(f"{ModelClass.__name__}/{job_id} finished in {model_instance.stats['run_duration']}s")
        return (job_id, True, ModelClass.__name__, run_results)
    
    except Exception as e:
        print("Got an error: ", flush=True)
        print(f"Error in {ModelClass.__name__}: {job_id}: {e}", flush=True)
        traceback.print_exc()

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
        print(f"{job_id} -- worker returns")
        return (job_id, False, ModelClass.__name__, empty_df) 
    

def run_lstm_chunk(jobs, chunk, total_chunks):
    
    """Run a subset of LSTM jobs sequentially — no multiprocessing needed."""
    chunk_size = len(jobs) // total_chunks
    start = chunk * chunk_size
    # last chunk gets any remainder
    end = start + chunk_size if chunk < total_chunks - 1 else len(jobs)
    chunk_jobs = jobs[start:end]

    print(f"LSTM chunk {chunk}/{total_chunks-1}: jobs {start}-{end-1} ({len(chunk_jobs)} total)", flush=True)

    results = []
    for job in chunk_jobs:
        result = run_worker(job)
        if result is not None:
            results.append(result)
    return results

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
    limit_bytes = int(mem_limit_gb * 1024 * 1024 * 1024)
    resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))

def limit_memory(maxsize_gb):
    # Convert GB to bytes
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    limit = maxsize_gb * 1024 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit, hard))


def main():
    # multiprocessing.set_start_method("spawn", force=True) #use spawn instead of sub-processes

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        help="Which model to run: arima, sarimax, prophet, lstm. Omit to run all.")
    parser.add_argument("--chunk", type=int, default=0,
                        help="Chunk index (0-based)")
    parser.add_argument("--total_chunks", type=int, default=1,
                        help="Total chunks to split LSTM jobs into")
    args = parser.parse_args()



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
    logger.info(f"Starting pool: {config.TOTAL_CORES} workers, {config.RAM_PER_WORKER:.2f}GB each.")


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
    #sampled_grid_lstm = sample_grid(full_grid_lstm, n_samples=10) #only sample lstm, others are fast enough to fully search
    full_grid_prophet = list(ParameterGrid(config.gs_config_prophet))


    #Create jobs list => tuple of Model class, input settings, df
    all_jobs = {}

    all_jobs["arima"] = add_model_grid_to_all_jobs(full_grid_arima, model.ModelArima, df, get_start_id(all_jobs))
    all_jobs["sarimax"] = add_model_grid_to_all_jobs(full_grid_sarimax, model.ModelSarimax, df, get_start_id(all_jobs))
    all_jobs["lstm"] = add_model_grid_to_all_jobs(full_grid_lstm, model.ModelLSTM, df, get_start_id(all_jobs))
    all_jobs["prophet"] = add_model_grid_to_all_jobs(full_grid_prophet, model.ModelProphet, df, get_start_id(all_jobs))

    #Sample jobs now, to still have continuous global ids
    sampled_jobs = all_jobs
    sampled_jobs["arima"] = sample_grid(sampled_jobs["arima"], n_samples=2) #delete
    sampled_jobs["sarimax"] = sample_grid(sampled_jobs["sarimax"], n_samples=config.sarimax_n_samples) #keep? 
    sampled_jobs["lstm"] = sample_grid(sampled_jobs["lstm"], n_samples=config.lstm_n_samples) #delete?
    
    # cores = 24#max(1, multiprocessing.cpu_count() - 1)
    # logger.info(f"---Using {cores} cores---")
    logger.info(f"---Using {config.TOTAL_CORES} cores with max {config.TOTAL_RAM_GB} GB RAM---")


    #------Run models (with multiprocessing) --------
    spawn_ctx = multiprocessing.get_context("spawn")

    if args.model in (None, "arima"):
        logger.info("---Starting ARIMA---")
        with spawn_ctx.Pool(
        # with multiprocessing.Pool(
            processes=config.TOTAL_CORES, 
            initializer=initialize_worker, 
            initargs=(config.RAM_PER_WORKER,)
        ) as pool:
            result_list_arima = pool.map(run_worker, sampled_jobs["arima"])# [0:100]) #to take subset

        valid_results_arima = [res[3] for res in result_list_arima if res is not None]
        final_result_df_arima = pd.concat(valid_results_arima).set_index("id")
        final_result_df_arima.to_csv("./results/Arima/grid_search_results.csv")
        logger.info("---Finished ARIMA---")


    if args.model in (None, "sarimax"):
        logger.info("---Starting SARIMAX---")
        with spawn_ctx.Pool(
        # with multiprocessing.Pool(
            processes=config.TOTAL_CORES, 
            initializer=initialize_worker, 
            initargs=(config.RAM_PER_WORKER,)
        ) as pool:
            result_list_sarimax = pool.map(run_worker, sampled_jobs["sarimax"])# [0:100]) #to take subset

        valid_results_sarimax = [res[3] for res in result_list_sarimax if res is not None]
        final_result_df_sarimax = pd.concat(valid_results_sarimax).set_index("id")
        final_result_df_sarimax.to_csv("./results/Sarimax/grid_search_results.csv")
        logger.info("---Finished SARIMAX---")


    if args.model in (None, "prophet"):
        logger.info("---Starting Prophet---")
        with spawn_ctx.Pool(
        # with multiprocessing.Pool(
            processes=config.TOTAL_CORES, 
            initializer=initialize_worker, 
            initargs=(config.RAM_PER_WORKER,)
        ) as pool:
            try:
                result_list_prophet = pool.map(run_worker, sampled_jobs["prophet"])#[0:100]) #to take subset
            except Exception as e:
                print(f"Model run Prophet failed: {e}", flush=True)
                # traceback.print_exc()
            # result_list_prophet = pool.map(run_worker, sampled_jobs["prophet"][0:100]) #all_jobs["sarimax"][0:8])

        print("finished with prophet, getting results")
        valid_results_prophet = [res[3] for res in result_list_prophet if res is not None]
        final_result_df_prophet = pd.concat(valid_results_prophet).set_index("id")
        final_result_df_prophet.to_csv("./results/Prophet/grid_search_results.csv")

        logger.info("---Finished Prophet---")
    
    
    # logger.info("Starting LSTM without mp")
    # m_lstm = model.ModelLSTM(df)
    # m_lstm.set_validation_rolling_window(**sampled_jobs["lstm"][1][1])
    # m_lstm.set_model_parameters(**sampled_jobs["lstm"][1][1])
    # m_lstm.set_exogenous_cols(**sampled_jobs["lstm"][1][1])
    # m_lstm.set_prediction_column(**sampled_jobs["lstm"][1][1])
    # m_lstm.model_run()



    # logger.info("Starting LSTM")
    # logger.info("Set thread cores to 4")
    # fork_ctx = multiprocessing.get_context("fork")
    # with fork_ctx.Pool(
    # # with multiprocessing.Pool(
    #     processes=1, 
    #     initializer=initialize_worker, 
    #     initargs=(config.RAM_PER_WORKER,)
    # ) as pool:
    #     # result_list_lstm = pool.map(run_worker, sampled_jobs["lstm"][0:5]) #all_jobs["sarimax"][0:8])
    #     try:
    #         result_list_lstm = pool.map(run_worker, sampled_jobs["lstm"][0:1]) #all_jobs["sarimax"][0:8])
    #     except Exception as e:
    #         print(f"--LSTM failed--: {e}")

    # valid_results_lstm = [res[3] for res in result_list_lstm if res is not None]
    # final_result_df_lstm = pd.concat(valid_results_lstm).set_index("id")
    # final_result_df_lstm.to_csv("./results/LSTM/grid_search_results.csv")
    # logger.info("---Finished LSTM---")


    # logger.info("Finished Running the pipeline")

    if args.model in ("lstm"):
        logger.info(f"---Starting LSTM chunk {args.chunk}/{args.total_chunks-1}---")
        result_list_lstm = run_lstm_chunk(
            sampled_jobs["lstm"],
            args.chunk,
            args.total_chunks
        )
        valid_results_lstm = [res[3] for res in result_list_lstm if res is not None]
        if valid_results_lstm:
            final_result_df_lstm = pd.concat(valid_results_lstm).set_index("id")
            out_path = f"./results/LSTM/grid_search_results_chunk_{args.chunk}.csv"
            final_result_df_lstm.to_csv(out_path)
            logger.info(f"---Finished LSTM chunk {args.chunk}, saved to {out_path}---")
        else:
            logger.warning(f"LSTM chunk {args.chunk} produced no valid results")





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



