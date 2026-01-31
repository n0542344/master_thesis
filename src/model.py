#TODO: is it necessary to load the modules here?
from src import data_model
from src import config
from src.utils import timer_func

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm #color palette
import gc

import os
import psutil

from sklearn.model_selection import train_test_split
import statsmodels
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import sklearn.metrics as metrics #error metrics (mae, mape etc)

#For LSTM
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from keras import Input, layers, Model

from prophet import Prophet

import time
from datetime import datetime
from pathlib import Path
import json

#delete:
import tensorflow as tf
# tf.debugging.set_log_device_placement(True)
num_cores = 6
tf.config.threading.set_intra_op_parallelism_threads(num_cores)
tf.config.threading.set_inter_op_parallelism_threads(num_cores)




#xXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# MARK: Model

#xXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

class Model:
    #methods for running are in the child classes for the individual models.
    # here only general methods needed for the models


    def __init__(self, data: data_model.Data, id: int=None): #TODO: maybe add configuration?
        # TODO: change parameter of model to be of 'Data' type
        self.data = data
        self.stats = {}

        #id for model run when using grid search
        if id == None:
            self.id = None
        else:
            self.id = f"{id:05d}" 

        #Train/Test sets:
        self.split_index = None #Index at split point
        self.train_data = None
        self.test_data = None
        self.train_test_data = None #list of tuples containg train/test df index pairs

        #Config and date sets for splitting data (train/test)
        self.validation_config = {}
        self.validation_sets = None
        #save forecasts as 1-day fc, 2-day-fc, for plotting:
        # when i set test_len as 7 days, i want to save all 1-day look aheads, all for the 2nd day etc
        
        self.stepwise_forecasts = pd.DataFrame()#TODO: in use anywhere or delete?

        #Model(s)
        self.models = []
        self.predictions = []
        #self.model_runs = list() #List of past runs
        #TODO: save model runs as well as their state (parameters etc.)
        # so they can be accessed later. save to file.

        self.alpha = None #for prediction interval

        #Forecast errors (stepwise, so for forecast 1 day ahead, 2 days ahead, etc.):
        #Values are pd.Dataframes, with stepwise errors for forecast timeframe:
        # ME -- Median error (shows model bias to be positive or negative)
        # MAE -- Mean absolute error
        # MedAE -- Median absolute error
        # MAPE -- Mean absolute percentage error
        # sMAPE -- Symmetric mean absolute percentage error
        # MASE -- Mean absolute scaled error
        # MSE -- Mean squared error
        # RMSE -- Root mean squared error
        # RMSSE -- root mean squared scaled error
        # MaxError -- maximum error
        self.stepwise_forecast_errors = pd.DataFrame()

        #Difference between stepwise forecast values to actual values
        self.stepwise_forecast_difference = pd.DataFrame()

        #Dont think is needed:
        #self.model_params = None #rename; prob better in base class.



        #Decompositions: (s.u., gehört eig. nciht hierher)
        # self.decomp = None






    #------------------------------------------------------------------------------------------------
    # Helper functions
    # MARK: HELPERS
    #------------------------------------------------------------------------------------------------

    #------------------------------------------------------------------------------------------------
    # GETTERS & Print
    # MARK: Getters

    def print_params(self):
        #TODO: probably better in Model class!
        for key, value in self.model_params.items():
            print(key, ": ", value)
        

    def get_params_df(self):
        """
        Returns params as df
        """
        #TODO: probably better in Model class!

        params_df = pd.DataFrame([self.model_params]) #bracket to keep everything in one row

        return params_df




    #------------------------------------------------------------------------------------------------
    # SETTERS
    # MARK: Setters

    def set_validation_expanding_window(self, train_percent: float, test_len: int, start_date: str=None):
        """
        Set parameters of validation_config variable (doesnt RUN validation) and create a
        set of indices (list of tuples) for train/test split by calling make_validation_set().
        There is only one config variable, so using setter again or for another method (rolling window)
        overwrites this setting!

        Expanding window validation means, that an initial train set is expanded,
        by x days each iteration (here x = 1 day).


        Args:
        :param train_percent: Percentage of whole dataset, that should be used as initial training set
        :type train_percent: float
        :param test_len: Number of days, to test into the future
        :type test_len: int
        :param start_date: Optional str in date format YYYY-MM-DD, to set start date of data set. Default None.
        :type start_date: str (, optional)

        :raises ValueError: if bla bla bla

        :return: Nothing, but sets validation_config member variable.
        :rtype: None
        """

        if not start_date:
            start_date = self.data.index.min()
        else:
            start_date = pd.to_datetime(start_date)

        print(start_date)

        train_len = len(self.data)

        self.validation_config = {
            "type" : "expanding window",
            "train_percent" : train_percent,
            "test_len" : test_len,
            "start_date" : start_date
        }

        self.make_validation_set()



    def set_validation_rolling_window(self, train_percent: float, test_len: int, start_date: str=None):
        """
        Set parameters of validation_config variable (doesnt RUN validation) and create a
        set of indices (list of tuples) for train/test split by calling make_validation_set().
        There is only one config variable, so using setter again or for another method (expanding window)
        overwrites this setting!

        Rolling window validation means, a fixed number of days is used for training, which
        rolls over dataset, and the following days of number <test_len> are used as test set.

        Parameters
        ----------
        train_percent : float
            float between 0-1, sets percentage of data to assign
            as training set.
        test_len : int
            Number of days to set as test set.
        start_data : str, optional
            Optional str in date format YYYY-MM-DD, to set
            start date of data set. Default None.
        """
        if not start_date:
            start_date = self.data.index.min()
        elif pd.to_datetime(start_date) < pd.to_datetime(self.data.index.min()):
            raise ValueError("'start_date' must be within data")
        else:
            start_date = pd.to_datetime(start_date)

        self.validation_config = {
            "type" : "rolling window",
            "train_percent" : train_percent,
            "test_len" : test_len,
            "start_date" : start_date
        }

        self.make_validation_set()



    def set_validation_single_split(self, train_percent: float, start_date: str=None):
        """
        Set parameters of validation_config variable (doesnt RUN validation) and create a
        set of indices (list of tuples) for train/test split by calling make_validation_set().
        There is only one config variable, so using setter again or for another method (rolling window)
        overwrites this setting!

        Single split validation means, that whole dataset is split a single time into
        train and test sets.

        Parameters
        ----------
        train_percent : float
            float between 0-1, sets percentage of data to assign as training set
        start_data : str
            Optional str in date format YYYY-MM-DD, to set start date of data set. Default None.
        """
        if not start_date:
            start_date = self.data.index.min()
        else:
            start_date = pd.to_datetime(start_date)

        self.validation_config = {
            "type" : "single split",
            "train_percent" : train_percent,
            "start_date" : start_date
        }

        self.make_validation_set()

    def set_alpha_prediction(self, alpha=0.05):
        # Even if alpha is not explicitly set (self.alpha==None), 
        # then get_prediction is coded, to set alpha inside fct by default to 0.05
        self.alpha = alpha


    def set_prediction_column(self, prediction_column: str=None):
        #TODO: moved here from LSTM
        """
        Docstring for set_prediction_column
        
        Args:
            prediction_column (str): Pass the string of the column name, whcih should be predicted.

        Raises:
            ValueError if no string supplied/argument is empty
        """

        if not prediction_column:
            raise ValueError("'column_to_predict' cant be empty") 
        
        self.model_params["prediction_column"] = prediction_column




    def set_exogenous_cols(self, exog_cols: list):
        """Set columns to use for exogenous variables with LSTM. 
        Sets member variables' "param" (dict) key "exog_cols" to value of input parameter.

        Args:
            exog_cols (list): List of strings containing column names for exog vars (in self.data)

        Raises:
            ValueError: If exog_cols list is empty. Doesnt check for data type.
            ValueError: If a column from exog_cols is not present in 'df'
            ValueError: If a list item is not a string.
        """
        #TODO: probably better in Model class!

        #Check df/exog_cols input for validity
        if len(exog_cols) == 0:
            raise ValueError("Need to pass col name present in 'df' to exog_cols")
        else:
            for col in exog_cols:
                if col not in self.data.columns:
                    raise ValueError(f"{col} not present df's columns: {self.data.columns}")
                elif type(col) != str:
                    raise ValueError(f"{col} not a string -- only string colnames allowed")
                
        self.model_params["exog_cols"] = exog_cols




    #------------------------------------------------------------------------------------------------
    # General
    # MARK: General

    def create_result_dir(self):
        #TODO: move to base class Model
        #Save model weights (first create dir, then save):
        #TODO: evaluate if this helps with memory leak, otherwise delete: --> creating once and reloading initial weights of model helps/solves mem leak
        # -> only build model once before window-loop, save weights on compile, then load 
        # again every iteration:
        #date+hh:mm+grid(id if doing grid search)
        self.date = datetime.now().strftime("%Y%m%d_%H%M")
        if self.id == None:
            self.dir_name = f"{self.date}-lstm"
        elif self.id != None:
            self.dir_name = f"{self.date}-{self.id}-lstm"


        #make directory with name (see above)
        print(f"Creating directory {self.dir_name}")
        print(f"Saving params to {self.dir_name}")
        self.file_path = "./results/"+self.dir_name
        Path(self.file_path).mkdir(parents=True, exist_ok=True)

    @timer_func
    def save_results(self):
        #Creates new directory where it saves self.model_params as a json and every dataframe for the results

        # -> Moved filepath creation to build_model (to save weights) and store in self.file_path
        #date+hh:mm+grid(id if doing grid search)
        # date = datetime.now().strftime("%Y%m%d_%H%M")
        # if self.id == None:
        #     dir_name = f"{date}-lstm"
        # elif self.id != None:
        #     dir_name = f"{date}-{self.id}-lstm"


        # #make directory with name (see above)
        # print(f"Creating directory {dir_name}")
        # print(f"Saving params to {dir_name}")
        # file_path = "./results/"+dir_name
        # Path(file_path).mkdir(parents=True, exist_ok=True)

        #make params json
        with open("./results/"+self.dir_name+"/params.json", "w") as f:
            json.dump(self.model_params, f)

        #Save self.predictions (each df individually)
        print(f"Saving forecasts to {self.dir_name}")
        for fc_day, df in self.predictions.items():
            print("Key: ", fc_day)
            print("Value: ", df)
            df.to_csv(path_or_buf=self.file_path+"/"+fc_day+".csv" ,sep=";")

        #Save Error Values df
        with open("./results/"+self.dir_name+"/forecast_errors.csv", "w") as f:
            self.forecast_errors

        print(f"Finished saving file to {self.dir_name}")




    @timer_func
    def get_start_end_days(self, window):
        #TODO: probably better in Model class!

        # since lstm works better with running multiple inputs (like a rolling window) and supplying y  so it can adjust
        # weights and biases more often. so we do an inner loop for the forecast window supplied by rolling/expanding window 
        # (=validations_sets).
        # This means we treat every validation set like its own timeframe, where we use , up to that day, and only
        self.train_start = window[0]
        self.train_end = window[1]
        self.test_start = window[2]
        self.test_end = window[3] - pd.Timedelta(days=1)

        self.forecast_days = (window[3] - window[2]).days #because window[2] and window[3] should be both included as days


    @timer_func
    def get_data(self):
        #TODO: probably better in Model class!
        """
        Using the input "window", which contains tuple of start + end date for both train and test sets, it returns
        numpy arrays for each. VAlues are in format of original data (not scaled!) --> "raw".
        Columns (exogenous, prediction) are taken from self.model_params.

        Args:
            window (tuple of pd.Datetime): Contains a tuple of pd.Datetime 
                for start/end date of train set, start/end date of test set.

        Raises:
            ValueError: If no column to predict was set previously.

        Returns:
            X_train_raw: Numpy array containing prediction+exogenous columns in original format for training
            y_train_raw: Numpy array containing one column for prediction column in original format for training
            X_test_raw: Numpy array containing prediction+exogenous columns in original format for testing
            y_test_raw: Numpy array containing one column for prediction column in original format for testing
        """

        #TODO: Description of get_data


        if not self.model_params["prediction_column"]:
            raise ValueError(f"Missing the column to predict ({self.model_params['predcition_column']})")
        
        columns = list(set([self.model_params["prediction_column"], *self.model_params["exog_cols"]]))#star* to unpack list, pred_cols is str only.
        # print("cols: ", columns)

        self.X_train_raw = self.data.loc[self.train_start : self.train_end, columns].values
        self.y_train_raw = self.data.loc[self.train_start : self.train_end, self.model_params["prediction_column"]].values.reshape(-1, 1)

        self.X_test_raw = self.data.loc[self.test_start : self.test_end, columns].values
        self.y_test_raw = self.data.loc[self.test_start : self.test_end, self.model_params["prediction_column"]].values.reshape(-1, 1)
      

    @timer_func
    def add_stepwise_difference(self):
        #todo: COPIED FROM lstm, SHOUDL BE in base Model class?
        #TODO: change ModelARIMA/ModelSARIMA so that it can use this function as well.
        # would need changes to structure of result.
        
        #This funcction adds a column with difference between Actual and Mean (=Prediction) to result dfs
        #Put it directly inside add_to_results(), so shouldnt be needed anymore.
        for fc_day, df in self.predictions.items():
            df["Difference"] = df["Actual"] - df["Mean"] 


    @timer_func
    def add_to_results(self):        
        #Add to existing self.predictions dictionary. self.predictions contains n keys of name "Day_"n_i where n=len(fc_days)
        # with the value of a dataframe with columns Actual, Mean, Lower, Upper and datetime index. 
        # Each dataframe gets expanded/filled in every window-iteration by the current (of the window) value of the day
        # and date. So in the first window with 14 fc days, 14 empty dataframes with keys of Day_1 to Day_14 get filled.
        # Say first day is 01.01., then Day_1 gets one row with actual/mean/lower/upper for 01.01., Day_2 for 02.01. etc.
        # In the next window, Day_1 gets a new row for day 02.01. and so on.

        #Moved here from ModelLSTM.
        #TODO: add proper description

        #Initialize self.predictions if not exists.
        # Creates n (=forecast_days) empty dataframes in a dict, each containing datetime index from day_n in the 
        # test/validation period.  
        #TODO: better to implement in __init__ or its own function?
        if not hasattr(self, "predictions") or self.predictions == None or type(self.predictions) == list:
            self.predictions = {}
            for fc_day in range(self.forecast_days):
                day_label = f"Day_{fc_day + 1}"

                start_date = self.validation_sets[0][2]
                end_date = self.validation_sets[-1][2] + pd.Timedelta(days=fc_day)

                self.predictions[day_label] = pd.DataFrame(
                    index=pd.date_range(start_date, end_date),
                    columns=["Actual", "Mean", "Lower", "Upper"]
                )

        #fill the dataframes
        for day in range(self.forecast_days):
            day_label = f"Day_{day + 1}"
            forecast_date = self.test_start + pd.Timedelta(days=day)

            day_predictions = self.all_predictions[:, 0, day] #shape of (np_iterations, 1, forecast_days)

            self.predictions[day_label].loc[forecast_date, "Actual"] = self.data.loc[forecast_date, self.model_params["prediction_column"]]
            # y_test_original_scale = self.scaler_y.inverse_transform(self.y_test) #old LSTM way: set this line above for-day range
            # self.predictions[day_label].loc[forecast_date, "Actual"] = y_test_original_scale[0, day]
            self.predictions[day_label].loc[forecast_date, "Mean"] = np.mean(day_predictions, axis=None) #alternative: axis=0)[0]
            self.predictions[day_label].loc[forecast_date, "Lower"] = np.percentile(day_predictions, 2.5, axis=None)
            self.predictions[day_label].loc[forecast_date, "Upper"] = np.percentile(day_predictions, 97.5, axis=None)

    
    @timer_func
    def get_stepwise_errors(self):
        #TODO move to base class Model
        #TODO: change ModelARIMA/ModelSARIMA so that it can use this function as well.
        #since lstm currently has different architecture of dataframes for results, ill make a new
        #function here. old (comparison, (s)arima) will be adjusted to be same as lstm model.


        #initialize empty df with structure like stepwise_forecasts (cols, indices, no content)
        forecast_steps = self.predictions.keys()
        errors = ["ME", "MAE", "MedAE", "MAPE", "RMSE", "MASE", "MaxError"]

        self.forecast_errors = pd.DataFrame(columns=errors, index=forecast_steps)

        self.forecast_errors = pd.DataFrame()
        for fc_day, df in self.predictions.items():

            y_pred = df["Mean"].dropna()
            y_true = df["Actual"].dropna()
            
            self.forecast_errors.loc[fc_day, "ME"] = np.median(y_pred - y_true) #median error -- shows bias (positive or negative)
            self.forecast_errors.loc[fc_day, "MAE"] = metrics.mean_absolute_error(y_pred=y_pred, y_true=y_true)
            self.forecast_errors.loc[fc_day, "MedAE"] = metrics.median_absolute_error(y_pred=y_pred, y_true=y_true)
            self.forecast_errors.loc[fc_day, "MAPE"] = metrics.mean_absolute_percentage_error(y_pred=y_pred, y_true=y_true)
            self.forecast_errors.loc[fc_day, "MSE"] = metrics.mean_squared_error(y_pred=y_pred, y_true=y_true)
            self.forecast_errors.loc[fc_day, "RMSE"] = metrics.root_mean_squared_error(y_pred=y_pred, y_true=y_true)
            # self.stepwise_forecast_errors.loc[fc_day, "MASE"] = metrics.mean_absolute_scaled_error(y_pred=y_pred, y_true=y_true) #MAE/
            self.forecast_errors.loc[fc_day, "MaxError"] = metrics.max_error(y_pred=y_pred, y_true=y_true)





    def make_validation_set(self, steps: int=1):
        """
        Make a list of tuples that contain (train_start, train_end, test_start, test_end), to use for validation.

        Args:
            validation_config (dict): member variable containing configs for which kind of validation, start date and length (percent) of train set
            steps (int, optional): Days to step ahead in each iteration. Defaults to 1.

        Returns nothing. Sets self.validation_sets as a list of tuples (in case of single split a single tuple)
        """
        #validation_config = self.validation_config

        # #calculate number of iterations/steps
        # train_len = train_len + validation_config["test_len"]
        # len_data = end_date - train_start #No. of days it data from start_date to the end
        # num_of_iterations = len_data - len_train_test + 1 #number of steps

        #Just with pd offset and while loop (no calculation of iterations)
        end_date = self.data.index.max() #last day in dataset

        train_start = self.validation_config["start_date"] #is (should be) pd datetime
        train_end = self.get_split_index_by_prct(start_date=self.validation_config["start_date"], prct=self.validation_config["train_percent"])
        #train_len = train_end - train_start

        test_start = train_end + pd.DateOffset(1)

        if self.validation_config["type"] == "single split":
            test_end = end_date
        else:
            test_end = test_start + pd.DateOffset(self.validation_config["test_len"])


        train_test_indices = []

        match self.validation_config["type"]:
            case "expanding window":
                while test_end <= end_date:
                    step_values = (train_start, train_end, test_start, test_end)
                    train_test_indices.append(step_values)

                    #Note: keep train_start always the same
                    train_end = train_end + pd.DateOffset(steps)
                    test_start = test_start + pd.DateOffset(steps)
                    test_end = test_end + pd.DateOffset(steps)


            case "rolling window":
                while test_end <= end_date:
                    step_values = (train_start, train_end, test_start, test_end)
                    train_test_indices.append(step_values)

                    train_start = train_start + pd.DateOffset(steps)
                    train_end = train_end + pd.DateOffset(steps)
                    test_start = test_start + pd.DateOffset(steps)
                    test_end = test_end + pd.DateOffset(steps)


            case "single split":
                train_test_indices.append((train_start, train_end, test_start, test_end))

        self.validation_sets = train_test_indices



    def add_stepwise_forecasts(self):
        """
        Add stepwise_forecasts variable: dictionary containing number of keys in the length of test_len,
        with values as pandas Series of forecasted values for respective days to look ahead.
        E.g. if test_len (in rolling/expanding window)
        is 7 days, keys "1" to "7" are added, containing the predicted value for predictions
        as far into the future as the key.
        Used for plotting and comparing, how well the prediction works into the future.

        Returns nothing, sets self.stepwise_forecasts as dataframe with days ahead as
        columns ("Days ahead: [val]") and prediction_mean as value/columns. Datetime index
        """
        #TODO: Fix wrong results: days ahead: 1 should have data until the end

        #not run if single split validation (has no test_len):
        if self.validation_config["test_len"]:
            #Add 'step' (day x) ahead
            for step in range(1, self.validation_config["test_len"] + 1):
                step_forecasts = pd.Series()
                for pred in self.predictions:
                    step_forecasts = pd.concat([step_forecasts, pred["Prediction"].iloc[[step-1]]])

                step_forecasts.sort_index(inplace=True)
                if step == 1:
                    self.stepwise_forecasts = step_forecasts.to_frame()
                else:
                    self.stepwise_forecasts = pd.concat([self.stepwise_forecasts, step_forecasts], axis=1)

                #Rename last col to string of days to look ahead:
                self.stepwise_forecasts.columns = [*self.stepwise_forecasts.columns[:-1], f"Days ahead: {step}"]


    #NOTE: not in use currently
    def split_by_percentage(self, percent: float=0.33, start_date=None):
        """
        percent = percent to assign as train data.
        """

        if percent >= 1 or percent <= 0:
            raise Exception(f"Training/test data ratio: test size: {percent}) must be float smaller than 1 (should be >0.5)")

        self.split_index = int(len(self.data)*percent)
        self.train_data = self.data.iloc[: self.split_index]
        self.test_data = self.data.iloc[self.split_index:]



    def add_stepwise_errors(self, col_pred: str="count"):
        """Calculate stepwise errors for stepwise forecasts and adds it to model variable.
        That means that for x days ahead, for all predicted values, the supplied error is calculated.
        Currently Supported error metrics: MAE, MAPE, MedAE (Median absolute error),
        MaxError, RMSE, MSE
        Args:
            col_pred (str, optional): Column name of original (test) data which is forecast, to compare preiction with
        """
        #initialize empty df with structure like stepwise_forecasts (cols, indices, no content)
        forecast_steps = self.stepwise_forecasts.columns
        errors = ["ME", "MAE", "MedAE", "MAPE", "RMSE", "MASE", "MaxError"]

        self.stepwise_forecast_errors = pd.DataFrame(columns=errors, index=forecast_steps)

        #add error measurements for each forecast step
        for col in self.stepwise_forecasts.columns:
            min_date = self.stepwise_forecasts[col].first_valid_index()
            max_date = self.stepwise_forecasts[col].last_valid_index()

            y_pred = self.stepwise_forecasts.loc[min_date:max_date, col]
            y_true = self.data.loc[min_date:max_date, col_pred]

            self.stepwise_forecast_errors.loc[col, "ME"] = np.median(y_pred - y_true) #median error -- shows bias (positive or negative)
            self.stepwise_forecast_errors.loc[col, "MAE"] = metrics.mean_absolute_error(y_pred=y_pred, y_true=y_true)
            self.stepwise_forecast_errors.loc[col, "MedAE"] = metrics.median_absolute_error(y_pred=y_pred, y_true=y_true)
            self.stepwise_forecast_errors.loc[col, "MAPE"] = metrics.mean_absolute_percentage_error(y_pred=y_pred, y_true=y_true)
            self.stepwise_forecast_errors.loc[col, "MSE"] = metrics.mean_squared_error(y_pred=y_pred, y_true=y_true)
            self.stepwise_forecast_errors.loc[col, "RMSE"] = metrics.root_mean_squared_error(y_pred=y_pred, y_true=y_true)
            self.stepwise_forecast_errors.loc[col, "MaxError"] = metrics.max_error(y_pred=y_pred, y_true=y_true)

    def add_stepwise_difference(self, col_pred: str="count"):
        """Get a df with the difference between the stepwise forecasted values and
        the actual values. By default it subtracts 'count' from the daily stepwise
        forecasted values and stores them in a new df called 'stepwise_forecast_difference'.

        Args:
            col (str, optional): Column name that should be subtracted from the forecasted values.
            Defaults to "count".
        """
        y_true = self.data[col_pred].loc[self.stepwise_forecasts.index]
        self.stepwise_forecast_difference = self.stepwise_forecasts.sub(y_true, axis=0)





    # Gehört eigentlich nichth zum model, passiert ja vorher:
    # def decompose(self, ):
    #     self.decomp = seasonal_decompose(series, model='additive')
    #     print(self.decomp.trend)
    #     print(self.decomp.seasonal)
    #     print(self.decomp.resid)
    #     print(self.decomp.observed)

    def make_stationary(self, data):
        #make stationary (remove trend)
        #maybe move to corresping model that needs it?
        # I think actually has to happen before, in data processing, but
        # if splitting also happens here, than i could do it here, before the split?
        # but would be needed before for autocorrelatio plots (data exploration!)
        # --> move stationary/detrend + splitting as functions in PROCESSING.
        # then only pass train and test data set to Model.
        # df = xxx
        # return df
        pass


    def validate_expanding_window(self, data, w):
        """
        w = window size in days
        """
        #split data


        #run model against test data set wtih expanding window
        pass




    def evaluate_model(self):
        #run evaluations, to get values like MAE, MAPE etc.
        pass

    def run_MAE():
        pass

    def run_MAPE():
        pass

    def run_MSRE():
        pass




    #------------------------------------------------------------------------------------------------
    # Getters
    # MARK: Getters
    def get_validation_type(self):
        print("Validation type:", self.validation_type["type"])



    def get_split_index_by_prct(self, start_date, prct: float=0.77):
        """
        returns datetime index of split position.
        """
        df_len = len(self.data[start_date:])
        if prct <= 0 or prct > 1:
            raise ValueError("must be between 0 and 1, not 0")
        if df_len <= 2:
            raise ValueError("df must be longer than 2 rows")

        idx = int(len(self.data[start_date:]) * prct)
        split_date = self.data[start_date:].index[idx]

        print(f"Split index/date with start_date ({start_date}: \n {split_date})")

        return split_date


    #UNCLEAR: add extra plotting here or just use the functions?




    #------------------------------------------------------------------------------------------------
    # Plotters
    # MARK: Plotters
    #TODO: implement selection for days to plot ahead
    def plot_stepwise(self, plot_type: str, df: pd.DataFrame=None, comparison=True, comparison_col="count", days=None):
        """Plot the stepwise predictions or difference against actual/true values.
        i.e. plot e.g. one-day-ahead prediction against test set, two-day-ahead, etc.
        Defaults to plot stepwise_forecasts data, but can also be used for stepwise_forecast_difference
        (which is alos stepwise).
        """
        if df is None:
            df = self.stepwise_forecasts

        comparison_data = self.data[comparison_col]


        colors = iter(cm.rainbow(np.linspace(1, 0.6, len(df.columns))))

        #original_data =
        plt.figure(figsize=(14,7))

        if plot_type == "forecast difference":
            plt.axhline(y= 0, linestyle = "dashed", color="lightgrey")
            for col in df.columns:
                print("original: ", col, "\n", df[col])
                df[col] = comparison_data - df[col]
                print("comparison: ", col, "\n", comparison_data)
                print("subtracted: ", col, "\n", df[col])


        if comparison == True:
            plot_start = df.index.min() - pd.DateOffset(60) #first element of first key
            plot_end = df.index.max()#self.stepwise_forecasts.keys()[-1][-1] #last element of last key
            plt.plot(self.data[plot_start:plot_end][comparison_col], label="original data")


        for col in df.columns:
            plt.plot(df[col], label=col, color=next(colors))

        plt.title(f"Stepwise {plot_type}; Model: {self.class_name}")
        plt.legend()
        plt.show()



    def plot_stepwise_forecast_errors(self):
        #TODO: change colors to be more different.
        # TODO: maybe add error names directly to lines instead of having a legend
        print(self.stepwise_forecast_errors.columns)
        print(len(self.stepwise_forecast_errors.columns))
        colors = iter(cm.gist_ncar(np.linspace(1, 0, len(self.stepwise_forecast_errors.columns))))

        for col in self.stepwise_forecast_errors.columns:
            if col == "MSE":
                continue
            plt.plot(self.stepwise_forecast_errors[col], label=col, color=next(colors))
        plt.tight_layout()
        plt.title(f"Forecast errors; Model: {self.class_name}")
        plt.legend()
        plt.show()








#--------------------------------------------------------------------
# Individual Models:
#--------------------------------------------------------------------


#xXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# MARK: Comparison Model

# - Comparison Model
# - single value
# - naive/persistent (n-1)
# - mean
# - seasonal naive (n-7)

#xXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

class ModelComparison(Model):
    class_name = "Comparison"

    def __init__(self, data):
        super().__init__(data)
        self.predictions = None #Dataframe

        #Needed parameters:
        self.col = None #throughout all runs
        self.single_value = None #for single_value
        self.forecast_window = None #for plotting
        self.start_date = None #for mean
        self.end_date = None #for mean

        #individual comp. models (dataframes)
        self.single_value_df = None
        self.naive_df = None
        self.mean_df = None
        self.seasonal_naive_df = None


    #------------------------------------------------------------------------------------------------
    # Setters/Getters
    #------------------------------------------------------------------------------------------------

    def set_forecast_window(self, days=14):
        """This is for plotting purposes only. Since models have forecast
        window, with days-ahead plotting/stats, i want to keep this
        model with the same window.

        Args:
            days (int, optional): Das ahead forecast. FOr plotting and stats. Defaults to 14.
        """

        self.forecast_window = days


    def set_column(self, col=config.COLUMN):
        """Sets column of interest, used in all comparison models. I want to set it in a single place for
        coherence reasons. if special column needed, you can just use a single 'run_' funciton. also, later i can
        still see the column that was used.

        Args:
            col (str, optional): Column name to use as base for naive.
               Defaults to config.COLUMN, which is a variable name used in main.py for column of interest.
        """

        self.col = col



    def set_single_value(self, single_val=100):
        """Sets column of interest, used in all comparison models. I want to set it in a single place for
        coherence reasons. if special column needed, you can just use a single 'run_' funciton. also, later i can
        still see the column that was used.

        Args:
            single_val (int, optional): Integer to set as comparison.
        """

        self.single_value = single_val

    def set_dates_mean(self, start_date: str=None, end_date: str=None, col=config.COLUMN):
        """Set start and end dates for run_mean(). If not set, defaults to min/max indices of self.data

        Args:
            start_date (str): Start date for run_mean(). Date as string in format "YYYY-MM-DD"
            end_date (str): Start date for run_mean(). Date as string in format "YYYY-MM-DD"
            col (str, optional): Which column to use for min/max date. (i think this could be removed)
        """
        if not start_date:
            self.start_date = self.data[col].index.min()
        else:
            self.start_date = start_date

        if not end_date:
            self.end_date = self.data[col].index.max()
        else:
            self.end_date = end_date



    def get_error_values(self):

        self.predictions_error_vals = pd.DataFrame({
            "error_val" : ["rmse", "mape", "mae", "medae", "maxerr"]
        })

        for column in self.predictions.columns[1:]: #first is og column
                comparison_col = self.predictions.dropna(subset=[column, self.col])[column]
                original_col = self.predictions.dropna(subset=[column, self.col])[self.col]#self.col = test data
            
                rmse = metrics.root_mean_squared_error(comparison_col, original_col)
                mape = metrics.mean_absolute_percentage_error(comparison_col, original_col)
                mae = metrics.mean_absolute_error(comparison_col, original_col)
                medae = metrics.median_absolute_error(comparison_col, original_col)
                maxerr = metrics.max_error(comparison_col, original_col)
                
                self.predictions_error_vals[column] = [rmse, mape, mae, medae, maxerr]

                print(column, ":\n")
                print("RMSE",  rmse)
                print("MAPE", mape)  
                print("MAE", mae)
                print("MedAE", mae)
                print("MaxErr", maxerr)




    def print_parameters(self):
        print(f"""
            Column: {self.col}
            single val: {self.single_value}
            fc window: {self.forecast_window}
            start_date: {self.start_date}
            end_date: {self.end_date}
              """)

    #------------------------------------------------------------------------------------------------
    # Models/Composite fct
    #------------------------------------------------------------------------------------------------


    def model_run(self):
        """Runs all comparison models (single value, naive, mean, seasonal naive) and creates a df called 'result',
        which has models as column names.
        You're not supposed to pass arguemnts with this function, use setters for that.

        """
        #initiate 'result'
        self.predictions = self.data[[config.COLUMN]]

        #Run all comparison models
        self.predictions["single_value"] = self.run_single_value(single_value=self.single_value, model_run=True)
        self.predictions["naive"] = self.run_naive(col=self.col, model_run=True)
        self.predictions["mean"] = self.run_mean(col=self.col, model_run=True)
        self.predictions["seasonal_naive"] = self.run_seasonal_naive(col=self.col, model_run=True)

        #TODO: make column with difference of pred type and actual value


    def run_single_value(self, single_value: int=100, model_run=False):
        """Sets ('runs calculation') for single value comparison model

        Args:
            single_value (int, optional): Value to be set. Defaults to 120.
            model_run (bool): If true, then function is ran inside of self.model_run(), so it
            returns the new column.
        """
        self.data["pred_single_val"] = single_value

        if model_run:
            return self.data["pred_single_val"]



    def run_naive(self, col=config.COLUMN, model_run=False):
        """Creates Naive (also called Persistance/persistent) forecast. that is to just take the last
        value available value to forecast for the next one. So if i know todays value X, then i say tomorrow its also x.
        Since we work with train/test data, i dont need to implement rolling/expanding window to calulcate this, it wouldnt
        make a difference

        Args:
            col (str, optional): Column name to use as base for naive.
               Defaults to config.COLUMN, which is a variable name used in main.py for column of interest.
            model_run (bool): If true, then function is ran inside of self.model_run(), so it
            returns the new column.

        """
        if col == None:
            col = self.col

        self.data["naive"] = self.data[col].shift(1)

        if model_run:
            return self.data["naive"]



    def run_mean(self, start_date: str=None, end_date: str=None, col=config.COLUMN, model_run=False):
        """Calculate the mean of historical data and use that as comparison.
        This would probably slightly profit from using rolling window, to only use more proximate data.
        But for simplicity reasons, i dont implement this.
        Caluclates mean for selected column, based on start_date to end_date period (or min/max dates if not passed).
        Sets this value for whole dataset, not just the passed period.

        Args:
            start_date (str, optional): Type start date for mean calculation as "YYYY-MM-DD". Defaults to min date of 'col'.
            end_date (str, optional): Type start date for mean calculation as "YYYY-MM-DD". Defaults to min date of 'col'.
            col (str, optional): Column name to use as base for naive.
                Defaults to config.COLUMN, which is a variable name used in main.py for column of interest.
            model_run (bool): If true, then function is ran inside of self.model_run(), so it
            returns the new column.

        """
        if col == None:
            col = self.col

        if not start_date:
            start_date = self.data[col].index.min()
        elif not end_date:
            end_date = self.data[col].index.max()

        data_mean = self.data.loc[start_date:end_date, col].mean()

        self.data["pred_mean"] = data_mean

        if model_run:
            return self.data["pred_mean"]


    def run_seasonal_naive(self, col=config.COLUMN, n=7, model_run=False):
        """Basically the same as naive (run_naive member function), but instead of taking the last value, take n-x values before.
        Since we have weekly seasonality, always take value from 7 days ago, so for monday, take last mondays value etc.

        Args:
            col (str, optional): Column name to use as base for naive.
                Defaults to config.COLUMN, which is a variable name used in main.py for column of interest.
            n (int, optional): How many rows/days before to use. For us it should be one week ago (7 days). Defaults to 7.
            model_run (bool): If true, then function is ran inside of self.model_run(), so it
            returns the new column.
            """
        if col == None:
            col = self.col

        self.data["seasonal_naive"] = self.data[col].shift(n)

        if model_run:
            return self.data["seasonal_naive"]





#xXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# ARIMA
# MARK: ARIMA

#xXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

class ModelArima(Model):
    class_name = "Arima"

    def __init__(self, data): #TODO: maybe add config, but more sense in base class imo
        super().__init__(data)
        self.p = None
        self.q = None
        self.d = None

    #------------------------------------------------------------------------------------------------
    # Setters
    #------------------------------------------------------------------------------------------------

    def set_model_parameters(self, p: int=1, d: int=1, q: int=1):
        self.p = p
        self.d = d
        self.q = q



    #composite function:
    def model_run(self, col: str="count", print_fit_summary=True, last_only=True, days=None):
        """Composite function that combines make_model, fit(), print_fit_summary(), predict(),
        add_stepwise_forecasts()

        Args:
            col (str, optional): column to make model and run prediction for.
            Defaults to "count".
            print_fit_summary (bool, optional): If true, prints the summary for the fit().
            Defaults to True
            last_only (bool, optional): If true prints only summary for last fit(). Otherwise prints
            summary for every fit (of rolling/expanding window).
            Only relevant if print_fit_summary argument is true. defaults to True.
            days (int, optional): Manually set days to look ahead (steps). Normally supplied via
            validation_config by setter functions for rolling/expanding window or single split.
            Defaults to None, which will then use abovementioned value.
        """

        self.make_model(col=col)
        self.fit()
        if print_fit_summary:
            self.print_fit_summary(last_only=last_only)
        self.get_prediction(days=days)

        #Get stepwise values:
        self.add_stepwise_forecasts()
        self.add_stepwise_errors(col_pred=col)
        self.add_stepwise_difference(col_pred=col)




    def make_model(self, col: str):
        """
        create model with trainign data + (hyper)parameters

        Parameters
        ----------
        col : string
            column (target) to use for for univariate forecasting
        """
        #Important!
        self.models = []

        #TODO: set up split AND validation
        # i think for validation, best option to have a list of lists with train_start, train_end, test_start, test_end
        # days (datetime), which i can cycle here (make_model, fit, print_fit_summary, predict), which is just
        # inplace filtering of df, so i dont have to store multiple dfs!
        series = self.data[col]
        #TODO: !use SARIMAX instead of ARIMA!

        for train_set in self.validation_sets:
            #Add exogenous, check for enforce_stationarity, enforce_invertibility
            self.models.append(ARIMA(
                endog=series[train_set[0] : train_set[1]],
                order=(self.p, self.d, self.q)))


    def fit(self):
        #Important!
        self.model_fits = []

        for model in self.models:
            self.model_fits.append(model.fit())


    def print_fit_summary(self, last_only=True):
        if last_only:
            print(self.model_fits[-1].summary())
        else:
            for fit in self.model_fits:
                print(fit.summary())

    def predict(self, days=None):
        """Predict x days ahead, where x == 'days'

        Args:
            days (_type_, optional): Days to predict ahead. Defaults to None, then
            days will be loaded from validation_config["test_len"].
        """
        # generate forecast for x time
        # (see base class)

        #Important!
        self.predictions = []

        if days == None:
            days = self.validation_config["test_len"]

        for fit in self.model_fits:
            self.predictions.append(fit
                .get_forecast(steps=days)
                .rename(columns={"predicted_mean":"Prediction"})
            )

    def get_prediction(self, days=None, alpha: int=None):

        if self.alpha == None:
            alpha = 0.05
        elif self.alpha != None:
            alpha = self.alpha
        
        self.predictions = []

        #iterate over both model_fits and validation_sets:
        for fit, validation_set in zip(self.model_fits, self.validation_sets): #'fit' is the fitted model
            test_start = validation_set[2]
            test_end = validation_set[3]

            self.predictions.append(fit
                .get_prediction(start=test_start, end=test_end, dynamic=True)
                .summary_frame(alpha=alpha)
                .rename(columns={"mean":"Prediction"})
            )





#xXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# SARIMAX
# MARK: SARIMAX

#xXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


class ModelSarimax(Model):
    class_name = "Arima"


    def __init__(self, data): #TODO: maybe add config, but more sense in base class imo
        super().__init__(data)
        #ARIMA part:
        self.p = None
        self.q = None
        self.d = None
        #Seasonal part:
        self.P = None
        self.Q = None
        self.D = None
        self.m = None
        #eXogenous part:
        self.exog_cols = None

    #------------------------------------------------------------------------------------------------
    # MARK: Setters
    #------------------------------------------------------------------------------------------------

    def set_model_parameters(
            self, p: int=1, d: int=1, q: int=1,
            P: int=1, D: int=1, Q: int=1, m: int=7):

        self.p = p
        self.d = d
        self.q = q

        self.P = P
        self.D = D
        self.Q = Q
        self.m = m


    def set_exogenous_vars(self, exog_cols: list):
        """Set columns to use for exogenous variables with SARIMAX.

        Args:
            exog_cols (list): List of strings containing column names for exog vars (in self.data)

        Raises:
            ValueError: If exog_cols list is empty. Doesnt check for data type.
            ValueError: If a column from exog_cols is not present in 'df'
        """
        #TODO: if no exog_cols supplied/argument is empty, set to None, and then do a check in model run, to run it w/o exog vars (SARIMA without X)
        #Check df/exog_cols input for validity
        if len(exog_cols) == 0:
            raise ValueError("Need to pass col name present in 'df' to exog_cols")
        else:
            for col in exog_cols:
                if col not in self.data.columns:
                    raise ValueError(f"{col} not present df's columns: {self.data.columns}")

        self.exog_cols = exog_cols


    #------------------------------------------------------------------------------------------------
    # MARK: General
    #------------------------------------------------------------------------------------------------


    #composite function:
    def model_run(self, pred_col: str="count", print_fit_summary=True, last_only=True, days=None): #exog: list=None,
        """Composite function that combines make_model, fit(), print_fit_summary(), predict(),
        add_stepwise_forecasts()

        Args:
            pred_col (str, optional): column to make model and run prediction for.
            Defaults to "count".
            print_fit_summary (bool, optional): If true, prints the summary for the fit().
            Defaults to True
            last_only (bool, optional): If true prints only summary for last fit(). Otherwise prints
            summary for every fit (of rolling/expanding window).
            Only relevant if print_fit_summary argument is true. defaults to True.
            days (int, optional): Manually set days to look ahead (steps). Normally supplied via
            validation_config by setter functions for rolling/expanding window or single split.
            Defaults to None, which will then use abovementioned value.
        """

        self.make_model(pred_col=pred_col)
        self.fit()
        if print_fit_summary:
            self.print_fit_summary(last_only=last_only)
        self.get_prediction(days=days)

        #Get stepwise values:
        self.add_stepwise_forecasts()
        self.add_stepwise_errors(col_pred=pred_col)
        self.add_stepwise_difference(col_pred=pred_col)



    def make_model(self, pred_col: str):
        """
        create model with trainign data + (hyper)parameters

        Parameters
        ----------
        pred_col : string
            columns (target) to use for for univariate forecasting
        """
        #Important!
        self.models = []

        #TODO: set up split AND validation
        # i think for validation, best option to have a list of lists with train_start, train_end, test_start, test_end
        # days (datetime), which i can cycle here (make_model, fit, print_fit_summary, predict), which is just
        # inplace filtering of df, so i dont have to store multiple dfs!
        series = self.data[pred_col]
        #TODO: !use SARIMAX instead of ARIMA!


        if self.exog_cols == None:
            for train_set in self.validation_sets:
                self.models.append(SARIMAX(
                    endog=series[train_set[0] : train_set[1]],
                    order=(self.p, self.d, self.q),
                    seasonal_order=(self.P, self.D, self.Q, self.m)))

        elif self.exog_cols != None:
            # exogenous = self.data[exog]
            for train_set in self.validation_sets:
                self.models.append(SARIMAX(
                    endog=series[train_set[0] : train_set[1]],
                    exog=self.data.loc[train_set[0]:train_set[1], self.exog_cols],
                    # exog=exogenous[train_set[0] : train_set[1]],
                    order=(self.p, self.d, self.q),
                    seasonal_order=(self.P, self.D, self.Q, self.m)))



    def fit(self):
        #Important!
        self.model_fits = []

        for model in self.models:
            self.model_fits.append(model.fit())



    def predict(self, days=None):
        """Predict x days ahead, where x == 'days'

        Args:
            days (_type_, optional): Days to predict ahead. Defaults to None, then
            days will be loaded from validation_config["test_len"].
        """
        # generate forecast for x days ahead
        # (see base class)

        #Important!
        self.predictions = []

        if days == None:
            days = self.validation_config["test_len"]

        for fit in self.model_fits:
            self.predictions.append(fit.get_forecast(steps=days))



    #------------------------------------------------------------------------------------------------
    # MARK: Getters/Print
    #------------------------------------------------------------------------------------------------


    def get_prediction(self, days=None, alpha: int=None):

        if self.alpha == None:
            alpha = 0.05
        elif self.alpha != None:
            alpha = self.alpha

        self.predictions = []

        #iterate over both model_fits and validation_sets:
        for fit, validation_set in zip(self.model_fits, self.validation_sets): #'fit' is the fitted model
            test_start = validation_set[2]
            test_end = validation_set[3]

            exog_prediction = self.data.loc[test_start:test_end, self.exog_cols]

            self.predictions.append(fit
                .get_prediction(start=test_start, end=test_end, exog=exog_prediction, dynamic=True)
                .summary_frame(alpha=alpha)
                .rename(columns={"mean":"Prediction"})
            )



    def print_fit_summary(self, last_only=True):
        if last_only:
            print(self.model_fits[-1].summary())
        else:
            for fit in self.model_fits:
                print(fit.summary())



    def print_params(self):
        print(f"p,d,q: {self.p}, {self.d}, {self.q}\nP,D,Q,m: {self.P},{self.D},{self.Q}{self.m}\nExogenous columns: {self.exog_cols}")






#xXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# LSTM
# MARK: LSTM

#xXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

class ModelLSTM(Model):

    def __init__(self, data): #TODO: maybe add config, but more sense in base class imo
        super().__init__(data)

        #LSTM-specific variable inits
        self.model = None

        self.model_params = {
            "prediction_column" : None,

            "memory_cells" : None,
            "epochs" : None,
            "batch_size" : None,
            "dropout" : None,
            "pi_iterations" : None,
            "optimizer" : None,
            "loss" : None,

            "activation_fct" : None,

            "lower_limit" : None,
            "upper_limit" : None,
            "exog_cols" : None
        } #rename; prob better in base class. | 




    #------------------------------------------------------------------------------------------------
    # MARK: Setters
    #------------------------------------------------------------------------------------------------





    def set_model_parameters(
            self, 
            inner_window: int=365,
            memory_cells: int=64,
            epochs: int=20,
            batch_size: int=32,
            dropout: float=0.5,
            pi_iterations: int=100, #how often to run, to calculate prediction intervals
            optimizer: str="adam",
            loss: str="mae",
            activation_fct: str="relu",
            lower_limit: float=2.5,
            upper_limit: float=97.5
            ):
        
        self.model_params["inner_window"] = inner_window
        self.model_params["memory_cells"] = memory_cells
        self.model_params["epochs"] = epochs
        self.model_params["batch_size"] = batch_size
        self.model_params["dropout"] = dropout
        self.model_params["pi_iterations"] = pi_iterations
        self.model_params["optimizer"] = optimizer
        self.model_params["loss"] = loss
        self.model_params["activation_fct"] = activation_fct
        self.model_params["lower_limit"] = lower_limit
        self.model_params["upper_limit"] = upper_limit




    #------------------------------------------------------------------------------------------------
    # MARK: Getters/Print
    #------------------------------------------------------------------------------------------------




    #------------------------------------------------------------------------------------------------
    # MARK: General
    # Main functions
    #------------------------------------------------------------------------------------------------


    #composite function:
    # Main function, that combines all other functions in right order for simple running of model + prediction
    def model_run(self, print_fit_summary=True, last_only=True): 
        # params here should only be for output to show (df, results, plot etc), no change in model run
        # expanding/rolling window needs to be set already!

        self.create_result_dir()

        #TODO: Write docstring
        all_windows_start = time.time()
        self.stats["windows_num"] = len(self.validation_sets)
        
        #initialize model for first time (needs X_train for shape, so need to run other fcts once)
        self.get_start_end_days(self.validation_sets[0])
        self.get_data()
        self.scale_data()
        self.get_training_test_set()
        self.build_model()

        for i, window in enumerate(self.validation_sets):
            print(f"\n\nWindow {i}/{len(self.validation_sets)}")
            
            print("Loading weights from: ", self.dir_name, " ", self.file_path)
            self.model.load_weights(self.file_path+"/initial.weights.h5")
            
            process = psutil.Process(os.getpid())
            print(f"Memory: {process.memory_info().rss/1024**2:.2f} MB")
            print(f"GC Objects: {len(gc.get_objects())}")

            window_start = time.time()
            
            self.reset_states()

            self.get_start_end_days(window)
            self.get_data() #get X_raw, y_raw data for every iteration in the sliding7expanding window (correct columns)
            self.scale_data()  #set scaler + transform to scaled data
            self.get_training_test_set()
            self.fit_model() #train model 
            self.get_prediction_intervalls() #iterations for prediction intervall
            self.add_to_results()
            
            window_end = time.time()
            print(f"Window {i} executed in {window_end - window_start}s\n")

        self.stats["total_duration"] = window_end - all_windows_start #TODO: move to setter?
        print(f"\nTotal time for all windows {window_end - all_windows_start}s")

        self.add_stepwise_difference()
        self.get_stepwise_errors()

        self.save_results()




    @timer_func
    def scale_data(self):
        """
        Fits and transforms X/y train and test data with StandardScaler().
        Fitted on train data, which is used to transform both train and test data.
        
        Args:
            X_train_raw (np.ndarray): Numpy array of shape [n, m] where n=exog cols + prediction col, m=number of train days (rows)
            y_train_raw (np.ndarray): Numpy array of shape [1, m] with prediction columnn and m=number of train days (rows)
            X_test_raw (np.ndarray): Numpy array of shape [n, m] where n=exog cols + prediction col, m=number of test days (rows)
            y_test_raw (np.ndarray): Numpy array of shape [1, m] with prediction columnn and m=number of test days (rows)

        Returns:
            np.ndarray: Numpy arrays for each input but scaled with StandardScaler.
        """
        #TODO: write docstring

        # Preprocessing stages
        # This is done for the whole raw dataset, and then later applied to each iteration of the 
        # rolling/expanding window(s)
        #Uses same scaler, fitted on train data to both test and training of X/y respectively.

        self.scaler_X = StandardScaler().fit(self.X_train_raw)
        self.scaler_y = StandardScaler().fit(self.y_train_raw)


        self.X_train_scaled = self.scaler_X.transform(self.X_train_raw)
        self.y_train_scaled = self.scaler_y.transform(self.y_train_raw)
        self.X_test_scaled = self.scaler_X.transform(self.X_test_raw)
        self.y_test_scaled = self.scaler_y.transform(self.y_test_raw)

        # self.X_train_scaled = self.X_train_scaled.reshape(1, self.X_train_scaled.shape[0], self.X_train_scaled.shape[1])
        # self.y_train_scaled = self.y_train_scaled.reshape(1, self.y_train_scaled.shape[0]) #TODO: shape[0] should be equal to self.forecast_days
        # self.X_test_scaled = self.X_test_scaled.reshape(1, self.X_test_scaled.shape[0], self.X_test_scaled.shape[1])
        # self.y_test_scaled = self.y_test_scaled.reshape(1, self.y_test_scaled.shape[0])



    @timer_func
    def get_training_test_set(self):
        #splits data of this window into many samples (basically nested rolling window for this window in buld_model)
        # into X_train with 3D shape of (samples, step_size=l)
        #This creates x training sets from X_train_scaled by iteration.


        # Create sliding window for our data (60days)
        sliding_size = self.model_params["inner_window"] #TODO: maybe rename?

        # Prep training features
        self.X_train, self.y_train = [], []                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
        for i in range(0, self.X_train_raw.shape[0] - sliding_size - self.forecast_days): #(sliding_size, training_data_len):
            self.X_train.append(self.X_train_scaled[i : i + sliding_size, :])
            self.y_train.append(self.y_train_scaled[i + sliding_size : i + sliding_size + self.forecast_days, 0])

        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)

        #Prep test data
        self.X_test = self.X_train_scaled[-sliding_size : , :]
        self.y_test = self.y_test_scaled

        self.X_test = np.reshape(self.X_test, (1, self.X_test.shape[0], self.X_test.shape[1]))
        self.y_test = np.reshape(self.y_test, (self.y_test.shape[1], self.y_test.shape[0])) #change (x, y) to (y, x) where y_test_scaled = (fc_window, 1)

        
    @timer_func
    def build_model(self):
        #TODO: description


        inputs = Input(shape=(self.X_train.shape[1], self.X_train.shape[2])) #TODO: was 1, 2
        # inputs = Input(shape=(self.X_train_scaled.shape[1], self.X_train_scaled.shape[2])) #TODO: was 1, 2
        x = layers.LSTM(self.model_params["memory_cells"], return_sequences=True, unroll=False)(inputs) #TODO: unroll=True (was before, testing if fixes memory leak)
        x = layers.LSTM(self.model_params["memory_cells"], return_sequences=False)(x)
        x = layers.Dense(2*self.model_params["memory_cells"], activation=self.model_params["activation_fct"])(x)
        #MC dropout
        x = layers.Dropout(0.5)(x, training=True)

        #output layer: x neurons for x days forecasting
        outputs = layers.Dense(self.forecast_days)(x)

        #make model
        self.model = keras.Model(inputs, outputs)
        self.model.compile(
            optimizer=self.model_params["optimizer"], 
            loss=self.model_params["loss"], 
            metrics=[keras.metrics.RootMeanSquaredError()]
            #jit_compile=True
        )
        self.model.save_weights(self.file_path+"/initial.weights.h5")



    
    @timer_func
    def fit_model(self):
        x_train = self.X_train.astype('float32')
        y_train = self.y_train.astype('float32')

        self.model.fit(
            x=self.X_train, 
            y=self.y_train, 
            epochs=self.model_params["epochs"], 
            batch_size=self.model_params["batch_size"], 
            verbose=0
        )


        # train_data = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train))
    
        # # Cache in memory and prefetch the next batch while the CPU is training
        # train_data = train_data.cache().shuffle(1000).batch(self.model_params["batch_size"]).prefetch(tf.data.AUTOTUNE)

        # self.model.fit(
        #     train_data, 
        #     epochs=self.model_params["epochs"], 
        #     verbose=0
        # )
    

    @timer_func
    def get_prediction_intervalls(self):      
        # Repeat X_test 'n' times along a new axis
        # If X_test is (samples, time, features), tiled becomes (iterations, samples, time, features)
        n_iter = self.model_params["pi_iterations"]
        
        # Reshape X_test to (n_iter * samples, time, features)
        # This processes all MC samples in one forward pass
        x_tiled = np.tile(self.X_test, (n_iter, 1, 1))
        
        # Get all predictions in one batch
        # preds = self.model.predict(x_tiled, batch_size=self.model_params["batch_size"], verbose=0)
        #Change to keep training set to true, which keeps the dropout layer necessary for MC (pred intervalls)
        preds = self.model(x_tiled, training=True, verbose=0).numpy()

        # Reshape back to (iterations, samples, forecast_days)
        preds = preds.reshape(n_iter, self.X_test.shape[0], self.forecast_days)
        
        # Inverse transform (optimized to handle the shape)
        #TODO: rename (like all_predictions_monte_carlo or something?) 
        self.all_predictions = np.array([
            self.scaler_y.inverse_transform(p).astype("float32") for p in preds
        ])





    @timer_func
    def reset_states(self):
        #resets all self values used in model_run
        if hasattr(self, 'model') and self.model is not None:
                # This helps break internal references
                #del self.model #TODO: deleted to test, if building model once helps with memory leak
                pass

        #resets all states generated by tensorflow-keras
        #keras.backend.clear_session()

        gc.collect()
        gc.collect()
        # tf.compat.v1.reset_default_graph() # TF graph isn't same as Keras graph


        self.train_start = None
        self.train_end = None
        self.test_start = None
        self.test_end = None
        self.forecast_days = None

        self.X_train_raw = None
        self.y_train_raw = None
        self.X_test_raw = None
        self.y_test_raw = None

        self.scaler_X = None
        self.scaler_y = None

        self.X_train_scaled = None
        self.y_train_scaled = None
        self.X_test_scaled = None
        self.y_test_scaled = None   

        # self.model = None

        self.forecast_mean = None
        self.forecast_lower = None
        self.forecast_upper = None
        #dont reset self.predictionss




#xXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# Prophet
# MARK: Prophet

#xXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
class ModelProphet(Model):
    def __init__(self, data): #TODO: maybe add config, but more sense in base class imo
        super().__init__(data)
        #prophet-specific variable inits
        self.model = None
        # self.model_params = None #rename; prob better in base class.
        self.model_params = {
            "prediction_column" : None,
            "exog_cols" : None,

            "lower_limit" : None,
            "upper_limit" : None
        } #rename; prob better in base class.


    def model_run(self):
        #create model with (hyper)parameters
        #params are model parameters
        # days are days to predict

        self.create_result_dir()

        all_windows_start = time.time()
        self.stats["windows_num"] = len(self.validation_sets)


        for i, window in enumerate(self.validation_sets):
            print(f"Window {i}/{len(self.validation_sets)}")
            window_start = time.time()
            
            # self.reset_states()

            self.get_start_end_days(window)
            self.get_data() #get X_raw, y_raw data for every iteration in the sliding7expanding window (correct columns)
            self.build_model()
            self.fit_model() #train model 
            self.predict()
            #TODO: make add_to_results()
            self.add_to_results()
            window_end = time.time()
            
            print(f"Window {i} executed in {window_end - window_start}s\n")

        all_windows_end = time.time()
        self.stats["run_duration"] = all_windows_end - all_windows_start #TODO: move to setter?
        print(f"\nTotal time for all windows {all_windows_end - all_windows_start}s")
        #TODO: make functions:
        self.add_stepwise_difference()
        self.get_stepwise_errors()
        self.save_results()
  





    def set_model_parameters(self, lower_limit: float=2.5, upper_limit: float=97.5):

        self.model_params["lower_limit"] = lower_limit
        self.model_params["upper_limit"] = upper_limit


    def print_params(self):
        #TODO: probably better in Model class!
        #copied from ModelLSTM:

        for key, value in self.model_params.items():
            print(key, ": ", value)
        

    def get_params_df(self):
        #TODO: probably better in Model class!
        #copied from ModelLSTM:

        """
        Returns params as df
        """
        params_df = pd.DataFrame([self.model_params]) #bracket to keep everything in one row

        return params_df



    @timer_func
    def get_start_end_days(self, window):
        #TODO: probably better in Model class! --> No, i dont think, because at least for 
        # Prophet i think should be different than lstm?


        self.train_start = window[0]
        self.train_end = window[1]
        self.test_start = window[2]
        self.test_end = window[3] + pd.Timedelta(days=1)

        self.forecast_days = (window[3] - window[2]).days #because window[2] and window[3] should be both included as days

    @timer_func
    def get_data(self):
        #TODO: probably better in Model class!
        #TODO: Description of get_data


        if not self.model_params["prediction_column"]:
            raise ValueError(f"Missing the column to predict ({self.model_params['predcition_column']})")
        
        columns = list(set([self.model_params["prediction_column"], *self.model_params["exog_cols"]]))#star* to unpack list, pred_cols is str only.

        # self.prophet_train = (self.data.loc[self.train_start : self.train_end, columns]
        self.prophet_train = (self.data.loc[self.train_start : self.train_end, columns]
                              #.values
                              .reset_index()
                              .rename(columns={"date":"ds", self.model_params["prediction_column"]:"y"})
                              )

      
    def build_model(self):
        interval_width = (self.model_params["upper_limit"] - self.model_params["lower_limit"]) / 100

        self.model = Prophet(weekly_seasonality=True, interval_width=interval_width)

        self.model.add_country_holidays(country_name="Austria")


        pass

    def fit_model(self):
        self.model.fit(self.prophet_train) #rename? But X_train doesnt really make sense imo

    def predict(self):
        #TODO: naming is not really descriptive!
        #initialize predictions-dataframe -- contains training + prediciton periods!
        self.forecast_df = self.model.make_future_dataframe(periods=self.forecast_days, freq="D")

        #predict
        self.forecast = self.model.predict(self.forecast_df)






# %%
