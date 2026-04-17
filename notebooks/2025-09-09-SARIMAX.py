#%%
import pandas as pd
import numpy as np
import sys
from pathlib import Path

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import statsmodels
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

import sklearn.metrics as metrics #error metrics (mae, mape etc)
from pmdarima import auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import MSTL #multiple seasonal decompose
from statsmodels.tools.eval_measures import rmse
import sklearn.metrics as metrics #error metrics (mae, mape etc)

from sklearn.preprocessing import MinMaxScaler

import importlib
import holidays
#---------------------------------------
#column to model !
COLUMN = "use_transfused" #column of interest (Y)
#---------------------------------------


#---------------------------------------
#Fixed values:

START_DATE = "2021-05-01"
END_DATE = "2025-05-01"
plt.rcParams['figure.dpi'] = 200



# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))
from src import viz

#from src import load


#%%
#Load cleaned data
df_original = pd.read_csv("data/03_transformed/output_transformed.csv")#../data/03_transformed/output_transformed.csv")
df_original = df_original.set_index("date")
df_original.index = pd.to_datetime(df_original.index)


#%%
# Add additional temporal features (workday, holiday, day of year, day of week)

#Add workday info
import holidays

vie_holidays = holidays.country_holidays('Austria', subdiv='W')


df_original["is_workday"] = df_original.index.to_series().apply(lambda x: vie_holidays.is_working_day(x))
#df_original = df_original.diff().dropna()
df_original["workday_enc"] = df_original["is_workday"].astype(int)
#alternative ways
#df_original["workday"] = df_original.where(df_original["is_workday"], 1, 0) 
#df_original["workday"] = df_original.apply(lambda workday: 1 if workday else 0)

#holiday encoding:
df_original["holiday"] = pd.Series(df_original.index).apply(lambda x: vie_holidays.get(x)).values
unique_holidays = df_original["holiday"].dropna().unique()
holiday_map = pd.DataFrame({
    "holiday" : unique_holidays,
    "holiday_enc" : range(1, len(unique_holidays)+1)
})
holiday_map = pd.concat([
    pd.DataFrame({"holiday": [np.nan], "holiday_enc": [0]}), 
    holiday_map],
    ignore_index=True
)
df_original = pd.merge(df_original.reset_index(), holiday_map, how="left", on="holiday").set_index("date")

#Add day of the week, day of the year, year columns
df_original["day_of_week"] = df_original.index.dayofweek
df_original["day_of_year"] = df_original.index.dayofyear
df_original["year"] = df_original.index.year

df_original.info()

#%%
# look at trends/seasonality 
result = seasonal_decompose(df_original.loc[START_DATE:END_DATE,COLUMN], model="additive", period=7)
result.plot()
plt.title(f"{COLUMN} seasonal decompose")
plt.show()

result.seasonal.plot(figsize=(16,8))
plt.title(f"{COLUMN} seasonal part")
plt.show()

result.resid.plot(figsize=(16,8))
plt.title(f"{COLUMN} residual part")
plt.show()

result.trend.plot(figsize=(16,8))
plt.title(f"{COLUMN} trend part")
plt.show()

# multiple seasonality:
multiple_seasonality = viz.multiple_decompose(col=COLUMN, df=df_original.loc[START_DATE:END_DATE], periods=[7, 365])


# importlib.reload(viz)
viz.seasonal_plot(df_original.loc[START_DATE:END_DATE], plot_type="daily", col_name=COLUMN)
viz.seasonal_plot(df_original.loc[START_DATE:END_DATE], plot_type="weekly", col_name=COLUMN)

plt.plot(df_original.loc[START_DATE:END_DATE, [COLUMN, "use_transfused", "use_discarded", "use_expired"]], linewidth=0.5, label=[COLUMN, "use_transfused", "use_discarded", "use_expired"])
plt.legend()
plt.show()

viz.seasonal_plot(df_original.loc[START_DATE:END_DATE], plot_type="daily", col_name="use_transfused")
viz.seasonal_plot(df_original.loc[START_DATE:END_DATE], plot_type="weekly", col_name="use_transfused")

viz.seasonal_plot(df_original.loc[START_DATE:END_DATE], plot_type="daily", col_name="use_discarded")
viz.seasonal_plot(df_original.loc[START_DATE:END_DATE], plot_type="weekly", col_name="use_discarded")

viz.seasonal_plot(df_original.loc[START_DATE:END_DATE], plot_type="daily", col_name="use_expired")
viz.seasonal_plot(df_original.loc[START_DATE:END_DATE], plot_type="weekly", col_name="use_expired")

# %%
#Setup



# Split data
df = df_original[START_DATE:END_DATE]

split_point = int(len(df) * 0.8) 
train_df = df.iloc[:split_point]
test_df = df.iloc[split_point:]



# ---------------------------------------------------------#
#                    COMPARISON MODELS                     #
# ---------------------------------------------------------#
# i didnt implement rolling forecast, so just do it 
# directly in df
plot_start_date = "2025-01-01"
plt.plot(df.loc[plot_start_date:, COLUMN], linewidth = 1.5, label=COLUMN, color="dimgrey")

# SINGLE VALUE
single_val_df = df
single_val_df["Prediction"] = 100 #took a number that covers most of the values but not the peaks (if that makes sense)

plt.plot(single_val_df.loc["2025-01-01":, ["Prediction"]], linewidth = 0.5, label="Single value", color="red")

print("\nSINGLE VALUE FORECAST")
print("RMSE",   metrics.root_mean_squared_error(single_val_df[COLUMN], single_val_df["Prediction"])) 
print("MAPE",   metrics.mean_absolute_percentage_error(single_val_df[COLUMN], single_val_df["Prediction"])) 
print("MAE",    metrics.mean_absolute_error(single_val_df[COLUMN], single_val_df["Prediction"])) 
print("MedAE",  metrics.median_absolute_error(single_val_df[COLUMN], single_val_df["Prediction"]))
print("MaxErr", metrics.max_error(single_val_df[COLUMN], single_val_df["Prediction"]))


#NAIVE/PERSISTANCE (n-1)
naive_df = df
naive_df["Prediction"] = naive_df[COLUMN].shift(1)
naive_df = naive_df.dropna(subset=["Prediction", COLUMN])

plt.plot(naive_df.loc["2025-01-01":, ["Prediction"]], linewidth = 0.5, label="Naive", color="royalblue")

print("\nNAIVE FORECAST")
print("RMSE",   metrics.root_mean_squared_error(naive_df[COLUMN], naive_df["Prediction"])) 
print("MAPE",   metrics.mean_absolute_percentage_error(naive_df[COLUMN], naive_df["Prediction"])) 
print("MAE",    metrics.mean_absolute_error(naive_df[COLUMN], naive_df["Prediction"])) 
print("MedAE",  metrics.median_absolute_error(naive_df[COLUMN], naive_df["Prediction"]))
print("MaxErr", metrics.max_error(naive_df[COLUMN], naive_df["Prediction"]))


plt.title("Comparison Forecasting Methods I")
plt.tight_layout()
plt.legend()
plt.show()



plt.plot(df.loc[plot_start_date:, COLUMN], linewidth = 1.5, label=COLUMN, color="dimgrey")

# MEAN
single_val_df = df
single_val_df["Prediction"] = single_val_df[COLUMN].mean() #took a number that covers most of the values but not the peaks (if that makes sense)

plt.plot(single_val_df.loc["2025-01-01":, ["Prediction"]], linewidth = 0.5, label="Mean", color="orange")

print("\nMEAN FORECAST")
print("RMSE",   metrics.root_mean_squared_error(single_val_df[COLUMN], single_val_df["Prediction"])) 
print("MAPE",   metrics.mean_absolute_percentage_error(single_val_df[COLUMN], single_val_df["Prediction"])) 
print("MAE",    metrics.mean_absolute_error(single_val_df[COLUMN], single_val_df["Prediction"])) 
print("MedAE",  metrics.median_absolute_error(single_val_df[COLUMN], single_val_df["Prediction"]))
print("MaxErr", metrics.max_error(single_val_df[COLUMN], single_val_df["Prediction"]))


# SEASONAL NAIVE (equal to persistance/naive, but shift by one period == 1 Week = 7days)
# (n-7)
seas_naive_df = df
seas_naive_df["Prediction"] = seas_naive_df[COLUMN].shift(7)
seas_naive_df = seas_naive_df.dropna(subset=["Prediction", COLUMN])

plt.plot(seas_naive_df.loc["2025-01-01":, ["Prediction"]], linewidth = 0.5,label="Seasonal Naive", color="darkgreen")

print("\nSEASONAL NAIVE FORECAST")
print("RMSE",   metrics.root_mean_squared_error(seas_naive_df[COLUMN], seas_naive_df["Prediction"])) 
print("MAPE",   metrics.mean_absolute_percentage_error(seas_naive_df[COLUMN], seas_naive_df["Prediction"])) 
print("MAE",    metrics.mean_absolute_error(seas_naive_df[COLUMN], seas_naive_df["Prediction"])) 
print("MedAE",  metrics.median_absolute_error(seas_naive_df[COLUMN], seas_naive_df["Prediction"]))
print("MaxErr", metrics.max_error(seas_naive_df[COLUMN], seas_naive_df["Prediction"]))


plt.title("Comparison Forecasting Methods II")
plt.tight_layout()
plt.legend()
plt.show()


# ---------------------------------------------------------#
#%%              SARIMAX FROM data_model.py                #
# ---------------------------------------------------------#
# #without exogenous (atm)
# from src import model
# from src import config
# from src import data_model

# importlib.reload(model)

# df = data_model.Data(data=df_original.loc[START_DATE:END_DATE, [COLUMN]]) #keep COLUMMNN inside braces to keep as df (instead of series)
# sarima = model.ModelSarima(df)
# # Test runs (it works as expected)
# # arima.set_validation_expanding_window(train_percent=0.992, test_len=7, start_date="2022-01-01")
# # arima.set_validation_single_split(train_percent=0.75)
# sarima.set_validation_rolling_window(train_percent=0.9000, test_len=14) #TODO: change date/remove it

# sarima.set_model_parameters(0, 1, 1, 0, 0, 2, 7) #7,1,1, #TODO: add hyperparam grid

# sarima.model_run(col=COLUMN)#, exog=["PAT_BG_0", "PAT_BG_A", "PAT_BG_AB", "PAT_BG_B"])

# #Try out stepwise error measurements (now only mae):
# sarima.plot_stepwise(plot_type="forecast", comparison_col=COLUMN) #forecast
# sarima.plot_stepwise(df=sarima.stepwise_forecast_difference, comparison=False, plot_type="forecast difference", comparison_col=COLUMN) #forecast difference
# sarima.plot_stepwise_forecast_errors()

# #%%
# # check accurarcy
# sarimax_test_df = df_original.loc["2025-04-11":"2025-04-17", [COLUMN]]
# prediction_sarimax = sarima.stepwise_forecasts
# print("SARIMAX")
# print("RMSE",   metrics.root_mean_squared_error(sarimax_test_df[COLUMN], prediction_sarimax["Days ahead: 1"].dropna())) #21.46 -- not so good i guess
# print("MAPE",   metrics.mean_absolute_percentage_error(sarimax_test_df[COLUMN], prediction_sarimax["Days ahead: 1"].dropna())) #21.46 -- not so good i guess
# print("MAE",    metrics.mean_absolute_error(sarimax_test_df[COLUMN], prediction_sarimax["Days ahead: 1"].dropna())) #21.46 -- not so good i guess
# print("MedAE",  metrics.median_absolute_error(sarimax_test_df[COLUMN], prediction_sarimax["Days ahead: 1"].dropna())) #21.46 -- not so good i guess
# print("MaxE",   metrics.max_error(sarimax_test_df[COLUMN], prediction_sarimax["Days ahead: 1"].dropna())) #21.46 -- not so good i guess






#%%  ---------------------------------------------------------#
#                        SARIMAX 1                         #
# ---------------------------------------------------------#
#without exogenous (atm)


# Make predictions



# Auto-arima
# auto_arima(df[COLUMN], m=7, trace=True, suppress_warnings=True).summary()


model_sarima1 = SARIMAX(
    train_df[COLUMN], #df[COLUMN], 
    order = (1,1,1), 
    seasonal_order=(1,0,2, 7)
)

res_sarima1 = model_sarima1.fit()
start = len(train_df)
end = len(train_df) + len(test_df) -1
prediction_sarima1 = res_sarima1.predict(start, end, dynamic=True).rename("Prediction")

prediction_sarima1 = prediction_sarima1.to_frame()
prediction_sarima1.index = pd.to_datetime(prediction_sarima1.index)#pd.to_datetime(prediction.index).dt.date
prediction_sarima1.index.name = "Date"
test_df[COLUMN].plot(legend=True, figsize=(16,8))
prediction_sarima1["Prediction"].plot(legend=True)
plt.title("sarima1X 1 (without): Test data & prediction")
plt.show()

difference_sarima1 = abs(test_df[COLUMN] - prediction_sarima1["Prediction"]).rename("Difference")
difference_sarima1.plot(legend=True, figsize=(16,8))
plt.title("sarima1X 1 (without): Absolute Difference between y - y_hat")
plt.show()
prediction_sarima1.head()

 
# check accuracy
print("SARIMAX 1 without exogenous")
print("RMSE",   metrics.root_mean_squared_error(test_df[COLUMN], prediction_sarima1["Prediction"])) #21.46 -- not so good i guess
print("MAPE",   metrics.mean_absolute_percentage_error(test_df[COLUMN], prediction_sarima1["Prediction"])) #21.46 -- not so good i guess
print("MAE",    metrics.mean_absolute_error(test_df[COLUMN], prediction_sarima1["Prediction"])) #21.46 -- not so good i guess
print("MedAE",  metrics.median_absolute_error(test_df[COLUMN], prediction_sarima1["Prediction"])) #21.46 -- not so good i guess
print("MaxE",   metrics.max_error(test_df[COLUMN], prediction_sarima1["Prediction"])) #21.46 -- not so good i guess


#%% ---------------------------------------------------------#
#                       SARIMAX   2                      #
# ---------------------------------------------------------#
#without exogenous (atm) 2
#its just the code below copied, and everything with exog removed.



# Auto-arima
# auto_arima(df[COLUMN], m=7, trace=True, suppress_warnings=True).summary()
# Make predictions




model_sarima2 = SARIMAX(
     train_df[COLUMN], 
     order = (1,1,1), 
     seasonal_order=(1,0,2,7)
)


res_sarima2 = model_sarima2.fit()#start_params=(0,0,0,0,0,1,1,1,1,1))

#Predict
start = len(train_df)
end = len(train_df) + len(test_df) -1
# start = test_df.index.min()
# end = test_df.index.max()
prediction_sarima2 = res_sarima2.predict(start, end, dynamic=True).rename("Prediction")

#forecast
# forecast_sarimax = res_sarima2.forecast(4, exog=exog_test)


prediction_sarima2 = prediction_sarima2.to_frame()
prediction_sarima2.index = pd.to_datetime(prediction_sarima2.index)#pd.to_datetime(prediction.index).dt.date
prediction_sarima2.index.name = "Date"
test_df[COLUMN].plot(legend=True, figsize=(16,8))
prediction_sarima2["Prediction"].plot(legend=True)
plt.title("SARIMAX 2(without): Test data & prediction")
plt.show()

difference_sarima2 = abs(test_df[COLUMN] - prediction_sarima2["Prediction"]).rename("Difference")
difference_sarima2.plot(legend=True, figsize=(16,8))
plt.title("SARIMAX 2 (without): Absolute Difference between y - y_hat")
plt.show()
prediction_sarima2.head()


# check accuracy
print("SARIMAX 2 without exogenous")
print("RMSE",   metrics.root_mean_squared_error(test_df[COLUMN], prediction_sarima2["Prediction"])) #21.46 -- not so good i guess
print("MAPE",   metrics.mean_absolute_percentage_error(test_df[COLUMN], prediction_sarima2["Prediction"])) #21.46 -- not so good i guess
print("MAE",    metrics.mean_absolute_error(test_df[COLUMN], prediction_sarima2["Prediction"])) #21.46 -- not so good i guess
print("MedAE",  metrics.median_absolute_error(test_df[COLUMN], prediction_sarima2["Prediction"])) #21.46 -- not so good i guess
print("MaxE",   metrics.max_error(test_df[COLUMN], prediction_sarima2["Prediction"])) #21.46 -- not so good i guess



#%% ---------------------------------------------------------#
#                       SARIMAX 3                         #
# ---------------------------------------------------------#
# WITH exogenous


# Auto-arima
# auto_arima(df[COLUMN], m=7, trace=True, suppress_warnings=True).summary()
# Make predictions



#define exogenous variables
exog_cols = ["tlmin", "workday_enc", "holiday_enc", "day_of_week", "day_of_year"]#, "influenza_daily"] #"tlmax", "influenza_weekly"
n_obs = len(train_df) #number of ENDOgenous observations
k_exog = len(exog_cols) #number of EXOG variables
exog = np.empty([n_obs, k_exog]) #empty array
for i, ex in enumerate(exog_cols):
     exog[:, i] = train_df[ex]



model_sarimax3 = SARIMAX(
     endog=train_df[COLUMN], 
     exog=train_df[exog_cols], 
     order = (0,0,1), 
     seasonal_order=(0,0,2,7)
)


res_sarimax3 = model_sarimax3.fit()#start_params=(0,0,0,0,0,1,1,1,1,1))

#Predict
# start = len(train_df)
# end = len(train_df) + len(test_df) -1
start = test_df.index.min()
end = test_df.index.max()
exog_prediction = test_df[exog_cols]
prediction_sarimax3 = res_sarimax3.predict(start=start, end=end, exog=exog_prediction, dynamic=True).rename("Prediction")
# prediction_sarimax3 = (res_sarimax3
#     .get_prediction(start=start, end=end, exog=exog_prediction, dynamic=True)
#     .summary_frame(alpha=0.05)
#     .rename(columns={"mean":"Prediction"})
# )
#forecast
# forecast_sarimax = res_sarimax3.forecast(4, exog=exog_test)


prediction_sarimax3 = prediction_sarimax3.to_frame()
prediction_sarimax3.index = pd.to_datetime(prediction_sarimax3.index)#pd.to_datetime(prediction.index).dt.date
prediction_sarimax3.index.name = "Date"


fig, ax = plt.subplots(figsize=(16,8))
ax.set(
    title="SARIMAX 3(with): Test data & prediction",
    xlabel="Date",
    ylabel=f"{COLUMN} (pred/actual)"
)
ax.plot(test_df[COLUMN], label="original data")
ax.plot(prediction_sarimax3["Prediction"], label="Prediction")
# ax.fill_between(
#     prediction_sarimax3.index, prediction_sarimax3['mean_ci_lower'], prediction_sarimax3['mean_ci_upper'],
#     color='r', alpha=0.1
# )
plt.legend()
plt.show()

difference_sarimax3 = abs(test_df[COLUMN] - prediction_sarimax3["Prediction"]).rename("Difference")
difference_sarimax3.plot(legend=True, figsize=(16,8))
plt.title("SARIMAX 3 (with): Absolute Difference between y - y_hat")
plt.show()


# check accuracy
print("SARIMAX 3 with exogenous")
print("RMSE",   metrics.root_mean_squared_error(test_df[COLUMN], prediction_sarimax3["Prediction"])) #21.46 -- not so good i guess
print("MAPE",   metrics.mean_absolute_percentage_error(test_df[COLUMN], prediction_sarimax3["Prediction"])) #21.46 -- not so good i guess
print("MAE",    metrics.mean_absolute_error(test_df[COLUMN], prediction_sarimax3["Prediction"])) #21.46 -- not so good i guess
print("MedAE",  metrics.median_absolute_error(test_df[COLUMN], prediction_sarimax3["Prediction"])) #21.46 -- not so good i guess
print("MaxE",   metrics.max_error(test_df[COLUMN], prediction_sarimax3["Prediction"])) #21.46 -- not so good i guess


#%% -------------------------------------------------------#
#                       SARIMAX 4                          #
# ---------------------------------------------------------#
# WITH exogenous, ONLY known future exogenous variables

# Auto-arima
#aa = auto_arima(df[COLUMN], m=7, trace=True, suppress_warnings=True).summary()
# Make predictions



#define exogenous variables
exog_cols = ["tlmin", "workday_enc", "holiday_enc", "day_of_week", "day_of_year"]#, "influenza_daily"] #"tlmax", "influenza_weekly"


model_sarimax3 = SARIMAX(
     endog=train_df[COLUMN], 
     exog=train_df[exog_cols], 
     order = (0,1,1), 
     seasonal_order=(0,0,2,7)
)


res_sarimax3 = model_sarimax3.fit()#start_params=(0,0,0,0,0,1,1,1,1,1))

#Predict
# start = len(train_df)
# end = len(train_df) + len(test_df) -1
start = test_df.index.min()
end = test_df.index.max()
exog_prediction = test_df[exog_cols]
# prediction_sarimax3 = res_sarimax3.predict(start=start, end=end, exog=exog_prediction, dynamic=True).rename("Prediction")
# prediction_sarimax3 = prediction_sarimax3.to_frame()
# prediction_sarimax3.index = pd.to_datetime(prediction_sarimax3.index)#pd.to_datetime(prediction.index).dt.date
# prediction_sarimax3.index.name = "Date"
prediction_sarimax3 = (res_sarimax3.
    get_prediction(start=start, end=end, exog=exog_prediction, dynamic=True)
    .summary_frame(alpha=0.5)
    .rename(columns={"mean": "Prediction"})
)

#forecast
# forecast_sarimax = res_sarimax3.forecast(4, exog=exog_test)


fig, ax = plt.subplots(figsize=(16,8))
ax.set(
    title="SARIMAX 3(with): Test data & prediction",
    xlabel="Date",
    ylabel=f"{COLUMN} (pred/actual)"
)
ax.plot(test_df[COLUMN], label="original data")
ax.plot(prediction_sarimax3["Prediction"], label="Prediction")
ax.fill_between(
    prediction_sarimax3.index, prediction_sarimax3['mean_ci_lower'], prediction_sarimax3['mean_ci_upper'],
    color='r', alpha=0.1
)
plt.legend()
plt.show()

difference_sarimax3 = abs(test_df[COLUMN] - prediction_sarimax3["Prediction"]).rename("Difference")
difference_sarimax3.plot(legend=True, figsize=(16,8))
plt.title("SARIMAX 3 (with): Absolute Difference between y - y_hat")
plt.show()


#old viz, without conf. int.
# test_df[COLUMN].plot(legend=True, figsize=(16,8))
# prediction_sarimax3["Prediction"].plot(legend=True)
# plt.title("SARIMAX 3(with): Test data & prediction")
# plt.show()

# difference_sarimax3 = abs(test_df[COLUMN] - prediction_sarimax3["Prediction"]).rename("Difference")
# difference_sarimax3.plot(legend=True, figsize=(16,8))
# plt.title("SARIMAX 3 (with): Absolute Difference between y - y_hat")
# plt.show()
# prediction_sarimax3.head()


# check accuracy
print("SARIMAX 3 with exogenous")
print("RMSE",   metrics.root_mean_squared_error(test_df[COLUMN], prediction_sarimax3["Prediction"])) #21.46 -- not so good i guess
print("MAPE",   metrics.mean_absolute_percentage_error(test_df[COLUMN], prediction_sarimax3["Prediction"])) #21.46 -- not so good i guess
print("MAE",    metrics.mean_absolute_error(test_df[COLUMN], prediction_sarimax3["Prediction"])) #21.46 -- not so good i guess
print("MedAE",  metrics.median_absolute_error(test_df[COLUMN], prediction_sarimax3["Prediction"])) #21.46 -- not so good i guess
print("MaxE",   metrics.max_error(test_df[COLUMN], prediction_sarimax3["Prediction"])) #21.46 -- not so good i guess



#
# ---------------------------------------------------------#
#%%                         LSTM                             #
# ---------------------------------------------------------#
# #without exogenous (atm)

# # Stationarity: TS is already stationary
# # Scale: TS must be scaled to activation function --> Not done
# #        Default for LSTM is hyperbolic tangent (tanh) with values between -1 and 1

# #%% Data prep
# X = df[[COLUMN]].values #double brackets to keeps as df (instead of series)

# train, test = X[:split_point], X[split_point:]

# #Scale
# # X = df[COLUMN].values
# # X = X.reshape(len(X), 1)
# scaler = MinMaxScaler(feature_range=(-1, 1))
# scaler = scaler.fit(X)
# scaled_X = scaler.transform(X)

# scaler.inverse_transform(scaled_X)
# #%% Make predictions
# X, y = train[:, 0:-1], test[:, -1]
# X.shape[1]
# X.reshape
# X = X.reshape(X.shape[0], 1, X.shape[1])
# print(X)

# #%%
# from sklearn.preprocessing import LabelEncoder

# # convert series to supervised learning
# def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
# 	n_vars = 1 if type(data) is list else data.shape[1]
# 	df = pd.DataFrame(data)
# 	cols, names = list(), list()
# 	# input sequence (t-n, ... t-1)
# 	for i in range(n_in, 0, -1):
# 		cols.append(df.shift(i))
# 		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
# 	# forecast sequence (t, t+1, ... t+n)
# 	for i in range(0, n_out):
# 		cols.append(df.shift(-i))
# 		if i == 0:
# 			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
# 		else:
# 			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
# 	# put it all together
# 	agg = pd.concat(cols, axis=1)
# 	agg.columns = names
# 	# drop rows with NaN values
# 	if dropnan:
# 		agg.dropna(inplace=True)
# 	return agg
 
# # load dataset
# # dataset = df_original.loc[START_DATE:, [COLUMN]]#read_csv('pollution.csv', header=0, index_col=0)
# dataset = df_original.loc[START_DATE:, ["EC_BG_0", "EC_BG_A", "EC_BG_B", "EC_BG_AB", "is_workday", COLUMN]]#read_csv('pollution.csv', header=0, index_col=0)
# values = dataset.values
# # integer encode direction
# encoder = LabelEncoder()
# values[:,4] = encoder.fit_transform(values[:,4])
# # ensure all data is float
# values = values.astype('float32')
# # normalize features
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled = scaler.fit_transform(values)
# # frame as supervised learning
# reframed = series_to_supervised(scaled, 1, 1)
# # drop columns we don't want to predict
# #reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
# print(reframed.head())


# # split into train and test sets
# values.shape
# values = reframed.values
# # n_train_hours = 365 * 24
# # train = values[:n_train_hours, :]
# # test = values[n_train_hours:, :]
# train = values[:1000, :]
# test = values[1000:, :]
# # split into input and outputs
# train_X, train_y = train[:, :-1], train[:, -1]
# test_X, test_y = test[:, :-1], test[:, -1]
# # reshape input to be 3D [samples, timesteps, features]
# train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
# test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
# print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# ...
# # design network
# from keras.models import Sequential
# from keras.layers import LSTM
# from keras.layers import Dense
# from math import sqrt

# #design network
# model = Sequential()
# model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
# model.add(Dense(1))
# model.compile(loss='mae', optimizer='adam')
# # fit network
# history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# # plot history
# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='test')
# plt.legend()
# plt.show()

# # make a prediction
# yhat = model.predict(test_X)
# test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# # invert scaling for forecast
# inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
# inv_yhat = scaler.inverse_transform(inv_yhat)
# inv_yhat = inv_yhat[:,0]
# # invert scaling for actual
# test_y = test_y.reshape((len(test_y), 1))
# inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
# inv_y = scaler.inverse_transform(inv_y)
# inv_y = inv_y[:,0]
# # calculate RMSE
# rmse = sqrt(metrics.mean_squared_error(inv_y, inv_yhat))
# print('Test RMSE: %.3f' % rmse)

# #%% check accuracy


#%%
# ---------------------------------------------------------#
# MARK:                   LSTM NEU                         #
# ---------------------------------------------------------#
#without exogenous (atm)

# This tutorial
# https://www.youtube.com/watch?v=94PlBzgeq90

# 3 Types of GATES:
# Forget Gate
# Input Gate
# Output Gate

from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import seaborn as sns

#%% load dataset
data = df_original.loc[START_DATE:END_DATE, ["EC_BG_0", "EC_BG_A", "EC_BG_B", "EC_BG_AB", "use_discarded", "use_expired", "is_workday", COLUMN]]#read_csv('pollution.csv', header=0, index_col=0)
data.head()
data.info()
data.describe()


# ------------------------
#%% initial visuallisations
# ------------------------

nonworking_days = data[data["is_workday"] == False]
plt.figure(figsize=(16,8))
plt.plot(data.index, data["use_transfused"], label="Transfused", color="black", linewidth=0.5)
plt.plot(data.index, data["use_discarded"], label="discarded", color="red", linewidth=0.5)
plt.plot(data.index, data["use_expired"], label="expired", color="green", linewidth=0.5)
# plt.scatter(nonworking_days.index, label="Non-working days", color="red")
#ax = plt.gca()
for holiday in nonworking_days.index:
	plt.axvspan(holiday, holiday + pd.Timedelta(days=1), color="red", alpha=0.05)
plt.title("Time series transfues EC")


#Check for correlation between features
numeric_data = data[["use_discarded", "use_expired", COLUMN]]
plt.figure(figsize=(16,8))
sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm")
plt.title("Feature correlation Heatmap")

#subset dates:
prediction_lstm = data.loc[
    (data.index > "2018-01-01") &
    (data.index < "2020-01-01")
]




#%% ------------------------
# Prepare for LSTM Model
# ------------------------

discarded = data["use_discarded"]
transfused = data.filter([COLUMN])


dataset = transfused.values # convert to np array

training_data_len = int(np.ceil(len(dataset) * 0.95)) # 95% of dataset

# Preprocessing stages
scaler = StandardScaler()
scaled_data = scaler.fit_transform(dataset) #TODO: this needs to be training only,otherwise information leak into scaler

training_data = scaled_data[:training_data_len] #95% of our data

#traiing features
#what we actually give the model to learn
X_train, y_train = [], []

# Create sliding window for our data (60days)
sliding_size = 60
forecast_days = 0 #days more than on day ahead
for i in range(sliding_size, len(training_data)):
    X_train.append(training_data[i - sliding_size:i, 0])
    y_train.append(training_data[i + forecast_days, 0])

#convert lists to np arrays (for tensorflow, needs arrays)
X_train, y_train = np.array(X_train), np.array(y_train)
#change to 3D array, tensorflow needs this to work better
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))





#%% ------------------------
# Build the model
# ------------------------

# Model with 5 Layers: 
# LSTM Layer 1 -> Layer 2 -> Layer 3 -> Dense Layer -> Dropout Layer -> Final Layer


# Actual model
memory_cells = 64 #32, 64, 128, etc
model_lstm = keras.models.Sequential()

# - - - - - 
# LAYERS 
# - - - - - 

# LSTM Layer 1
# First Layer helps understands model understand patterns
#return_sequence -> give back "ideas" after its done (list of "ideas")
model_lstm.add(keras.layers.LSTM(memory_cells, return_sequences=True, input_shape=(X_train.shape[1], 1)))

# LSTM Layer 2
# return_sequence = False --> only give back single output
#gives back final prediciton, so dont give everything back
model_lstm.add(keras.layers.LSTM(64, return_sequences=False))

# Dense Layer
# Turns patterns into a decision (decision making)
# 128 = neurons to make final decision
# activation = relu --> non-linearity (recitvied linear unit)
# this layer helps model complicated patterns
model_lstm.add(keras.layers.Dense(128, activation="relu"))


#Dropout Layer
#0.5 --> randomly drops out 50% of neurons during traiing --> prevents overfitting
# keeps model from being to sensitive to training data
model_lstm.add(keras.layers.Dropout(0.5))

#Final Layer (Dense layer as well)
# (final decision making)
# simple layer with 1 neuron that outputs one value -- the predicted value
model_lstm.add(keras.layers.Dense(1)) 


model_lstm.summary()

#Model compilation
# puts all pieces together
# compiles moodel to tellit how to learn.
model_lstm.compile(optimizer="adam", loss="mae", metrics=[keras.metrics.RootMeanSquaredError()])



#%% ------------------------
# Train the model
# ------------------------

#epochs = hwo many times is it gonna run to find the best solution
# batch size = how much data is in each batch when it runs
training = model_lstm.fit(X_train, y_train, epochs=20, batch_size=32)

#Prep test data
test_data = scaled_data[training_data_len - sliding_size : ]
X_test, y_test = [], dataset[training_data_len : ]

for i in range(sliding_size, len(test_data)):
    X_test.append(test_data[i - sliding_size : i + forecast_days, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


# Make predictions
predictions_lstm = model_lstm.predict(X_test)
predictions_lstm = scaler.inverse_transform(predictions_lstm)


#%% ------------------------
# Plot results
# ------------------------

train = data[ : training_data_len]
test = data[training_data_len : ]

test = test.copy()

test["Predictions"] = predictions_lstm

plt.figure(figsize=(16,8))
plt.plot(train.loc["2019-10-01": ].index, 
		 train.loc["2019-10-01": ]["use_transfused"], 
		 label="Train (Actual)", 
		 color = "blue")
plt.plot(test.index, test["use_transfused"], label="Test (Actual)", color = "orange")
plt.plot(test.index, test["Predictions"], label="Predictions", color = "red")
plt.title("EC trnasufison prediction")
plt.xlabel("Date")
plt.ylabel("number of EC")
plt.legend()



#%% ------------------------
# Get evaluation values
# ------------------------
print("LSTM")
print("RMSE:", metrics.root_mean_squared_error(y_pred=test["Predictions"], y_true=test["use_transfused"]))
print("MAPE:", metrics.mean_absolute_percentage_error(y_pred=test["Predictions"], y_true=test["use_transfused"]))
print("MAE: ", metrics.mean_absolute_error(y_pred=test["Predictions"], y_true=test["use_transfused"]))
print("MdAE:", metrics.median_absolute_error(y_pred=test["Predictions"], y_true=test["use_transfused"]))
print("MaxE:", metrics.max_error(y_pred=test["Predictions"], y_true=test["use_transfused"]))










#%% ------------------------
# MARK: MULTIVARIATE LSTM
# ------------------------



#%% load dataset
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import seaborn as sns

data = df_original.loc[
    START_DATE:END_DATE, 
    [COLUMN, "use_discarded", "use_expired", 'ward_AN',
     'ward_CH', 'ward_I1', 'ward_I3', 'ward_Other', 'ward_UC', 
     "workday_enc", "holiday_enc", "day_of_week", "day_of_year", "year", "tlmin", "tlmax"]
    ]
data.head()
data.info()
data.describe()

col_Y = "use_transfused" #column to predict (y)
cols_multivariate = ["use_discarded", "used_expired"]



# ------------------------
#%% Prepare for multivariate LSTM Model
# ------------------------



dataset = data.values # convert to np array

training_data_len = int(np.ceil(len(dataset) * 0.95)) # 95% of dataset

# Preprocessing stages
scaler = StandardScaler()
scaled_data = scaler.fit_transform(dataset[:training_data_len, :]) #TODO: this needs to be training only,otherwise information leak into scaler

training_data = scaled_data[:training_data_len, :] #95% of our data

#training features
#what we actually give the model to learn
# wher x train are the input variables and y train is the variable, the model uses in the training stage
# to adjust weights and baises to conform to the result (compare result in training to y_train)
X_train, y_train = [], []

# Create sliding window for our data (60days)
sliding_size = 60 #
forecast_days = 0 #days more than on day ahead
for i in range(sliding_size, len(training_data)):
    X_train.append(training_data[i - sliding_size:i, :])
    y_train.append(training_data[i + forecast_days, 0]) #0 = the position of our variable to forecast, is only one variable, as we only forecast COLUMN
    

#convert lists to np arrays (for tensorflow, needs arrays)
X_train, y_train = np.array(X_train), np.array(y_train)
#change to 3D array, tensorflow needs this to work better
# X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))



#%% ------------------------
# Build the model
# ------------------------

# Model with 5 Layers: 
# LSTM Layer 1 -> Layer 2 -> Layer 3 -> Dense Layer -> Dropout Layer -> Final Layer


# Actual model
memory_cells = 64 #32, 64, 128, etc
model_lstm_mv = keras.models.Sequential() #mv = multivariate

# LSTM Layers

#this is different in multivariate (mv) lstm: the input shape
model_lstm_mv.add(keras.layers.LSTM(memory_cells, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model_lstm_mv.add(keras.layers.LSTM(64, return_sequences=False))
model_lstm_mv.add(keras.layers.Dense(128, activation="relu"))
model_lstm_mv.add(keras.layers.Dropout(0.5))
model_lstm_mv.add(keras.layers.Dense(1)) 

model_lstm_mv.compile(optimizer="adam", loss="mae", metrics=[keras.metrics.RootMeanSquaredError()])
model_lstm_mv.summary()



# ------------------------
#%% Train the model
# ------------------------

#epochs = hwo many times is it gonna run to find the best solution
# batch size = how much data is in each batch when it runs
training = model_lstm_mv.fit(X_train, y_train, epochs=20, batch_size=32)

#Plot training:
plt.plot(training.history["loss"], label="training loss")

#Prep test data
test_data = scaled_data[training_data_len - sliding_size : ]
X_test, y_test = [], dataset[training_data_len : ]

# for i in range(sliding_size, len(test_data)): #sliding_size
for i in range(0, len(test_data)): #sliding_size
    X_test.append(test_data[i - sliding_size : i + forecast_days, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


# Make predictions
predictions_lstm_mv = model_lstm_mv.predict(X_test)
predictions_lstm_mv = scaler.inverse_transform(predictions_lstm_mv)


#%% ------------------------
# Plot results
# ------------------------

train = data[ : training_data_len]
test = data[training_data_len : ]

test = test.copy()

test["Predictions"] = predictions_lstm_mv

plt.figure(figsize=(16,8))
plt.plot(train.loc["2019-10-01": ].index, 
		 train.loc["2019-10-01": ]["use_transfused"], 
		 label="Train (Actual)", 
		 color = "blue")
plt.plot(test.index, test["use_transfused"], label="Test (Actual)", color = "orange")
plt.plot(test.index, test["Predictions"], label="Predictions", color = "red")
plt.title("EC trnasufison prediction")
plt.xlabel("Date")
plt.ylabel("number of EC")
plt.legend()



#%% ------------------------
# Get evaluation values
# ------------------------
print("LSTM")
print("RMSE:", metrics.root_mean_squared_error(y_pred=test["Predictions"], y_true=test["use_transfused"]))
print("MAPE:", metrics.mean_absolute_percentage_error(y_pred=test["Predictions"], y_true=test["use_transfused"]))
print("MAE: ", metrics.mean_absolute_error(y_pred=test["Predictions"], y_true=test["use_transfused"]))
print("MdAE:", metrics.median_absolute_error(y_pred=test["Predictions"], y_true=test["use_transfused"]))
print("MaxE:", metrics.max_error(y_pred=test["Predictions"], y_true=test["use_transfused"]))








# MARK: -
# Prophet
#%% -------------------------------------------------------#
#                         PROPHET (temp)                   #
# ---------------------------------------------------------#
# #without exogenous (atm)


from prophet import Prophet

start_date = pd.to_datetime(START_DATE)#("2020-01-01")
split_date = pd.to_datetime("2025-03-09")#("2024-12-31")
end_date = pd.to_datetime(END_DATE)#("2024-12-31")

pred_col = "count"
regressor_cols = ['EC_BG_0', 'EC_BG_A', 'EC_BG_AB', 'EC_BG_B', 'EC_RH_Rh_negative',
       'EC_RH_Rh_positive', 'EC_TYPE_EKF', 'EC_TYPE_EKFX', 'EC_TYPE_Other',
       'PAT_BG_0', 'use_discarded', 'use_expired', 'use_transfused']
sel_cols = [pred_col] + regressor_cols

df = df_original
train_df = df.loc[start_date:split_date, sel_cols]
test_df = df.loc[split_date:end_date, sel_cols]


df.info()
train_df.info()
test_df.info()


#%%
#try to run model (Prophet)
prophet_train = (
    train_df[pred_col]
    .reset_index()
    .rename(columns={"date":"ds", "count":"y"})
)

prophet_train.info()
print(prophet_train.head())

#%%
# m = Prophet(weekly_seasonality=True, interval_width=0.95)
m = Prophet(weekly_seasonality=True, interval_width=0.95)
m.add_country_holidays(country_name="Austria")

m.fit(prophet_train)
future_dates = m.make_future_dataframe(periods=365, freq="D")

fc = m.predict(future_dates)
#%%
m.plot(fc, uncertainty=True)

fc.head()

#%% 
# plot components
fig = m.plot_components(fc)
m.train_holiday_names




#%% Make predictions
#%%
# 
prophet_fc = fc.set_index("ds")["2025-03-09":"2025-05-01"]
# check accuracy
print("prophet")
print("RMSE:", metrics.root_mean_squared_error(y_pred=prophet_fc["yhat"], y_true=test["use_transfused"]))
print("MAPE:", metrics.mean_absolute_percentage_error(y_pred=prophet_fc["yhat"], y_true=test["use_transfused"]))
print("MAE: ", metrics.mean_absolute_error(y_pred=prophet_fc["yhat"], y_true=test["use_transfused"]))
print("MdAE:", metrics.median_absolute_error(y_pred=prophet_fc["yhat"], y_true=test["use_transfused"]))
print("MaxE:", metrics.max_error(y_pred=prophet_fc["yhat"], y_true=test["use_transfused"]))

# %%
prophet_result = fc[["yhat", "ds"]].set_index("ds").loc[START_DATE:END_DATE]
plot_start_date = pd.to_datetime(END_DATE) - pd.Timedelta(days=20)
fig, ax = plt.subplots()
ax.plot(prophet_result, linewidth=0.5, color="orange")
ax.plot(test_df.loc[plot_start_date:END_DATE, "count"], linewidth=0.5, color="blue")
plt.show()
# %%
