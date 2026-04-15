Code and Text for the master thesis. Since data is sensitive, it is hosted only locally

# Install right package versions
To let the grid search pipeline run, make sure the correct versions of all packages are installed:
Use the environment.yml file to create a conda environment, which installs the right python version and will install requirements.txt



# Run ./clean_data.py
To get cleaned data from raw data, run ./clean_data.py, this will create two files:
- output_cleaned.csv
- output_transformed.csv
where output_transformed.csv is daily aggregated and will be used in the models.
(should take a few minutes, not too long).

# (Data exploration) (WIP)
For data visualisation of raw/cleaned/transformed data, use the interactive (jupyter) data_exploration.py file. 
Still WIP!

# Run pipeline
To run the pipeline (forecasting only), use ./run_all.sh.
Multiprocessing of s/arima, prophet is done with the main.py file (settings see below)
Multiprocessing is not working with LSTM model. 
To still run LSTM in parallel, for LSTM, multiple python processes (main.py) are started after completing the other models (when using run_all.sh) and the results of LSTM are then merged at the end.

Important:
- specify the cores for s/arima, prophet multiprocessing in main.py (needs to be set at the top of main.py)
- specify the cores for LSTM in run_all.sh
- specify the python version/path to python in run_all.sh
- **Set parameters in src/config.py**, be wary of the number of combinations this results in and use [model]_n_samples accordingly (set to below zero or delete line in main.py)

Depending on the parameter settings, this will take multiple days!
Some combinations are non-sensical, so will yield no output, resulting in an empty line in the grid_search_results.csv file of the model, with only the ID of the run present.

# Output
A folder ./results is created with subfolders for each model.
In each model, for every parameter combination is a separate subfolder, starting with the format [run_ID]_[YYYYMMDD].
Inside the runs directory, for every every day of the forecast period is a separate .csv, showing the actual value for that day, the mean of the forecast, the upper and lower CI.
The forecast_errors.csv is for that model run only and then aggregated into the models grid_search_results.csv (first day forecast only!).
params.json shows the parameters used for that model run.
stats.json shows the runs id, which model was used, run duration in seconds, the start day of the model, the split date (where the rolling window starts), the window num (number of rolling windows), end date.
For LSTM, also the initial weights are stored.

# Results (WIP)
Use the interactive (with jupyter) result_evaluation.py file, to get some stats and and visualisations.
