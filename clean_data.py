#main file for reading, cleaning, transforming raw data.
#moved code from main.py to here
#%%
from src import clean
from src import load
from src import transform
from src import config_cleaning

#%%
#Logic to check if cleaned/transformed data exists, inside method calls. 
df_raw = load.load_data(path=config_cleaning.RAW_DATA_PATH)
df_clean = clean.clean_data(df_raw, existing_file_path=config_cleaning.CLEANED_DATA_PATH)
df_processed = transform.transform_data(df_clean, existing_file_path=config_cleaning.TRANSFORMED_DATA_PATH)


