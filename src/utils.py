import pandas as pd
from time import time
from pathlib import Path


from src import config





#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# Global functions
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

#wrapper function to time function duration
def timer_func(func):
    #todo: add either own logging func+decorators or put it inside here?
    def wrap_func(*args, **kwargs):
        if config.ENABLE_TIMING:
            t1 = time()
            result = func(*args, **kwargs)
            t2 = time()
            print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
            return result
        else:
            return func(*args, **kwargs)
    return wrap_func


#Slice df by date range:
def subset_df(df, start: str=config.DEV_START_DATE, end: str=config.DEV_END_DATE):
    #Only keep subset on (for dev purposes)
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)

    df = df[start_date:end_date]
    
    #different option:
    # mask = (df.index >= start_date) & (df.index <= end_date)
    # mask = (df.index >= "2018-01-01") & (df.index <= "2024-12-31")
    # df = df.loc[mask]

    return df

#Sample random days (all of the days rows) (for testing purposes)
def sample_days(df, n: int=500):
    #supposed the df has a datetimeindex
    sampled_values = df.index.to_series().sample(n=n, random_state=10)
    df_sample = df[df.index.isin(sampled_values)]
    return df_sample

#Sample random rows (not whole days)
def sample_rows(df, n: int=90000):
    #supposed the df has a datetimeindex
    # n = number of rows to sample
    df_sample = df.sample(n=n, random_state=10) 
    return df_sample


#----------------------------------------------------------------------------------------------
# GENERAL 
#----------------------------------------------------------------------------------------------

#TODO: save_figs is not done yet -- maybe scrap it altogether

def save_plots(fig, filename_general: str, filename_suffix: str, location: str, foldername: str=""):
    """Function to save files to specified location, creating folder with subfolder of 
    name 'foldername' if not exists.

    Args:
        filename (str): naming scheme of the file. Date will be appended.
        location (str): location in ./plots/
        foldername (str): foldername to create addtional subfolder inside ./plots/00_subfolder/~
        If not specified, no new subfolder is created and files are stored in 'location'
    """
    # my_dir = Path(location)

    # #TODO: super confusing with fname_general, fname_suffix, foldername/location.
    # #--> simplify and think of usecases, i.e. put in non-existing, new folder or just pass complete path minus filename
    # #Create new subdirectory, if foldername is specified and not exists:
    # if foldername != "" and not my_dir.is_dir():
    #     new_dir = Path.joinpath(my_dir, foldername)
    #     new_dir.mkdir(parents=False, exist_ok=False)
    #     print(f"Created new directory: {new_dir}")
    #     flocation = "/".join(location, foldername, filename_general + filename_suffix) 
    # else:
    #     flocation = location + 

    # #Save plot:
    # fig.savefig(fname=)

