# Functions for splitting datasets into test/training/validation etc.
# - transforms df from long to wide (aggregation)
# - adds exogenous variables from csv: workday, influenza cases vienna, weather data
# - removes unnecessary cols
# - 
import pandas as pd
from datetime import datetime, timedelta
from src import load
import holidays
import numpy as np

from sklearn.preprocessing import StandardScaler

from pathlib import Path



#-----------------------------------------------------------------------------
# Main wrapper function for transforming/processing:
#-----------------------------------------------------------------------------
def transform_data(df,
                   existing_file_path: str="./data/03_transformed/output_transformed.csv", 
                   new_file_path: str="./data/03_transformed/output_transformed.csv"):
    #load cleaned file if exists (so i can skip cleaning step):
    my_file = Path(existing_file_path)

    if my_file.is_file():
        df = pd.read_csv(existing_file_path, sep=",")
        #set again to datetime, because first time didnt work? (i think because of merging/missing values)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df = df.sort_index()
        print(f"Reading existing transformed file at {existing_file_path}")
    else:
        #remove unnecessary cols:
        cols_to_remove = ["EC_ID_O_hash", "EC_ID_I_hash"]#, "PAT_WARD"] #"Unnamed: 0"
        df = df.drop(columns=cols_to_remove, axis=1)
        #colnames which i want to sum (all except 'date')
        df = df.reset_index()

        #Combines individual wards into top5 wards and rest into 'Other'
        df = combine_wards(df)

        #sum all above cols and split into category columns
        cols_to_sum = list(df.columns)
        cols_to_sum.remove('date')
        df = aggregate_categorical_cols(df, cols_to_sum)


        #Total count of EC for that day (expired, transfused etc.)
        #df = add_daily_total(df)

        df = add_temporal_features(df)


        df = add_weather_data(df)

        df = add_influenza_data(df)
        df = add_covid_data(df)


        #Scale data
        df["year_scaled"] = df["year"] - df["year"].min()

        scaler = StandardScaler()
        df[["covid_daily_scaled", "influenza_daily_scaled"]] = scaler.fit_transform(df[["covid_daily", "influenza_daily"]])




        #Save new file:
        print(f"Write new file to {new_file_path}")
        df.to_csv(path_or_buf=new_file_path, sep=",", index_label="date")



    return df


#-----------------------------------------------------------------------------
# helper functions for main steps
#-----------------------------------------------------------------------------

def aggregate_categorical_cols(df, cols_to_sum: dict):
    #Split columns with n categorical values into n columns.
    # Aggregate daily sum of occurences to each column.
    # cols_to_sum list of all col names with categorical values which i want to sum.  
    
    results = []

    for col in cols_to_sum:
        res = pd.crosstab(df['date'], df[col])
        res = res.add_prefix(col + "_")
        res.columns = res.columns.str.replace(" ", "_") #replace spaces

        results.append(res)

    #wide_df = results[0].join([res for res in results[1:]])
    wide_df = pd.concat(results, axis=1)


    #Add daily total amounts:
    #test_df = df.set_index('date')['use'].resample('D').count()
    total_df = df.groupby('date').size()#.reset_index(name='count')
    total_df.name = "count"
    #Add to wide_df
    wide_df = wide_df.join(total_df)#.set_index('date'))
    
    return wide_df

def combine_wards(df, ward_map_path="./data/00_external_data/wards_mapping.csv", top_N=5):
    """
    Combines wards first to main wards (around 50?), then calculates top 5 and puts the rest in other.

    Explanation for the steps
    1. In main df, 
    1.1 map ID_Kostenstelle (short 2letter code) onto PAT_WARD with Kostenstelle (long code)
    1.2 fill missing/non-matching with 'Other'
    2. Count by ID_Kostenstelle and Rank in main df
    3. Assign 'Other' to ID_Kostenstelle bigger than rank 5
    --> mapping df with ID_Kostenstelle, amount, rank and Categorization to top5 + 'other' ('pat_wards_ranking_df')
    4. Join categorization top5+Other from 'pat_wards_ranking_df' onto main df column ID_Kostenstelle
    (So first 2-letter code gets mapped onto long code, missing filled with other, then the ranking applied onto 2-letter code in main df)
    Explanation/Reasoning: This way, some information might be lost, because sometimes a valid 2-letter code could be extracted from the long code in main df,
    but i only want to map existing long codes from Alex data onto valid long codes in main df. 

    Args:
        df (DataFrame): Main dataframe
        ward_map_path (str): Location of csv file for mapping of wards
        top_N (int): Amount of individual wards to keep, rest to Other
    """

    #Load necessary data
    ward_map = pd.read_csv(ward_map_path, sep="\t")
    ward_map["Kostenstelle"] = ward_map["Kostenstelle"].str.strip() #remove whitespaces around strings
    ward_map = ward_map.drop_duplicates(subset="Kostenstelle")
    # ward_map = pd.concat(ward_map, pd.DataFrame(["Other", "Other"]))

    #NOTE:Apparently decided not to do (see docstring):
    #NOTE: i think the idea was, to only map 'perfect' matches.
    # keep only firsst 5 letters of 'Kostenstelle'/'PAT_WARD', to remove submapping
    # as not all full codes from 'PAT_WARDS' are in 'Kostenstelle' 
    # ward_map["Kostenstelle"] = ward_map["Kostenstelle"].str[:4] 
    # df["PAT_WARD"] = df["PAT_WARD"].str[:4] 
    # print(f"vorher: {len(df)}", flush=True)
    # Merge short code (2-letter code) onto df, fill missing values with NA
    df = (
        pd.merge(
            left=df, #.reset_index(), #so date still persists after merge
            right=ward_map[["ID_Kostenstelle", "Kostenstelle"]], 
            left_on="PAT_WARD", 
            right_on="Kostenstelle", 
            how="left"
        )
        #.drop("Kostenstelle", axis=1)
        .set_index("date")
        .fillna({"ID_Kostenstelle":"Other"}) 
    )
    # print(f"nachher: {len(df)},  flush=True")
    # df = (
    #     df.join(
    #         other=ward_map[["ID_Kostenstelle", "Kostenstelle"]].set_index("Kostenstelle"), 
    #         on="PAT_WARD", 
    #         how="left"
    #     )
    #     #.drop("Kostenstelle", axis=1)
    #     .set_index("date")
    #     .fillna({"ID_Kostenstelle":"Other"}) 
    # )


    # Helper variable: Create count+ranking and Map of ID_Kostenstelle/Kostenstelle/top_wards
    # (top_wards = top 5 ranking wards keep their names, rest get assigned as 'Other')
    pat_wards_ranking_df = (
        df
        .groupby("ID_Kostenstelle")
        .size().to_frame("amount")
        .sort_values(by="amount", ascending=False).reset_index()
        .assign(rank = lambda x: x["amount"].where(x["ID_Kostenstelle"] != "Other").rank(ascending=False)) #Exclude 'Other' from ranking.  Other == where no (existing) ID_Kostenstelle was matching.
        .assign(top_wards = lambda x: np.where(
            (x["rank"] <= top_N) | (x["rank"].isna()), x["ID_Kostenstelle"], "Other"))
        #.assign(top_wards = lambda x: np.where(x["ID_Kostenstelle"] == "Other", "Other"))
    )

    # merge directly onto ID_Kostenstelle (2-letter code)
    df = (
        pd.merge(
            left=df.reset_index(),
            right=pat_wards_ranking_df.drop(["amount", "rank"], axis=1),
            left_on="ID_Kostenstelle",
            right_on="ID_Kostenstelle",
            how="left"
        )
        #remove unncecessary cols
        .drop(["PAT_WARD", "ID_Kostenstelle", "Kostenstelle"], axis=1)
        .rename(columns={"top_wards":"ward"})
    )

    return df



def add_temporal_features(df):
    """Adds temporal (some encoded) features to dataframe:
    - Holidays (name, encoding)
    - is workday (includes holidays as non-working days)
    - day of the week, day of the year, year columns

    Args:
        df (dataframe): Dataframe with datetime index

    Returns:
        _type_: _description_
    """

    #Add workday + encoding
    vie_holidays = holidays.country_holidays(country="AT", subdiv="W", years=range(2020,2025), language="de")
    df["is_workday"] = df.index.to_series().apply(lambda x: vie_holidays.is_working_day(x)) #boolean
    df["workday_enc"] = df["is_workday"].astype(int)

    #Add additional temporal features (holiday name + encoding, day of year, day of week)
    
    #holiday name + encoding:
    df["holiday"] = pd.Series(df.index).apply(lambda x: vie_holidays.get(x)).values
    unique_holidays = df["holiday"].dropna().unique()
    #create encoding map (NaN=0)
    holiday_map = pd.DataFrame({
        "holiday" : unique_holidays,
        "holiday_enc" : range(1, len(unique_holidays)+1)
    })
    holiday_map = pd.concat([
        pd.DataFrame({"holiday": [np.nan], "holiday_enc": [0]}), 
        holiday_map],
        ignore_index=True
    )
    df = pd.merge(df.reset_index(), holiday_map, how="left", on="holiday").set_index("date")

    #Add day of the week, day of the year, year columns
    df["day_of_week"] = df.index.dayofweek
    df["day_of_year"] = df.index.dayofyear
    df["year"] = df.index.year

    return df


def add_weather_data(df, weather_data_path="data/00_external_data/Messstation_InnereStadt_Tagesdaten_Datensatz_20000101_20250703.csv", cols_to_add=["tlmin", "tlmax"]):
    #add daily precipitation (mm) and the daily average temperature (°C) -- no min/max values.
    # Data from https://data.hub.geosphere.at/dataset/klima-v2-1d 
    # respectively: https://dataset.api.hub.geosphere.at/app/frontend/station/historical/klima-v2-1d
    # downloaded 03.07.2025. Data station Innere Stadt
    # 24h sum rainfall, temp max, temp min, temp I (7:00), II (14:00), III (19:00 MEZ), temp avg.

    weather_df = load.load_data(path=weather_data_path, sep=",")
    weather_df["time"] = pd.to_datetime(weather_df['time']).dt.tz_localize(None) #remove timezone, to enable merging


    #subset, only keep tmin, tmax:
    cols_to_add.append("time")
    weather_df = weather_df[cols_to_add].set_index("time")
    #weather_df = weather_df.set_index("time")

    df = pd.merge(df, weather_df, left_index=True, right_index=True)

    return df


#Add Covid/influenca data:
def add_influenza_data(df, influenza_data_path="data/00_external_data/", filename="grippemeldedienst", file_ending=".csv"):
    """checks if imputated data for influenza exists, else it opens existing non-interpolated file and
     interpolates/imputates linearly to daily. adds resulting influenza_df/columns to 'df' and rturns 'df'

    Args:
        df (_type_): main dataframe, which influenza data is added to
        influenza_data_path (str, optional): file path to the csv. Defaults to "data/00_external_data/".
        filename (str, optional): filename of the csv. Defaults to "grippemeldedienst".
        file_ending (str, optional): file ending. Defaults to ".csv".

    Returns:
        DataFrame: Returns dataframe with added interpolated influenza data
    """
    
    influenza_interpolated_df = Path(influenza_data_path+filename+"-interpolated"+file_ending)

    if influenza_interpolated_df.is_file():
        influenza_df = pd.read_csv(influenza_data_path+filename+"-interpolated"+file_ending)
        influenza_df = influenza_df.set_index("date")
        influenza_df.index = pd.to_datetime(influenza_df.index)
    else:
        influenza_file = Path(influenza_data_path+filename+file_ending)
        #interpolates & saves file
        influenza_df = interpolate_influenza_data(
            # timeframe = df.index,
            filepath=influenza_data_path, 
            filename=filename, 
            file_ending=file_ending, 
            save_file=True
        ) 

    #add to df
    df = pd.merge(df, influenza_df, left_index=True, right_index=True, how="left")

    return df


def add_covid_data(df, data_path="data/00_external_data/", filename="abwassermonitoring", file_ending=".csv"):
    #Load covid data from Abwassermonitoring dataset.
    # If not yet interpolated, load the raw data abwassermonitoring.csv, forward fill 
    # and divide by 7 to get daily data. 
    # Scaled down by dividing through 1e14.
    # Data is Fracht (GC/Tag), whole year (influenza from grippemeldung is only winter!)
    # and influenza/RSA only start in 2022 in abwassermonitoring.

    covid_interpolated_df = Path(data_path+filename+"-interpolated"+file_ending)

    if covid_interpolated_df.is_file():
        covid_interpolated_df = pd.read_csv(data_path+filename+"-interpolated"+file_ending, index_col="date")
        # covid_interpolated_df = data_path.set_index("date")
        covid_interpolated_df.index = pd.to_datetime(covid_interpolated_df.index)
    else:
        #Read raw and clean, interpolate linearly with forward fill to daily, dividy by 1E14, to scale down
        covid_interpolated_df = pd.read_csv(
            f"{data_path}{filename}{file_ending}", 
            index_col="Datum", 
            sep=";")
        covid_interpolated_df.index = pd.to_datetime(covid_interpolated_df.index, format="%d.%m.%Y")
        covid_interpolated_df = covid_interpolated_df.drop(covid_interpolated_df.columns[-1], axis=1)
        covid_interpolated_df = covid_interpolated_df.query("Target == 'SARS-CoV-2'")
        covid_interpolated_df = covid_interpolated_df.rename({covid_interpolated_df.columns[-1]:"covid_weekly", "Datum":"date"}, axis=1)
        covid_interpolated_df["covid_weekly"] = covid_interpolated_df["covid_weekly"].str.replace(",", ".")
        covid_interpolated_df["covid_weekly"] = pd.to_numeric(covid_interpolated_df["covid_weekly"])

        #make wide, only use SARS-CoV-2, as others only start in 2022 
        #and have some missing values
        covid_interpolated_df = covid_interpolated_df.pivot_table(
                index=covid_interpolated_df.index, 
                columns="Target", 
                values="covid_weekly"
        )
        covid_interpolated_df = covid_interpolated_df.rename({"SARS-CoV-2":"covid_weekly"}, axis=1)
            
        #forward fill to daily:
        covid_interpolated_df = covid_interpolated_df.resample("D").ffill() #no limit to fill gap in 2022
        covid_interpolated_df["covid_daily"] = covid_interpolated_df["covid_weekly"].div(7).round(3)

        #fill rest of the rows with zero (if present, should be none)
        covid_interpolated_df = covid_interpolated_df.fillna(0)

        #Save to file abwassermonitoring-interpolated.csv
        covid_interpolated_df.to_csv(data_path+filename+"-interpolated"+file_ending, index_label="date")

    df = pd.merge(df, covid_interpolated_df, left_index=True, right_index=True, how="left")
    
    return df




#Make wide df, where every col value becomes its own col with daily freq count; prefix of original col name
def make_wide(df):
    #see aggregate_categorical_cols!

    #if date is not index
    daily_total = df.groupby("date").size()
    #if is index:
    #df = df.set_index("date")
    daily_total = df.groupby(df.index.date).size()
    daily_total.name = "total"


    res = []
    for col in df.columns:
        if col == "date":
            continue
        pivotted = pd.crosstab([df["date"]], columns=[df[col]])
        pivotted = pivotted.add_prefix(col + "_")
        res.append(pivotted)
    final = pd.concat(res, axis=1)
    final = final.join(daily_total)

    return final


#-----------------------------------------------------------------------------
# Sub-Helper functions
#-----------------------------------------------------------------------------

def interpolate_influenza_data(filepath, filename, file_ending, save_file=True):
    """_summary_

    Args:
        filepath (_type_): _description_
        filename (_type_): _description_
        file_ending (_type_): _description_
        save_file (bool, optional): _description_. Defaults to True.

    Returns:
        df (DataFrame): DataFrame stripped to necessary columns with daily interpolated values
    """
    # OLD:
    # #open (non-interpolated) file 
    # df = pd.read_csv(filepath+filename+file_ending)

    # #convert date to datetime
    

    # #convert Kalenderwoche to strings
    # df['weeknum'] = df['Kalenderwoche'].str.extract(r'(\d+)').astype(int)
    # df['year'] = df["Jahr"]#pd.to_datetime(df['Jahr'], format="%Y")

    # # Apply the function to create the 'date' column
    # df['date'] = df.apply(lambda row: get_first_monday(row['year'], row['weeknum']), axis=1)
        
    # #rename + remove unnecessary cols (maybe could need Schwankungsbreite?)
    # df.rename(columns={"Neuerkrankungen pro Woche": "influenza_weekly"})
    # df = df.loc[["date", "influenza_weekly"]]

    # #TODO: linear interpolate
    # # Add missing days
    # # df.index = timeframe
    # df = df.set_index("date")
    # df = df.reindex(timeframe, fill_value=None)
    
    # # Fill missing days with nan
    # # Add new col influenza_daily, which takes existing vals/7, then
    # # interpolate these linearly



    # #open (non-interpolated) file 
    df = pd.read_csv(filepath+filename+file_ending)

    #convert Kalenderwoche to strings
    df['weeknum'] = df['Kalenderwoche'].str.extract(r'(\d+)').astype(int)
    df['year'] = df["Jahr"]#pd.to_datetime(df['Jahr'], format="%Y")

    # Apply the function to create the 'weekDate' column
    df['date'] = df.apply(lambda row: get_first_monday(row['year'], row['weeknum']), axis=1)
    
    #rename + remove unnecessary cols (maybe could need Schwankungsbreite?)
    df = df.rename(columns={"Neuerkrankungen pro Woche": "influenza_weekly"})
    df.index = pd.to_datetime(df["date"])
    df = df[["influenza_weekly"]]

    #Convert to int; add daily cases
    df["influenza_weekly"] = pd.to_numeric(df["influenza_weekly"], errors="coerce")

    df = df[["influenza_weekly"]].resample("D").ffill() # no limit, influenza doesnt go down to 0 in summer, just no monitoring!
    # df["influenza_weekly"] = df["influenza_weekly"].ffill(limit=6)
    df["influenza_daily"] = df["influenza_weekly"].div(7).round(3)

    #Extend to last date, fill with last value:
    end_date = "2025-07-03"
    df = df.reindex(pd.date_range(df.index[0], end_date, freq="D")).ffill()


    #fill rest of the rows with zero
    df = df.fillna(0)

    # df["influenza_daily"] = df["influenza_weekly"].div(7)

    # # Add missing days
    # df = df.resample("D")

    # #Extend by one week (to interpolate last week which has no rows)
    # # new_index = pd.date_range(start=df.index[0], end=df.index[-1] + pd.Timedelta(days=6), freq='D')
    # # df = df.reindex(new_index)
    # # df.index = pd.to_datetime(df.index)

    # #forward fill new_cases weekly
    # df["influenza_weekly"] = df["influenza_weekly"].ffill(limit=6)
    # #linearly interpolate influenza_daily
    # # df["influenza_daily"] = df["influenza_daily"].interpolate(method="linear", limit=6, limit_direction="forward").round()
    # df["influenza_daily"] = df["influenza_daily"].div(7)
    # #fill rest of the rows with zero
    # df = df.fillna(0)



    #save file
    if save_file:
        df.to_csv(filepath+filename+"-interpolated"+file_ending, index_label="date")

    return df



# Function to get the date of the first Monday of a given week and year
def get_first_monday(year, week):
    date_str = f'{year}-{week}'
    date_obj = datetime.strptime(date_str + '-1', "%G-%V-%u") #use these for ISO week number
    return date_obj.strftime('%Y-%m-%d')