#%%
import pandas as pd
from datetime import datetime, timedelta
filepath, filename, file_ending = "../data/00_external_data/", "grippemeldedienst", ".csv"

#%%
df = pd.read_csv(filepath+filename+file_ending)
main_df = pd.read_csv("../data/03_transformed/output_transformed.csv")
timeframe = main_df["date"]
# Function to get the date of the first Monday of a given week and year
def get_first_monday(year, week):
    date_str = f'{year}-{week}'
    date_obj = datetime.strptime(date_str + '-1', "%G-%V-%u")
    return date_obj.strftime('%Y-%m-%d')

#convert Kalenderwoche to strings
df['weeknum'] = df['Kalenderwoche'].str.extract(r'(\d+)').astype(int)
df['year'] = df["Jahr"]#pd.to_datetime(df['Jahr'], format="%Y")

# for index, row in df.iterrows():
#     get_first_monday(row['year'], row['weeknum'])
# Apply the function to create the 'weekDate' column
df['date'] = df.apply(lambda row: get_first_monday(row['year'], row['weeknum']), axis=1)
#%%
df = df.rename(columns={"Neuerkrankungen pro Woche": "influenza_weekly"})
df.index = pd.to_datetime(df["date"])
df = df[["influenza_weekly"]]
df["influenza_weekly"] = pd.to_numeric(df["influenza_weekly"], errors="coerce")
df["influenza_daily"] = df["influenza_weekly"]//7
#TODO: linear interpolate
# Add missing days
# df.index = timeframe
# df = df.set_index("date")
df.resample("D").asfreq()#.fillna()

#Extend by one week (to interpolate last week which has no rows)
new_index = pd.date_range(start=df.index[0], end=df.index[-1] + pd.Timedelta(days=6), freq='D')
df = df.reindex(new_index)

#forward fill new_cases weekly
df["influenza_weekly"] = df["influenza_weekly"].ffill()
#linearly interpolate influenza_daily
df["influenza_daily"] = df["influenza_daily"].interpolate(method="linear", limit=6, limit_direction="forward").round()


