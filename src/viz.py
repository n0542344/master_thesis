import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import MSTL #multiple seasonal decompose
from pandas.plotting import autocorrelation_plot

from src.config import SAVE_FIGS
from src.utils import save_plots


def line_plot(data):
    # TODO
    pass


def seasonal_plot(data, plot_type: str, col_name = "count"): #New: col_name (from data_model.plot_seasonal)
    #Old version
    # """
    # periods = periods to make seasonal plots for (e.g. week, months)
    # can be "W", "M", "Y" 
    # """
    # fig, ax = plt.subplots(ncols=1, nrows=len(periods))

    # data['day_of_week'] = data['date'].dt.dayofweek
    # data['week'] = data['date'].dt.isocalendar().week
    # data['week_str'] = data['week'].astype(str)
    # data['month'] = data['date'].dt.isocalendar().month

    # for period in periods:
    #     pass
    
    # #TODO
    # fig, ax = plt.subplots()
    # ax.plot(label = ["year"])
    # pass



    #New version, copied from data_model.plot_seasonal:

    #seasonal plot (days of week, week of year, years)
    # 'column': str name of column to plot. Column values must be float or integer

    accepted_types = ["daily", "weekly"]
    if plot_type not in accepted_types:
        raise ValueError("'plot_type' must be 'daily' or 'weekly'")
    
    #drop all except 'date' and column
    series = data[col_name].to_frame()

    # df = data[["date", column]]

    #Get data depending on 'plot_type'
    # Days in a week
    if plot_type == 'daily':
        #Set plotting & naming values:      
        x = 'day_of_week'
        ref_frame = 'week' #comparison period; name of column
        ref_frame_str = 'week_str' #comparison period string;  name of column
        xlabel = 'Day of week'
        xticks_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        title = 'Daily'
        print(type(series))
        print(series.head())
        #Resample daily:
        df = series.resample("D").sum()
        # df = series[col_name].resample("D").sum()
        # df = df.reset_index()
        print(df)
        #Add new columns:
        df[x] = df.index.dayofweek
        df[ref_frame] = df.index.isocalendar().week
        # df[x] = df.index.day_of_week
        # df[ref_frame] = df.index.isocalendar().week #need to make it string later
        df[ref_frame_str] = df[ref_frame].astype(str) #need string for 'hue'


    #Weeks in year
    elif plot_type == 'weekly':
        #Set plotting & naming values:
        x = 'week_of_year'
        ref_frame = 'year' #comparison period; name of column
        ref_frame_str = 'year_str' #comparison period string;  name of column
        xlabel = 'Week number'
        xticks_labels = [str(week) for week in range(1,53)]
        title = 'Weekly'

        #Resample weekly:
        df = series.resample('W').sum()
        # df = df.reset_index()
        #Add new columns:
        df[x] = df.index.isocalendar().week
        df[ref_frame] = df.index.year
        # df[x] = df.index.isocalendar().week
        # df[ref_frame] = df.index.year
        df[ref_frame_str] = df[ref_frame].astype('str')
    # NOTE: could add daily in year, daily in month

    #Settings for plot:
    num_of_lines = df[ref_frame_str].nunique()
    color_palette = sns.color_palette("mako", n_colors=num_of_lines)
    color_palette_reversed = color_palette[::-1]

    #Plotting:
    # ax = sns.scatterplot(x=x, y=col_name, data=df, hue=ref_frame_str, palette=color_palette_reversed, linewidth=0.75, alpha=0.3)
    ax = sns.lineplot(x=x, y=col_name, data=df, hue=ref_frame_str, errorbar=('ci', False), palette=color_palette_reversed, linewidth=0.75)
    ax.set_title(f'{title} seasonality plot for {col_name}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('value')
    ax.set_xticks(ticks=range(len(xticks_labels)), labels=xticks_labels)
    ax.legend(title=plot_type, loc='upper right', bbox_to_anchor=(1, 1))

    if plot_type == 'daily':
        ax.legend([],[], frameon=False)



    #if more than 12 xticks, show only every third label
    if len(ax.get_xticklabels()) > 12:
        for i, label in enumerate(ax.xaxis.get_major_ticks()):
            if i % 3 != 0:
                label.set_visible(False)
                

            
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    



def heatmap(data):
    #TODO
    # plt.figure(figsize=(7, 5))
    # sns.lineplot(x=data.index.month, y=data['count'], ci=None)
    # plt.xlabel('Month')
    # plt.ylabel('Number of XXX') ä#TODO
    # plt.title('Seasonal Plot')
    # plt.xticks(
    #     range(1, 13), 
    #     labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    # )
    # plt.grid(True)
    # plt.show()
    # return plt
    pass




#----------------------------------------------------------------------------------------------
# DECOMPOSITION
#----------------------------------------------------------------------------------------------
#TODO: how to do it? do i want ot have a function, that iterates+decomposes every value?
# Or only the main value (total count), or main values (total count, ec_bg/rh_count, pat_bg/rh_count, etc?)


def decompose_all(df, model, period=7):
    #maybe function to decompose multiple/all columns? or just one fct, 
    # where it iterates over models (and i can pass df.columns minus date)?

    result = seasonal_decompose(df, model='additive', period=period)
    print(result.trend)
    print(result.seasonal)
    print(result.resid)
    print(result.observed)

    # Visualize:
    #TODO: tune plot
    result.plot()
    plt.show()

    return None

def decompose_one(df, model, column, period=7):
    #maybe function to target only one column to decompose?
    result = seasonal_decompose(df[column], model='additive', period=period)
    print(result.trend)
    print(result.seasonal)
    print(result.resid)
    print(result.observed)

    # Visualize:
    #TODO: tune plot
    result.plot()
    plt.show()

    #zoomed in seasonality subplot:
    result.seasonal.plot(figsize=(16,8))
    plt.show()
    
    #zoomed in residual subplot:
    result.resid.plot(figsize=(16,8))
    plt.show()


def multiple_decompose(df, col: str, periods: list):
    # col = col to decompose, i.e. y, for example "count"
    mstl = MSTL(df[col], periods=periods)
    res = mstl.fit()

    # res.plot()
    fig = res.plot()
    fig.autofmt_xdate()
    axes = fig.get_axes()
    for ax in axes:
        for line in ax.get_lines():
            line.set_linewidth(0.5)
            if line.get_marker() not in (None, 'None'):
                line.set_markersize(1)
        # for dot in ax.collections:
        #     dot.set_sizes([1])
    plt.show()

    return mstl

#----------------------------------------------------------------------------------------------
# AUTOCORRELATION
#----------------------------------------------------------------------------------------------

def autocorr(column: pd.Series):
    autocorrelation_plot(column)

    plt.show()


#----------------------------------------------------------------------------------------------
# GENERAL VIZ
#----------------------------------------------------------------------------------------------

def plot_patient_wards(df: pd.DataFrame, n: int, save_figs=SAVE_FIGS, filename="pat_wards_counts", foldername="PAT_WARDS", location="./plots/02_cleaned"):
    """Plots each patients wards time series of daily transfusion count.
    Supposes that df is a wide df, where each ward has its own column with counts.
    Only prints columns that have more than n transfusions.

    Args:
        df (pd.DataFrame): wide dataframe, with separate column for each ward. Column names
        must start with 'PAT_WARD'.
        n (int): number of transfusions must be above n for the column to be plotted.
    """
    for ward in df["PAT_WARD"].unique():

        df_pat_ward_daily = df[df['PAT_WARD'] == ward]
        df_pat_ward_daily.info()
        df_filtered = df_pat_ward_daily.groupby(df_pat_ward_daily.index.date).count()
        if len(df_filtered["PAT_WARD"]) < 500:
            continue

        fig, ax = plt.subplots()
        ax.plot(df_filtered['PAT_WARD'])
        ax.set_xlabel('date')
        ax.set_title(ward)
        if save_figs:
            fig.savefig(fname="/".join([location, "02-" + filename + "_" + ward]))

        fig.show()

        #Old verions: delete
        # df_filtered['PAT_WARD'].plot(x='date')
        # plt.title(ward)
        # plt.show()

        #TODO: save_figs is not done yet -- maybe scrap it altogether
        # if save_figs:
        #     save_plots(fig=fig, filename_general=filename, filename_suffix=ward, location=location, foldername=foldername)

