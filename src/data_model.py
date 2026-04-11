#%% Make DATA class for dataframe
# so i can add methods for plotting etc.
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
import seaborn as sns
import pandas as pd
import dayplot as dp 
import calplot


from statsmodels.graphics.tsaplots import month_plot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import MSTL #multiple seasonal decompose

month_names_short = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
day_names_short = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

plt.rcParams.update({
    'axes.titlesize': 20,
    'axes.labelsize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'axes.titleweight': 'bold'#,
    #'axes.labelweight': 'bold'
})


class Data(pd.DataFrame):
    # Class for the processed data, which contains methods for plotting
    # and transforming, which will be passed into class 'Model'
    # TODO: change parameter of model to be of 'Data' type 
    
    #Subclass pandas DataFrame, to still use df methods (info(), head(), slicing[], ...)
    def __init__(self, data: pd.DataFrame):
        #index is datetime 
        super().__init__(data)
        self.data = data.sort_index()
        #self.add_year_month_day()
        
    @property
    def _constructor(self):
        return Data

    def add_year_month_day(self) -> None:
        """Add 'year', 'month', 'day' as separate columns"""
        self.data['year'] = self.data.index.year 
        self.data['month'] = self.data.index.month 
        self.data['day'] = self.data.index.day
        self.data = self.data.reset_index()
        self.data['day_of_year'] = self.data['date'].dt.dayofyear #.timetuple().tm_yday
        self.data = self.data.set_index('date')

    #Methods:
    def print_head(self): 
        print(type(self.data))
        print(self.data.head())


    def plot_line(self, col_name: str):
        #simple line plot
        fig, ax = plt.subplots()
        if col_name == None:
            ax.plot(self.data[0])
        else:
            ax.plot(self.data[col_name])

        plt.show()

    def plot_boxplots(self, col_name: str, max_y=None):
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 12), sharey=True, constrained_layout = True)
        
        if not max_y:
            max_y = self.data[col_name].max() * 1.1
        # for a in ax.flatten():
        #     a.set_ylim(top=max_y)
        plt.setp(ax, ylim=(0, max_y))

        #yearly
        sns.boxplot(x=self.data.index.year, y=self.data[col_name], ax=ax[0][0])
        ax[0][0].set_title('Year-wise Box Plot', loc='center')
        ax[0][0].set_xlabel('Year')
        ax[0][0].set_ylabel(col_name)
        #if more than 12 xticks, show only every third label
        if len(ax[0][0].get_xticklabels()) > 12:
            for i, label in enumerate(ax[0].xaxis.get_major_ticks()):
                if i % 4 != 0:
                    label.set_visible(False)

        #Monthly
        sns.boxplot(x=self.data.index.month, y=self.data[col_name], ax=ax[0][1])
        #interval for xticks + rename numbers to months
        # ax[0][1].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        #ax[0][1].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax[0][1].set_title('Month-wise Box Plot', loc='center')
        ax[0][1].set_xticks(range(0, 12))
        ax[0][1].set_xticklabels(month_names_short)
        ax[0][1].set_xlabel('Month')
        ax[0][1].set_ylabel(col_name)

        #weekly over year
        sns.boxplot(x=self.data.index.isocalendar().week, y=self.data[col_name], ax=ax[1][0])
        ax[1][0].set_title('Weekly Box Plot', loc='center')
        ax[1][0].xaxis.set_major_locator(ticker.MultipleLocator(7))
        ax[1][0].set_xlabel('Week')
        ax[1][0].set_ylabel(col_name)

        #weekdays
        sns.boxplot(x=self.data.index.dayofweek, y=self.data[col_name], ax=ax[1][1])
        ax[1][1].set_title('Day of the week Box Plot', loc='center')
        ax[1][1].set_xticks(range(7))
        ax[1][1].set_xticklabels(day_names_short)
        ax[1][1].set_xlabel('Week')
        ax[1][1].set_ylabel(col_name)



    def plot_seasonal(self, plot_type: str, col_name: str, fig_location=None):
        #seasonal plot (days of week, week of year, years)
        # 'column': str name of column to plot. Column values must be float or integer

        accepted_types = ["daily", "weekly"]
        if plot_type not in accepted_types:
            raise ValueError("'plot_type' must be 'daily' or 'weekly'")
        
        #drop all except 'date' and column
        series = self.data[col_name].to_frame()


        # df = self.data[["date", column]]

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
            df = series[col_name].resample("D").sum()
            df = df.reset_index()

            #Add new columns:
            df[x] = df['date'].dt.day_of_week
            df[ref_frame] = df['date'].dt.isocalendar().week.astype('str') #need to make it string later
            #df[ref_frame_str] = df[ref_frame].astype(str) #need string for 'hue'

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
            df = df.reset_index()
            #Add new columns:
            df[x] = df['date'].dt.isocalendar().week
            df[ref_frame] = df['date'].dt.year.astype('str')
            #df[ref_frame_str] = df[ref_frame].astype('str')
        
        # NOTE: could add daily in year, daily in month
        #rename ref_frame_str to ref_name for proper legend title
        #df = df.drop(ref_frame, axis=1).rename(columns={ref_frame:ref_frame_str})


        #Get min/max dates for title
        min_date = pd.to_datetime(df['date'].min()).date()
        max_date = pd.to_datetime(df['date'].max()).date()

        # Numeric ref_frame values for colormap
        df['_ref_num'] = df[ref_frame].astype(int)
        ref_values = sorted(df['_ref_num'].unique())
        cmap = plt.get_cmap('mako')
        norm = matplotlib.colors.Normalize(vmin=ref_values[0], vmax=ref_values[-1])

        #Plotting: one line per ref_frame value, colored by colormap
        fig, ax = plt.subplots(figsize=(12, 6))
        for val in ref_values:
            subset = df[df['_ref_num'] == val]
            grouped = subset.groupby(x)[col_name].mean()
            ax.plot(grouped.index, grouped.values, color=cmap(norm(val)), linewidth=0.75)

        ax.set_title(f'{title} seasonality plot for {col_name}: {min_date} to {max_date}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('count')
        ax.set_xticks(ticks=range(len(xticks_labels)), labels=xticks_labels)

        #if more than 12 xticks, show only every third label
        if len(ax.get_xticklabels()) > 12:
            for i, label in enumerate(ax.xaxis.get_major_ticks()):
                if i % 3 != 0:
                    label.set_visible(False)

        ax.grid(True)

        # Colorbar outside plot
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label(ref_frame.replace('_', ' ').title())
        # For few distinct values (e.g. years), label each tick explicitly
        if len(ref_values) <= 15:
            cbar.set_ticks(ref_values)
            cbar.set_ticklabels([str(v) for v in ref_values])

        plt.tight_layout()
        if fig_location:
            plt.savefig(fname="/".join([fig_location, "_" + plot_type]))

        plt.show()

    def plot_seasonal_subseries(self, col_name: str):
        #plot seasonal subseries
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 6))
        # fig, ax = plt.subplot(figsize=(16, 6))
        data_temp = self.data[[col_name]].resample("MS").sum() #double [[]] to keep as df
        month_plot(data_temp[col_name], ylabel="cOUNT", ax=ax) #col_name.capitalize()
        #month_plot(self.data[col_name], ylabel=col_name.capitalize(), ax=ax[0])

    def plot_daily_heatmap(self, col_name: str):
        #plot heatmap for daily values:

        #Avg. per day of year:
        #avg_per_day_of_year = self.data["count"].groupby([self.data.index.month, self.data.index.day]).mean() #, as_index=False
        

        dp.calendar(
            dates=self.data.index,
            values=self.data[col_name]
        )





    def plot_overview(self):
        # aggregate method to plot multiple functions
        pass

    #----------------------------------------------------------------------------------------------
    # TIME SERIES PLOTS (acf etc.)
    #----------------------------------------------------------------------------------------------


    def plot_autocorrelation(self, col_name: str):
        # from pandas.plotting import autocorrelation_plot
        # autocorrelation_plot(self.data[col_name])
        # plt.show()

        #new:

        plot_acf(x=self.data[col_name])
        plt.show()



    def plot_partial_autocorrelation(self, col_name: str):

        plot_pacf(x=self.data[col_name])
        plt.show()


    #----------------------------------------------------------------------------------------------
    # DECOMPOSITION
    #----------------------------------------------------------------------------------------------
    #TODO: how to do it? do i want ot have a function, that iterates+decomposes every value?
    # Or only the main value (total count), or main values (total count, ec_bg/rh_count, pat_bg/rh_count, etc?)

    def decompose_one(self, col_name: str, model: str='additive', period=7):
        #maybe function to target only one column to decompose?

        result = seasonal_decompose(self.data[col_name], model=model, period=period)
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
        plt.title("Seasonal part zoomed in")
        plt.show()
        
        #zoomed in residual subplot:
        result.resid.plot(figsize=(16,8))
        plt.title("Residuals part zoomed in")
        plt.show()

    def decompose_all(self, model: str='additive', period: int=7):
        #maybe function to decompose multiple/all columns? or just one fct, 
        # where it iterates over models (and i can pass df.columns minus date)?


        #TODO: add functionality to decompose all columns
        # i think now its the same as decompose one? (see original in 'viz.py')
        result = seasonal_decompose(self.data, model=model, period=period)
        print(result.trend)
        print(result.seasonal)
        print(result.resid)
        print(result.observed)

        # Visualize:
        #TODO: tune plot
        result.plot()
        plt.show()


    def multiple_decompose(self, col_name: str, periods: list):
        # col = col to decompose, i.e. y, for example "count"


        mstl = MSTL(self.data[col_name], periods=periods)
        res = mstl.fit()

        res.plot()
        plt.show()

        #zoomed in seasonality subplot:
        res.seasonal.plot(figsize=(16,8))
        plt.title("Seasonal part zoomed in")
        plt.show()
        
        #zoomed in residual subplot:
        res.resid.plot(figsize=(16,8))
        plt.title("Residuals part zoomed in")
        plt.show()

        #return mstl

