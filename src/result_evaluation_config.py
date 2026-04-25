from matplotlib import pyplot as plt
IMG_PATH = "../thesis_text/images"
TBL_PATH = "../thesis_text/tables"

CHAPTER = "05"
RESULTS_PATH = "./results" #"./results_FirstRun"

PATH_RAW_DATA = "./data/01_raw"

START_DATE = "2025-01-01"

#start/end date, when predictions first start in models.
# used for comparsion model.
FC_START_DAY_1 = "2024-07-05"
FC_END_DAY_1 = "2025-06-19"
FC_END_DAY_14 = "2025-07-02"

#Initial load of results
PATH_ARIMA = f"{RESULTS_PATH}/Arima"
PATH_SARIMAX = f"{RESULTS_PATH}/Sarimax"
PATH_LSTM = f"{RESULTS_PATH}/LSTM"
PATH_PROPHET = f"{RESULTS_PATH}/Prophet"
SEP = ","
INDEX_COL = "id"
#today = datetime.today().strftime('%Y_%m_%d')



#For data_exploration:
PIPE_START = "2020-07-05"
PIPE_END = "2025-07-03"

SUBSET_START = "2024-01-01"
SUBSET_END = "2024-12-31"


#FORMATTING
DATE_FRMT = '%Y-%m-%d'


#Graphical settings:
plt.style.use('seaborn-v0_8-deep')

plt.rcParams.update({
    'font.size': 24, #12,
    'font.family': 'serif',
    #Axes
    'axes.labelsize': 32, #14,
    'axes.titlesize': 32, #16,
    #Ticks
    'xtick.labelsize': 24, #11,
    'ytick.labelsize': 24, #11,
    #Legend
    'legend.fontsize': 32, #11,
    #'figure.figsize': (10, 6),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'figure.titlesize': 32,
    # 'axes.grid': True,
    # 'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

m_cmap = {
    "arima" : "darkorange",
    "sarimax" : "firebrick",
    "lstm" : "dodgerblue",
    "prophet" : "lightseagreen"
}

