#Global Variables and settings
import pandas as pd
from time import time
ENABLE_TIMING = False #If true, print messages. #TODO: add logging module to do that
ENABLE_LOGGING = True
SAVE_FIGS = False

#Slice data while developing
DEV_START_DATE = "2024-05-01"
DEV_END_DATE = "2025-07-01"
COLUMN = "use_transfused"


# DATE: Mapping of column names with their respective format
# DONE: change all date formats to iso without time
date_format_map = {
    "T_XL" : {"unit" : "D", "origin" : "1899-12-30"},
    "T_ISO_T" : {"yearfirst" : True}, 
    "T_DE" : {"dayfirst" : True}, 
    "T_US_T" : {"format" : "%m/%d/%y %H:%M"}, #%H for 24h clock, %I for 12h clock
    # "T_DE_S" : {"format" : "%d.%m.%y"}, #y for short year
    "T_DE_S" : {"format" : "%m-%d-%y"} #in new data (2025-07-16) its mm-dd-yy
    #"T_US" : {"format" : "%m-%d-%y"}, #in new data (2025-07-16) doesnt exist anymore
    #"T_DE_T" : {"format" : "%d.%m.%y %H:%M"}, #in new data (2025-07-16) doesnt exist anymore
    #"T_ISO" : {"yearfirst" : True} #in new data (2025-07-16) doesnt exist anymore
}

#Columns with info about EC status ('discarded', 'expired' etc.)
transfusion_cols = ["ToD", "ToD_N"]# "ToD_O" doesnt exist anymore (2025-07-16 data)

#Mapping of transfusion status raw --> processed
#See Mail from 14.7.25
transfusion_status_map = {
        "Transfundiert" : "transfused",
        "VER" : "transfused", #'Verabreicht' = Zum patienten gekommen. Ob tatsächlich verabreicht ist unbekannt
        "Verkauft": "transfused", #verkauft an andere krankenanstalt; wie 'ausgegeben'
        "Ausgegeben": "transfused", #Zum patienten gekommen. Ob tatsächlich verabreicht ist unbekannt


        "AUS" : "discarded", # 'Ausgegeben'
        "Entsorgt" : "discarded",

        "ABG" : "expired", # Abgelaufen == expired
        "Abgelaufen": "expired", 
        "expired" : "expired",
        "VRN" : "discarded", #Vernichtet


         #nan : "???" ,
        "BER" : "???", #NOTE: vermutlich 'bereitgestellt' (wie ausgegeben?)
        "END" : "???", #NOTE: mapping?
        "RES" : "???", #NOTE: vermutlich 'reserviert'
        "RET" : "???" #NOTE: retourniert -- wie klassifizieren?
}

#NOTE: it would be better imo to have the cleaned value as key and original values as values,
# so that you dont repeat it so often. But then dict needs to be reversed, for replacement.
# see solution here https://stackoverflow.com/questions/35491223/inverting-a-dictionary-with-list-values (not implemented!)

#Rhesus factor (EC/PAT) mapping:
rhesus_factor_map = {
    "Rh negativ" : "Rh negative",
    "-" : "Rh negative",
    "N" : "Rh negative", #TODO: is N Negative or Nicht bestimmt?
    
    "Rh positiv" : "Rh positive",
    "+" : "Rh positive",

    
    "nan" : "NB",
    # "Not applicable" : "NB",
    "NBN" : "NB",
    "Rh nicht bestimmb." : "NB",
    "KMT Rh n. bestimmb." : "NB",
    "Sonderfall" : "NB",

    "Rh D weak" : "NB",
    "Rh D var" : "NB",
    "Rh Du" : "NB"
}

#Blood group (EC/PAT) mapping:
blood_group_map = {
    "A" : "A",
    "A2" : "A", #Subgroup of A, one occurence
    "0" : "0",
    "0.0" : "0",
    "B" : "B",
    "AB" : "AB",

    "nan" : "NB",
    # "Not applicable" : "NB",
    "NB" : "NB",
    #"NBN" : "NBN",
    "BG nicht bestimmb." : "NB",
    "KMT BG n. bestimmb." : "NB",
    "BG Unbekannt" : "NB"
}





#EC type 
# Values to keep, all other change to "Other"
# (full bag, split bag etc) -- only two values in relevant amounts
ec_type_keep = ["EKF", "EKFX"]


# WARDs
wards = {

}




