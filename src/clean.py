# Conforming dates, etc.

import pandas as pd
import datetime as dt
from time import time

from numpy import nan

from pathlib import Path
from numpy import nan

from src import config
from src import config_cleaning
from src.utils import timer_func
from os.path import commonprefix


#-----------------------------------------------------------------------------
# Main wrapper function for cleaning:
#-----------------------------------------------------------------------------
def clean_data(df, 
               existing_file_path: str="./data/02_cleaned/output_cleaned.csv", 
               new_file_path: str="./data/02_cleaned/output_cleaned.csv"):
    #load cleaned file if exists (so i can skip cleaning step):
    my_file = Path(existing_file_path)

    if my_file.is_file():
        df = pd.read_csv(existing_file_path, sep=",")
        #set again to datetime, because first time didnt work? (i think because of merging/missing values)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df.sort_index(inplace=True)
        print(f"Reading existing cleaned file at {existing_file_path}")
    else:
        # ✓ 1. clean date 
        #      ✓ clean different date formatted cols
        #      ✓ merge, set index, sort index
        # x 2. transfusion status
        #      ✓ Get correct mappings from Alex TODO/WAITING
        #      ✓ Implement mapping
        #      x Deal with nan
        #      ✓ merge cols
        # x 3. split 
        print("Reading and cleaning raw data")

        df = clean_dates(df) #DONE
        #set again to datetime, because first time didnt work? (i think because of merging/missing values)
        #df['date'] = pd.to_datetime(df['date'])



        df = clean_transfusion_status(df) #DONE

        # Split columns BG_RH: PAT_BG_RH --> PAT_BG + PAT_RH / EC_BG_RH --> EC_BG + EC_RH
        #TODO: wrap whole thing in function
        #split cols EC_BG_RH and PAT_BG_RH to separate EC_BG_temp/EC_RH_temp and PAT_BG_temp/PAT_RH_temp
        df = split_BG_RH(df, origin="EC_BG_RH", temp_cols=["EC_BG_temp", "EC_RH_temp"], target_cols=["EC_BG", "EC_RH"])
        df = split_BG_RH(df, origin="PAT_BG_RH", temp_cols=["PAT_BG_temp", "PAT_RH_temp"], target_cols=["PAT_BG", "PAT_RH"])
        
        #Merge into EC_BG, EC_RH, PAT_BG, PAT_RH, drop old columns
        df = merge_to_new_col(df, columns_to_merge=["EC_BG", "EC_BG_temp"], new_name="EC_BG")
        df = merge_to_new_col(df, columns_to_merge=["EC_RH", "EC_RH_temp"], new_name="EC_RH")
        df = merge_to_new_col(df, columns_to_merge=["PAT_BG", "PAT_BG_temp"], new_name="PAT_BG")
        df = merge_to_new_col(df, columns_to_merge=["PAT_RH", "PAT_RH_temp"], new_name="PAT_RH")
        
        #parse all _BG/_RH(+temp) columns
        #BG_RH_cols = ["EC_BG", "EC_RH", "PAT_BG", "PAT_RH", "EC_BG_temp", "EC_RH_temp", "PAT_BG_temp", "PAT_RH"]
        #NOTE: coddddduld add top level to dict to map to BG/RH
        #NOTE: could wrap in function 
        for col in ["EC_BG", "PAT_BG"]:
            df[col] = df[col].replace(nan, "nan") #np.nan
            df[col] = df[col].replace(config_cleaning.blood_group_map)
        for col in ["EC_RH", "PAT_RH"]:
            df[col] = df[col].replace(nan, "nan") #np.nan
            df[col] = df[col].replace(config_cleaning.rhesus_factor_map)
            print(f"unique values in for loop: {pd.unique(df[col])}")
        print(f"unique values after for loop: {df.apply(lambda col: col.unique())}")

        #Transform EC_Type to other if not in ["EKF", "EKFX"]
        # (these two make up 54%/38% of values, next is only 2%)
        allowed_ec_types = ["EKF", "EKFX"]
        df.loc[~df["EC_TYPE"].isin(allowed_ec_types), "EC_TYPE"] = "Other"


        #add 'Not applicaple' to PAT_BG, PAT_RH, PAT_WARD where type==expired 
        #TODO: better fct name
        df = add_not_applicable(df)

        df = merge_BG_Rh(df)

        #Save new file:
        print(f"Write new file to {new_file_path}")
        df.to_csv(path_or_buf=new_file_path, sep=",")

    return df

#-----------------------------------------------------------------------------
# (wrapper) functions for main steps
#-----------------------------------------------------------------------------

@timer_func
def clean_dates(df):
    # Date columns: Unify and merge all date columns
    # DONE Merge all date columns
    # DONE wrap in function
    date_cols = [col for col in df.columns if col.startswith("T_")] # or use map: date_cols = date_format_map.keys()
    for col, kwargs in config_cleaning.date_format_map.items():
        df[col] = pd.to_datetime(df[col], **kwargs).dt.date

    df = merge_to_new_col(df, date_cols, "date")

    df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    return df

@timer_func
def clean_transfusion_status(df):
    # Transfusion status: Rename and merge
    # DONE wrap in function

    # for col, val in config_cleaning.transfusion_status_map.items():
    #     df[col].replace(val, inplace=True) # NOTE'replace' doesnt change non-matching vals, 'map' changes them to NaN
    # transfusion_status_columns = [col for col in df.columns if col.startswith("ToD")]

    for col in config_cleaning.transfusion_cols:
        df[col] = df[col].replace(config_cleaning.transfusion_status_map) # NOTE'replace' doesnt change non-matching vals, 'map' changes them to NaN

    # DONE Rename to "DoT" or "Transfusion_date"
    df = merge_to_new_col(df, columns_to_merge=config_cleaning.transfusion_cols, new_name="use")

    return df


@timer_func
def split_BG_RH(df, origin, temp_cols, target_cols):
    df[temp_cols[0]] = df[origin].str[:2]#.apply(extract_BG_from_detailed_notation, args=("bg", )) #with .map() conversion_type="bg"
    df[temp_cols[0]] = df[temp_cols[0]].str.strip()
    #extract rhesus factor
    df[temp_cols[1]] = df[origin].str[2]#.apply(extract_BG_from_detailed_notation, args=("rh", ))
    df[temp_cols[1]] = df[temp_cols[1]].str.strip()

    df = df.drop(origin, axis="columns")
    return df


@timer_func
def parse_BG(df, column: str):
    #parses 'column' for BG and maps new values
    df[column] = df[column] #TODO:
    #use config_cleaning.blood_group_map

@timer_func
def parse_RH(df, column: str):
    #parses 'column' for RH and maps new values
    df[column] = df[column] #TODO:
    #use config_cleaning.rhesus_factor_map


# def split_BG_RH(df, origin, temp_cols, target_cols):
#     """splits column of PAT or EC that contains bg+rh in AB+ccddee-notation to two AB0/Rhesus factor columns,
#     then drops the origin column and the two created columns after merging the extraced columns"""

#     #extract bloodgroup
#     #TODO: delete time measurements
#     start = time() 
#     df[temp_cols[0]] = df[origin].apply(extract_BG_from_detailed_notation, args=("bg", )) #with .map() conversion_type="bg"
#     end1 = time()
#     #extract rhesus factor
#     df[temp_cols[1]] = df[origin].apply(extract_BG_from_detailed_notation, args=("rh", ))
#     end2 = time() 
#     print(f"BG took {end1-start}s,\nRh took {end2-end1}s\ntotal of {end2-start}s.") 

#     df[temp_cols[0]] = df[origin].str.extract(r'(NBN|DSL|fiber|LTE)', expand=False).fillna('Other')

#     # df[temp_cols[0]] = df[origin].replace(regex={r'^NBN.$': 'NBN', 'foo': 'xyz'})

#     #Replace values with map. convert non-matching to 'Other'
#     # for col in config_cleaning.transfusion_cols:
#     #     df[col] = df[col].replace(config_cleaning.transfusion_status_map) # NOTE'replace' doesnt change non-matching vals, 'map' changes them to NaN

#     #         df[col] = df[col].where(df[col].isin(config_cleaning.))




#     return df


@timer_func
def add_not_applicable(df):
    """Fills 'fill_columns' with 'Not applicable', because they're logically cant have a value,
    since no patient was in contact with that EC"""
    #TODO: what about dicarded, why do they have a Patient (bg, rh, ward) assigned?

    fill_columns = ["PAT_BG", "PAT_RH", "PAT_WARD"]
    df[fill_columns] = df[fill_columns].where(df['use'] != 'expired', 'Not applicable')
    
    return df

#-----------------------------------------------------------------------------
# Helper functions
#-----------------------------------------------------------------------------


@timer_func
def merge_to_new_col(df, columns_to_merge: list, new_name: str):
    # Merge multiple non-overlapping columns into one with new_name as the new name
    #WARNING: make sure, there are no rows with overlapping values (like 2 dates)
    
    #Check for overlapping values:
    non_null_counts = df[columns_to_merge].notna().sum(axis=1)
    # mask if two or more columns overlap (with non-na values)
    overlap_mask = non_null_counts >= 2
    overlapping_rows = df[overlap_mask]
    if not overlapping_rows.empty:
        print("Rows with at least two non-null values:")
        print(overlapping_rows)
        raise ValueError("One or more rows are overlapping")

    df[new_name] = df[columns_to_merge].bfill(axis=1).iloc[:, 0]
    #remove new_name from columns_to_merge (which get dropped)
    while new_name in columns_to_merge: columns_to_merge.remove(new_name)    
    df = df.drop(columns_to_merge, axis=1) #, inplace = True)
    return df

@timer_func
def merge_cols(df, cols):
    "cols=List; Merge two non-overlapping cols with values/nan into the first one in cols, REMOVES second column!"
    print(df[cols[0]])
    print(df[cols[1]])
    df[cols[0]].fillna(df[cols[1]], inplace=True)
    df = df.drop(cols[1], axis=1)#, inplace=True)

    return df

#Detailed Rh phenotype notation (ccddee) Format to AB0/Rh 
@timer_func
def extract_BG_from_detailed_notation(original, conversion_type):
    """Use in df.apply() to extract blood group or rhesus from input string,
    depending if conversion type is bg or rh"""
    if pd.isna(original):
        return nan

    #Extract Bloodgroup or rhesus factor
    if original[:3] == "NBN":
        return original[:3]
    
    if conversion_type == "bg":
        bg = original[0:2].strip()
        return bg
    elif conversion_type == "rh":
        try:
            rh = original[2]
        except IndexError:
            rh = "NBN"

        if rh == "-":
            return "Rh positiv"
        elif rh == "+":
            return "Rh negativ"
        elif rh == "NBN":
            return rh
        #TODO: do i need to differntiate between NBN and nan/None?
        else:
            return None




@timer_func
def check_unique_values(df, hidden_cols=["date", "EC_ID_I_hash", "EC_ID_O_hash"]):
    """Hidden cols for cols that have lots of unique values, like date, hash"""
    #TODO: wrap in function, keep fct call, but comment it out.
    # Check (print) unique columns:
    print(df.columns)
    for col in df.columns:
        if col not in hidden_cols:
            print(f"{col}:\n{df[col].unique()}\n")


def merge_BG_Rh(df, merge_cols=[("EC_BG", "EC_RH"),("PAT_BG", "PAT_RH")], suffix="_BG_RH"):
    """Merge back Bloodgroup (BG) with Rhesus factor (RH) column for both EC_BG+EC_RH
    and PAT_BG+PAT_RH respectively. This makes sense, since they belong together.
    Its not of interest, if we have X transfusions with BG A, but hwo many of A+, AB- etc.
    Columns are merged so that BG is first, then Rh (in string).

    Args:
        df (DataFrame): DataFrame with all other cleaning steps done. 
        Must include columns pairs of columns in order of merging in format prefix_suffix, 
        where suffix is differentiating between bg and rh (for prefix finding letter)
        Here/Default: 'EC_BG', 'EC_RH' and 'PAT_BG', 'PAT_RH'
    """

    for pair in merge_cols:
        bg = pair[0]
        rh = pair[1]

        #get prefix:
        # prefix = ""
        # for letter in bg:
        #     if bg[letter] == rh[letter]:
        #         prefix.append(bg[letter])
        #         print(f"prefix: {prefix}")
        # if len(prefix) == 0:
        #     prefix = bg + rh
        prefix = commonprefix([bg, rh])
        new_name = prefix[:-1] + suffix 

        #actual merging: [:-1] to remove '_'
        df[new_name] = df[bg].astype(str) + " " + df[rh].astype(str)
        # df[f"{prefix[:-1]}_BG_RH"] = df[[bg, rh]].agg(' '.join, axis=1)
        df[new_name] = df[new_name].replace(to_replace="Not applicable Not applicable", value="Not applicable")
        #remove old columns
        df = df.drop([bg, rh], axis=1)


        

    return df


