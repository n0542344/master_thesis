#Needs to be extra file to prevent circular import, because src/utils.py imports src/config.py
from itertools import combinations

#----------------------------------------------------------------------------------------------
# CONFIG / Grid search 
#----------------------------------------------------------------------------------------------

    

#Gets combinations of values() -- Lists, not the items -- to return as possible parameters in gs_configs.
def get_exog_list_combinations(input_dict):
    keys = list(input_dict.keys())

    #Get combinations of all keys of exog_types dict.
    all_key_combinations = []
    for k in range(1, len(keys) + 1):
        combos = list(combinations(keys, k))
        all_key_combinations.extend(combos)

    #Get lists corresponding to the key (pairs) of all_combinations
    combos_list_of_tuples = []
    for combo in all_key_combinations:
        merge = []
        for v in combo:
            merge.extend(input_dict.get(v))
        combos_list_of_tuples.append((list(combo), merge))

    return combos_list_of_tuples
