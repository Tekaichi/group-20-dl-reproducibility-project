import pandas as pd
import numpy as np
file_location = "data/original_clinical_dataset.json"
full_data = pd.read_json(path_or_buf = file_location)

def clinical_survival():
    demographic = []
    survival = []
    keys = ["gender","race","age_at_index"]
    keys_2 = ["vital_status","days_to_death"]
    keys_3 = ["vital_status"]
    for x in full_data.values:
        df = pd.DataFrame(x[1],index=[0])
        reldata = df[keys].iloc[0].values
        try:
            rel_data = df[keys_2].iloc[0].values
        except:
            rel_data = df[keys_3].iloc[0].values
        #case_id gender rage age_at_index
        demographic.append(np.concatenate(([x[0]],reldata)))
        survival.append(np.concatenate(([x[0]],rel_data)))
    return demographic, pd.DataFrame(survival)        
    


