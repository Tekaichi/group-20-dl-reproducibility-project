import pandas as pd
from lifelines import KaplanMeierFitter
from pandas import json_normalize
import matplotlib.pyplot as plt

# From the paper: Each patient has a time of death recorded [...] up to a maximum of 11 000 days
# after diagnosis across all cancer sites
MAX_DAYS_TO_DEATH = 11000
ALIVE = 0
DEAD = 1

biospecimen_file = "pancancer_biospecimen.csv"
clinical_data_file = "original_clinical_dataset.json"

biospecimen = pd.read_csv(biospecimen_file, "\t")
clinical_data = pd.read_json(path_or_buf = clinical_data_file)
demographic = json_normalize(clinical_data.demographic)

# data cleaning
drop_keys_specimen = ["aliquot", "sample_type", "ffpe"]
biospecimen = biospecimen.drop(drop_keys_specimen, axis="columns")
biospecimen["barcode"] = biospecimen["barcode"].str.slice(0, 12)
biospecimen["project"] = biospecimen["project"].str.slice(5, 9)
biospecimen = biospecimen.drop_duplicates()

relevant_keys_demographic = ["submitter_id", "days_to_death", "vital_status"]
demographic = demographic.filter(relevant_keys_demographic)
demographic["submitter_id"] = demographic["submitter_id"].str.slice(0, 12)
demographic = demographic.rename(columns = {"submitter_id": "barcode"})

data = pd.merge(demographic, biospecimen, on = "barcode", how = "inner")
data = data.replace({"vital_status" : {"Alive": ALIVE, "Dead": DEAD}})

# drop patients without information about vital_status
data = data[(data.vital_status == 0) | (data.vital_status == 1)]

# get different cancer types
cancer_types = data["project"].unique()
data_by_cancer_type = pd.DataFrame({"cancer_type" : cancer_types})

# check how many patients have/had each cancer type and how many didn't survive
for t in cancer_types:
    aux = data.loc[data["project"] == t]
    data_by_cancer_type.loc[(data_by_cancer_type.cancer_type == t), "total_count"] = len(aux)
    dead_count = aux["vital_status"].value_counts()[DEAD]
    data_by_cancer_type.loc[(data_by_cancer_type.cancer_type == t), "deaths_count"] = dead_count

# calculate mortality rate for each cancer type
data_by_cancer_type["mortality_rate"] = data_by_cancer_type["deaths_count"] / data_by_cancer_type["total_count"]
# sort cancer type by mortality rate and reset index
data_by_cancer_type = data_by_cancer_type.sort_values(by = ["mortality_rate"])
data_by_cancer_type = data_by_cancer_type.reset_index(drop = True)
cancer_types = data_by_cancer_type["cancer_type"].tolist()
cancer_highest_survival_rate = sorted(cancer_types[0:16])
cancer_lowest_survival_rate = sorted(cancer_types[17:33])

def plot_kaplan_meier(kmf, cancer_type_list):
    kmf = KaplanMeierFitter()
    for c in cancer_type_list:
        print(c)
        aux = data.loc[data["project"] == c]
        print(aux)
        duration = aux["days_to_death"]
        observed  = aux["vital_status"]
        # fill days_to_death of patients alive with the maximum value of patients not alive
        duration = duration.fillna(duration.max())
        kmf.fit(duration, observed, label = c)
        kmf.plot(ci_show = False)

# create plot highest mean overall survival
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
kmf_higest = KaplanMeierFitter()
plot_kaplan_meier(kmf_higest, cancer_highest_survival_rate)
plt.xlabel("Time")
plt.ylabel("Survival probability")
plt.legend(bbox_to_anchor=(0.9, 1.3), ncol = 4)

# create plot highest mean overall survival

plt.subplot(1, 2, 2)
kmf_lowest = KaplanMeierFitter()
plot_kaplan_meier(kmf_lowest, cancer_lowest_survival_rate)
plt.xlabel("Time")
plt.ylabel("Survival probability")
plt.legend(bbox_to_anchor=(0.9, 1.3), ncol = 4)
 
