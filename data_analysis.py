import pandas as pd
from pandas import json_normalize
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import glob
import math
import random

class Loader:
    
    BIOSPECIMEN_FILE = "./data/pancancer_biospecimen.csv"
    FILE_CLINICAL_DATA = "./data/original_clinical_dataset.json"
    DIR_MIRNA = "./data/miRNA/"
    DIR_RNASEQ = "./data/rnaseq/"
    
    def __init__(self,data = None):
        self.end = data
        self.files = []
        self.frame = self.__load_frame()
       
    def __load_frame(self):
        clinical_data = self.__load_clinical_data()
        miRNA = self.__load_miRNA()
        rnaseq = self.__load_rnaseq()
        
        clinical_data = self.__process_clinical_data(clinical_data)
        miRNA, rnaseq = self.__process_miRNA_rnaseq(miRNA, rnaseq)
        
        patient_cancer_type = self.__process_biospecimen()
        
        joined = clinical_data.join(miRNA,how="inner")
        joined = joined.join(rnaseq,how="outer")
     
        joined = patient_cancer_type.join(joined,how="inner")
        joined  = joined.fillna(0)
        
        return joined

    def __load_miRNA(self):
        #miRNA = self.__load_pkl_single_file(self.DIR_MIRNA)
        miRNA = self.__load_pkl(self.DIR_MIRNA)
        miRNA = miRNA.sort_index()
        return miRNA

    def __load_rnaseq(self):
        #rnaseq = self.__load_pkl_single_file(self.DIR_RNASEQ)
        rnaseq = self.__load_pkl(self.DIR_RNASEQ)
        rnaseq = rnaseq.sort_index()
        return rnaseq
    
    #This should retrieve paired files.
    def __load_pkl(self, dir):
        files = [f for f in glob.glob(dir + "*.pkl") if "annot" not in f]
        files.sort()
        if(self.end is not None):
            files = files[0:self.end]
        frames = []
        self.files = self.files + files
        for file in files:
            frames.append(pd.read_pickle(file, 'gzip'))
        return pd.concat(frames)
        
    def __load_pkl_single_file(self, dir):
        if (dir == self.DIR_MIRNA):
            return pd.read_pickle(dir + "thyroid_THCA_miRNA_data.pkl", 'gzip')
        else:
            return pd.read_pickle(dir + "thyroid_THCA_rnaseq_data.pkl", 'gzip')

    def __load_clinical_data(self):
        clinical_data = pd.read_json(path_or_buf = self.FILE_CLINICAL_DATA)
        return clinical_data

    def __truncate_barcode(self, data): 
        return data.index.map(lambda x: x[0:12])

    def __log_transform(self, data):
        return np.log(data+1)

    def __standard_scaler(self, data):
        scaler = StandardScaler()
        np_data = scaler.fit_transform(data)
        return pd.DataFrame(np_data, index = data.index.values)

    def __process_biospecimen(self):
        biospecimen = pd.read_csv(self.BIOSPECIMEN_FILE, "\t")
        drop_keys_specimen = ["aliquot", "sample_type", "ffpe"]
        biospecimen = biospecimen.drop(drop_keys_specimen, axis="columns")
        biospecimen["barcode"] = biospecimen["barcode"].str.slice(0, 12)
        biospecimen["project"] = biospecimen["project"].str.slice(5, 9)
        biospecimen = biospecimen.drop_duplicates()
        biospecimen = biospecimen.set_index("barcode")
        biospecimen = biospecimen.sort_index()
        return biospecimen
    
    def __process_clinical_data(self, clinical_data):
        demographic = json_normalize(clinical_data.demographic)
        relevant_keys_demographic = ["submitter_id", "race", "gender", "age_at_index", "days_to_death", "vital_status"]
        
        demographic = demographic.filter(relevant_keys_demographic)
        demographic["tumor_stage"] = clinical_data["diagnoses"].astype(str).str.extract("'tumor_stage': '(.*?)'", expand = True)
        demographic["tumor_stage"] = demographic["tumor_stage"].fillna("not reported")
        demographic["tumor_stage"] = demographic["tumor_stage"].str.extract("(.*[div])", expand = True)
        
        demographic["submitter_id"] = demographic["submitter_id"].str.slice(0, 12)
        demographic = demographic.set_index("submitter_id")
        demographic.index = demographic.index.rename("barcode")        
        demographic = demographic.sort_index()
        
        demographic["days_to_death"] = demographic["days_to_death"].fillna(math.inf)
        demographic = demographic.replace({"vital_status" : {"Alive": 0, "Dead": 1, "Not Reported": 0}})
        
        # one hot encoding
        demographic = pd.get_dummies(demographic, prefix=['race'], columns = ['race'])
        demographic = pd.get_dummies(demographic, prefix=['gender'], columns = ['gender'])
        demographic = pd.get_dummies(demographic, prefix=['tumor_stage'], columns = ['tumor_stage'])
        
        demographic = demographic.add_prefix("c_")
        demographic = demographic.rename(columns={"c_vital_status": "vital_status", "c_days_to_death": "days_to_death"})
        
        return demographic
    
    def __process_miRNA_rnaseq(self, miRNA, rnaseq):
        miRNA.index = self.__truncate_barcode(miRNA)
        rnaseq.index = self.__truncate_barcode(rnaseq)

        miRNA = self.__log_transform(miRNA)    
        rnaseq = self.__log_transform(rnaseq)

        miRNA = self.__standard_scaler(miRNA)
        rnaseq = self.__standard_scaler(rnaseq)
        
        miRNA = miRNA.add_prefix("m_")
        rnaseq = rnaseq.add_prefix("g_")
        
        return miRNA, rnaseq
    
    def load_clinical_data(self):
        filter_col = [col for col in self.frame if col.startswith('c_')]
        return self.frame[filter_col]
    
    def load_target(self):
        return self.frame[["vital_status", "days_to_death"]]
    
    def load_miRNA(self):
        filter_col = [col for col in self.frame if col.startswith('m_')]
        return self.frame[filter_col]

    def load_rnaseq(self):
        filter_col = [col for col in self.frame if col.startswith('g_')]
        return self.frame[filter_col]

    def load_patient_cancer_type(self):
        return self.frame[["project"]]
