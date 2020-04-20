import time
import torch
from data_analysis import Loader
from models import Model
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split

print("Loading data")
ld = Loader()
mirna = ld.load_miRNA().values.tolist()
clinical = ld.load_clinical_data().values.tolist()
rnaseq = ld.load_rnaseq().values.tolist()
patient_cancer_type = ld.load_patient_cancer_type().values.tolist()
target = ld.load_target()

print("Data loaded")
#---- Dont do this manually
gene_expression_input =60483
mirna_input = 1881
clinical_input = 17
#-----

import numpy as np
patient_cancer_type = np.array(patient_cancer_type).reshape(-1)


for cancer_type in set(patient_cancer_type):
    if(cancer_type in ignore):
        continue
    f = open("table2_single_cancer-3 modalities_v2.txt","a")
    print("Cancer type:",cancer_type,"starting")
    indexes = [index for index, value in enumerate(patient_cancer_type) if value == cancer_type]
    type_rnaseq = np.array(rnaseq)[indexes]
    type_rnaseq = [list(i) for i in type_rnaseq]
    type_mirna =  np.array(mirna)[indexes]
    type_mirna = [list(i) for i in type_mirna]
    type_clinical = np.array(clinical)[indexes]
    type_clinical = [list(i) for i in type_clinical]
    type_target = target.iloc[indexes]
    f.write("\n")
    f.write(str(cancer_type))
    f.write("\n{0} data\n".format(len(type_target)))
    print("{0} data".format(len(type_target)))
    clinical_train, clinical_test, rnaseq_train, rnaseq_test, mirna_train,mirna_test,target_train,target_test = train_test_split(type_clinical,type_rnaseq,type_mirna,type_target, test_size = 0.15)
    Mo = Model(clinical_input = clinical_input,gene_expression_input =gene_expression_input,mirna_input = mirna_input)
    
    
    device = "cpu"
    target_train.index = [i for i in range(len(target_train))]
    days_to_death = target_train["days_to_death"].values
    
    data = {"gene_expression" :torch.tensor(rnaseq_train,device = device),
            "mirna":    torch.tensor(mirna_train,device = device),
            "clinical": torch.tensor(clinical_train,device = device),
            }
          
    now = time.time()
    Mo.train(data,target_train)
    took = time.time()-now
    print("Train time:",took)
    
    print("Testing Data---for", cancer_type)
    days_to_death = target_test["days_to_death"].values
    vital_status = target_test["vital_status"].values
    
    data = {"gene_expression" :torch.tensor(rnaseq_test),
            "mirna":    torch.tensor(mirna_test),
            "clinical": torch.tensor(clinical_test),
            }
    hazard = Mo(data)["hazard"].detach()
    
   
    try:
        c_index_1= concordance_index(days_to_death,-hazard)
    except:
        c_index_1 = "None"
    try:
        c_index_2= concordance_index(days_to_death,hazard)
    except:
        c_index_2= "None"
    try:
        c_index_3 =concordance_index(days_to_death,-hazard,np.logical_not(vital_status))
    except:
        c_index_3 = "None"
       
        
    write = "C_index:{0} {1} {2}".format(c_index_1,c_index_2,c_index_3)
    f.write(write)
    f.close()
   