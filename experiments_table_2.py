import time
import torch
from data_analysis import Loader
from models import Model
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split
import numpy as np
import threading

print("Loading data")
ld = Loader(data = 6)
mirna = ld.load_miRNA().values.tolist()
clinical = ld.load_clinical_data().values.tolist()
rnaseq = ld.load_rnaseq().values.tolist()
patient_cancer_type = ld.load_patient_cancer_type().values.tolist()
patient_cancer_type = list(np.array(patient_cancer_type).reshape(-1))
target = ld.load_target()
print("Data loaded")
#---- Dont do this manually
gene_expression_input =60483
mirna_input = 1881
clinical_input = 17
#-----


#10 batches
mods = ["mirna","gene_expression",""]


     
        
def run_comb(i):
     
    
    clinical_train, _, rnaseq_train, _, mirna_train,_,target_train,target_test = train_test_split(clinical,rnaseq,mirna,target, test_size = 0.15,stratify = patient_cancer_type)
    Mo = Model(clinical_input = clinical_input,gene_expression_input =gene_expression_input,mirna_input = mirna_input)
       
    
    device = "cpu"
    target_train.index = [i for i in range(len(target_train))]
    days_to_death = target_train["days_to_death"].values
    
    data = {"gene_expression" :torch.tensor(rnaseq_train,device = device),
            "mirna":    torch.tensor(mirna_train,device = device),
            "clinical": torch.tensor(clinical_train,device = device),
            }
    if i != "":
        del data[i]
    f = open("table2_{0}.txt".format(data.keys()),"a")
    f.write("\nFiles used: {0}".format(ld.files))
  
    f.write("{0}".format(data.keys()))
    now = time.time()
    Mo.train(data,target_train,n_batches = 10)
    took = time.time()-now
    print("Train time:",took)
    f.write("Took {0}".format(took))
    
    for cancer_type in set(patient_cancer_type):
        

        indexes = [index for index, value in enumerate(patient_cancer_type) if value == cancer_type]
        type_rnaseq = np.array(rnaseq)[indexes]
        type_rnaseq = [list(i) for i in type_rnaseq]
        type_mirna =  np.array(mirna)[indexes]
        type_mirna = [list(i) for i in type_mirna]
        type_clinical = np.array(clinical)[indexes]
        type_clinical = [list(i) for i in type_clinical]
        print("\nTesting Data---for", cancer_type)
        f.write("\nTesting Data---for{0}--{1}".format(cancer_type,i))
        days_to_death = target_test["days_to_death"].values
        vital_status = target_test["vital_status"].values
        
        data = {"gene_expression" :torch.tensor(type_rnaseq),
                "mirna":    torch.tensor(type_mirna),
                "clinical": torch.tensor(type_clinical),
                }
        if i != "":
            del data[i]
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
           
            
        write = "\nC_index:{0} {1} {2}".format(c_index_1,c_index_2,c_index_3)
        f.write(write)
    f.close()


for i in mods:
    
    print("Starting Thread:",i)
    x = threading.Thread(target= run_comb, args=(i,))
    x.start()
    
