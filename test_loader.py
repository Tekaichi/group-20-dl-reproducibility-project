import time
import torch
from data_loader import clinical_data
from data_analysis import Loader
from models import Model
from lifelines.utils import concordance_index
import numpy as np
from sklearn.model_selection import train_test_split

ld = Loader(data = 3)
mirna = ld.load_miRNA()
clinical = ld.load_clinical_data()
rnaseq = ld.load_rnaseq()
patient_cancer_type = ld.load_patient_cancer_type()
target = ld.load_target()
