import time
import torch
from data_loader import clinical_survival
from models import Model
gene_expression =60483
mirna = 1881
clinical = 4
N_patients = 100 #Testing purposes

Mo = Model(gene_expression,mirna,clinical)

_,survival = clinical_survival()

survival = survival[:N_patients]

data = {"gene_expression" :torch.rand(N_patients,gene_expression),
        "mirna":    torch.rand(N_patients,mirna),
        "clinical": torch.rand(N_patients,clinical),
        }
now = time.time()
Mo.train(data,survival= survival)
print("Train time:",time.time() - now)
