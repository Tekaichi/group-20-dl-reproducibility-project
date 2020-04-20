from data_loader import clinical_survival
import torch

_, survival_data = clinical_survival()


n = 30



survival = survival_data[:n]


filter_ = (survival[1] == "Dead").values

#---
import random
x = [random.randint(0,1)/1.0 for i in range(n)]
x = [x[i] for i in range(0,len(x)) if filter_[i]]
x = torch.tensor(x)
#---
y = torch.tensor(survival[filter_][2].values)


def l_cox(x,y):
    total = torch.tensor(0,dtype=torch.float64)
    for i in range(0,len(x)):
        if(y[i] == 0):
            continue
        log_total = torch.tensor(0,dtype=torch.float64)
        for j in range(0,len(x)):
            if(y[j] > y[i]):
                log_total = torch.add(log_total,torch.exp(x[j]))
                
        if(log_total == 0): #If there is no y[j] bigger than y[i]..
            log = torch.tensor(0)
        else:
            log = torch.log(log_total)
        add_ = torch.add(x[i],-1.0*log)
        total = torch.add(total,add_)

     
    return -1.0*total


l=l_cox(x,y)