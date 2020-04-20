import torch
import torch.nn.functional as F

def l_sim(values):
  
    total = torch.tensor(0,requires_grad=True,dtype=torch.float64)
  
    for x in range(0,len(values)):
        for y in range(x,len(values)):
            total = total + l_theta(values[x],values[y])
           
    return total

def l_theta(x,y,M = 0.1):
    return max(0,M -sim_theta(x,y)+sim_theta(x,x))

#patient x
#patient y
#sim between their modalities
def sim_theta(x,y):
  
   
    total = torch.tensor(0,requires_grad=True,dtype=torch.float64)
    for i in range(0,len(x)):
        modality_i = x[i]
        for j in range(i,len(y)):
            modality_j = y[j]
            total = total + (torch.dot(modality_i,modality_j)/(torch.norm(modality_i)*torch.norm(modality_j)+1e-10))
    
    return total


#This can be optimized
def l_cox(prediction,target):
    total = torch.tensor(0,dtype=torch.float64,requires_grad=True)
    vital_status = target["vital_status"].values #.values is needed so that the df indexes are dropped.
    days_to_death = target["days_to_death"].values #^
 
        
    for i in range(0,len(prediction)):
        if(vital_status[i] == 0): #If alive don't use it.
            continue
        log_total = torch.tensor(0,dtype=torch.float64)
        for j in range(0,len(prediction)):
          
            if(days_to_death[j]>days_to_death[i]):
                log_total = log_total + torch.exp(prediction[j])
                
        if(log_total == 0):
            log = torch.tensor(1)
        else:
            log = torch.log(log_total)
        add_ = prediction[i] + -1.0*log
        total = total +add_

     
    return -1.0 * total

#Copied from authors code
def loss(self, pred, target):

    vital_status = target["vital_status"]
    days_to_death = target["days_to_death"]
    hazard = pred["hazard"].squeeze()

    loss = F.nll_loss(pred["score"], vital_status)

    _, idx = torch.sort(days_to_death)
    hazard_probs = F.softmax(hazard[idx].squeeze()[1-vital_status.byte()])
    hazard_cum = torch.stack([torch.tensor(0.0)] + list(accumulate(hazard_probs)))
    N = hazard_probs.shape[0]
    weights_cum = torch.range(1, N)
    p, q = hazard_cum[1:], 1-hazard_cum[:-1]
    w1, w2 = weights_cum, N - weights_cum

    probs = torch.stack([p, q], dim=1)
    logits = torch.log(probs)
    ll1 = (F.nll_loss(logits, torch.zeros(N).long(), reduce=False) * w1)/N
    ll2 = (F.nll_loss(logits, torch.ones(N).long(), reduce=False) * w2)/N
    loss2 = torch.mean(ll1 + ll2)

    #loss3 = pred["ratio"].mean()
    
    return loss + loss2 #+ loss3*0.3