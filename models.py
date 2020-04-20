import torch
import torch.nn as nn
import random
from loss import l_sim, l_cox
import math

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

    def forward(self,x):
       
        return self.model(x)
    
    def parameters(self):
        return self.model.parameters()

    def zero_grad(self):
        self.model.zero_grad()


class FeedForward(Net):
    def __init__(self, D_in, D_out):
        super(FeedForward, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(D_in, 100),
            torch.nn.Dropout(p = 0.3),
            torch.nn.Sigmoid(),
            torch.nn.Linear(100, D_out),

        )
    
   
    def forward(self,x):       
        return self.model(x)
  

#Squeezenet
class DCNN(Net):
    # WIP
    def __init__(self):
        super(DCNN, self).__init__()
        self.model = torch.nn.Sequential(
            torch.hub.load('pytorch/vision:v0.5.0', 'squeezenet1_0', pretrained=False),
            torch.nn.Linear(1,512)
            )
  
        

class HighwayLayer(nn.Module):
    def __init__(self, D_in):
        super(HighwayLayer, self).__init__()
        self.activation = torch.nn.Sigmoid()
        self.plain = torch.nn.Linear(D_in, D_in)
        self.gate_activation = torch.nn.Sigmoid()
        self.gate = torch.nn.Linear(D_in, D_in)

    def forward(self, x):
        h_out = self.activation(self.plain(x))
        t_out = self.gate_activation(self.gate(x))
       
        return torch.add(torch.mul(h_out, t_out), torch.mul((1.0-t_out), x))


class Highway(Net):
    def __init__(self, D_in, D_out):
        super(Highway, self).__init__()
    

        modules = []
        # 10 cycles of Highway Layer + FC
        for i in range(0, 10):
            modules.append(HighwayLayer(D_out))

        self.model = nn.Sequential(torch.nn.Linear(
            D_in, D_out),torch.nn.Sigmoid(), *modules,torch.nn.Dropout(p=0.3))

    def forward(self,x):
    
        return self.model(x)

   


class Model(Net):

    def __init__(self,clinical_input, gene_expression_input = None, mirna_input = None,wsi_input = None ):
        super(Model, self).__init__()

        if(gene_expression_input is None and mirna_input is None and clinical_input is None and wsi_input is None):
            raise Exception("No input data, no model.")
        self.model = {}
        if(gene_expression_input is not None):
            self.model["gene_expression"] = Highway(gene_expression_input, 512)
        if(mirna_input is not None):
            self.model["mirna"] = Highway(mirna_input, 512)
        if(clinical_input is not None):
            self.model["clinical"] =FeedForward(clinical_input, 512)
        if(wsi_input is not None):
            self.model["wsi"] = DCNN()
            
   
        self.hazard = torch.nn.Sequential(
            torch.nn.Dropout(p =0.3),
            torch.nn.Linear(512,1))
        self.score = torch.nn.Sequential(torch.nn.Linear(512,2),
                                         torch.nn.LogSoftmax(dim= 1)
        )
 

    def forward(self, data):
     
        feature_vectors = []
      
        #Alt feature_vectors
        for modality in data:
            network = self.model[modality]
            network.eval() #Prevent training only layers (dropout, batch n..)
            prediction = network(data[modality])
            feature_vectors.append(prediction)
            
        y = torch.stack(feature_vectors,dim = 1)
      
        y = torch.mean(y,dim = 1)
    
        self.hazard.eval()
        hazard = self.hazard(y)
        score =  self.score(y)
        
        return  {"score":score,"hazard":hazard}

    def multimodal_dropout(self, n, p=0.25):
        for i in range(0, len(n)):
            prob = random.random()
            if(prob <= p):
                n[i] = torch.zeros(512)
          

    def concatenate(self, size, feature_vectors):
        result = []
        for i in range(size):
            patient = []
            for j in feature_vectors:
                if(j[i].sum() != 0):
                    patient.append(j[i])
            result.append(patient)
        return result

    def mean(self, x):
       
        means = []
        
        for i in range(0, len(x)):
            if(len(x[i]) == 0): #For some reason all the feature vectors were dropped out 
                mean = torch.zeros(512)
            else:
                mean = torch.mean(torch.stack(x[i]), dim=0)
          
            means.append(mean)
            
        return torch.stack(means)

    def train(self, data, target, learning_rate=1e-4,n_batches = 5,multimodal = True):
        
    
        
        params = list(self.hazard.parameters())
        for x in self.model.values():
            params = params + list(x.parameters())
            
        optimizer = torch.optim.Adam(params,learning_rate)
     
        n_patients = len(target)
      
        batch_size = int(n_patients/n_batches)+1 #Assumption
        epochs = 40
        for t in range(epochs):
            print("Epoch:", t,"/",epochs)

            # Get a random order of indexes.
            permutation = torch.randperm(n_patients)
            
            #permutation = [i for i in range(n_patients)] The batch in each epoch is always the same
            for i in range(0, n_patients, batch_size):
             
       
                
                indices = permutation[i:i+batch_size]
                
                batch = target.loc[indices]
                
           

                # Call the network only if the modality is present. Exceptions?
                #The networks should be in a vector, so it is easier to iterate and to call if needed.
                feature_vectors = []
                for modality in data:
                    network = self.model[modality]
                    #Predict
                    prediction = network(data[modality][indices])
                    #Dropout with p = 0.25
                    if(multimodal):
                        self.multimodal_dropout(prediction)
                    feature_vectors.append(prediction)
                                  
              
                if(multimodal):
                    #Needed because torch.stack requires the tensors to all have the same shape.
                    #In the case of a multimodal dropout, one of the vectors is deleted, so the shape is not the same.
                    y = self.concatenate(len(feature_vectors[0]), feature_vectors)
                else:
                    y = torch.stack(feature_vectors,dim = 1)
                sim_loss = l_sim(y)
                y = self.mean(y)
               
                #y = self.dropout(y)
                y_pred = self.hazard(y)
               
                y_real = batch
                cox_loss = l_cox(y_pred, y_real)
               
                loss = cox_loss + sim_loss
                #print("l_cox:",cox_loss.data," l_sim:",sim_loss.data)
                print("Batch:",int((i+1)/batch_size), "Loss:",loss.data)
                if(cox_loss == -math.inf):
                    raise Exception("Impossibru")
                self.hazard.zero_grad()
                
                for modality in data:
                    self.model[modality].zero_grad()
            
         
                loss.backward()
                optimizer.step()

                 
                  
               

