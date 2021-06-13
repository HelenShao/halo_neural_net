import numpy as np
import matplotlib.pyplot as plt
import sys, os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler, WeightedRandomSampler
import optuna
import architecture, data 

################################# Objective Function #############################
class objective(object):
    def __init__(self, input_size, n_min, n_max, min_layers,
                 max_layers, f_rockstar, device, num_epochs, seed, batch_size): 
        
        self.input_size         = input_size
        self.max_layers         = max_layers
        self.min_layers         = min_layers
        self.n_min              = n_min
        self.device             = device
        self.num_epochs         = num_epochs
        self.seed               = seed 
        self.n_max              = n_max
        self.f_rockstar         = f_rockstar
        self.batch_size         = batch_size
    
    def __call__(self, trial):
        
        #Files for saving results and best model
        f_text_file   = 'VRMS_FEAT_SEL_%d.txt'%(trial.number)
        f_best_model  = 'VRMS_FEAT_SEL_%d.pt'%(trial.number)
        f_all_models  = 'VRMS_Models.txt'

        # Generate the model.
        model = architecture.neural_net(trial, input_size, n_min, 
                                         n_max, min_layers, max_layers).to(device)

        # Define loss function
        criterion = nn.MSELoss()
        
        # Define optimizer
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        wd = trial.suggest_float("wd", 1e-5, 1e3, log=True)
        L1 = trial.suggest_float("L1", 1e-5, 1e3, log=True)
        # L1 = 0 # First train without L1
        
        optimizer = getattr(optim, "Adam")(model.parameters(), lr=lr, weight_decay = wd)
    
        #Create datasets
        train_Dataset, valid_Dataset, test_Dataset = data.create_datasets(seed, f_rockstar)

        #Create Dataloaders
        train_loader = DataLoader(dataset=train_Dataset, 
                                  batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(dataset=valid_Dataset, 
                                  batch_size=batch_size, shuffle=True)
        test_loader  = DataLoader(dataset=test_Dataset,  
                                  batch_size=batch_size, shuffle=True)
        # Train the model
        min_valid = 1e40
        for epoch in range(num_epochs):
            model.train()
            count, loss_train = 0, 0.0
            for input, TRUE in train_loader:        
                # Forward Pass
                input = input.to(device=device)
                TRUE = TRUE.to(device=device)
                output = model(input)
                
                # Get the sparsity loss for L1 regularization
                model_children = list(model.children())
                w1 = model_children[0].weight
                sparse_loss = torch.mean(F.relu(w1))
                
                # Compute loss
                loss    = criterion(output, TRUE) + L1*sparse_loss
                loss_train += loss.cpu().detach().numpy()

                # Backward propogation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validation of the model.
            model.eval() 
            count, loss_valid = 0, 0.0
            for input, TRUE in valid_loader:
                input = input.to(device=device)
                TRUE = TRUE.to(device=device)
                output = model(input)
                
                # Get the sparsity loss for L1 regularization
                model_children = list(model.children())
                w1 = model_children[0].weight
                sparse_loss = torch.mean(torch.abs(w1))
                
                # Compute loss
                loss    = criterion(output, TRUE) + L1*sparse_loss
                loss_valid += loss.cpu().detach().numpy()
                count += 1
            loss_valid /= count

            if loss_valid<min_valid:  
                min_valid = loss_valid
                torch.save(model.state_dict(), f_best_model)
            f = open(f_text_file, 'a')
            f.write('%d %.5e %.5e\n'%(epoch, loss_valid, min_valid))
            f.close()

        # Record the loss and the sparsity for plot
        f = open(f_all_models, 'a')
        
        model_layers = list(model.children())
        weights1 = model_layers[0].weight
        sparsity = torch.sum(F.relu(weights1))/input_size
        
        f.write('%.5e %.5e\n'%(min_valid, sparsity))
        f.close()
        
        return min_valid

##################################### INPUT #######################################
# Data Parameters
#n_halos      = 3674
#n_properties = 10
seed         = 4
mass_per_particle = 6.56561e+11 
f_rockstar = "Rockstar_z=0.0.txt"

# Training Parameters
num_epochs    = 500
batch_size    = 64

# Architecture Parameters
input_size = 9          # Number of input features 
n_min = 2               # Minimum number of neurons in hidden layers
n_max = 1000            # Maximum number of neurons in hidden layers
min_layers = 1          # Minimum number of hidden layers
max_layers = 5          # Maximum number of hidden layers

# Optuna Parameters
n_trials   = 1000 
study_name = 'VRMS_FEAT_SEL_params'
n_jobs     = 1

############################## Start OPTUNA Study ###############################

# Use GPUs if avaiable
if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda')
else:
    print('CUDA Not Available')
    device = torch.device('cpu')

if __name__ == "__main__":
    
    # define the optuna study and optimize it
    objective = objective(input_size, n_min, n_max, min_layers,
                 max_layers, f_rockstar, device, num_epochs, seed, batch_size)
    
    # !! Optimization direction = minimize valid_loss !!
    # Change initial sample of parameters to 300 
    sampler = optuna.samplers.TPESampler(n_startup_trials=30) 
    study = optuna.create_study(study_name=study_name, sampler=sampler,direction="minimize", load_if_exists = True)
    study.optimize(objective, n_trials=n_trials, n_jobs = n_jobs)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    # Print parameters of the best trial
    trial = study.best_trial
    print("Best trial: number {}".format(trial.number))
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
