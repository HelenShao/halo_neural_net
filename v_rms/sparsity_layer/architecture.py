import torch 
import torch.nn as nn
import numpy as np
import sys, os, time
import optuna
import math
import torch.nn.functional as F

# First Layer connects each input feature to one output
class FirstLayer(nn.Module):
    def __init__(self, n_features): 
        super(FirstLayer, self).__init__()
        self.n_features = n_features
        # Define weight as a nn.Parameter to be trained
        # Initialize weights as ones
        self.weight = nn.Parameter(torch.ones(n_features))  
            
    def forward(self, input):
        # Instead of multiplying the weight with halo input, multiply weight with 
        # a row of ones, get the relu of that, and multiply that with the actual input 
        output = input*F.leaky_relu(self.weight)
        return output

# Define the model architecture 
def neural_net(trial, input_size, n_min, n_max, min_layers, max_layers):
    # define container for layers
    layers = []
    
    # Allow optuna to find the number of layers
    n_layers = trial.suggest_int("n_layers", min_layers, max_layers)
    
    # Define initial in_features and final output_size
    in_features = input_size
    output_size = 1    # for last layer
    
    # Define the first layer (1-1 + activation)
    # Output of 1-1 layer is multiplied (element-wise) by the actual input (halo data), 
    # before getting fed to rest of network.
    layer1 = FirstLayer(in_features)
    layers.append(layer1)  
    
    for i in range(n_layers):
        # Add to layers container linear layer + activation layer
        
        # Define out_features
        out_features = trial.suggest_int("n_units_{}".format(i), n_min, n_max)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.LeakyReLU(0.2))
        
        # Turn in_features to out_features for next layer
        in_features = out_features
        
    # last layer doesnt have activation!
    layers.append(nn.Linear(in_features, output_size))
    
    # return the model
    return nn.Sequential(*layers)
