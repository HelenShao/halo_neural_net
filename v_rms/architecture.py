import torch 
import torch.nn as nn
import numpy as np
import sys, os, time
import optuna

# Define the model architecture 
def neural_net(trial, input_size, n_min, n_max, min_layers, max_layers):
    # define container for layers
    layers = []
    
    # Allow optuna to find the number of layers
    n_layers = trial.suggest_int("n_layers", min_layers, max_layers)
    
    # Define initial in_features and final output_size
    in_features = input_size
    output_size = 1 
    
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
