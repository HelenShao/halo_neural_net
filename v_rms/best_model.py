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
import data

"""Trial 314 finished with value: 0.006314551457762718 and parameters: {'n_layers': 6, 
'n_units_0': 113, 'n_units_1': 223, 'n_units_2': 194, 'n_units_3': 246, 'n_units_4': 265, 'n_units_5': 209, 
'lr': 0.0012739829692055532, 'wd': 9.099651665739195e-05}. Best is trial 314 with value: 0.006314551457762718.[0m"""

################################### INPUT #####################################
# Data parameters
seed         = 4
mass_per_particle = 6.56561e+11
f_rockstar   = 'Rockstar_z=0.0.txt'
n_properties = 10

# Training Parameters
batch_size    = 1
learning_rate = 0.0012739829692055532
weight_decay  = 9.099651665739195e-05

# Architecture parameters
input_size    = 9
n_layers      = 6
out_features  = [113, 223, 194, 246, 265, 209]
f_best_model  = 'VRMS_NN_314.pt'

#################################### DATA #################################
#Create datasets
train_Dataset, valid_Dataset, test_Dataset = data.create_datasets(seed, f_rockstar, n_properties)

#Create Dataloaders
train_loader = DataLoader(dataset=train_Dataset, 
                          batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_Dataset,
                          batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(dataset=test_Dataset,
                          batch_size=batch_size, shuffle=True)

########################################### DEFINE MODEL ###########################################
def neural_net(n_layers, out_features):
    # define container for layers
    layers = []
    
    # Define initial in_features and final output_size
    in_features = 9
    output_size = 1 
    
    for i in range(n_layers):
        # Add to layers container linear layer + activation layer
        
        # Define out_features
        layers.append(nn.Linear(in_features, out_features[i]))
        layers.append(nn.LeakyReLU(0.2))
        
        # Turn in_features to out_features for next layer
        in_features = out_features[i]
        
    # last layer doesnt have activation!
    layers.append(nn.Linear(in_features, output_size))
    
    # return the model
    return nn.Sequential(*layers)

# Use GPUs 
if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda')
else:
    print('CUDA Not Available')
    device = torch.device('cpu')

# Load model
model = neural_net(n_layers, out_features).to(device)
if os.path.exists(f_best_model):
    print("loading model")
    model.load_state_dict(torch.load(f_best_model, map_location=torch.device('cpu')))
    
from torchsummary import summary
summary(model, (1,9))

# Find validation and test loss
print(f_best_model)
criterion = nn.MSELoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, 
                             weight_decay=weight_decay)
########################################## VALIDATION AND TEST ERROR ########################################
# Validation of the model.
model.eval() 
count, loss_valid = 0, 0.0
for input, TRUE in valid_loader:
    input = input.to(device=device)
    TRUE = TRUE.to(device=device)
    output = model(input)
    loss    = criterion(output, TRUE)  
    loss_valid += loss.cpu().detach().numpy()
    count += 1
loss_valid /= count

    
# TEST
count, loss_test = 0, 0.0
for input, TRUE in test_loader:
    input = input.to(device=device)
    TRUE = TRUE.to(device=device)
    output  = model(input)
    loss    = criterion(output, TRUE)  
    loss_test += loss.cpu().detach().numpy()
    count += 1
loss_test /= count
    
print('%.4e %.4e'%(loss_valid, loss_test))

################################## Predicted V_RMS vs. True ################################
original_halo_stats_log10 = {
    'm_vir': [14.698086, 0.1589484],
    'v_max': [1259.5973, 171.71756],
    'v_rms': [1342.5425, 197.71458],
    'r_vir': [1619.2311, 216.03847],
    'r_s'  : [351.91614, 199.3251],
    'vel'  : [546.7821, 265.44467],
    'ang'  : [16.614807, 0.38230082],
    'spin' : [.030354135, 0.017895017],
    'b_to_a': [0.5825688, 0.15415642],
    'c_to_a': [0.42461935, 0.11053585],
    't_u'   : [0.6614023, 0.055876903]
}

# Denormalize v_rms predicted 
true_vrms = np.zeros((367, 1), dtype = np.float32)
pred_vrms = np.zeros((367, 1), dtype = np.float32)

i = -1 
for input, TRUE in test_loader:
    i +=1 
    input = input.to(device=device)
    TRUE = TRUE.numpy()
    true_vrms[i] = TRUE
    output  = model(input)
    pred_vrms[i] = output.cpu().detach().numpy()
    
# Denormalize
mean_vrms = original_halo_stats_log10.get('v_rms')[0]
std_vrms = original_halo_stats_log10.get('v_rms')[1]
denorm_pred_vrms = (pred_vrms * std_vrms) + mean_vrms
denorm_true_vrms = (true_vrms * std_vrms) + mean_vrms

import corr_coef
# Calculate r_squared value with respect to y=x line
r_squared = corr_coef.r_squared(denorm_pred_vrms[:,0], denorm_true_vrms[:,0])

# Make plot with the r_squared value 
figname = "Predicted_vs_True:_V_RMS"
plt.scatter(denorm_pred_vrms[:,0], denorm_true_vrms[:,0])
plt.xlabel("Predicted V_RMS", fontsize=11)
plt.ylabel("True V_RMS", fontsize=11)
plt.title(figname, fontsize=11)

# y=x line
min = np.min([np.min(denorm_pred_vrms[:,0]), np.min(denorm_true_vrms)])
max = np.max([np.max(denorm_pred_vrms[:,0]), np.max(denorm_true_vrms)])
x = np.linspace(min, max, 1000)   
plt.plot(x,x, '-r')

# textbox with r_squared value
textstr = str(r_squared)
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
plt.text(np.median([max,min]), min,textstr, fontsize=12, bbox=props, horizontalalignment="center")

# Save and show figure
plt.savefig(figname)
plt.show()

############################################ COMPUTE SALIENCY ############################################
# Saliency for test data
saliency_vrms = np.zeros((367, 9), dtype=np.float32)
test_input = np.zeros((367, 9), dtype=np.float32)

model.eval()
i = -1
for input, output in test_loader:
    test_input[i] = input.numpy()
    i += 1
    input = input.to(device)
    output = output.to(device)
    
    # Get gradient and send pred through back prop
    input.requires_grad_()
    prediction = model(input)
    prediction.backward()
    
    # Print saliency
    saliency = input.grad.cpu().detach().numpy()
    # print(saliency)
    saliency_vrms[i] = saliency
    
# take abs value of each column and take average saliency for each property
saliency_avg = np.zeros((1,9), dtype=np.float32)
for i in range(9):
    saliency_vrms[:,i] = np.abs(saliency_vrms[:,i])
    saliency_avg[:,i]   = np.mean(saliency_vrms[:,i])
    
print(saliency_avg)

properties = ['v_max', "t_u", 'r_vir', 'scale_radius', 'velocity',
             "J", "spin", "b_to_a", "c_to_a"]
colors = ["red", "blue", "green", "magenta", "black",
         "darkorange", "purple", "brown", "cyan"]
x = np.arange(len(properties))

i = -1
for x, color, property in zip(x, colors, properties):
    i += 1
    plt.scatter(x, saliency_avg[:,i], c=color, label = property)

plt.legend(bbox_to_anchor=(1.33, 1))
plt.xlabel("Property")
plt.ylabel("Saliency Values")
plt.title("NN: V_RMS")
plt.savefig("Saliency_v_rms")
plt.show()
