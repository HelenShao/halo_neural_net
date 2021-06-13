#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[3]:


"""Trial 203 finished with value: 0.005279603336627285 and parameters: 
{'n_layers': 1, 'n_units_0': 158, 'lr': 0.0034624465593182845, 'wd': 1.9519255507635358e-05, 
'L1': 4.184157757590173e-05}. Best is trial 203 with value: 0.005279603336627285"""

################################### INPUT #####################################
# Data parameters
seed         = 4
mass_per_particle = 6.56561e+11
f_rockstar   = 'Rockstar_z=0.0.txt'
n_properties = 10

# Training Parameters
batch_size    = 1
learning_rate = 0.0034624465593182845
weight_decay  = 1.9519255507635358e-05
l1 = 4.184157757590173e-05

# Architecture parameters
input_size    = 9
n_layers      = 1
out_features  = [158]
f_best_model  = "VRMS_FEAT_SEL_203.pt"


# In[ ]:


#################################### DATA #################################
#Create datasets
train_Dataset, valid_Dataset, test_Dataset = data.create_datasets(seed, f_rockstar)

#Create Dataloaders
torch.manual_seed(seed)
train_loader = DataLoader(dataset=train_Dataset, 
                          batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_Dataset,
                          batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(dataset=test_Dataset,
                          batch_size=batch_size, shuffle=True)


# In[ ]:


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
        output = input*F.relu(self.weight)
        return output

def neural_net(n_features, n_layers, out_features):
    # define container for layers
    layers = []
    
    # Define initial in_features and final output_size
    in_features = n_features
    output_size = 1 # for last layer
    
    # Define the first layer (1-1 + activation)
    # Output of 1-1 layer is multiplied (element-wise) by the actual input (halo data), 
    # before getting fed to rest of network.
    layer1 = FirstLayer(in_features)
    layers.append(layer1)
    
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
model = neural_net(input_size, n_layers, out_features).to(device)
if os.path.exists(f_best_model):
    print("loading model")
    model.load_state_dict(torch.load(f_best_model, map_location=torch.device('cpu')))
    
from torchsummary import summary
summary(model, (1,9))


# In[5]:


# Find validation and test loss
print(f_best_model)
criterion = nn.MSELoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, 
                             weight_decay=weight_decay)

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


# In[8]:


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


# In[9]:


import corr_coef
# Calculate r_squared value with respect to y=x line
r_squared = corr_coef.r_squared(denorm_pred_vrms[:,0], denorm_true_vrms[:,0])

# Make plot with the r_squared value 
figname = "Predicted_vs_True:_V_RMS (with feature selection)"
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


# In[10]:


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
plt.figure(figsize=(11,5))
plt.bar(height=saliency_avg.reshape(9,), x=properties)
plt.xlabel("Property")
plt.ylabel("Saliency Values")
plt.title("NN: V_RMS (with feature selection)")
plt.savefig("Saliency_v_rms_ft")
plt.show()


# In[11]:


# Print Weights of First Layer
model_layers = list(model.children())
print(model_layers[0].weight)

################################# Bar Graph for Weights ###############################
weights = model_layers[0].weight.cpu().detach().numpy()

fig, ax = plt.subplots(figsize=(11,5))

# Save the chart so we can loop through the bars below.
bars = ax.bar(height= weights, x=properties, width=0.5)

# Axis formatting.
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#DDDDDD')
ax.tick_params(bottom=False, left=False)
ax.set_axisbelow(True)
ax.yaxis.grid(True, color='#868687')
ax.xaxis.grid(False)

plt.title("J: Weights of First Layer", fontsize="12")
plt.ylabel("Weight Values", fontsize="14")
plt.xlabel("Features", fontsize="14")

fig.tight_layout()
plt.savefig("J_Weights")


# In[ ]:




