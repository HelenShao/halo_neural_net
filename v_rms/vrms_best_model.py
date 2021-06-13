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


# In[6]:


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


# In[7]:


#################################### DATA #################################
#Create datasets
train_Dataset, valid_Dataset, test_Dataset = data.create_datasets(seed, f_rockstar)

#Create Dataloaders
train_loader = DataLoader(dataset=train_Dataset, 
                          batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_Dataset,
                          batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(dataset=test_Dataset,
                          batch_size=batch_size, shuffle=True)


# In[8]:


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


# In[9]:


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


# In[10]:


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


# In[35]:


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


# In[16]:


# Saliency from all of halo data
halo_data = data.read_data(f_rockstar, 10)
saliency_vrms = np.zeros((367, 9), dtype=np.float32)
model.eval()
i = -1

for input in halo_data[:,1:10]:
    input = input.to(device)
    
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

# Make the plot
properties = ['v_max', "t_u", 'r_vir', 'scale_radius', 'velocity',
             "J", "spin", "b_to_a", "c_to_a"]
colors = ["red", "blue", "green", "magenta", "black",
         "darkorange", "purple", "brown", "cyan"]
x = np.arange(len(properties))

i = -1
for x, color, property in zip(x, colors, properties):
    i += 1
    plt.scatter(x, saliency_avg[:,i], c=color, label = property)

plt.legend()
plt.xlabel("Property")
plt.ylabel("Saliency Values")
plt.title("NN: V_RMS")
plt.savefig("Saliency_v_rms")
plt.show()


# In[43]:


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
plt.title("NN: V_RMS")
plt.savefig("Saliency_v_rms")
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


# In[29]:


saliency_avg = saliency_avg.reshape(9,)


# In[34]:


def saliency_histogram(saliency):
    property_name = ["v_max", "T_U", "r_vir", 
                     "scale_radius", "velocity", "Angular momentum", "spin",
                     "b_to_a", "c_to_a"]
    fig = plt.figure(figsize=(77, 5))
    
    for i in range(9):
        ax = plt.subplot(1, 9, i+1)
        plt.hist(saliency[:,i], color = "purple")
        plt.title(property_name[i])
        variance = np.var(saliency[:,i])
        plt.xlabel("variance: " + str(variance) + ", mean: " + str(saliency_avg[i]))
        
    plt.savefig("SALIENCY_VRMS_HISTOGRAM")
    return fig 


# In[35]:


saliency_histogram(saliency_vrms)


# In[ ]:


# Saliency Histogram: x-axis are the bins for that property
# y axis is the frequency


# In[23]:


# Get the average saliency using Captum
import captum
from captum.attr import Saliency

# Store the saliency values for all inputs in test loader
saliency_vrms = np.zeros((367, 9), dtype=np.float32)

i = -1
saliency = Saliency(model)
for input, output in test_loader:
    input = input.to(device).requires_grad_()
    i += 1
    attribution = saliency.attribute(input)
    saliency_vrms[i] = attribution.cpu().numpy()
    
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
plt.title("NN: V_RMS (Captum)")
plt.savefig("Saliency_v_rms_captum")
plt.show()


# In[28]:


# Captum: Integrated Gradients
import captum
from captum.attr import IntegratedGradients

method_name = "integrated_grad"
# Store the gradient values for all inputs in test loader
integrated_grad = np.zeros((367, 9), dtype=np.float32)

i = -1
method = IntegratedGradients(model)
for input, output in test_loader:
    input = input.to(device).requires_grad_()
    i += 1
    attribution = method.attribute(input)
    integrated_grad[i] = attribution.cpu().detach().numpy()
    
# take abs value of each column and take average attribution for each property
method_avg = np.zeros((1,9), dtype=np.float32)
for i in range(9):
    integrated_grad[:,i] = np.abs(integrated_grad[:,i])
    method_avg[:,i]   = np.mean(integrated_grad[:,i])
    
print(method_avg)

properties = ['v_max', "t_u", 'r_vir', 'scale_radius', 'velocity',
             "J", "spin", "b_to_a", "c_to_a"]
colors = ["red", "blue", "green", "magenta", "black",
         "darkorange", "purple", "brown", "cyan"]
x = np.arange(len(properties))

i = -1
for x, color, property in zip(x, colors, properties):
    i += 1
    plt.scatter(x, method_avg[:,i], c=color, label = property)

plt.legend(bbox_to_anchor=(1.33, 1))
plt.xlabel("Property")
plt.ylabel("Saliency Values")
plt.title(method_name)
plt.savefig("VRMS_"+method_name)
plt.show()


# In[30]:


# Captum: InputXGradient
import captum
from captum.attr import InputXGradient

method_name = "InputXGradient"
# Store the gradient values for all inputs in test loader
input_x_grad = np.zeros((3674, 9), dtype=np.float32)

i = -1
method = InputXGradient(model)
for input, output in test_loader:
    input = input.to(device).requires_grad_()
    i += 1
    attribution = method.attribute(input)
    input_x_grad[i] = attribution.cpu().detach().numpy()
    
# take abs value of each column and take average attribution for each property
method_avg = np.zeros((1,9), dtype=np.float32)
for i in range(9):
    input_x_grad[:,i] = np.abs(input_x_grad[:,i])
    method_avg[:,i]   = np.mean(input_x_grad[:,i])
    
print(method_avg)

properties = ['v_max', "t_u", 'r_vir', 'scale_radius', 'velocity',
             "J", "spin", "b_to_a", "c_to_a"]
colors = ["red", "blue", "green", "magenta", "black",
         "darkorange", "purple", "brown", "cyan"]
x = np.arange(len(properties))

i = -1
for x, color, property in zip(x, colors, properties):
    i += 1
    plt.scatter(x, method_avg[:,i], c=color, label = property)

plt.legend(bbox_to_anchor=(1.33, 1))
plt.xlabel("Property")
plt.ylabel("Saliency Values")
plt.title(method_name)
plt.savefig("VRMS_"+method_name)
plt.show()


# In[33]:


# Captum: DeepLift
import captum
from captum.attr import DeepLift

method_name = "DeepLift"
# Store the gradient values for all inputs in test loader
deep_lift = np.zeros((3674, 9), dtype=np.float32)

i = -1
method = DeepLift(model)
for input, output in test_loader:
    input = input.to(device).requires_grad_()
    i += 1
    attribution = method.attribute(input)
    deep_lift[i] = attribution.cpu().detach().numpy()
    
# take abs value of each column and take average attribution for each property
method_avg = np.zeros((1,9), dtype=np.float32)
for i in range(9):
    deep_lift[:,i] = np.abs(deep_lift[:,i])
    method_avg[:,i]   = np.mean(deep_lift[:,i])
    
print(method_avg)

properties = ['v_max', "t_u", 'r_vir', 'scale_radius', 'velocity',
             "J", "spin", "b_to_a", "c_to_a"]
colors = ["red", "blue", "green", "magenta", "black",
         "darkorange", "purple", "brown", "cyan"]
x = np.arange(len(properties))

i = -1
for x, color, property in zip(x, colors, properties):
    i += 1
    plt.scatter(x, method_avg[:,i], c=color, label = property)

plt.legend(bbox_to_anchor=(1.33, 1))
plt.xlabel("Property")
plt.ylabel("Saliency Values")
plt.title(method_name)
plt.savefig("VRMS_" + method_name)
plt.show()


# In[34]:


# Captum: DeepLiftsShap
import captum
from captum.attr import DeepLiftShap

method_name = "DeepLiftShap"
# Store the gradient values for all inputs in test loader
dls = np.zeros((3674, 9), dtype=np.float32)

i = -1
method = DeepLiftShap(model)
for input in halo_data[:,1:10]:
    input = input.to(device)
    i += 1
    attribution = method.attribute(input, baselines = torch.randn(9))
    dls[i] = attribution.cpu().detach().numpy()
    
# take abs value of each column and take average attribution for each property
method_avg = np.zeros((1,9), dtype=np.float32)
for i in range(9):
    dls[:,i] = np.abs(dls[:,i])
    method_avg[:,i]   = np.mean(dls[:,i])
    
print(method_avg)

properties = ['v_max', "t_u", 'r_vir', 'scale_radius', 'velocity',
             "J", "spin", "b_to_a", "c_to_a"]
colors = ["red", "blue", "green", "magenta", "black",
         "darkorange", "purple", "brown", "cyan"]
x = np.arange(len(properties))

i = -1
for x, color, property in zip(x, colors, properties):
    i += 1
    plt.scatter(x, method_avg[:,i], c=color, label = property)

plt.legend()
plt.xlabel("Property")
plt.ylabel("Saliency Values")
plt.title(method_name)
plt.savefig("VRMS_"+method_name)
plt.show()


# In[31]:


# Captum: GradientShap
import captum
from captum.attr import GradientShap

method_name = "GradientShap"
# Store the gradient values for all inputs in test loader
grad_shap = np.zeros((3674, 9), dtype=np.float32)
baselines = torch.randn([1,9])

i = -1
method = GradientShap(model)
for input, output in test_loader:
    input = input.to(device)
    i += 1
    attribution = method.attribute(input, baselines)
    grad_shap[i] = attribution.cpu().detach().numpy()
    
# take abs value of each column and take average attribution for each property
method_avg = np.zeros((1,9), dtype=np.float32)
for i in range(9):
    grad_shap[:,i] = np.abs(grad_shap[:,i])
    method_avg[:,i]   = np.mean(grad_shap[:,i])
    
print(method_avg)

properties = ['v_max', "t_u", 'r_vir', 'scale_radius', 'velocity',
             "J", "spin", "b_to_a", "c_to_a"]
colors = ["red", "blue", "green", "magenta", "black",
         "darkorange", "purple", "brown", "cyan"]
x = np.arange(len(properties))

i = -1
for x, color, property in zip(x, colors, properties):
    i += 1
    plt.scatter(x, method_avg[:,i], c=color, label = property)

plt.legend(bbox_to_anchor=(1.33, 1))
plt.xlabel("Property")
plt.ylabel("Saliency Values")
plt.title(method_name)
plt.savefig("VRMS_"+method_name)
plt.show()


# In[38]:


# Captum: InputXGradient
import captum
from captum.attr import Occlusion

method_name = "Occlusion"
# Store the gradient values for all inputs in test loader
occlusion_array = np.zeros((3674, 9), dtype=np.float32)

i = -1
method = Occlusion(model)
for input, output in test_loader:
    input = input.to(device).requires_grad_()
    i += 1
    attribution = method.attribute(input, sliding_window_shapes=(1,))
    occlusion_array[i] = attribution.cpu().detach().numpy()
    
# take abs value of each column and take average attribution for each property
method_avg = np.zeros((1,9), dtype=np.float32)
for i in range(9):
    occlusion_array[:,i] = np.abs(occlusion_array[:,i])
    method_avg[:,i]   = np.mean(occlusion_array[:,i])
    
print(method_avg)

properties = ['v_max', "t_u", 'r_vir', 'scale_radius', 'velocity',
             "J", "spin", "b_to_a", "c_to_a"]
colors = ["red", "blue", "green", "magenta", "black",
         "darkorange", "purple", "brown", "cyan"]
x = np.arange(len(properties))

i = -1
for x, color, property in zip(x, colors, properties):
    i += 1
    plt.scatter(x, method_avg[:,i], c=color, label = property)

plt.legend(bbox_to_anchor=(1.33, 1))
plt.xlabel("Property")
plt.ylabel("Saliency Values")
plt.title(method_name)
plt.savefig("VRMS_"+method_name)
plt.show()


# In[63]:


v_max = np.zeros((367,), dtype=np.float32)
t_u = np.zeros((367,), dtype=np.float32)
r_vir = np.zeros((367,), dtype=np.float32)
J = np.zeros((367,), dtype=np.float32)
i = -1
for input, output in test_loader:
    i += 1
    v_max[i] = input[:,0].numpy()
    t_u[i]   = input[:,2].numpy()
    r_vir[i] = input[:,3].numpy()
    J[i] = input[:,6].numpy()


# In[64]:


plt.scatter(v_max, t_u, c=v_max, cmap="seismic")
plt.ylabel("T_U")
plt.xlabel("V_MAX")
plt.title("Projection of VMAX")


# In[65]:


plt.scatter(v_max, t_u, c=t_u, cmap="seismic")
plt.ylabel("T_U")
plt.xlabel("V_MAX")
plt.title("Projection of T_U")


# In[67]:


plt.scatter(r_vir, J, c=r_vir, cmap="seismic")
plt.ylabel("R_Vir")
plt.xlabel("Angular Momentum")
plt.title("Projection of R_VIR")


# In[68]:


plt.scatter(r_vir, J, c=J, cmap="seismic")
plt.ylabel("R_Vir")
plt.xlabel("Angular Momentum")
plt.title("Projection of J")


# In[ ]:




