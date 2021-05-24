import numpy as np
import matplotlib.pyplot as plt
import sys, os
import torch
from pysr import pysr, best, get_hof
import data
import corr_coef

# Load normalized halo data 
f_rockstar = "Rockstar_z=0.0.txt"
halo_data = data.read_data(f_rockstar, normalize=True, tensor=False)

# load input variables into container for PYSR
input_data = np.zeros((3674, 2), dtype = np.float32)
input_data[:,1:11] = halo_data

# Load output variable
v_rms = halo_data[:,0]

####################################### PYSR ###################################
property_names   = ["r_vir", "J"]
binary_operators = ["plus", "mult", "sub", "pow", "div"]
unary_operators  = ["exp", "logm", "tan", "sin", "cos"]

x = input_data    # normalized input
y = v_rms         # normalized v_rms (output)
cores = 1         # request more cores on slurm

# Initiate Symbolic regression with PYSR
equations = pysr(x, y, niterations=1000, binary_operators= binary_operators,
                 unary_operators= unary_operators, variable_names = property_names, procs = cores)

# Get best equation and plot it 
print(best(equation))

pred = v_max + np.sin(0.24053788*t_u) # Best equation
plt.figure(figsize = (18,10))
plt.scatter(pred, v_rms)  # Plot predicted vs true

# plot y=x line
min = np.min(pred)
max = np.max(pred)
x = np.linspace(min, max, 1000)
plt.plot(x, x, '-r')

# Compute r_squared value
r_squared_value = corr_coef.r_squared(v_rms, pred)
textstr = str(r_squared_value)
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
plt.text(np.median([max,min]), min, textstr, fontsize=14, bbox=props, horizontalalignment="center") 
plt.title("$v_(max) + sin(0.24053788*t_u)$", fontsize = 18)
plt.ylabel("True v_rms", fontsize=14)
plt.xlabel("PYSR_Pred v_rms", fontsize=14)
plt.savefig("PYSR_BEST")
plt.show()
