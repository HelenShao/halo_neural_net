import torch 
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import sys, os, time


# This function reads the rockstar file
def read_data(f_rockstar, normalize = True, tensor = True):
    n_properties = 10
    # Halo Mask Array
    PIDs = np.loadtxt(f_rockstar, usecols=41)       # Array of IDs (Halos have ID = -1)
    is_halo  = np.array([x == -1 for x in PIDs])    # Conditional array to identify halos from subhalos

    # Number of Particles Per Halo >500 
    mass_per_particle = 6.56561e+11
    m_vir    = np.loadtxt(f_rockstar, skiprows = 16, usecols = 2)[is_halo]
    n_particles = m_vir / mass_per_particle
    np_mask     = np.array([x>500 for x in n_particles])

    # Get the number of halos and properties
    n_halos = np.size(m_vir[np_mask])

    #################################### LOAD DATA ###################################
    # Define container for data 
    data = np.zeros((n_halos, n_properties), dtype=np.float32)

    # v_rms
    data[:,0] = np.loadtxt(f_rockstar, skiprows = 16, usecols = 4)[is_halo][np_mask]

    #v_max
    data[:,1] = np.loadtxt(f_rockstar, skiprows = 16, usecols = 3)[is_halo][np_mask]

    #m_vir
    #data[:,2] = np.loadtxt(f_rockstar, skiprows = 16, usecols = 2)[is_halo][np_mask]
    
    # Ratio of kinetic to potential energies T/|U|
    data[:,2] = np.loadtxt(f_rockstar, skiprows = 16, usecols = 37)[is_halo][np_mask]

    # r_vir
    data[:,3] = np.loadtxt(f_rockstar, skiprows = 16, usecols = 5)[is_halo][np_mask]

    # r_s
    data[:,4] = np.loadtxt(f_rockstar, skiprows = 16, usecols = 6)[is_halo][np_mask]

    # Velocities 
    v_x      = np.loadtxt(f_rockstar, skiprows = 16, usecols = 11)[is_halo][np_mask]
    v_y      = np.loadtxt(f_rockstar, skiprows = 16, usecols = 12)[is_halo][np_mask]
    v_z      = np.loadtxt(f_rockstar, skiprows = 16, usecols = 13)[is_halo][np_mask]
    v_mag    = np.sqrt((v_x**2) + (v_y**2) + (v_z**2))
    data[:,5] = v_mag

    # Angular momenta 
    J_x      = np.loadtxt(f_rockstar, skiprows = 16, usecols = 14)[is_halo][np_mask]
    J_y      = np.loadtxt(f_rockstar, skiprows = 16, usecols = 15)[is_halo][np_mask]
    J_z      = np.loadtxt(f_rockstar, skiprows = 16, usecols = 16)[is_halo][np_mask]
    J_mag    = np.sqrt((J_x**2) + (J_y**2) + (J_z**2))
    data[:,6] = J_mag

    # Spin
    data[:,7] = np.loadtxt(f_rockstar, skiprows = 16, usecols = 17)[is_halo][np_mask]

    # b_to_a
    data[:,8] = np.loadtxt(f_rockstar, skiprows = 16, usecols = 27)[is_halo][np_mask]

    # c_to_a
    data[:,9] = np.loadtxt(f_rockstar, skiprows = 16, usecols = 28)[is_halo][np_mask]

    ############################# NORMALIZE DATA ##############################
    # This function normalizes the input data
    def normalize_data(data):
        # Data_shape: (n_samples, n_features)
        n_halos = data.shape[0]       #n_samples
        n_properties = data.shape[1]  #n_features
        
        # Create container for normalized data
        data_norm = np.zeros((n_halos, n_properties), dtype=np.float32)
        
        # Take log10 of J_mag (m_vir removed)
        #data[:,2]  = np.log10(data[:,2]+1)
        data[:,6]  = np.log10(data[:,6]+1)
        
        for i in range(n_properties):
            mean = np.mean(data[:,i])
            std  = np.std(data[:,i])
            normalized = (data[:,i] - mean)/std
            data_norm[:,i] = normalized
        
        return(data_norm)

    # Normalize each property
    if normalize == True:
        data = normalize_data(data)

    # Convert to torch tensor
    if tensor == True:
        data = torch.tensor(data, dtype=torch.float)
    
    return data


###################################### Create Datasets ###################################
class make_Dataset(Dataset):
    
    def __init__(self, name, seed, f_rockstar):
        
        # Get the data
        halo_data = read_data(f_rockstar, normalize = True, tensor = True)
        n_properties = halo_data.shape[1]
        n_halos = halo_data.shape[0] 
        
        # shuffle the halo number (instead of 0 1 2 3...999 have a 
        # random permutation. E.g. 5 9 0 29...342)
        # n_halos = 3674
        # halo_data shape = (n_halo, number of properties) = (3674, 10)
        
        np.random.seed(seed)
        indexes = np.arange(n_halos)
        np.random.shuffle(indexes)
        
        # Divide the dataset into train, valid, and test sets
        if   name=='train':  size, offset = int(n_halos*0.8), int(n_halos*0.0)
        elif name=='valid':  size, offset = int(n_halos*0.1), int(n_halos*0.8)
        elif name=='test' :  size, offset = int(n_halos*0.1), int(n_halos*0.9)
        else:                raise Exception('Wrong name!')
        
        self.size   = size
        self.input  = torch.zeros((size, 9), dtype=torch.float) # Each input has a shape of (9,) (flattened)
        self.output = torch.zeros((size, 1), dtype=torch.float) # Each output has shape of (1,) 
        
        # do a loop over all elements in the dataset
        for i in range(size):
            j = indexes[i+offset]                 # find the halo index (shuffled)
            self.input[i] = halo_data[:,1:10][j]  # load input
            self.output[i] = halo_data[:,0][j]    # Load output (v_rms)
            
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.input[idx], self.output[idx]

    
#This function creates datasets for train, valid, test
def create_datasets(seed, f_rockstar):
    
    train_Dataset = make_Dataset('train', seed, f_rockstar)
    valid_Dataset = make_Dataset('valid', seed, f_rockstar)
    test_Dataset  = make_Dataset('test',  seed, f_rockstar)
    
    return train_Dataset, valid_Dataset, test_Dataset

############################## Load Original Data for Mean & STD ###########################
def get_halo_data():
    # Halo Mask Array
    f_rockstar   = 'Rockstar_z=0.0.txt'
    PIDs = np.loadtxt(f_rockstar, usecols=41)       # Array of IDs (Halos have ID = -1)
    is_halo  = np.array([x == -1 for x in PIDs])  # Conditional array to identify halos from subhalos
    # Number of Particles Per Halo >500 
    mass_per_particle = 6.56561e+11
    m_vir    = np.loadtxt(f_rockstar, skiprows = 16, usecols = 2)[is_halo]
    n_particles = m_vir / mass_per_particle
    np_mask     = np.array([x>500 for x in n_particles])

    # Get the number of halos and properties
    n_halos = np.size(m_vir[np_mask])
    n_properties = 11

    # Define container for data 
    data = np.zeros((n_halos, n_properties), dtype=np.float32)

    #m_vir
    data[:,0] = np.loadtxt(f_rockstar, skiprows = 16, usecols = 2)[is_halo][np_mask]

    #v_max
    data[:,1] = np.loadtxt(f_rockstar, skiprows = 16, usecols = 3)[is_halo][np_mask]

    # v_rms
    data[:,2] = np.loadtxt(f_rockstar, skiprows = 16, usecols = 4)[is_halo][np_mask]

    # r_vir
    data[:,3] = np.loadtxt(f_rockstar, skiprows = 16, usecols = 5)[is_halo][np_mask]

    # r_s
    data[:,4] = np.loadtxt(f_rockstar, skiprows = 16, usecols = 6)[is_halo][np_mask]

    # Velocities 
    v_x      = np.loadtxt(f_rockstar, skiprows = 16, usecols = 11)[is_halo][np_mask]
    v_y      = np.loadtxt(f_rockstar, skiprows = 16, usecols = 12)[is_halo][np_mask]
    v_z      = np.loadtxt(f_rockstar, skiprows = 16, usecols = 13)[is_halo][np_mask]
    v_mag    = np.sqrt((v_x**2) + (v_y**2) + (v_z**2))
    data[:,5] = v_mag

    # Angular momenta 
    J_x      = np.loadtxt(f_rockstar, skiprows = 16, usecols = 14)[is_halo][np_mask]
    J_y      = np.loadtxt(f_rockstar, skiprows = 16, usecols = 15)[is_halo][np_mask]
    J_z      = np.loadtxt(f_rockstar, skiprows = 16, usecols = 16)[is_halo][np_mask]
    J_mag    = np.sqrt((J_x**2) + (J_y**2) + (J_z**2))
    data[:,6] = J_mag

    # Spin
    data[:,7] = np.loadtxt(f_rockstar, skiprows = 16, usecols = 17)[is_halo][np_mask]

    # b_to_a
    data[:,8] = np.loadtxt(f_rockstar, skiprows = 16, usecols = 27)[is_halo][np_mask]

    # c_to_a
    data[:,9] = np.loadtxt(f_rockstar, skiprows = 16, usecols = 28)[is_halo][np_mask]

    # Ratio of kinetic to potential energies T/|U|
    data[:,10] = np.loadtxt(f_rockstar, skiprows = 16, usecols = 37)[is_halo][np_mask]

    # Take the log10 of m_vir and J_mag
    data[:,0]  = np.log10(data[:,0]+1)
    data[:,6]  = np.log10(data[:,6]+1)

    return data

# This function finds the mean and std values used to normalize data
def find_mean_std(data):
    mean = []
    std  = []    
    
    n_halos = data.shape[0]
    n_properties = data.shape[1]
        
    for i in range(n_properties):
        mean.append(np.mean(data[:,i]))
        std.append(np.std(data[:,i]))
        
    return mean, std

""" Mean and STD Values for TESTLOADER Halo Data:
    529296740000000.0  243236260000000.0
    1252.6361          162.23611
    1332.9749          183.21971
    1610.7006          206.09721
    344.88223          193.82602
    533.7007           246.06735
    5.526106e+16       5.482941e+16
    0.029700194        0.016971707
    0.59044486         0.16346285
    0.4307732          0.11112411
    0.6583044          0.055268593 """

""" Mean and STD Values for ORIGINAL Halo Data:
    540465400000000.0 274823620000000.0
    1259.5973         171.71756
    1342.5425         197.71458
    1619.2311         216.03847
    351.91614         199.3251
    546.7821          265.44467
    6.330108e+16      8.754468e+16
    0.030354135       0.017895017
    0.5825688         0.15415642
    0.42461935        0.11053585
    0.6614023         0.055876903"""
