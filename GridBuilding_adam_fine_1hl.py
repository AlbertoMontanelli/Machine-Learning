import numpy as np
from itertools import product

from Functions import activation_functions_grid, d_activation_functions_grid
from CUPDataProcessing import CUP_data_splitter

'''
This code aims at building the grid in the hyperparameters space in order to allow the implementation of 
grid building algorithm using adam optimizator.
The hyperparameters taken into consideration are:
- number of hidden layers has been fixed to 1;
- number of units per layer has been fixed to 256;
- type of activation function, from the dictionary activation_functions_grid (leaky_ReLU or tanh);
- learning_rate, ranging from 1e-5 to 1e-4.;
- lambda has been fixed to 1e-5;
- alpha has been fixed to 0.5;
- batch_size has been fixed to 1
'''

# Splitting CUP data
x_trains, target_trains, x_vals, target_vals = CUP_data_splitter.train_val_split()

# Adam Grid
N_layer = [1]
N_units = [256]

nn_architecture = []

for i in N_layer:
    unit_combo = product(N_units, repeat = i)
    for combo in unit_combo:
        nn_architecture.append({'N_layer' : i, 'N_units' : combo})

param_grid = {
    'opt_type' : ['adam'], 
    'activation_function' : list(activation_functions_grid.keys()),
    'd_activation_function' : list(d_activation_functions_grid.keys()),
    'learning_rate' : [1e5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5, 1e-4], 
    'lambda': [1e-5],
    'alpha': [0.5]
    }

batch_size = [1]

# Genera tutte le combinazioni
all_combinations = list(product(*param_grid.values()))

# Mette tutte le combinazioni
valid_combinations = [
    dict(zip(param_grid.keys(), values))
    for values in all_combinations
]

combinations_grid = [
    {**nn_architecture_list, **valid_combinations_list}
    for nn_architecture_list, valid_combinations_list in product(nn_architecture, valid_combinations)
]