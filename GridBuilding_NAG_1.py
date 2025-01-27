import numpy as np
from itertools import product

from Functions import activation_functions_grid, d_activation_functions_grid
from CUPDataProcessing import CUP_data_splitter

'''
This code aims at building the grid in the hyperparameters space in order to allow the implementation of 
successive halvings algorithm.
The hyperparameters taken into consideration are:
- number of hidden layers, ranging from 1 to 2;
- number of units per layer, ranging from 32 to 256 in steps of powers of 2;
- type of activation function, from the dictionary activation_functions_grid;
(specifically, number of hidden layers + number of units per layer + type of activation function form the architecture
of the neural network)
- learning_rate, ranging from 1e-5 to 1e-3 in coarse grids, ranging within the decade in finer grids;
- lambda, between 1e-2 and 0;
- alpha from [0, 0.33, 0.66, 1];
- batch_size, generally from the array [1, 40, 160]
Separate grids have been built for different optimizers.
For NAG, momentum has been fixed to 0.9.
For adam, beta_1 has been fixed to 0.9, beta_2 has been fixed to 0.999 and epsilon to 1e-8.
'''

# Splitting CUP data
x_trains, target_trains, x_vals, target_vals = CUP_data_splitter.train_val_split()

# NAG Grid
N_layer = [1, 2]
N_units = [32, 64, 128, 256]

nn_architecture = []

for i in N_layer:
    unit_combo = product(N_units, repeat = i)
    for combo in unit_combo:
        nn_architecture.append({'N_layer' : i, 'N_units' : combo})

param_grid = {
    'opt_type' : ['NAG'], 
    'activation_function' : list(activation_functions_grid.keys()),
    'd_activation_function' : list(d_activation_functions_grid.keys()),
    'learning_rate' : np.logspace(-5, -3, num = 3),  
    'lambda': [0, 1e-2, 1e-4, 1e-6],
    'alpha' : [0, 0.33, 0.66, 1]
    }

batch_size = [1, 40, 160]

# Genera tutte le combinazioni
all_combinations = list(product(*param_grid.values()))

# Mette tutte le combinazioni
valid_combinations = [
    dict(zip(param_grid.keys(), values))
    for values in all_combinations
    if not (dict(zip(param_grid.keys(), values))['lambda'] == 0 and dict(zip(param_grid.keys(), values))['alpha'] != 0)
]


combinations_grid = [
    {**nn_architecture_list, **valid_combinations_list}
    for nn_architecture_list, valid_combinations_list in product(nn_architecture, valid_combinations)
]