from itertools import product

from Functions import activation_functions_grid, d_activation_functions_grid
from CUPDataProcessing import CUP_data_splitter

'''
This code aims at building the grid in the hyperparameters space in order to allow the implementation of 
a grid search algorithm.
The hyperparameters taken into consideration are:
- number of hidden layers, generally ranging from 1 to 5;
- number of units per layer, ranging from 16 to 256 in steps of powers of 2;
- type of activation function, from the dictionary activation_functions_grid;
(specifically, number of hidden layers + number of units per layer + type of activation function form the architecture
of the neural network)
- learning_rate, ranging from 1e-6 to 1e-1 in coarse grids, ranging within the decade in finer grids;
- lambda, between 1e-5 and 1e-3;
- alpha, generally from the array [0, 0.25, 0.5, 0.75, 1];
- batch_size, generally from the array [1, 16, 32, 40, 64, 80, 160]
Separate grids have been built for different optimizers.
For NAG, momentum has been fixed to 0.9.
For adam, beta_1 has been fixed to 0.9 and beta_2 has been fixed to 0.999.
'''

# Splitting CUP data
x_trains, target_trains, x_vals, target_vals = CUP_data_splitter.train_val_split()

# Adam Grid
N_layer = [1, 2, 3]
N_units = [32, 64, 128, 256]

# possible architectures
nn_architecture = []

for i in N_layer:
    unit_combo = product(N_units, repeat = i) # product returns all the possible combinations
    for combo in unit_combo:
        nn_architecture.append({'N_layer' : i, 'N_units' : combo})

# partial grid of the "training" parameters
param_grid = {
    'opt_type' : ['adam'], 
    'activation_function' : list(activation_functions_grid.keys()),
    'd_activation_function' : list(d_activation_functions_grid.keys()),
    'learning_rate' : [5e-3, 1e-3, 5e-4], 
    'lambda': [0, 1e-3, 1e-5],
    'alpha': [0.5]
    }

batch_size = [1, 40]

# Generating all possible combinations of the "training" parameters
all_combinations = list(product(*param_grid.values()))

valid_combinations = [
    dict(zip(param_grid.keys(), values))
    for values in all_combinations
]

# the whole grid has been formed combining the possible architectures with the training parameters combinations
combinations_grid = [
    {**nn_architecture_list, **valid_combinations_list}
    for nn_architecture_list, valid_combinations_list in product(nn_architecture, valid_combinations)
]