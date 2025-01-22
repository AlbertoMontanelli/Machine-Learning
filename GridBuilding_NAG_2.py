import numpy as np
from itertools import product

from Functions import activation_functions_grid, d_activation_functions_grid
from CUPDataProcessing import CUP_data_splitter

'''
MOMENTUM:
0.9 è il valore più frequentemente utilizzato e funziona bene in molti casi pratici.
0.99 può essere usato per problemi con gradienti molto rumorosi, poiché enfatizza maggiormente l'accumulo della direzione passata.
0.8-0.85 è scelto in situazioni in cui un valore più basso aiuta a stabilizzare l'ottimizzazione, specialmente nelle fasi iniziali.

SONO FISSATI:
BETA_1 = 0.9
BETA_2 = 0.999
'''

# Splitting CUP data
x_trains, target_trains, x_vals, target_vals = CUP_data_splitter.train_val_split()

# Adam Grid
N_layer = [3]
N_units = [32, 64, 128]

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
    'alpha': [0, 0.33, 0.66, 1]
    }

batch_size = [1, 40, 160]

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