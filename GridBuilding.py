import numpy as np
from itertools import product

from Functions import activation_functions_grid, d_activation_functions_grid
from CUPDataProcessing import CUP_data_splitter

'''
IPERPARAMETRI

numero epoche forse no

NR DI LAYER
NR DI UNITÀ PER LAYER
ACTIVATION_FUNCTION
numero batch
LEARNING RATE_W
LEARNING RATE_B
tipo di regolarizzazione{
LAMBDA
ALPHA
}

MOMENTUM:
0.9 è il valore più frequentemente utilizzato e funziona bene in molti casi pratici.
0.99 può essere usato per problemi con gradienti molto rumorosi, poiché enfatizza maggiormente l'accumulo della direzione passata.
0.8-0.85 è scelto in situazioni in cui un valore più basso aiuta a stabilizzare l'ottimizzazione, specialmente nelle fasi iniziali.

SONO FISSATI:
BETA_1 = 0.9
BETA_2 = 0.999
reg_type = 'elastic'
'''

# Splitting CUP data
x_trains, target_trains, x_vals, target_vals = CUP_data_splitter.train_val_split()

# baby griglia

N_layer = [1]
N_units = [16]

nn_architecture = []

for i in N_layer:
    unit_combo = product(N_units, repeat = i)
    for combo in unit_combo:
        nn_architecture.append({'N_layer' : i, 'N_units' : combo})

param_grid = {
    'opt_type' : ['none', 'NAG', 'adam'], 
    'activation_function' : list(activation_functions_grid.keys()),
    'd_activation_function' : list(d_activation_functions_grid.keys()),
    'batch_size' : [1, 64, len(x_trains[0])],
    'learning_rate' : np.logspace(-5, -3, num = 3),  
    'lambda': np.logspace(-4, -3, num = 2),
    'alpha': [0.2, 0.8]
    }

'''
N_layer = [1, 2, 3]
N_units = [16, 32, 64, 128, 256]

nn_architecture = []

for i in N_layer:
    unit_combo = product(N_units, repeat = i)
    for combo in unit_combo:
        nn_architecture.append({'N_layer' : i, 'N_units' : combo})

param_grid = {
    'opt_type' : ['none', 'NAG', 'adam'], 
    'activation_function' : list(activation_functions_grid.keys()),
    'd_activation_function' : list(d_activation_functions_grid.keys()),
    'batch_size' : [1, 16, 32, 64, len(x_trains[0])],
    'learning_rate' : np.logspace(-5, -2, num = 4),  
    'lambda': np.logspace(-4, -1, num = 4),
    'alpha': [0, 0.25, 0.5, 0.75, 1]
    }
'''

# Genera tutte le combinazioni
all_combinations = list(product(*param_grid.values()))


# Filtra le combinazioni in base alla validità delle coppie attivazione-derivata
valid_combinations = [
    dict(zip(param_grid.keys(), values))
    for values in all_combinations
]

combinations_grid = [
    {**nn_architecture_list, **valid_combinations_list}
    for nn_architecture_list, valid_combinations_list in product(nn_architecture, valid_combinations)
]
